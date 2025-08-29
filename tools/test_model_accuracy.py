import os
import platform
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr.dataset.dataset_v1 import CAPTCHADatasetTraining, read_json_file
from ocr.models.crnn.model import CRNN
from ocr.models.crnn.traning import (
    TrainConfig, get_test_set, remove_git_keep, 
    decode_predictions, correct_prediction
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_model_accuracy():
    """Test accuracy of existing best.bin model"""
    print("🧪 Testing Model Accuracy...")
    
    # Load config
    CFG = TrainConfig()
    
    # Get model path first
    if platform.system() == "Windows":
        path_file = "/".join(os.path.abspath(__file__).split("\\")[:-2])
    else:
        path_file = "/".join(os.path.abspath(__file__).split("/")[:-2])
    
    model_path = path_file + "/ocr/models/crnn/save/best.bin"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    try:
        # First, load the checkpoint to see what vocabulary size it expects
        checkpoint = torch.load(model_path, 
                               map_location=torch.device('cpu' if not torch.cuda.is_available() else 'cuda'),
                               weights_only=True)
        
        # Get the vocabulary size from the checkpoint's linear2 layer
        checkpoint_vocab_size = checkpoint['linear2.weight'].shape[0]
        print(f"Checkpoint expects vocabulary size: {checkpoint_vocab_size}")
        
        # Load character mapping
        mapping_path = path_file + "/ocr/dataset/mapping_char.json"
        if os.path.exists(mapping_path):
            idx2char = read_json_file()
        else:
            print(f"❌ Mapping file not found: {mapping_path}")
            return
        
        # If mapping has different size than checkpoint, use only the first N characters
        if len(idx2char) != checkpoint_vocab_size:
            print(f"⚠️ Mapping size ({len(idx2char)}) != checkpoint size ({checkpoint_vocab_size})")
            print("Using first {} characters from mapping".format(checkpoint_vocab_size))
            
            # Create new mapping with only the required number of characters
            sorted_keys = sorted(idx2char.keys(), key=int)[:checkpoint_vocab_size]
            idx2char = {k: idx2char[k] for k in sorted_keys}
        
        char2idx = {v: k for k, v in idx2char.items()}
        print(f"Using vocabulary size: {len(idx2char)}")
        print(f"Characters: {''.join(sorted(idx2char.values()))}")
        
        # Get test dataset
        image_fns_test, label_fns_test = get_test_set(CFG)
        print(f"Test dataset size: {len(image_fns_test)}")
        
        # Filter test data to only include samples with valid characters
        valid_samples = []
        valid_labels = []
        checkpoint_chars = set(idx2char.values())
        
        for img_path, label in zip(image_fns_test, label_fns_test):
            if set(label).issubset(checkpoint_chars):
                valid_samples.append(img_path)
                valid_labels.append(label)
            else:
                invalid_chars = set(label) - checkpoint_chars
                print(f"⚠️ Skipping '{label}' - contains invalid characters: {invalid_chars}")
        
        print(f"Valid test samples: {len(valid_samples)} / {len(image_fns_test)}")
        
        if not valid_samples:
            print("❌ No valid test samples found!")
            return
        
        # NOW initialize model with the exact vocabulary size from checkpoint
        crnn = CRNN(num_chars=checkpoint_vocab_size, rnn_hidden_size=256)
        crnn.load_state_dict(checkpoint)
        crnn = crnn.to(DEVICE)
        crnn.eval()
        print("✅ Model loaded successfully")
        
        # Create test dataset with filtered data
        testset = CAPTCHADatasetTraining(CFG.TEST_PATH, valid_samples, valid_labels, 'test')
        test_loader = DataLoader(testset, batch_size=16, num_workers=1, shuffle=False)
        
        # Run inference
        results_test = pd.DataFrame(columns=['actual', 'prediction'])
        
        print("🔍 Running inference on test set...")
        with torch.no_grad():
            for image_batch, text_batch in tqdm(test_loader, desc="Testing"):
                text_batch_logits = crnn(image_batch.to(DEVICE))
                text_batch_pred = decode_predictions(text_batch_logits.cpu(), idx2char)
                
                # Create batch results
                df = pd.DataFrame({
                    'actual': text_batch,
                    'prediction': text_batch_pred
                })
                results_test = pd.concat([results_test, df], ignore_index=True)
        
        # Calculate accuracy
        results_test['prediction_corrected'] = results_test['prediction'].apply(correct_prediction)
        
        raw_accuracy = accuracy_score(results_test['actual'], results_test['prediction'])
        corrected_accuracy = accuracy_score(results_test['actual'], results_test['prediction_corrected'])
        
        # Display results
        print(f"\n=== Test Results ===")
        print(f"📊 Total test samples: {len(results_test)}")
        print(f"🎯 Raw accuracy: {raw_accuracy:.4f} ({raw_accuracy*100:.2f}%)")
        print(f"✨ Corrected accuracy: {corrected_accuracy:.4f} ({corrected_accuracy*100:.2f}%)")
        
        # Show some examples
        print(f"\n=== Sample Predictions ===")
        sample_results = results_test.head(10)
        for idx, row in sample_results.iterrows():
            status = "✅" if row['actual'] == row['prediction_corrected'] else "❌"
            print(f"{status} Actual: '{row['actual']}' | Raw: '{row['prediction']}' | Corrected: '{row['prediction_corrected']}'")
        
        # Show incorrect predictions
        incorrect = results_test[results_test['actual'] != results_test['prediction_corrected']]
        if len(incorrect) > 0:
            print(f"\n=== Incorrect Predictions (showing first 10) ===")
            for idx, row in incorrect.head(10).iterrows():
                print(f"❌ Expected: '{row['actual']}' | Got: '{row['prediction_corrected']}'")
        
        # Save detailed results
        results_file = "test_accuracy_results.csv"
        results_test.to_csv(results_file, index=False)
        print(f"\n💾 Detailed results saved to: {results_file}")
        
        return corrected_accuracy
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return None

if __name__ == "__main__":
    test_model_accuracy()