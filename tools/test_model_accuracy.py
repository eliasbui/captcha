import os
import platform
import torch
import torch.nn as nn
import pandas as pd
import numpy as np  # Missing import
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
    decode_predictions
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
    
    model_path = path_file + CFG.BEST_MODEL_PATH
    if not os.path.exists(model_path):
        print(f"❌ Saved model not found: {model_path}")
        raise
    
    # Load character mapping from checkpoint file location
    mapping_path = path_file + CFG.MAPPING_CHARACTER_PATH
    
    try:
        # Use the exact same mapping that was used to train the model
        if os.path.exists(mapping_path):
            idx2char = read_json_file()  # This should load from mapping_char.json
        else:
            print(f"❌ Mapping file not found: {mapping_path}")
            return
            
        char2idx = {v: k for k, v in idx2char.items()}
        checkpoint_vocab_size = len(idx2char)
        
        print(f"Checkpoint vocabulary size: {checkpoint_vocab_size}")
        print(f"Characters: {''.join(sorted(idx2char.values()))}")
        
        # Get testset
        image_fns_test, label_fns_test = get_test_set(CFG)
        print(f"Test dataset size: {len(image_fns_test)}")
        # Filtered data
        testset = CAPTCHADatasetTraining(CFG.TEST_PATH, image_fns_test, label_fns_test, 'test')
        test_loader = DataLoader(testset, batch_size=CFG.BATCH_SI ZE, num_workers=1, shuffle=False)
        
        # Initialize model 
        crnn = CRNN(num_chars=checkpoint_vocab_size, rnn_hidden_size=CFG.RNN_HIDDEN_SIZE)
        crnn.load_state_dict(torch.load(model_path, 
                                       map_location=torch.device('cpu' if not torch.cuda.is_available() else 'cuda'),
                                       weights_only=True
                                       ))
        crnn = crnn.to(DEVICE)
        print("✅ Model loaded successfully")
        # Run inference
        results_test = pd.DataFrame(columns=['actual', 'prediction'])
        print("🔍 Running inference on test set...")
        
        crnn.eval()
        with torch.no_grad():
            for image_batch, text_batch in tqdm(test_loader, leave=True):
                text_batch_logits = crnn(image_batch.to(DEVICE)) # [T, batch_size, num_classes==num_features]
                text_batch_pred = decode_predictions(text_batch_logits.cpu(), idx2char)
                df = pd.DataFrame(columns=['actual', 'prediction'])
                df['actual'] = text_batch
                df['prediction'] = text_batch_pred
                results_test = pd.concat([results_test, df])
        
        results_test = results_test.reset_index(drop=True)
        raw_accuracy = accuracy_score(results_test['actual'], results_test['prediction'])

        print("\n=== Test Results ===")
        print(f"📊 Total test samples: {len(results_test)}")
        print(f"🎯 Raw accuracy: {raw_accuracy:.4f} ({raw_accuracy*100:.2f}%)")

        print("\n=== Sample Predictions ===")
        sample_results = results_test.head(10)
        for idx, row in sample_results.iterrows():
            status = "✅" if row['actual'] == row['prediction'] else "❌"
            print(f"{status} Actual: '{row['actual']}' | Predicted: '{row['prediction']}'")
        
        # Show incorrect predictions
        incorrect = results_test[results_test['actual'] != results_test['prediction']]
        if len(incorrect) > 0:
            print("\n=== Incorrect Predictions (showing first 10) ===")
            for idx, row in incorrect.head(10).iterrows():
                print(f"❌ Expected: '{row['actual']}' | Got: '{row['prediction']}'")
        
        # Save detailed results
        results_file = "test_accuracy_results.csv"
        results_test.to_csv(results_file, index=False)
        print(f"\n💾 Detailed results saved to: {results_file}")
        
        return raw_accuracy
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return None

if __name__ == "__main__":
    test_model_accuracy()