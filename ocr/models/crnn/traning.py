import os
import cv2
import yaml
import glob
import string
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
from tqdm import tqdm
from typing import List, Tuple
import multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from ocr.dataset.dataset_v1 import CAPTCHADatasetTraining
from ocr.dataset.dataset_v1 import read_json_file, create_mapping_char
from .model import CRNN

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TrainConfig:
    def __init__(self):
        if platform.system() == "Windows":
            path_file = "/".join(os.path.abspath(__file__).split("\\")[:-1])
        else:
            path_file = "/".join(os.path.abspath(__file__).split("/")[:-1])
            
        with open(path_file + "/config.yml", 'r') as file:
            config = yaml.safe_load(file)
        
        for k, v in config.items():
            setattr(self, k, v)

def weights_init(m):
    classname = m.__class__.__name__
    if type(m) in [nn.Linear, nn.Conv2d, nn.Conv1d]:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def remove_git_keep(list_path):
    try:
        list_path.remove(".gitkeep")
    except:
        list_path = list_path
    return list_path

def get_test_set(CFG):
    image_fns_test = os.listdir(CFG.TEST_PATH)
    image_fns_test = remove_git_keep(image_fns_test)
    label_test = [image_fn.split(".")[0].lower() for image_fn in image_fns_test]
    return image_fns_test, label_test

def get_train_set(CFG):
    image_fns_train = os.listdir(CFG.TRAIN_PATH)
    image_fns_train = remove_git_keep(image_fns_train)
    label_train = [image_fn.split(".")[0].lower() for image_fn in image_fns_train]
    return image_fns_train, label_train

def encode_text_batch(text_batch, char2idx):
    text_batch_targets_lens = [len(text) for text in text_batch]
    text_batch_targets_lens = torch.LongTensor(text_batch_targets_lens)
    text_batch_concat = "".join(text_batch)
    text_batch_targets = [char2idx[c] for c in text_batch_concat]
    text_batch_targets = torch.LongTensor(text_batch_targets)
    return text_batch_targets, text_batch_targets_lens

def compute_loss(text_batch, text_batch_logits, criterion, char2idx):
    """
    text_batch: list of strings of length equal to batch size
    text_batch_logits: Tensor of size([T, batch_size, num_classes])
    """
    text_batch_logps = F.log_softmax(text_batch_logits, 2) # [T, batch_size, num_classes]  
    text_batch_logps_lens = torch.full(size=(text_batch_logps.size(1),), 
                                       fill_value=text_batch_logps.size(0), 
                                       dtype=torch.long).to(DEVICE) # [batch_size] 
    text_batch_targets, text_batch_targets_lens = encode_text_batch(text_batch, char2idx)
    loss = criterion(text_batch_logps, text_batch_targets, text_batch_logps_lens, text_batch_targets_lens)

    return loss

# def decode_predictions(text_batch_logits, idx2char):
#     text_batch_tokens = F.softmax(text_batch_logits, 2).argmax(2) # [T, batch_size]
#     text_batch_tokens = text_batch_tokens.numpy().T # [batch_size, T]
#     text_batch_tokens_new = []
#     for text_tokens in text_batch_tokens:
#         text = [idx2char[idx] for idx in text_tokens]
#         text = "".join(text)
#         text_batch_tokens_new.append(text)
#     return text_batch_tokens_new

def decode_predictions(text_batch_logits, idx2char, blank_idx=0):
    """
    Greedy CTC-style decode: argmax per timestep, collapse repeats, remove blank.
    Works if idx2char has int keys or string keys.
    """
    text_batch_tokens = F.softmax(text_batch_logits, 2).argmax(2)  # [T, batch_size]
    text_batch_tokens = text_batch_tokens.cpu().numpy().T  # [batch_size, T]
    text_batch_tokens_new = []
    for seq in text_batch_tokens:
        prev = None
        chars = []
        for idx in seq:
            if idx == prev:
                prev = idx
                continue
            prev = idx
            if int(idx) == blank_idx:
                continue
            # handle mapping keys that might be ints or strings
            ch = None
            if isinstance(idx2char, dict):
                idx_key = int(idx)
                if idx_key in idx2char:
                    ch = idx2char[idx_key]
                elif str(idx_key) in idx2char:
                    ch = idx2char[str(idx_key)]
            if ch is None:
                ch = "?"
            chars.append(ch)
        text_batch_tokens_new.append("".join(chars))
    return text_batch_tokens_new

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        self.best_weights = model.state_dict().copy()

def validate_model(model, val_loader, criterion, char2idx, device, idx2char):
    """Validate the model and return average validation loss and accuracy."""
    model.eval()
    val_losses = []
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for image_batch, text_batch in val_loader:
            text_batch_logits = model(image_batch.to(device))
            loss = compute_loss(text_batch, text_batch_logits, criterion, char2idx)
            
            if not (np.isnan(loss.item()) or np.isinf(loss.item())):
                val_losses.append(loss.item())
            
            # Get predictions for accuracy calculation
            text_batch_pred = decode_predictions(text_batch_logits.cpu(), idx2char)

            predictions.extend(text_batch_pred)
            actuals.extend(text_batch)
    
    model.train()
    avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
    val_accuracy = accuracy_score(actuals, predictions) if predictions else 0.0
    
    return avg_val_loss, val_accuracy

def get_training(debug: bool = True):
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
    else:
        print("GPU Not found. Using CPU - training will be slower")
    
    CFG = TrainConfig()
    
    image_fns_test, label_fns_test = get_test_set(CFG)
    image_fns_train, label_fns_train = get_train_set(CFG)
    
    trainset = CAPTCHADatasetTraining(CFG.TRAIN_PATH, image_fns_train, label_fns_train, 'train') 
    testset = CAPTCHADatasetTraining(CFG.TEST_PATH, image_fns_test, label_fns_test, 'test')
    if debug:
        for i in range(5):
            image, label = trainset[i]
            plt.imshow(image.squeeze(), cmap='gray')
            plt.title(label)
            plt.show()
    
    if debug:
        print("test size: ", len(image_fns_test))
        print("train size: ", len(image_fns_train))
        
    if platform.system() == "Windows":
        path_file = "/".join(os.path.abspath(__file__).split("\\")[:-1])
    else:
        path_file = "/".join(os.path.abspath(__file__).split("/")[:-1])
    
    label_fns  = label_fns_train + label_fns_test
    label_fns  = "".join(label_fns)
    letters    = sorted(list(set(list(label_fns))))
    vocabulary = ["-"] + letters
    
    idx2char   = {k:v for k,v in enumerate(vocabulary, start=0)}
    idx2char_checkpoint = read_json_file()
    
    # Convert to sets for easier comparison
    current_vocab = set(idx2char.values())
    checkpoint_vocab = set(idx2char_checkpoint.values())
    
    # Find new characters
    new_characters = current_vocab - checkpoint_vocab
    removed_characters = checkpoint_vocab - current_vocab
    
    if debug:
        print(f"\n=== Vocabulary Checking...===")
        print(f"Checkpoint vocabulary size: {len(checkpoint_vocab)}")
        print(f"Current dataset vocabulary size: {len(current_vocab)}")
        print(f"Checkpoint characters: {''.join(sorted(checkpoint_vocab))}")
        print(f"Current dataset characters: {''.join(sorted(current_vocab))}")
        
        if new_characters:
            print(f"\n🆕 NEW CHARACTERS FOUND: {new_characters}")
            print(f"New characters: {''.join(sorted(new_characters))}")
        else:
            print("\n✅ No new characters found")

    # Use current vocabulary to maintain model compatibility
    idx2char = idx2char.copy()
    char2idx = {v: k for k, v in idx2char.items()}
    num_chars = len(char2idx)
    
    try:
        # Split training data into train/validation
        train_size = int(0.85 * len(trainset))
        val_size = len(trainset) - train_size
        train_subset, val_subset = torch.utils.data.random_split(trainset, [train_size, val_size])
        
        train_loader = DataLoader(train_subset, 
                                  batch_size=CFG.BATCH_SIZE, 
                                  num_workers=2, 
                                  persistent_workers=True, 
                                  pin_memory=True if torch.cuda.is_available() else False, 
                                  shuffle=True
                                  )
        val_loader = DataLoader(val_subset, 
                                batch_size=CFG.BATCH_SIZE, 
                                num_workers=2, 
                                persistent_workers=True, 
                                pin_memory=True if torch.cuda.is_available() else False, 
                                shuffle=False
                                )
        test_loader = DataLoader(testset, 
                                 batch_size=CFG.BATCH_SIZE, 
                                 num_workers=2, 
                                 persistent_workers=True, 
                                 pin_memory=True if torch.cuda.is_available() else False, 
                                 shuffle=False
                                 )

        crnn = CRNN(num_chars=num_chars, rnn_hidden_size=CFG.RNN_HIDDEN_SIZE)
        crnn.apply(weights_init)
        crnn = crnn.to(DEVICE)

        criterion = nn.CTCLoss(blank=0)
        optimizer = optim.AdamW(
                                crnn.parameters(), 
                                lr=CFG.LR, 
                                weight_decay=CFG.WEIGHT_DECAY
                                )
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(crnn.parameters(), max_norm=5.0)
        
        # Learning rate scheduler to help convergence
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Initialize early stopping
        early_stopping = EarlyStopping(patience=CFG.EARLY_STOPPING_PATIENCE, 
                                       min_delta=CFG.EARLY_STOPPING_MIN_DELTA, 
                                       restore_best_weights=True
                                       )
        
        epoch_losses = []
        val_losses = []
        val_accuracies = []
        iteration_losses = []
        num_updates_epochs = []
        best_val_loss = float('inf')
        
        for epoch in tqdm(range(1, CFG.EPOCHS+1), desc="Epochs total:"):
            epoch_loss_list = [] 
            num_updates_epoch = 0
            crnn.train()
            # Training phase
            for image_batch, text_batch in tqdm(train_loader, leave=False, desc="Epoch train:{}".format(epoch)):
                optimizer.zero_grad()
                text_batch_logits = crnn(image_batch.to(DEVICE))
                loss = compute_loss(text_batch, text_batch_logits, criterion=criterion, char2idx=char2idx)
                iteration_loss = loss.item()

                if np.isnan(iteration_loss) or np.isinf(iteration_loss):
                    continue
                
                num_updates_epoch += 1
                iteration_losses.append(iteration_loss)
                epoch_loss_list.append(iteration_loss)
                loss.backward()
                nn.utils.clip_grad_norm_(crnn.parameters(), CFG.CLIP_NORM)
                optimizer.step()

            # Validation phase
            val_loss, val_accuracy = validate_model(crnn, val_loader, criterion, char2idx, DEVICE, idx2char)
            
            epoch_loss = np.mean(epoch_loss_list)
            epoch_losses.append(epoch_loss)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            num_updates_epochs.append(num_updates_epoch)
            
            print(f"Epoch: {epoch}")
            print(f"  Train Loss: {epoch_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Accuracy: {val_accuracy:.4f}")
            print(f"  NumUpdates: {num_updates_epoch}")
            
            # if val_loss < best_val_loss and epoch > 10:
            #     best_val_loss = val_loss
            #     torch.save(crnn.state_dict(), path_file + f"/save/best_{val_loss:.2f}_{val_accuracy:.2f}.bin")
            #     print(f"  💾 Saved best model (val_loss: {val_loss:.4f})")
            
            lr_scheduler.step(val_loss)
            
            # Early stopping check
            if early_stopping(val_loss, crnn) and epoch > 50:
                print(f"\n🛑 Early stopping triggered after {epoch} epochs!")
                print(f"Best validation loss: {early_stopping.best_loss:.4f}")
                
                # Save the best model after early stopping is triggered
                torch.save(crnn.state_dict(), path_file + "/save/best.bin")
                print(f"  💾 Saved best model (val_loss: {early_stopping.best_loss:.4f})")
                break
        # # Save best model
        # torch.save(crnn.state_dict(), path_file + f"/save/best_{val_loss:.2f}_{val_accuracy:.2f}.bin")
        # print(f"  💾 Saved best model (val_loss: {val_loss:.4f})")
        
        # Final evaluation on test set
        print("\n📊 Final evaluation on test set...")
        results_test = pd.DataFrame(columns=['actual', 'prediction'])
        test_loader = DataLoader(testset, batch_size=16, num_workers=1, shuffle=False)
        
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
        results_test['prediction_corrected'] = results_test['prediction']
        test_accuracy = accuracy_score(results_test['actual'], results_test['prediction_corrected'])
        
        print(f"Test Accuracy: {test_accuracy:.4f}")
        # Plot training history
        if debug:
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.plot(epoch_losses, label='Train Loss')
            plt.plot(val_losses, label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Training and Validation Loss')
            
            plt.subplot(1, 3, 2)
            plt.plot(val_accuracies)
            plt.xlabel('Epoch')
            plt.ylabel('Validation Accuracy')
            plt.title('Validation Accuracy')
            
            plt.subplot(1, 3, 3)
            plt.plot(iteration_losses)
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title('Training Loss per Iteration')
            
            plt.tight_layout()
            plt.show()
        
        return ""
    except Exception as e:
        create_mapping_char(idx2char_checkpoint)
        print(str(e))
        return str(e)