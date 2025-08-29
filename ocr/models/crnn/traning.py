import os
import glob
import string
import platform
from PIL import Image
import multiprocessing as mp
from typing import List, Tuple

import cv2
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
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
    label_test = [image_fn.split(".")[0] for image_fn in image_fns_test]
    return image_fns_test, label_test

def get_train_set(CFG):
    image_fns_train = os.listdir(CFG.TRAIN_PATH)
    image_fns_train = remove_git_keep(image_fns_train)
    label_train = [image_fn.split(".")[0] for image_fn in image_fns_train]
    return image_fns_train, label_train

def encode_text_batch(text_batch, char2idx):
    text_batch_targets_lens = [len(text) for text in text_batch]
    text_batch_targets_lens = torch.IntTensor(text_batch_targets_lens)
    text_batch_concat = "".join(text_batch)
    text_batch_targets = [char2idx[c] for c in text_batch_concat]
    text_batch_targets = torch.IntTensor(text_batch_targets)
    return text_batch_targets, text_batch_targets_lens

def compute_loss(text_batch, text_batch_logits, criterion, char2idx):
    """
    text_batch: list of strings of length equal to batch size
    text_batch_logits: Tensor of size([T, batch_size, num_classes])
    """
    text_batch_logps = F.log_softmax(text_batch_logits, 2) # [T, batch_size, num_classes]  
    text_batch_logps_lens = torch.full(size=(text_batch_logps.size(1),), 
                                       fill_value=text_batch_logps.size(0), 
                                       dtype=torch.int32).to(DEVICE) # [batch_size] 
    text_batch_targets, text_batch_targets_lens = encode_text_batch(text_batch, char2idx)
    loss = criterion(text_batch_logps, text_batch_targets, text_batch_logps_lens, text_batch_targets_lens)

    return loss

def decode_predictions(text_batch_logits, idx2char):
    text_batch_tokens = F.softmax(text_batch_logits, 2).argmax(2) # [T, batch_size]
    text_batch_tokens = text_batch_tokens.numpy().T # [batch_size, T]
    text_batch_tokens_new = []
    for text_tokens in text_batch_tokens:
        text = [idx2char[idx] for idx in text_tokens]
        text = "".join(text)
        text_batch_tokens_new.append(text)
    return text_batch_tokens_new

def remove_duplicates(text):
    if len(text) <= 5:
        letters = [text[i] for i in range(len(text))]
        return "".join(letters)
    
    while len(text) > 5:
        isCut = False
        # Find the first duplicate character
        for i in range(len(text) - 1):
            if text[i] == text[i+1]:
                # Remove the duplicate character
                text = text[:i] + text[i+1:]
                isCut = True
                break
                
        if isCut == False:
            if len(text) > 5:
                break
                
    letters = [text[i] for i in range(len(text))]
    return "".join(letters)

def correct_prediction(word):
    parts = word.split("-")
    parts = [remove_duplicates(part) for part in parts]
    corrected_word = "".join(parts)
    return corrected_word

def get_training(debug: bool = True):
    CFG = TrainConfig()
    
    image_fns_test, label_fns_test = get_test_set(CFG)
    image_fns_train, label_fns_train = get_train_set(CFG)
    
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
        print(f"\n=== Vocabulary Analysis ===")
        print(f"Checkpoint vocabulary size: {len(checkpoint_vocab)}")
        print(f"Current dataset vocabulary size: {len(current_vocab)}")
        print(f"Checkpoint characters: {''.join(sorted(checkpoint_vocab))}")
        print(f"Current dataset characters: {''.join(sorted(current_vocab))}")
        
        if new_characters:
            print(f"\n🆕 NEW CHARACTERS FOUND: {new_characters}")
            print(f"New characters: {''.join(sorted(new_characters))}")
        else:
            print("\n✅ No new characters found")
            
        # if removed_characters:
        #     print(f"\n🗑️ REMOVED CHARACTERS: {removed_characters}")
        #     print(f"Removed characters: {''.join(sorted(removed_characters))}")
        # else:
        #     print("\n✅ No characters removed")
    
    # Option 1: Use only checkpoint vocabulary (recommended)
    if new_characters and debug:
        print(f"\n⚠️ WARNING: Found {len(new_characters)} new characters!")
        print("Options:")
        print("1. Use only checkpoint vocabulary (recommended)")
        print("2. Expand vocabulary and retrain from scratch")
        
        # Validate dataset against checkpoint vocabulary
        invalid_labels = []
        for label in label_fns_train + label_fns_test:
            if not set(label).issubset(checkpoint_vocab):
                invalid_chars = set(label) - checkpoint_vocab
                invalid_labels.append((label, invalid_chars))
        
        if invalid_labels:
            print(f"\n❌ Found {len(invalid_labels)} labels with invalid characters:")
            for label, chars in invalid_labels[:10]:  # Show first 10
                print(f"  '{label}' contains: {chars}")
            if len(invalid_labels) > 10:
                print(f"  ... and {len(invalid_labels) - 10} more")
    
    # Use checkpoint vocabulary to maintain model compatibility
    idx2char = idx2char.copy()
    char2idx = {v: k for k, v in idx2char.items()}
    num_chars = len(char2idx)
    
    try:
        trainset = CAPTCHADatasetTraining(CFG.TRAIN_PATH, image_fns_train, label_fns_train, 'train') 
        testset = CAPTCHADatasetTraining(CFG.TEST_PATH, image_fns_test, label_fns_test, 'test')
        train_loader = DataLoader(trainset, batch_size=CFG.BATCH_SIZE, num_workers=3, shuffle=True)
        test_loader = DataLoader(testset, batch_size=CFG.BATCH_SIZE, num_workers=3, shuffle=False)

        crnn = CRNN(num_chars=num_chars, rnn_hidden_size=CFG.RNN_HIDDEN_SIZE)
        crnn.apply(weights_init)
        crnn = crnn.to(DEVICE)
        if not new_characters:
            crnn.load_state_dict(torch.load(path_file + "/save/best.bin", 
                                            map_location=torch.device('cpu' if not torch.cuda.is_available() else 'cuda'),
                                            weights_only=True
                                            ))
        criterion = nn.CTCLoss(blank=0)
        optimizer = optim.Adam(crnn.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=5)
        
        epoch_losses = []
        iteration_losses = []
        num_updates_epochs = []
        for epoch in tqdm(range(1, CFG.EPOCHS+1), desc="Epochs total:"):
            epoch_loss_list = [] 
            num_updates_epoch = 0
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

            epoch_loss = np.mean(epoch_loss_list)
            print("Epoch:{}    Loss:{}    NumUpdates:{}".format(epoch, epoch_loss, num_updates_epoch))
            epoch_losses.append(epoch_loss)
            num_updates_epochs.append(num_updates_epoch)
            lr_scheduler.step(epoch_loss)
        
        results_test = pd.DataFrame(columns=['actual', 'prediction'])
        test_loader = DataLoader(testset, batch_size=16, num_workers=1, shuffle=False)
        with torch.no_grad():
            for image_batch, text_batch in tqdm(test_loader, leave=True):
                text_batch_logits = crnn(image_batch.to(DEVICE)) # [T, batch_size, num_classes==num_features]
                text_batch_pred = decode_predictions(text_batch_logits.cpu(), idx2char)
                df = pd.DataFrame(columns=['actual', 'prediction'])
                df['actual'] = text_batch
                df['prediction'] = text_batch_pred
                results_test = pd.concat([results_test, df])
        results_test = results_test.reset_index(drop=True)
        results_test['prediction_corrected'] = results_test['prediction'].apply(correct_prediction)
        test_accuracy = accuracy_score(results_test['actual'], results_test['prediction_corrected'])
        
        if debug == False:
            time_str = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
            if test_accuracy > CFG.ACC_THRESHOLD:
                os.rename(path_file + "/save/best.bin", f"{path_file}/save/last_before_{time_str}.bin")
                torch.save(crnn.state_dict(), path_file + "/save/best.bin")
        print("Test Accuracy: ", test_accuracy)
        print("Done training")
        return ""
    except Exception as e:
        create_mapping_char(idx2char_checkpoint)
        print(str(e))
        return str(e)