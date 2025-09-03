import os
import json
import platform
from PIL import Image 
from io import BytesIO

import cv2
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset

def preprocess(image):
    """
    Adaptive preprocessing that handles different image conditions
    """
    # Convert to grayscale
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Check if inversion is needed (dark text on light background)
    # Compute mean pixel value in the center region where text likely exists
    h, w = image.shape
    center_region = image[h//4:3*h//4, w//4:3*w//4]
    mean_value = np.mean(center_region)
    
    # If mean is high (light background), invert
    if mean_value > 127:
        image = 255 - image
    
    # Apply Otsu's threshold (automatically finds optimal threshold)
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Normalize
    binary = binary.astype(np.float32) / 255.0
    return binary
class CAPTCHADataset(Dataset):
    def __init__(self, data_dir, image_fns):
        self.data_dir = data_dir
        self.image_fns = image_fns
        
    def __len__(self):
        return len(self.image_fns)
    
    def __getitem__(self, index):
        image_fn = self.image_fns[index]
        image_fp = os.path.join(self.data_dir, image_fn)
        image = cv2.imread(image_fp, cv2.IMREAD_UNCHANGED)[:,:,3]
        image = 255 - image
        image = preprocess(image)

        image = self.transform(image)
        text = image_fn.split(".")[0]
        return image, text
    
    def transform(self, image):
        
        transform_ops = transforms.Compose([
            transforms.ToTensor(),
        ])
        return transform_ops(image)
    
class CAPTCHADatasetTraining(Dataset):
    def __init__(self, data_dir, image_fns, label_fns, type = "train", target_size=(50, 130), preprocess = True):
        self.data_dir  = data_dir
        self.image_fns = image_fns
        self.label_fns = label_fns
        self.target_height, self.target_width = target_size 
        self.preprocess = preprocess
        self.type      = type
        
    def __len__(self):
        return len(self.image_fns)
    
    def __getitem__(self, index):
        image_fn = self.image_fns[index]
        
        image_fp = os.path.join(self.data_dir, image_fn)
        image = cv2.imread(image_fp, cv2.IMREAD_UNCHANGED)
        
        if image is None:
            print(f"Skipping corrupted file: {image_fp}")
            return self.__getitem__((index + 1) % len(self.label_fns))  # Load the next image
        
        if len(image.shape) > 2:
            image = image[:,:,-1]
        
        image = 255 - image
        if self.preprocess:
            image = preprocess(image)
        image = self.transform(image)
        
        label = str(self.label_fns[index])
        return image, label
    
    def transform(self, image):
        transform_ops = transforms.Compose([
        # transforms.ToPILImage(),  # Convert NumPy array to PIL Image
        # transforms.Resize((self.target_height, self.target_width), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),  # Convert to PyTorch tensor
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
        ])
        return transform_ops(image)

# from training dataset
def read_json_file():
    if platform.system() == "Windows":
        path_file = "/".join(os.path.abspath(__file__).split("\\")[:-1])
    else:
        path_file = "/".join(os.path.abspath(__file__).split("/")[:-1])
        
    with open(path_file + "/mapping_char.json", "r") as f:
        data = json.load(f)
        
    new_dict = {
        int(k):v for k, v in data.items()
    }
        
    return new_dict

def create_mapping_char(new_dict):
    if platform.system() == "Windows":
        path_file = "/".join(os.path.abspath(__file__).split("\\")[:-1])
    else:
        path_file = "/".join(os.path.abspath(__file__).split("/")[:-1])
        
    idx2char_old = read_json_file()
    with open(path_file + "/mapping_char_old.json", "w") as f:
        json.dump(idx2char_old, f)
        
    dict_str = {
        str(k):v for k, v in new_dict.items()
    }
    with open(path_file + "/mapping_char.json", "w") as f:
        json.dump(dict_str, f)
    return new_dict

idx2char = read_json_file()
char2idx = {v:k for k,v in idx2char.items()}

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

def post_process_v1(text_batch_logits):
    text_batch_tokens = F.softmax(text_batch_logits, 2).argmax(2) # [T, batch_size]
    text_batch_tokens = text_batch_tokens.numpy().T # [batch_size, T]
    text_batch_tokens_new = []
    for text_tokens in text_batch_tokens:
        text = [idx2char[idx] for idx in text_tokens]
        text = "".join(text)
        text = correct_prediction(text)
        text_batch_tokens_new.append(text)
    return text_batch_tokens_new    

class CAPTCHADatasetInferenceV1(Dataset):
    def __init__(self, image_buffers):
        self.image_buffers = image_buffers
        
    def __len__(self):
        return len(self.image_buffers)
    
    def __getitem__(self, index):
        image_content = self.image_buffers[index]
        png_buffer = BytesIO(image_content)
        png_image = Image.open(png_buffer)
        image = np.array(png_image)
        
        if len(image.shape) >= 3:
            image = image[:,:,-1]

        image = 255 - image
        image = preprocess(image)

        image = self.transform(image)
        return image
    
    def transform(self, image):
        transform_ops = transforms.Compose([
            transforms.ToTensor(),
        ])
        return transform_ops(image)