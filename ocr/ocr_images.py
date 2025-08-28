from typing import List

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from .config import OCR_CONFIG
from .models import get_model, get_training
from .dataset import get_dataset_post_process

class OCRImages:
    def __init__(self):
        self.images = []
        self.cfg    = OCR_CONFIG
        
        # take model
        self.model  = get_model(self.cfg)
        
        # take training
        self.training = get_training(self.cfg)
        
        # take dataset
        self.dataset, self.post_process = get_dataset_post_process(self.cfg)
        print("OCRImages initialized")
        pass

    def clear_cache(self):
        self.images = []
    
    def add_image(self, image_content):
        self.images.append(image_content)
        
    def run(self) -> List[str]:
        if "Pytorch" in self.cfg.model_name:
            return self.__run_pytorch()
        
    def run_training(self, debug=True):
        print("START TRAINING")
        self.training(debug)
        print("END TRAINING")
    
    def __run_pytorch(self) -> List[str]:
        loader_data = DataLoader(
            self.dataset(self.images), 
            batch_size=self.cfg.batch_size, 
            num_workers=self.cfg.num_workers,
            shuffle=False)
        
        results = []
        with torch.no_grad():
            for _, image_batch in enumerate(loader_data):
                image_batch = image_batch.to(self.cfg.device)
                logits = self.model(image_batch)
                texts = self.post_process(logits)
                results.extend(texts)
                
        return results
    
    