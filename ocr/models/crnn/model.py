import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b4
from ocr.dataset.dataset_v1 import CAPTCHADatasetTraining, read_json_file
import os
import platform

class AttentionModule(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionModule, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size * 2, 1)
        
    def forward(self, rnn_outputs):
        # rnn_outputs: (batch_size, seq_len, hidden_size * 2)
        attention_weights = F.softmax(self.attention(rnn_outputs), dim=1)
        context = torch.sum(attention_weights * rnn_outputs, dim=1)
        return context, attention_weights

class CRNN(nn.Module):
    def __init__(self, num_chars, rnn_hidden_size=256, dropout=0.1, use_attention=True):
        super(CRNN, self).__init__()
        self.num_chars = num_chars
        self.rnn_hidden_size = rnn_hidden_size
        self.dropout = dropout
        self.use_attention = use_attention

        base = efficientnet_b4(weights=None)
        # Modify first conv layer for grayscale
        base.features[0][0] = nn.Conv2d(1, 48, kernel_size=3, stride=1, padding=1, bias=False)
        self.cnn_p1 = nn.Sequential(*list(base.features[:6]))  # First 6 layers
        
        # Conv layer with residual connections
        self.cnn_p2 = nn.Sequential(
            nn.Conv2d(160, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        )
        # Adaptive pooling to handle variable heights
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))
            
        self.rnn1 = nn.LSTM(input_size=128, 
                            hidden_size=128,
                            bidirectional=True, 
                            batch_first=True,
                            dropout=dropout if dropout > 0 else 0)
        
        self.rnn2 = nn.LSTM(input_size=128 * 2, 
                            hidden_size=128,
                            bidirectional=True, 
                            batch_first=True,
                            dropout=dropout if dropout > 0 else 0)
        
        # Optional
        if self.use_attention:
            self.attention = AttentionModule(rnn_hidden_size)
        # Ouput
        self.output = nn.Linear(256, num_chars)
        
    def forward(self, x):
        x = self.cnn_p1(x)
        x = self.cnn_p2(x)
        x = self.adaptive_pool(x)  # (B, C=256, H=1, W=time)
        
        # Reshape for RNN: (B, W, C)
        x = x.squeeze(2).permute(0, 2, 1)  # (B, time, 256)
        
        # RNN layers 
        rnn1_out, _ = self.rnn1(x)
        rnn2_out, _ = self.rnn2(rnn1_out)
        
        # Add residual connection
        if rnn1_out.size(-1) == rnn2_out.size(-1):
            rnn_out = rnn1_out + rnn2_out
        else:
            rnn_out = rnn2_out
        
        output = self.output(rnn_out)
        # Permute for CTC loss: (time, batch, classes)
        output = output.permute(1, 0, 2)
        return output
    
def get_model_checkpoint():
    if platform.system() == "Windows":
        path_file = "/".join(os.path.abspath(__file__).split("\\")[:-1])
    else:
        path_file = "/".join(os.path.abspath(__file__).split("/")[:-1])
    
    try:
        # Use the exact same mapping that was used to train the model
        idx2char = read_json_file()  # This should load from mapping_char.json
        char2idx = {v: k for k, v in idx2char.items()}
        checkpoint_vocab_size = len(idx2char)
        model = CRNN(checkpoint_vocab_size, 256)
        
        # Determine device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model.load_state_dict(torch.load(path_file + "/save/best.bin", 
                                        map_location=device,
                                        weights_only=True
                                        ))
        # Move model to the correct device
        model = model.to(device)
        model.eval()
    except Exception as e:
        print("❌ Error LOADING model.")
        raise
    return model