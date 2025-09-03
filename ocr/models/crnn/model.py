import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, efficientnet_b4, mobilenet_v3_small
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
    def __init__(self, vocab_size, rnn_hidden_size=256, dropout=0.5, lstm_dropout=0.2):
        super().__init__()
        # self.vocab_size = vocab_size
        # self.chars = chars
        # self.char2idx, self.idx2char = self.char_idx()
        
        self.convlayer = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Conv2d(512, 512, 3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        
        self.mapSeq = nn.Linear(512 * 2, 256)

        self.lstm_0 = nn.LSTM(256, 256, bidirectional=True, dropout=lstm_dropout)
        self.lstm_1 = nn.LSTM(512, 256, bidirectional=True, dropout=lstm_dropout)

        self.out = nn.Linear(512, vocab_size)
        
    def forward(self, x):
        x = self.convlayer(x)
        x = x.permute(3, 0, 1, 2)
        x = x.view(x.size(0), x.size(1), -1)
        x = self.mapSeq(x)

        x, _ = self.lstm_0(x)
        x, _ = self.lstm_1(x)

        x = self.out(x)
        return x
    
def get_model_checkpoint():
    if platform.system() == "Windows":
        path_file = "/".join(os.path.abspath(__file__).split("\\")[:-1])
    else:
        path_file = "/".join(os.path.abspath(__file__).split("/")[:-1])
    
    model = CRNN(25, 256) # fix size dictionary is 25
    model.load_state_dict(torch.load(path_file + "/save/best.bin", 
                                     map_location=torch.device('cpu' if not torch.cuda.is_available() else 'cuda'),
                                     weights_only=True
                                     ))
    model.eval()
    return model