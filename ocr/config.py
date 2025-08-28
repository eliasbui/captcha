import torch

class OCR_CONFIG:
    # OCR Config
    batch_size   = 8
    model_name   = "CRNN_Pytorch"
    dataset_name = "CaptchaV1"
    num_workers  = 3
    device       = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    