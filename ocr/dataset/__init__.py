
from .dataset_v1 import CAPTCHADatasetInferenceV1 as CaptchaV1
from .dataset_v1 import post_process_v1 as CaptchaV1_post_process

def get_dataset_post_process(config: dict):
    if config.dataset_name == "CaptchaV1":
        return CaptchaV1, CaptchaV1_post_process
    else:
        raise ValueError("Dataset not found")