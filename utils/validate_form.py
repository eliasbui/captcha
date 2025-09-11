from pydantic import BaseModel
from fastapi.responses import JSONResponse

class StatusMessage:
    MESS_DONE          = ""
    MESS_TIME_LOWER    = "Time must be greater than 0"
    MESS_FAILED_ACCESS = "Failed to access the link"
    MESS_NONE_COOKIES  = "Cookies must not be empty"
    MESS_EMPTY_URL     = "URL must not be empty"
    MESS_EMPTY_IMAGE   = "Image must not be empty"
    MESS_OCR_FAILED    = "OCR model fail"
    
    MESS_EMPTY_IDS     = "Images must not be empty"
    MESS_EMPTY_IMG     = "Images must not be empty"
    MESS_EMPTY_LABELS  = "Labels must not be empty"
    MESS_NOT_EQUAL     = "Number of images and labels must be equal"

    MESS_DB_ACCESS     = "Failed to access the database"
    MESS_DB_FAILED     = "Failed to insert data the database"
    MESS_NO_IDX        = "Some image id not found in the database"
    MESS_DECODE_FAILED = "Failed to decode the image"
    MESS_TRAIN_FAILED  = "Failed to train the model"
    
    MESS_NOT_ENOUGH    = "Number of images must be greater than input"
class ValidateRecognitionResponse:
    mapping_status_code = {
        StatusMessage.MESS_DONE: 200,
        StatusMessage.MESS_TIME_LOWER: 400,
        StatusMessage.MESS_FAILED_ACCESS: 400,
        StatusMessage.MESS_NONE_COOKIES: 400,
        StatusMessage.MESS_EMPTY_URL: 400,
        StatusMessage.MESS_OCR_FAILED: 400,
        StatusMessage.MESS_EMPTY_IDS: 400,
        StatusMessage.MESS_EMPTY_LABELS: 400,
        StatusMessage.MESS_DB_FAILED: 400,
        StatusMessage.MESS_NO_IDX: 400,
        StatusMessage.MESS_EMPTY_IMAGE: 400,
        StatusMessage.MESS_TRAIN_FAILED: 400,
        StatusMessage.MESS_DECODE_FAILED: 400,
    }
    
    def __init__(self, message: str = "", error: str = "", content: dict = {}):
        self.message = message
        self.error   = error
        self.content = content
        
    def __str__(self):
        return f"Message: {self.message}, Error: {self.error}"
    
    def get_response(self):
        content_res = {
            k:v for k, v in self.content.items()
        }
        content_res["message"] = self.message,
        content_res["error"]   = self.error,
        return JSONResponse(
            status_code=self.mapping_status_code[self.message],
            content=content_res
        )

class RequestImageQualification(BaseModel):
    image_url: str
    time: int
    cookies: str
    
class RequestImageCaptureQualification(BaseModel):
    image_base64: str

class ResultValidation(BaseModel):
    list_id: list
    list_label: list
    
class TrainingValidation(BaseModel):
    list_images_base64: list
    list_label: list
    debug: bool = True