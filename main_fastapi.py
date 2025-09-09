from io import BytesIO
from PIL import Image 
import sqlite3
import base64

import cv2
import ssl
import uuid
import requests
from urllib.request import Request, urlopen
import uvicorn
import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import matplotlib.pyplot as plt

from utils.adapter_http import get_legacy_session
from utils.validate_form import *
from utils.db_connect import *
from ocr import OCRImages
from utils import *

app = FastAPI()
db  = LogDataBase()
session = get_legacy_session()
# Create a secure SSL context
ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS)  

@app.get("/")
async def root():
    return JSONResponse(
        status_code=200,
        content={"message": "Home Page"}
    )

@app.post("/captcha_image")
async def read_item(item: RequestImageQualification):
    # message = StatusMessage.MESS_DONE
    # if item.time < 1:
    #     message = StatusMessage.MESS_TIME_LOWER
    # if not item.cookies or len(item.cookies) == 0:
    #     message = StatusMessage.MESS_NONE_COOKIES
    # if not item.image_url or len(item.image_url) == 0:
    #     message = StatusMessage.MESS_EMPTY_URL
    # ocr_instance = OCRImages()
    # if len(message) != 0:
    #     return ValidateRecognitionResponse(message=message).get_response()

    print(f"Request received successfully: {item}")
    message = StatusMessage.MESS_DONE
    if item.time < 1:
        message = StatusMessage.MESS_TIME_LOWER
        print(f"Validation failed: time < 1")
    if not item.cookies or len(item.cookies) == 0:
        message = StatusMessage.MESS_NONE_COOKIES
        print(f"Validation failed: empty cookies")
    if not item.image_url or len(item.image_url) == 0:
        message = StatusMessage.MESS_EMPTY_URL
        print(f"Validation failed: empty URL")
    
    print(f"Validation message: '{message}'")
    
    ocr_instance = OCRImages()
    if len(message) != 0:
        print(f"Returning error response: {message}")
        return ValidateRecognitionResponse(message=message).get_response()
    
    print("Starting image processing...")
    error        = ""
    list_image   = []
    list_id      = []
    list_label   = []
    list_img_path = []
    
    for index in range(int(item.time)):
        print(f"Processing image {index + 1}/{item.time}")
        try:
            print(f"Making request to: {item.image_url}")
            req = Request(item.image_url)
            req.add_header('Cookie', f"cookies={item.cookies}")
            response = urlopen(
                req, context=ssl_context
                )
            content_buffer = response.read()            
            image_id = str(uuid.uuid4())
            print(f"Successfully downloaded image, ID: {image_id}")
        except Exception as e:
            print(f"Error downloading image: {e}")
            message        = StatusMessage.MESS_FAILED_ACCESS
            error          = str(e)
            return ValidateRecognitionResponse(
                message=message, 
                error=error,
            ).get_response()
            
        print("Processing image data...")
        png_buffer     = BytesIO(content_buffer)
        png_image      = Image.open(png_buffer)
        pixel_data     = np.array(png_image)[:,:,3]

        # Debug: Check image properties
        print(f"Image mode: {png_image.mode}")
        print(f"Image size: {png_image.size}")
        print(f"Image format: {png_image.format}")
        print(f"Image shape: {pixel_data.shape}")
        
        pixel_data     = np.array(png_image)[:,:,3]
        print(f"Alpha channel shape: {pixel_data.shape}")
        print(f"Alpha channel min/max: {pixel_data.min()}/{pixel_data.max()}")
        print(f"Alpha channel unique values: {np.unique(pixel_data)}")
        
        image_path     = f"./image_crawl/ocr_images/{image_id}.png"
        list_image.append(pixel_data)
        list_id.append(image_id)
        ocr_instance.add_image(content_buffer) 
        cv2.imwrite(image_path, pixel_data)
        list_img_path.append(f"{image_id}.png")
        print(f"Image processed and saved to: {image_path}")
        
    print("Running OCR...")
    try:
        list_label = ocr_instance.run()     
        data = {
            'list_id'    : list_id,
            'list_label' : list_label,
            'list_path'  : list_img_path
        }
        print(f"OCR completed successfully: {list_label}")
    except Exception as e:
        print(f"OCR failed: {e}")
        data       = {
            'list_id'    : [],
            'list_label' : [],
            'list_path'  : []
        }
        message    = StatusMessage.MESS_OCR_FAILED
        error      = str(e)
        # Return early if OCR fails completely
        return ValidateRecognitionResponse(message=message, error=error).get_response()
        
    print("Checking database...")
    # Only check database if we have valid IDs
    if len(data['list_id']) > 0:
        list_id_have = db.check_idx(data['list_id'])
        if len(list_id_have) != 0:
            for idx in list_id_have:
                idx_find = data['list_id'].index(idx)
                data['list_label'].remove(data['list_label'][idx_find])
                data['list_path'].remove(data['list_path'][idx_find])
                data['list_id'].remove(idx)
        
        print("Saving to database...")
        try:
            db.add_images(data['list_id'], data['list_path'])
            db.add_preds(data['list_id'], data['list_label'])
            print("Database operations completed successfully")
        except Exception as e:
            print(f"Database error: {e}")
            message        = StatusMessage.MESS_DB_FAILED
            error          = str(e)
            return ValidateRecognitionResponse(message=message).get_response()
    else:
        print("No valid data to save to database")
    
    print("Clearing cache and returning response...")
    ocr_instance.clear_cache()
    return ValidateRecognitionResponse(
        message=message, 
        error=error,
        content=data
    ).get_response()
    
@app.post("/captcha_recap_image")
async def read_recap_item(item: RequestImageCaptureQualification):
    message = StatusMessage.MESS_DONE
    if not item.image_base64 or len(item.image_base64) == 0:
        message = StatusMessage.MESS_EMPTY_IMAGE
    ocr_instance = OCRImages()
    if len(message) != 0:
        return ValidateRecognitionResponse(message=message).get_response()
    
    error          = ""
    list_image     = []
    list_id        = []
    list_label     = []
    list_img_path  = []
    
    # decode image
    content_buffer = item.image_base64   
    content_buffer = base64.b64decode(content_buffer)  
    image_id       = str(uuid.uuid4())
    png_buffer     = BytesIO(content_buffer)
    png_image      = Image.open(png_buffer)
    
    # preprocess image 
    pixel_data     = np.array(png_image)
    pixel_data     = cv2.cvtColor(pixel_data, cv2.COLOR_BGR2GRAY)
    pixel_data     = 255 - pixel_data
    content_buffer = cv2.imencode(".png", pixel_data)[1].tobytes()
    image_path     = f"./image_crawl/ocr_images/{image_id}.png"
    
    list_image.append(pixel_data)
    list_id.append(image_id)
    ocr_instance.add_image(content_buffer) 
    cv2.imwrite(image_path, pixel_data)
    list_img_path.append(f"{image_id}.png") 
    
    try:
        list_label = ocr_instance.run()     
        data = {
            'list_id'    : list_id,
            'list_label' : list_label,
            'list_path'  : list_img_path
        }
    except Exception as e:
        data       = {
            'list_id'    : [],
            'list_label' : [],
            'list_path'  : []
        }
        message    = StatusMessage.MESS_OCR_FAILED
        error      = str(e)
        return ValidateRecognitionResponse(message=message,
                                           error=error).get_response()
        
    list_id_have = db.check_idx(data['list_id'])
    if len(list_id_have) != 0:
        for idx in list_id_have:
            idx_find = data['list_id'].index(idx)
            data['list_label'].remove(data['list_label'][idx_find])
            data['list_path'].remove(data['list_path'][idx_find])
            data['list_id'].remove(idx)
    
    try:
        db.add_images(data['list_id'], data['list_path'])
        db.add_preds(data['list_id'], data['list_label'])
    except Exception as e:
        message        = StatusMessage.MESS_DB_FAILED
        error          = str(e)
        return ValidateRecognitionResponse(message=message,
                                           error=error).get_response()
    
    ocr_instance.clear_cache()
    return ValidateRecognitionResponse(
        message=message, 
        error=error,
        content=data
    ).get_response()

@app.post("/captcha_retrain_model")
def running_train(item: TrainingValidation):
    ocr_instance = OCRImages()
    
    if len(item.list_images_base64) == 0:
        return ValidateRecognitionResponse(
            message=StatusMessage.MESS_EMPTY_IMG
        ).get_response()
    if len(item.list_label) == 0:
        return ValidateRecognitionResponse(
            message=StatusMessage.MESS_EMPTY_LABELS
        ).get_response()
    if len(item.list_images_base64) != len(item.list_label):
        return ValidateRecognitionResponse(
            message=StatusMessage.MESS_NOT_EQUAL
        ).get_response()
        
    for elm in item.list_images_base64:
        if len(elm) == 0:
            return ValidateRecognitionResponse(
                message=StatusMessage.MESS_EMPTY_IMG
            ).get_response()
    
    for elm in item.list_label:
        if len(elm) == 0:
            return ValidateRecognitionResponse(
                message=StatusMessage.MESS_EMPTY_LABELS
            ).get_response()
                    
    # decode image
    try:
        for index_content in range(len(item.list_images_base64)):
            content_buffer = base64.b64decode(item.list_images_base64[index_content])  
            image_buff = Image.open(BytesIO(content_buffer))
            image_buff = np.array(image_buff)
            cv2.imwrite(f'./image_crawl/train_images/{item.list_label[index_content]}.png', image_buff)
    except Exception as e:
        return ValidateRecognitionResponse(
            message=StatusMessage.MESS_DECODE_FAILED,
            error=str(e)
        ).get_response()
    
    try:
        ocr_instance.run_training(debug=item.debug)
    except Exception as e:
        return ValidateRecognitionResponse(
            message=StatusMessage.MESS_TRAIN_FAILED,
            error=str(e)
        ).get_response()
    
    return ValidateRecognitionResponse(
        message=StatusMessage.MESS_DONE
    ).get_response()

if __name__ == "__main__":
    uvicorn.run("main_fastapi:app", host="0.0.0.0", port=8000, reload=True)