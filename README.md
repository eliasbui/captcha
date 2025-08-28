## Contents

- [Contents](#contents)
- [OCR API](#ocr-api)
- [Endpoints](#endpoints)
  - [1. Submit Captcha Image](#1-submit-captcha-image)
  - [2. Submit Recap Captcha Image](#2-submit-recap-captcha-image)
  - [3. Retrain OCR Model](#3-retrain-ocr-model)
- [Status Messages and Codes](#status-messages-and-codes)
  - [Status Messages](#status-messages)
  - [Status Code Mapping](#status-code-mapping)
  - [Summary](#summary)
- [ISSUE](#issue)

## OCR API 

### Build Image

```bash
docker build -t {your_docker_repo}/ocr_captcha:{version} .

docker run -p 8000:8000 {your_docker_repo}/ocr_captcha:{version}
```

### Run Image

```bash
docker pull namsiunhon2811/ocr_captcha

docker run -p 8000:8000 namsiunhon2811/ocr_captcha
```

Docs automatic for api: http://localhost:8000/docs

## Endpoints

### 1. Submit Captcha Image

- **POST** `/captcha_image`

  **Description**: Call url `time` captcha image for processing and recognition.

  **Request Body**:
  - **Content-Type**: application/json
  - **Schema**:
    ```json
    {
      "session_id": "string",
      "image_url": "string",
      "time": "integer"
    }
    ```

  **Response**:
  - **Status Code**: 200 or 400
  - **Content**:
    ```json
    {
      "message": "string",
      "error": "string",
      "content": {
        "list_id": ["string"],
        "list_label": ["string"],
        "list_path": ["string"]
      }
    }
    ```

---

### 2. Submit Recap Captcha Image

- **POST** `/captcha_recap_image`

  **Description**: Call url captcha image `base64` string for processing and recognition.

  **Request Body**:
  - **Content-Type**: application/json
  - **Schema**:
    ```json
    {
      "image_base64": "string",
    }
    ```

  **Response**:
  - **Status Code**: 200 or 400
  - **Content**:
    ```json
    {
      "message": "string",
      "error": "string",
      "content": {
        "list_id": ["string"],
        "list_label": ["string"],
        "list_path": ["string"]
      }
    }
    ```

---

### 3. Retrain OCR Model

- **POST** `/captcha_retrain_model`

  **Description**: Initiates the training of the OCR model with specified images.

  **Request Body**:
  - **Content-Type**: application/json
  - **Schema**:
    ```json
    {
      "list_images_base64": ["integer"],
      "list_label": ["integer"],
      "debug": "boolean"
    }
    ```

  **Response**:
  - **Status Code**: 200 or 400
  - **Content**:
    ```json
    {
      "message": "string",
      "error": "string"
    }
    ```

---

## Status Messages and Codes

### Status Code Mapping

The `ValidateRecognitionResponse` class maps each status message to an appropriate HTTP status code. Most of the error messages return a status code of **400**, indicating a "Bad Request." This signifies that the client has submitted an invalid request due to various reasons (e.g., missing data, incorrect values).

The only successful response is mapped to **200**, indicating that the request was processed successfully without issues.

### Summary

- **200**: Success (Operation completed)
- **400**: Client Error (Various reasons such as invalid input, missing data, or access issues)

This structured approach helps in clearly communicating the outcome of API requests, facilitating easier error handling and debugging for clients interacting with the API.

## ISSUE

open ssl fix error when call request: https://github.com/python-poetry/install.python-poetry.org/issues/112

## About Me

Hi, I'm @namladuc! I'm a passionate developer with a focus on MLOps and Data Science. 

### GitHub Profile

You can find my GitHub profile [here](https://github.com/namladuc). On my profile, you'll see:

- **Repositories**: A collection of my projects, including [mention some notable projects].
- **Contributions**: My contributions to various open-source projects.
- **Stars**: Projects that I've starred and found interesting.
- **Followers**: Connect with me and follow my work.

Feel free to explore my repositories and reach out if you have any questions or collaboration ideas!
