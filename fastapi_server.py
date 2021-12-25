import json
import os

from celery.result import AsyncResult
from fastapi import FastAPI, File, UploadFile, Path, status
from fastapi.responses import JSONResponse

from app.app_utils import is_valid_image, request_to_pil, pil_to_base64
from app.exception import CustomException
from app.exception_handler import add_exception_handler
from app.middleware_config import add_middleware
from app.models import Base64Input
from celery_task.tasks import submit_task

# APP Setup

app = FastAPI()
add_middleware(app)
add_exception_handler(app)

log_dir = os.getcwd() + "/upload/"
if not os.path.exists(log_dir):
    os.mkdir(log_dir)


# API Definition

@app.post("/api/check/v2", status_code=status.HTTP_202_ACCEPTED)
async def predict(face_img: UploadFile = File(...),
                  card_img: UploadFile = File(...)):
    """
    Attendance check API with file upload
    """

    # validate input
    if not is_valid_image([face_img.content_type, card_img.content_type]):
        raise CustomException(message="Invalid image file",
                              status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)

    # create task submission
    task_args = {
        'face-img-b64': pil_to_base64(request_to_pil(await face_img.read())),
        'card-img-b64': pil_to_base64(request_to_pil(await card_img.read())),
    }

    # insert celery queue
    task = submit_task.delay(task_args)
    result_id = str(task)
    return {"result_id": result_id}


@app.post("/api/check", status_code=status.HTTP_202_ACCEPTED)
async def predict_base64(body: Base64Input):
    """
    Attendance check API with base64 image
    """

    # create task submission
    task_args = {
        'face-img-b64': body.face_img,
        'card-img-b64': body.card_img,
    }

    # insert to celery queue
    task = submit_task.delay(task_args)
    result_id = str(task)
    return {"result_id": result_id}


@app.get("/api/results/{result_id}", status_code=200)
async def get_result(result_id: str = Path(...)):
    """
    Fetch result for a given result_id
    """

    task = AsyncResult(result_id)
    # return result if task has executed successfully
    if task.successful():
        result = task.get()
        return result

    # return error if any exception raised
    if task.failed():
        try:
            task.get()
        except Exception as e:
            return JSONResponse(content=json.loads(str(e)),
                                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)

    # return current task's status
    if not task.ready():
        return JSONResponse(content={
            "status": str(task.status)
        }, status_code=status.HTTP_202_ACCEPTED)


# HEALTH CHECK API


@app.get("/api/health", status_code=200)
async def health_check():
    return {"status": "OK"}
