import os
from datetime import datetime

from celery.result import AsyncResult
from fastapi import FastAPI, File, UploadFile, Form, Path, Request, status
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
async def predict(request: Request,
                  face_img: UploadFile = File(...),
                  card_img: UploadFile = File(...),
                  examCode: str = Form(...)):
    """
    Create a celery task then return task_id to client in order to fetch result later
    """

    # validate input
    if not is_valid_image([face_img.content_type, card_img.content_type]):
        raise CustomException(message="Invalid image file",
                              status_code=status.HTTP_400_BAD_REQUEST)

    # create task submission
    task_args = {
        'face-img-b64': pil_to_base64(request_to_pil(await face_img.read())),
        'card-img-b64': pil_to_base64(request_to_pil(await card_img.read())),
        'face-fn': face_img.filename,
        'card-fn': card_img.filename,
        "exam_code": examCode,
        "remote_addr": request.client.host,
        "user_agent": request.headers.get('User-Agent'),
        "log_dir": log_dir
    }

    # insert celery queue
    task = submit_task.delay(task_args)
    result_id = str(task)
    return {"result_id": result_id}


@app.post("/api/check", status_code=status.HTTP_202_ACCEPTED)
async def predict_base64(request: Request,
                         body: Base64Input):
    """
    API when submit with base64 image
    """

    # create task submission
    log_prefix = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    task_args = {
        'face-img-b64': body.face_img,
        'card-img-b64': body.card_img,
        'face-fn': f"{log_prefix}_{body.examCode}_face.jpg",
        'card-fn': f"{log_prefix}_{body.examCode}_card.jpg",
        "exam_code": body.examCode,
        "remote_addr": request.client.host,
        "user_agent": request.headers.get('User-Agent'),
        "log_dir": log_dir
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
        result = task.get()
        return JSONResponse(content=result,
                            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)

    # return current task's status
    if not task.ready():
        return JSONResponse(content={
            "status": str(task.status)
        }, status_code=status.HTTP_102_PROCESSING)


# HEALTH CHECK API


@app.get("/api/health", status_code=200)
async def health_check():
    return {"status": "OK"}


@app.post("/api/health/v1", status_code=200)
async def base64_img_test(body: Base64Input):
    return {"status": "OK"}
