import os

from celery import Celery
from dotenv import load_dotenv

load_dotenv()

BROKER_URI = os.getenv("BROKER_URI")
BACKEND_URI = os.getenv("BACKEND_URI")

app = Celery(
    'celery_app',
    broker=BROKER_URI,
    backend=BACKEND_URI,
    include=['celery_task.task',
             'celery_task.task.task_exception',
             'celery_task.task.task_utils',
             'src.resource_loader']
)

app.conf.update({
    'task_compression': 'gzip',
    'task_serializer': 'json',
    'result_serializer': 'json',
    'accept_content': ['json'],
    'task_track_started': True,
    'task_time_limit': 60,
    'result_expires': int(os.getenv("RESULT_EXPIRE", "100")),
    'worker_prefetch_multiplier': 1,
    'task_reject_on_worker_lost': True,
    'task_acks_late': True
})

app.control.purge()
