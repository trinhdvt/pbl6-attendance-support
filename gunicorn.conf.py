import multiprocessing
import os

bind = f"0.0.0.0:{os.getenv('PORT', '8080')}"
workers = multiprocessing.cpu_count() * 2 if os.getenv("ENV") == "DEPLOY" else 1
timeout = 90
wsgi_app = "fastapi_server:app"
loglevel = "info"
capture_output = False
worker_class = os.getenv("WORKER_CLASS", "uvicorn.workers.UvicornWorker")
accesslog = "-"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(T)s %(M)s %(D)s %(L)s'
