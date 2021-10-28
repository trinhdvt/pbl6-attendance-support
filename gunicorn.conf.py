import multiprocessing

bind = "0.0.0.0:8080"
# workers = multiprocessing.cpu_count() * 2 + 1
workers = 1
timeout = 90
wsgi_app = "server:app"
loglevel = "debug"
capture_output = True
