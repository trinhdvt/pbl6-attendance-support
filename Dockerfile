FROM python:3.9-alpine

RUN apk update && apk add build-base cmake pkgconfig

RUN pip3 install --upgrade pip

WORKDIR /app

COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY . .

#CMD ["gunicorn", "--bind","0.0.0.0:8080", "server:app"]
CMD ["python3","server.py"]
