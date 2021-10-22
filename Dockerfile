FROM animcogn/face_recognition:cpu-latest

RUN python3 -m pip install --upgrade pip setuptools wheel

WORKDIR /app

COPY requirements.txt .

RUN pip3 install -r requirements.txt --no-cache-dir

RUN pip3 install torch==1.9.1+cpu torchvision==0.10.1+cpu torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html --no-cache-dir

COPY . .

CMD ["gunicorn", "--bind","0.0.0.0:8080", "server:app"]
#CMD ["python3","server.py"]
