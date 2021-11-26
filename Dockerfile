FROM animcogn/face_recognition:cpu-latest

RUN python3 -m pip install --upgrade pip setuptools wheel --no-cache-dir

WORKDIR /app

COPY requirements.txt .

RUN pip3 install -r requirements.txt --upgrade --no-cache-dir \
    && pip3 install torch==1.10.0+cpu torchvision==0.11.1+cpu torchaudio==0.10.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html --no-cache-dir

COPY . .

ENV TZ=Asia/Ho_Chi_Minh

CMD ["gunicorn"]
