import gdown
import time


def download_model(drive_id, output_path):
    url = f"https://drive.google.com/uc?id={drive_id}"
    gdown.download(url, output_path)
    print("Sleeping ...")
    time.sleep(1)
