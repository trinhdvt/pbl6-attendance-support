import gdown


def download_model(drive_id, output_path):
    url = f"https://drive.google.com/uc?id={drive_id}"
    gdown.download(url, output_path)
