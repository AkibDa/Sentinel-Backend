import requests
import uuid

def download_file(url):
    filename = f"temp_{uuid.uuid4()}.mp4"

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(url, headers=headers,stream=True)

    if response.status_code != 200:
        raise Exception(f"Download failed with status {response.status_code}")
    
    with open(filename, "wb") as f:
        for chunk in response.iter_content(1024):
            f.write(chunk)

    return filename