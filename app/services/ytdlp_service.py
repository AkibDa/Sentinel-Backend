import yt_dlp
import uuid
import os
import glob


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
COOKIES_PATH = os.path.join(BASE_DIR, "cookies.txt")

def download_media_ytdlp(url: str) -> str:
    base_name = f"temp_{uuid.uuid4()}"
    
    ydl_opts = {
        'outtmpl': f'{base_name}.%(ext)s',
        'format': 'best', 
        'noplaylist': True,
        'cookiesfile': COOKIES_PATH,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(url, download=True)
        
        downloaded_files = glob.glob(f"{base_name}.*")
        
        if not downloaded_files:
            raise Exception("Download appeared to succeed, but the file was not saved to the disk.")
            
        return downloaded_files[0]

    except Exception as e:
        raise Exception(f"yt-dlp error: {str(e)}")
