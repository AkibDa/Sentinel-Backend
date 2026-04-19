import yt_dlp
import uuid
import os
import glob

def download_media_ytdlp(url: str) -> str:
    """
    Uses yt-dlp to download video.
    """
    base_name = f"temp_{uuid.uuid4()}"
    
    ydl_opts = {
        'outtmpl': f'{base_name}.%(ext)s',
        'format': 'best', 
        'noplaylist': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Using extract_info forces yt-dlp to throw a real Python exception if it fails
            ydl.extract_info(url, download=True)
        
        # Use glob to find the exact file yt-dlp created (e.g., temp_123.mp4, temp_123.mkv)
        downloaded_files = glob.glob(f"{base_name}.*")
        
        if not downloaded_files:
            raise Exception("Download appeared to succeed, but the file was not saved to the disk.")
            
        return downloaded_files[0]

    except Exception as e:
        # This will now capture the EXACT reason Twitter/Reddit rejected the download
        raise Exception(f"yt-dlp error: {str(e)}")