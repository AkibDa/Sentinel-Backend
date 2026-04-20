import subprocess

def get_raw_image_url(social_url: str) -> str:

    try:
        result = subprocess.run(
            ["gallery-dl","--cookies","cookies.txt", "-g",social_url],
            capture_output=True,
            text=True,
            check=True
        )
        urls = result.stdout.strip().split('\n')

        if urls and urls[0] and "http" in urls[0]:
            return urls[0]
        
        else:
            raise Exception("No direct image URL could be found on this page")
        
    except subprocess.CalledProcessError as e:
        raise Exception(f"Failed to scrape image: {e.stderr.strip()}")
    