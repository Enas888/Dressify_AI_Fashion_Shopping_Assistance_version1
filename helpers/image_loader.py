import os
import requests
from PIL import Image
from config import Config

def get_or_download_image(image_id: str, image_url: str) -> str | None:
    """
    Downloads image if not cached.
    Returns local image path or None if failed.
    """
    local_path = os.path.join(Config.IMAGES_PATH, f"{image_id}.jpg")

    if os.path.exists(local_path):
        return local_path

    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()

        with open(local_path, "wb") as f:
            f.write(response.content)

        # Verify image
        Image.open(local_path).convert("RGB")
        return local_path

    except Exception as e:
        print(f"❌ Failed to download {image_url}: {e}")
        return None
