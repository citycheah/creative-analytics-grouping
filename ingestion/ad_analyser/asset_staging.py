import logging
import mimetypes
import os
import tempfile
import time

import requests

from .exceptions import ContentDownloadError
from .storage import StorageManager

logger = logging.getLogger(__name__)

mimetypes.add_type('video/webm', '.webm')
mimetypes.add_type('image/webp', '.webp')

def to_https(url: str) -> str:
    if not url: 
        return url
    if not url.startswith("http"): 
        return url
    return "https://" + url[len("http://") :] if url.startswith("http://") else url

def stage_file(url: str, storage_manager: StorageManager) -> tuple[str, str]:
    max_retries = 2
    last_exception = None

    for attempt in range(max_retries):
        tmp_name = None
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            content_type = response.headers.get('Content-Type')
            mime_type = content_type.split(';')[0].strip() if content_type else None
            
            ext = None
            if mime_type:
                ext = mimetypes.guess_extension(mime_type)
            
            if not ext:
                mime_type_url, _ = mimetypes.guess_type(url)
                if mime_type_url:
                    ext = mimetypes.guess_extension(mime_type_url)
                    mime_type = mime_type or mime_type_url
                else:
                    ext = ".bin"
                    mime_type = mime_type or "application/octet-stream"

            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp_name = tmp.name
                total_downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        tmp.write(chunk)
                        total_downloaded += len(chunk)
                
            expected_size = response.headers.get('Content-Length')
            if expected_size and not response.headers.get('Content-Encoding'):
                if total_downloaded != int(expected_size):
                    raise ContentDownloadError("Incomplete download.")

            if total_downloaded == 0:
                raise ContentDownloadError("Downloaded file is empty.")

            try:
                gcs_uri = storage_manager.upload_to_gcs(tmp_name, mime_type)
                return gcs_uri, mime_type

            finally:
                if tmp_name and os.path.exists(tmp_name):
                    try:
                        os.remove(tmp_name)
                    except OSError:
                        pass
        
        except Exception as e:
            err_msg = str(e)
            if ("403" in err_msg or "Forbidden" in err_msg) and ".webm" in url:
                 url = url.replace(".webm", ".mp4")
                 continue
            last_exception = e
            time.sleep(2 ** attempt)

    raise ContentDownloadError(f"Download Error: {last_exception}") from last_exception
