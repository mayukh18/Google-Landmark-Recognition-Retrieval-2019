import numpy as np
import cv2
import urllib
import requests
from tqdm import tqdm



def check_size(url):
    """
    Helper method to check the size of the file from the url
    """
    r = requests.get(url, stream=True)
    return int(r.headers['Content-Length'])


def download_file(url, filename, bar=True):
    """
    Helper method handling downloading large files from `url` to `filename`. Returns a pointer to `filename`.
    """
    try:
        chunkSize = 1024
        r = requests.get(url, stream=True)
        with open(filename, 'wb') as f:
            if bar:
                pbar = tqdm(unit="B", total=int(r.headers['Content-Length']))
            for chunk in r.iter_content(chunk_size=chunkSize):
                if chunk:  # filter out keep-alive new chunks
                    if bar:
                        pbar.update(len(chunk))
                    f.write(chunk)
        return filename
    except Exception as e:
        print(e)
        return


def download_image_cv2_urllib(url):
    """
    Modifying the url to download the 360p or 720p version actually slows it down.
    """
    try:
        resp = urllib.request.urlopen(url)
        foo = np.asarray(bytearray(resp.read()), dtype="uint8")
        foo = cv2.imdecode(foo, cv2.IMREAD_COLOR)
        foo = cv2.resize(foo, (192, 192), interpolation=cv2.INTER_AREA)
        foo = cv2.cvtColor(foo, cv2.COLOR_BGR2RGB)
        return foo
    except:
        return np.array([])
