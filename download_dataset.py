import os
import urllib.request
import zipfile
from pathlib import Path

def download_and_extract():
    url = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
    zip_path = "kagglecatsanddogs_5340.zip"
    extract_dir = Path.cwd()
    
    if not (extract_dir / "PetImages").exists():
        print(f"Downloading dataset from {url}...")
        urllib.request.urlretrieve(url, zip_path)
        
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
            
        print("Cleaning up zip file...")
        os.remove(zip_path)
        print("Done! PetImages folder is ready.")
    else:
        print("PetImages directory already exists!")

if __name__ == "__main__":
    download_and_extract()
