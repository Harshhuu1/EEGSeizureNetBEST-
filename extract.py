import os
import requests
from bs4 import BeautifulSoup
import concurrent.futures

# Config
base_url = "https://physionet.org/files/chbmit/1.0.0/chb02/"
save_dir = ""
os.makedirs(save_dir, exist_ok=True)

# Get list of files
response = requests.get(base_url)
soup = BeautifulSoup(response.text, 'html.parser')
file_links = [link.get('href') for link in soup.find_all('a') if link.get('href').endswith(('.edf', '.txt'))]


# Download one file
def download_file(file_name):
    url = base_url + file_name
    path = os.path.join(save_dir, file_name)

    if os.path.exists(path):
        print(f"✅ Already exists: {file_name}")
        return

    print(f"⬇️ Downloading: {file_name} ...")  # ⬅️ Progress line added here

    try:
        r = requests.get(url, timeout=30)
        with open(path, 'wb') as f:
            f.write(r.content)
        print(f"✅ Downloaded: {file_name}")
    except Exception as e:
        print(f"❌ Failed: {file_name} - {e}")


# Use ThreadPoolExecutor to download files in parallel
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
    executor.map(download_file, file_links)

print("\n✅ All downloads completed.")
