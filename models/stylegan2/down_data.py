import requests
 
from tqdm import tqdm
 
def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
 
        return None
 
    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
 
        with open(destination, "wb") as f:
            with tqdm(unit='B', unit_scale=True, unit_divisor=1024) as bar:
                for chunk in response.iter_content(CHUNK_SIZE):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
                        bar.update(CHUNK_SIZE)
 
    URL = "https://drive.google.com/file/d/12rtaChb4ON9FRaA03Z0R9zLvYzR1VACr/view?usp=sharing"
 
    session = requests.Session()
 
    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)
 
    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)
 
    save_response_content(response, destination)    
 
 
if __name__ == "__main__":
    import sys
    print("start")
    if len(sys.argv) is not 3:
        print("Usage: python google_drive.py drive_file_id destination_file_path")
    if len(sys.argv) is not 3:
        # TAKE ID FROM SHAREABLE LINK
        print("开始")
        file_id = "https://drive.google.com/file/d/12rtaChb4ON9FRaA03Z0R9zLvYzR1VACr/view?usp=sharing"
        # DESTINATION FILE ON YOUR DISK
        destination = ".//data"
        download_file_from_google_drive(file_id, destination)