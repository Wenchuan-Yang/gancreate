import numpy as np
import requests
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

url = 'http://127.0.0.1:8000/predict'

def predict_result(data):
    # data = open(data,'rb').read()
    # print(type(data))
    data = str(data)
    payload = {'image':data}
    
    re = requests.post(url, files=payload).json()

    re = np.array(re)
    final_im = Image.fromarray((re * 255).astype(np.uint8)).resize((500,500),Image.LANCZOS)
    return final_im

if __name__ == "__main__":
    for i in range(6):
        data = {
            'seed':'ncjcysri',
            'truncation': 0.5,
            # 属性
            'gender': i,
            'Trueness': 0,
            'hair': 0,
            'race': 0,
            'hairColor': 0,
            'color': 0,
            'hairLength': 0,
            
            'start_layer': 0,
            'end_layer': 14
            }
        image = predict_result(data)
        image.show()
        # image.save(f'im1.png')


