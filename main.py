from typing import Union

from fastapi import FastAPI
import os
import random as rd
import base64
import subprocess
from pydantic import BaseModel
app = FastAPI()

class Item(BaseModel):
    categoryName: str
    imageSource: str
    selectedPortion: str

@app.get("/")
def read_root():
    return {"Hello": "World"}

def generate_inferences(file_name, category_name):
    cnt = 0
    # generate inferences
    result = subprocess.run(
        f'python yolov7\detect_and_count.py --weights yolov7\mybest_40.pt --conf 0.1 --classes {category_name} --source inference\images\{file_name}',
        stdout=subprocess.PIPE, text=True)
    print("\n\n result.stdout -- \n\n")
    print(result.stdout)
    for l in result.stdout.splitlines():
        print(type(l))
        if 'total_counts' in l:
            cnt = l.split('=')[1]
        print(l)

    with open(os.path.join("runs\detect\exp", file_name), "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        return {'status': int(cnt) > 0, 'result': encoded_string if int(cnt) > 0 else None, 'count':int(cnt)}

@app.post("/upload")
def read_item(item: Item):
    souce_file = base64.b64decode(item.imageSource)
    file_name = f'source_img_{rd.randint(1, 500)}.jpg'
    print('--' * 50, '\n File uploaded with name: ', file_name, '--' * 50)

    with open(os.path.join('inference\images', file_name), 'wb') as f:
        f.write(souce_file)

    return generate_inferences(file_name, item.categoryName)
    # return {"image_id": item.categoryName}

# @app.put("/items/{item_id}")
# def update_item(item_id: int, item: Item):
#     return {"item_name": item.name, "item_id": item_id}
