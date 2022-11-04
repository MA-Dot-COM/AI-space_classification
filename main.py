from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

import tensorflow as tf
classification_model = tf.keras.models.load_model('./space_classification/space_classification_model/space_classification_model.hdf5')

class Item(BaseModel):
    name: Union[str, None] = None
    price: Union[float, None] = None
    url: str
    is_offer: Union[bool, None] = None

@app.get("/")
def read_root():
    return {"Hello": "World"}


from space_classification.space_classification import space_classification, img_download
@app.put("/test")
def test_model(item: Item):
    url = item.url
    # url = "https://dispatch.cdnser.be/cms-content/uploads/2020/04/09/a26f4b7b-9769-49dd-aed3-b7067fbc5a8c.jpg"
    img_path = img_download(url)
    print(img_path)
    return space_classification(img_path,classification_model)

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_name": item.price, "item_id": item_id}
