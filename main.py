from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
import json

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
    image_path = img_download(url)
    category, score = space_classification(image_path, classification_model)

    # json으로 호환 가능하게 데이터 타입을 바꿔주는 인코더
    # category_jsonable = jsonable_encoder(category)
    # category_jsonable = json.dumps(category_jsonable)
    #
    # score_jsonable = jsonable_encoder(score)
    # score_jsonable = json.dumps(score_jsonable)

    return {"lifing": category, "score": score}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_name": item.price, "item_id": item_id}
