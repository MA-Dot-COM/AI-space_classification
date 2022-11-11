from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder

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
    classification = space_classification(img_path, classification_model)
    # json으로 호환 가능하게 데이터 타입을 바꿔주는 인코더
    classification_jsonable = jsonable_encoder(classification)
    # json.dumps(classification_jsonable)
    return {"lifing":classification_jsonable}
