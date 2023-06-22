from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
from pathlib import Path
from keras.utils import pad_sequences
from typing import List
import numpy as np
#import uvicorn


app = FastAPI()

BASE_DIR = Path(__file__).resolve(strict=True).parent

tokenizer = load(f'{BASE_DIR}/model/tokenizer_embedding_fasttext.joblib')
model = load(f'{BASE_DIR}/model/embedding_fasttext.joblib')

dic_y_mapping = {
    0: 'All Electronics',
    1: 'Amazon Fashion',
    2: 'Amazon Home',
    3: 'Arts, Crafts & Sewing',
    4: 'Automotive',
    5: 'Books',
    6: 'Camera & Photo',
    7: 'Cell Phones & Accessories',
    8: 'Computers',
    9: 'Digital Music',
    10: 'Grocery',
    11: 'Health & Personal Care',
    12: 'Home Audio & Theater',
    13: 'Industrial & Scientific',
    14: 'Movies & TV',
    15: 'Musical Instruments',
    16: 'Office Products',
    17: 'Pet Supplies',
    18: 'Sports & Outdoors',
    19: 'Tools & Home Improvement',
    20: 'Toys & Games',
    21: 'Video Games'
}


class TextIn(BaseModel):
    also_buy: List[str]
    also_view: List[str]
    asin: str
    brand: str
    category: List[str]
    description: List[str]
    feature:  List[str]
    image: List[str]
    price: str
    title: str
    main_cat: str


class PredictionOut(BaseModel):
    category: str


@app.get('/')
def home():
    return {'health_check': 'OK'}


@app.post('/predict', response_model=PredictionOut)
def predict(payload: TextIn):
    category = predict_text(str(payload.title + " " + payload.description[0]))
    return {'category': category}


def predict_text(text):
    res = tokenizer.texts_to_sequences([text])
    res = pad_sequences(res, maxlen=100)
    res = model.predict(res)
    res = [dic_y_mapping[np.argmax(pred)] for pred in res]
    return str(res)


#if __name__ == '__main__':
    #uvicorn.run(app, host='0.0.0.0', port=8000)