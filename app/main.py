from typing import Union
from fastapi import FastAPI, Request 
from pydantic import BaseModel
import cv2
import numpy as np
import base64
from matplotlib import pyplot as plt
from keras.models import load_model

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "Hello World !!!"}

# API ของการทำงานหลัก โดยรันที่ port 5000 ซึ่งจะรับข้อมูลรูปภาพเเบบ base64
# มาจาก Container 1 ที่ทำการ resize ในขนาด 28x28 เพื่อให้ตรงกับข้อมูลรูปภาพที่ทำการ Train Model
@app.post("/api/image/predicted")
async def Image_Predicted(image_base64 : Request):

    image_base64_json = await image_base64.json() # ทำการเเปลงข้อมูลที่ส่งมาให้เป็น json

    # ทำการเเปลงข้อมูลรูปภาพ ให้สามารถใช้ในการประมวลผลหรือแสดงผลได้
    image_data = base64.b64decode(image_base64_json['image_base64']) # ถอดรหัส Base64 เพื่อให้เราได้ข้อมูลรูปภาพในรูปแบบ binary data
    image_array = np.frombuffer(image_data, np.uint8) # แปลงข้อมูลรูปภาพในรูปแบบ binary data เป็น NumPyarray
    img = cv2.imdecode(image_array, cv2.COLOR_BGR2RGB) # แปลง NumPyarray ที่เก็บรูปภาพในรูปแบบ binary data ให้เป็นรูปภาพเเบบ RGB เพื่อให้ตรงกับข้อมูลรูปภาพที่ทำการ Train Model
    # print(img)
    # plt.imshow(img, cmap = 'gray')
    # plt.show()
    
    model = load_model('/work/model/model_flower.h5') # ทำการโหลด Model CNN ที่ได้ Train ไว้มาใช้งาน
    yhat = model.predict(np.expand_dims(img, axis=0)) # ทำการทำนาย ประเภทดอกไม้จาก Model ที่ได้โหลดมาโดยผลลัพธ์เก็บในตัวเเปร yhat = มีค่า 0 - 9 เเทนประเภทดอกไม้ชนิดต่างๆ 10 ประเภท
    print(f'Predicted: class={np.argmax(yhat)}')

    if np.argmax(yhat) == 0:
        class_name = 'bellflower'
    elif np.argmax(yhat) == 1:
        class_name = 'black_eyed_susan'
    elif np.argmax(yhat) == 2:
        class_name = 'calendula'
    elif np.argmax(yhat) == 3:
        class_name = 'carnation'
    elif np.argmax(yhat) == 4:
        class_name = 'common_daisy'
    elif np.argmax(yhat) == 5:
        class_name = 'iris'
    elif np.argmax(yhat) == 6:
        class_name = 'rose'
    elif np.argmax(yhat) == 7:
        class_name = 'sunflower'
    elif np.argmax(yhat) == 8:
        class_name = 'tulip'
    elif np.argmax(yhat) == 9:
        class_name = 'water_lily'
    else:
        class_name = 'Not in category'  

    return {"Flower Type": class_name} # ทำการ return ประเภทดอกไม้กลับไปให้ Container 1
    