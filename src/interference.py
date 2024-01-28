from ultralytics import YOLO
import easyocr
from utils import extract_feature_from_image, postprocessing
from pycaret.classification import load_model, predict_model
import pandas as pd
import numpy as np



image_dir = "data/07-2021 - 07-2022/"
filename = "1.jpg"
box_model_path = "data_extracter/graph_table/runs/detect/yolov8n/weights/best.pt"
symbol_model_path = "data_extracter/symbol_detection/runs/detect/yolov8n/weights/best.pt"
classifier_path = "classification\models\extract_model.pkl"

box_model = YOLO(box_model_path)
symbol_model = YOLO(symbol_model_path)
classifier = load_model(classifier_path)
reader = easyocr.Reader(["en"], gpu=True)

# extract data from image
predict_rt_df, predict_lt_df = extract_feature_from_image(
    filename, image_dir, box_model, symbol_model, reader
)
predict_rt_df = predict_rt_df.rename(columns={0: "db"})
predict_lt_df = predict_lt_df.rename(columns={0: "db"})
print("right_extarcted_data:")
print(predict_rt_df)
print("\n")
print("left_extarcted_data:")
print(predict_lt_df)
print("\n")

# post processing right and left data
predict_rt_df = postprocessing(predict_rt_df)
predict_lt_df = postprocessing(predict_lt_df)
predict_right_hearing_loss = predict_model(classifier, data=predict_rt_df)
predict_left_hearing_loss = predict_model(classifier, data=predict_lt_df)

# concat right and left data and rename index
result = pd.concat([predict_right_hearing_loss, predict_left_hearing_loss], axis=0)
result.index = ['right', 'left']
print('result:')
print(result)