from src.utils import extract_feature_from_image, postprocessing
from ultralytics import YOLO
from pycaret.classification import load_model, predict_model
import easyocr
import pandas as pd


def feature_extraction(image_dir, filename, box_model_path = "models/yolov8_box.pt", symbol_model_path = "models/yolov8_symbol.pt"):
    """
    Extracts the features from the given image.
    
    Args:
        image_dir (str): Directory of the image.
        filename (str): Filename of the image.
        box_model_path (str): Path to the box model.
        symbol_model_path (str): Path to the symbol model.

    Returns:
        pandas.DataFrame: extracted features for the right ear.
        pandas.DataFrame: extracted features for the left ear.
    """
    # load model
    box_model = YOLO(box_model_path)
    symbol_model = YOLO(symbol_model_path)
    reader = easyocr.Reader(["en"], gpu=True)

    # extract data from image
    predict_rt_df, predict_lt_df = extract_feature_from_image(
        filename, image_dir, box_model, symbol_model, reader
    )

    return predict_rt_df, predict_lt_df


def preprocessing_feature(predict_rt_df, predict_lt_df, interpolation=True):
    """
    Preprocesses the extracted features from the given image.

    Args:
        predict_rt_df (pandas.DataFrame): extracted features for the right ear.
        predict_lt_df (pandas.DataFrame): extracted features for the left ear.
        interpolation (bool): If True, the data will be interpolated.
    
    Returns:
        pandas.DataFrame: preprocessed features for the right ear.
        pandas.DataFrame: preprocessed features for the left ear.
    """
    # post processing right and left data
    predict_rt_df = postprocessing(predict_rt_df, interpolation=interpolation)
    predict_lt_df = postprocessing(predict_lt_df, interpolation=interpolation)

    return predict_rt_df, predict_lt_df


def classified_feature(predict_rt_df, predict_lt_df, model_path="models/extract_model"):
    """
    Classifies the degree of hearing loss for the given right and left ear data.

    Args:
        predict_rt_df (pandas.DataFrame): right ear data to be classified.
        predict_lt_df (pandas.DataFrame): left ear data to be classified.


    Returns:
        pandas.DataFrame: classified degree of hearing loss for the right and left ears.
    """
    classifier_path = model_path
    classifier = load_model(classifier_path)
    label = ['Normal', 'Mild', 'Moderate', 'Moderately Severe', 'Severe', 'Profound']

    # classified degree of hearing loss
    predict_right_hearing_loss = predict_model(classifier, data=predict_rt_df)
    predict_left_hearing_loss = predict_model(classifier, data=predict_lt_df)

    # concat right and left data and rename index
    result = pd.concat([predict_right_hearing_loss, predict_left_hearing_loss], axis=0)
    result.index = ['right', 'left']

    # rename label
    result['prediction_label'] = result['prediction_label'].apply(lambda x: label[x])
    return result



