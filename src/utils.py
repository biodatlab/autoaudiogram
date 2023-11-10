import os
import pandas as pd
import numpy as np
import cv2
from deskew import determine_skew
from skimage.transform import rotate
import re


def strighten_image(img):
    """
    img: np.array

    rotate image to strighten
    """
    angle = determine_skew(img)
    rotated = rotate(img, angle, resize=True) * 255
    rotated_img = rotated.astype(np.uint8)
    return rotated_img, angle


def get_text_from_image(reader, image, transform=True):
    """
    reader: easyocr.Reader
    image: np.array
    transform: bool, if True, return int, else return str

    extract text from image using reader
    """
    results = reader.readtext(image, allowlist="0123456789")
    # results = reader.readtext(image)
    if len(results) == 0:
        text = ""
    else:
        text = results[0][1]
    if transform:
        match = re.search(r"\d+", text)
        if match:
            text = int(match.group())
    return text


def get_positions(results, label):
    """
    results: list, results from model.predict
    label: int, label of class

    get position from results as xyxy format
    """
    cls = results[0].boxes.cls.cpu().numpy()
    index = np.where(cls == label)[0]
    positions = results[0].boxes.xyxy[index].cpu().numpy()  # [x1, y1, x2, y2]
    return positions


def shift_positions(position, margin):
    """
    position: np.array, [x1, y1, x2, y2]

    shift position to left and up
    """
    position[:, 0] = position[:, 0] - margin
    position[:, 1] = position[:, 1] - margin
    position[:, 2] = position[:, 2] - margin
    position[:, 3] = position[:, 3] - margin
    return position


def scale_positions(positions, image_width, image_height):
    """
    positions: np.array, [x1, y1, x2, y2]
    image_width: int
    image_height: int

    scale positions to fit decibel and frequency of audiogram
    """
    positions[:, [0, 2]] = (positions[:, [0, 2]] / image_width) * 6  # shift x
    positions[:, [1, 3]] = ((positions[:, [1, 3]] / image_height) * 130) - 10  # shift y
    return positions


def get_centers(positions):
    """
    positions: np.array, [x1, y1, x2, y2]

    get centers of positions to find the actual position of symbol
    """
    centers = (positions[:, [0, 1]] + positions[:, [2, 3]]) / 2  # x, y
    return centers


def predict_db(graph, results, label, margin):
    """
    graph: np.array, image of audiogram
    results: list, results from model.predict
    label: int, label of class
    margin: int, margin of crop image

    using results to extract db from graph
    """

    image = graph.copy()
    image = image[margin:-margin, margin:-margin]
    positions = get_positions(results, label)
    positions = shift_positions(positions, margin)
    positions = scale_positions(positions, image.shape[1], image.shape[0])
    centers = get_centers(positions)  # [x,y]
    # sort by y
    centers = np.array(sorted(centers, key=lambda x: x[0]))
    if centers.shape[0] == 0:
        return [np.NaN]
    return centers


def load_image(image_path):
    """
    image_path: str

    load image from image_path
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image, angle = strighten_image(image)
    return image


def crop_graph_table(image, model, margin=20):
    """
    image: np.array, image of audiogram
    model: ultralytics.YOLO, model to predict box

    return crop garph and table from image using model
    """
    box_results = model.predict(image)
    x1, y1, x2, y2 = get_positions(box_results, 0).astype(int)[0]  # [x1, y1, x2, y2]
    graph = image[y1 - margin : y2 + margin, x1 - margin : x2 + margin]
    x1, y1, x2, y2 = get_positions(box_results, 1).astype(int)[0]  # [x1, y1, x2, y2]
    table = image[y1 - margin : y2 + margin, x1 - margin : x2 + margin]
    table = cv2.resize(table, (431, 1023), interpolation=cv2.INTER_AREA)
    return graph, table


def extract_text_from_table(table, reader):
    """
    table: np.array, image of table
    reader: easyocr.Reader

    extract text from table using reader
    """
    acrt = table[75:200, 220:400]
    aclt = table[190:325, 220:400]
    bcrt = table[300:425, 220:400]
    bclt = table[415:545, 220:400]

    srtrt = table[680:850, 110:260]
    slrt = table[680:760, 220:320]
    pbrt = table[760:850, 220:360]

    srtlt = table[840:1000, 110:260]
    sllt = table[840:920, 220:320]
    pblt = table[920:1000, 220:360]

    acrt_text = get_text_from_image(reader, acrt)
    aclt_text = get_text_from_image(reader, aclt)
    bcrt_text = get_text_from_image(reader, bcrt)
    bclt_text = get_text_from_image(reader, bclt)

    srtrt_text = get_text_from_image(reader, srtrt)
    slrt_text = get_text_from_image(reader, slrt)
    pbrt_text = get_text_from_image(reader, pbrt)

    srtlt_text = get_text_from_image(reader, srtlt)
    sllt_text = get_text_from_image(reader, sllt)
    pblt_text = get_text_from_image(reader, pblt)

    return (
        acrt_text,
        aclt_text,
        bcrt_text,
        bclt_text,
        srtrt_text,
        slrt_text,
        pbrt_text,
        srtlt_text,
        sllt_text,
        pblt_text,
    )


def detect_symbols(graph, model):
    """
    class_dict = {0: 'Air Rt Unmasked', 1: 'Air Lt Unmasked', 2: 'Air Rt Masked',
              3: 'Air Lt masked', 4: 'Bone Rt Unmasked', 5: 'Bone Lt Unmasked',
              6: 'Bone Rt Masked', 7: 'Bone Lt Masked'}
    graph: np.array, image of audiogram
    model: ultralytics.YOLO, model to predict symbol

    predcict all symbols from graph using model
    """
    symbol_results = model(graph)
    air_rt_um = pd.DataFrame(predict_db(graph, symbol_results, 0, 20))
    air_lt_um = pd.DataFrame(predict_db(graph, symbol_results, 1, 20))
    air_rt_m = pd.DataFrame(predict_db(graph, symbol_results, 2, 20))
    air_lt_m = pd.DataFrame(predict_db(graph, symbol_results, 3, 20))
    bone_rt_um = pd.DataFrame(predict_db(graph, symbol_results, 4, 20))
    bone_lt_um = pd.DataFrame(predict_db(graph, symbol_results, 5, 20))
    bone_rt_m = pd.DataFrame(predict_db(graph, symbol_results, 6, 20))
    bone_lt_m = pd.DataFrame(predict_db(graph, symbol_results, 7, 20))

    air_rt_df = (
        pd.concat([air_rt_um, air_rt_m], axis=0)
        .sort_values(by=0)
        .rename(columns={0: "x", 1: "y"})
    )
    air_lt_df = (
        pd.concat([air_lt_um, air_lt_m], axis=0)
        .sort_values(by=0)
        .rename(columns={0: "x", 1: "y"})
    )
    bone_rt_df = (
        pd.concat([bone_rt_um, bone_rt_m], axis=0)
        .sort_values(by=0)
        .rename(columns={0: "x", 1: "y"})
    )
    bone_lt_df = (
        pd.concat([bone_lt_um, bone_lt_m], axis=0)
        .sort_values(by=0)
        .rename(columns={0: "x", 1: "y"})
    )

    return air_rt_df, air_lt_df, bone_rt_df, bone_lt_df  # [x, y]


def select_symbol(df, c_type):
    """
    df: pd.DataFrame, [x, y]
    c_type: str, 'air' or 'bone' (conduct_type)

    select symbol from df to support multiple detection
    """
    if c_type == "air":
        x_map = [1, 2, 3, 4, 5, 5.5, 6]

    if c_type == "bone":
        x_map = [2, 3, 4, 5]

    df = df.dropna()
    xs = df["x"].values
    if xs.shape[0] == 0:
        return pd.DataFrame([[np.NaN] * 2] * (len(x_map)), columns=["x", "y"])
    x_needs = [(np.abs(xs - i)).argmin() for i in x_map]
    return df.iloc[x_needs]


def select_symbols(air_rt_df, air_lt_df, bone_rt_df, bone_lt_df):
    """
    air_rt_df: pd.DataFrame, [x, y]
    air_lt_df: pd.DataFrame, [x, y]
    bone_rt_df: pd.DataFrame, [x, y]
    bone_lt_df: pd.DataFrame, [x, y]

    select symbols from all df
    """
    air_rt_df = select_symbol(air_rt_df, "air").reset_index(drop=True)
    air_rt_df = air_rt_df.rename(
        index={
            0: "250",
            1: "500",
            2: "1000",
            3: "2000",
            4: "4000",
            5: "6000",
            6: "8000",
        }
    )
    air_lt_df = select_symbol(air_lt_df, "air").reset_index(drop=True)
    air_lt_df = air_lt_df.rename(
        index={
            0: "250",
            1: "500",
            2: "1000",
            3: "2000",
            4: "4000",
            5: "6000",
            6: "8000",
        }
    )
    bone_rt_df = select_symbol(bone_rt_df, "bone").reset_index(drop=True)
    bone_rt_df = bone_rt_df.rename(index={0: "500", 1: "1000", 2: "2000", 3: "4000"})
    bone_lt_df = select_symbol(bone_lt_df, "bone").reset_index(drop=True)
    bone_lt_df = bone_lt_df.rename(index={0: "500", 1: "1000", 2: "2000", 3: "4000"})
    return air_rt_df, air_lt_df, bone_rt_df, bone_lt_df


def cal_pta(df):
    """
    df: pd.DataFrame, [x, y]

    calculate some pta from df
    """
    f_500 = df.loc["500", "y"]
    f_1000 = df.loc["1000", "y"]
    f_2000 = df.loc["2000", "y"]
    c = (f_500 + f_1000 + f_2000) / 3
    return c


def cal_all_pta(air_rt_df, air_lt_df, bone_rt_df, bone_lt_df):
    """
    air_rt_df: pd.DataFrame, [x, y]
    air_lt_df: pd.DataFrame, [x, y]
    bone_rt_df: pd.DataFrame, [x, y]
    bone_lt_df: pd.DataFrame, [x, y]

    calculate all pta from all df
    """
    acrt_cal = cal_pta(air_rt_df)
    aclt_cal = cal_pta(air_lt_df)
    bcrt_cal = cal_pta(bone_rt_df)
    bclt_cal = cal_pta(bone_lt_df)
    return acrt_cal, aclt_cal, bcrt_cal, bclt_cal


def extract_feature_from_image(image_file, image_dir, box_model, symbol_model, reader):
    """
    image_file: str, image file name
    image_dir: str, image directory
    box_model: ultralytics.YOLO, model to predict box
    symbol_model: ultralytics.YOLO, model to predict symbol
    reader: easyocr.Reader

    complete extract feature from image
    """
    image_path = os.path.join(image_dir, image_file)
    image = load_image(image_path)
    graph, table = crop_graph_table(image, box_model)
    (
        acrt_text,
        aclt_text,
        bcrt_text,
        bclt_text,
        srtrt_text,
        slrt_text,
        pbrt_text,
        srtlt_text,
        sllt_text,
        pblt_text,
    ) = extract_text_from_table(table, reader)
    air_rt_df, air_lt_df, bone_rt_df, bone_lt_df = detect_symbols(graph, symbol_model)
    air_rt_df, air_lt_df, bone_rt_df, bone_lt_df = select_symbols(
        air_rt_df, air_lt_df, bone_rt_df, bone_lt_df
    )
    acrt_cal, aclt_cal, bcrt_cal, bclt_cal = cal_all_pta(
        air_rt_df, air_lt_df, bone_rt_df, bone_lt_df
    )
    pta_rt_df = pd.DataFrame(
        [acrt_cal, bcrt_cal, slrt_text, srtrt_text, pbrt_text],
        index=["PTA_AC", "PTA_BC", "SL", "SRT", "PB"],
    )
    pta_lt_df = pd.DataFrame(
        [aclt_cal, bclt_cal, sllt_text, srtlt_text, pblt_text],
        index=["PTA_AC", "PTA_BC", "SL", "SRT", "PB"],
    )
    # concat pta air bone
    predict_rt_df = pd.concat([pta_rt_df, air_lt_df["y"], bone_lt_df["y"]], axis=0)
    predict_lt_df = pd.concat([pta_lt_df, air_rt_df["y"], bone_rt_df["y"]], axis=0)
    return predict_rt_df, predict_lt_df


def get_feature_from_df(df, idx):
    """
    df: pd.DataFrame, [id, feature, label]
    idx: int, id of feature

    selected id from df
    """
    feature_x = df[df.id == idx].iloc[:, 1:-1].T
    feature_y = df[df.id == idx].iloc[:, -1:].T
    return feature_x, feature_y


def concat_feature(predict_df, true_df):
    """
    predict_df: pd.DataFrame, [id, feature, label]
    true_df: pd.DataFrame, [id, feature, label]

    concat feature from predict_df and true_df
    """
    x_true_df_ls = []
    y_true_df_ls = []
    for idx in predict_df.index:
        x_true, y_true = get_feature_from_df(true_df, idx)
        if x_true.shape[1] == 0:
            print(f"not found in true_df: {idx}")
            continue
        if x_true.shape[1] > 1:
            print(f"multiple id: {idx}")
            continue
        x_true.columns = [idx]
        y_true.columns = [idx]
        x_true_df_ls.append(x_true.T)
        y_true_df_ls.append(y_true.T)
    x_true_dfs = pd.concat(x_true_df_ls, axis=0)
    y_true_dfs = pd.concat(y_true_df_ls, axis=0)
    return x_true_dfs, y_true_dfs
