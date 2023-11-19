# Autoaudiogram: Object detection-based audiogram classification.

AutoAudiogram is an object detection-based audiogram classification system. The model uses object detection to detect the audiological symbols
from the audiogram and then use them to classify the degree of hearing loss severity.

<img src="figure/figure1.png" alt="drawing" width="1000"/>

For the data collection, we annotation 200 audiograms including graphs, tables, and 8 audiological symbols.

<img src="figure/figure2.png" alt="drawing" width="400"/>

## Get Started

Clone the repository and navigate into the project directory

```sh
git clone https://github.com/biodatlab/autoaudiogram.git
cd autoaudiogram
```

## Graph and table training

Navigate to the directory `cd src/data_extracter/graph_table` and organize the folder in this directory as follows

```
graph_table
|____data
| |____images
| |____labels
|____runs
|____config.yaml
|____eval.ipynb
|____train.ipynb
|____train.py
```

Then run `train.py` using

```sh
python -m train
```

## Audiological symbol training

Navigate to the directory `cd src/data_extracter/symbol_detection` and organize the folder as follows

```
symbol_detection
|____data
| |____images
| |____labels
|____runs
|____config.yaml
|____eval.ipynb
|____train.ipynb
|____train.py
```

Then run `train.py` using

```
python -m train
```

## Severity classification

Navigate to the directory `cd src/classification` and organize folder as follows

```
|classification
|____data
| |____train_extract.csv # example file name
| |____test_extract.csv
| |____train_true.csv
| |____test_true.csv
|____evalute
| |____extract
| |____true
|____model
|____eval.ipynb
|____train.ipynb
```

```
python -m train --path <path to train file> # python -m train --path "data/data_extract.csv"
```

# Evaluation

## Object Detection

mAP score of each audiological symbol

| Symbols | :o:       | :x:      | < | > | :white_square_button: | △ | [ | ] |
|---------|-----------|----------|--------------|---------------|------------------------|-----------------------|------------------------|-------------------------|
| YOLOv5  | 0.957     | 0.925    | 0.848        | 0.841         | 0.833                  | 0.986                 | 0.826                  | 0.875                   |
| YOLOv8  | 0.945     | 0.945    | 0.857        | 0.876         | 0.915                  | 0.85                  | 0.817                  | 0.944                   |


## Hearing severity classification

Hearing loss classification performance of top-5 models using AutoML.

**Automatic feature extraction**
| **Model**                        | **Test Accuracy** | **Test AUC** | **Test Recall** | **Test Precision** | **Test F1 Score** |
| -------------------------------- | ----------------- | -------- | ----------- | -------------- | -------------- |
| Gradient Boosting Classifier      | 0.9401           | 0.9856   | 0.9401      | 0.9419         | 0.9388         |
| Light Gradient Boosting Machine   | 0.9472           | 0.9842   | 0.9472      | 0.9472         | 0.9472         |
| Random Forest Classifier          | 0.933            | 0.9806   | 0.933       | 0.9352         | 0.9317         |
| Decision Tree Classifier          | 0.9344           | 0.9522   | 0.9344      | 0.9346         | 0.9335         |
| Extra Trees Classifier            | 0.9044           | 0.9815   | 0.9044      | 0.9043         | 0.9033         |

**Manual feature extraction**
| **Model**                        | **Test Accuracy** | **Test AUC** | **Test Recall** | **Test Precision** | **Test F1 Score** |
|----------------------------------|-------------------|--------------|-----------------|----------------|---------------|
| Gradient Boosting Classifier      | 0.9629          | 0.9892     | 0.9629       | 0.9631         | 0.9629         |
| Light Gradient Boosting Machine   | 0.9629          | 0.9861     | 0.9629       | 0.9630         | 0.9629         |
| Random Forest Classifier          | 0.9643          | 0.9865     | 0.9643       | 0.9645         | 0.9643         |
| Decision Tree Classifier          | 0.9272          | 0.9528     | 0.9272       | 0.9286         | 0.9275         |
| Extra Trees Classifier            | 0.9501          | 0.9834     | 0.9501       | 0.9506         | 0.9500         |

<img src="figure/confusion_matrix.png" alt="drawing" width="600"/>


