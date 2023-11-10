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
| Model                          | Test    | Test AUC | Test Recall | Test Precision | Test F1 Score |
|-------------------------------|---------|----------|-------------|----------------|---------------|
| Gradient Boosting              | 0.9472  | 0.9873   | 0.9472      | 0.9473         | 0.9471        |
| Light Gradient Boosting Machine| 0.9401  | 0.9883   | 0.9401      | 0.941          | 0.9401        |
| Random Forest                  | 0.9415  | 0.9866   | 0.9415      | 0.9417         | 0.9414        |
| Decision Tree                  | 0.9215  | 0.9479   | 0.9215      | 0.9221         | 0.9213        |
| Extra Trees                    | 0.9116  | 0.9856   | 0.9116      | 0.9126         | 0.9111        |

**Manual feature extraction**
| Model                          | Test    | Test AUC | Test Recall | Test Precision | Test F1 Score |
|-------------------------------|---------|----------|-------------|----------------|---------------|
| Gradient Boosting              | 0.9615  | 0.9901   | 0.9615      | 0.9616         | 0.9614        |
| Light Gradient Boosting Machine| 0.9615  | 0.9877   | 0.9615      | 0.9618         | 0.9615        |
| Random Forest                  | 0.9643  | 0.9839   | 0.9643      | 0.9645         | 0.9643        |
| Decision Tree                  | 0.9144  | 0.9452   | 0.9144      | 0.9165         | 0.915         |
| Extra Trees                    | 0.9501  | 0.9814   | 0.9501      | 0.9505         | 0.95          |

<img src="figure/confusion_matrix.png" alt="drawing" width="600"/>


