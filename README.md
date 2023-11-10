# Autoaudiogram
Object detection-based audiogram classification

## Overview
AutoAudiogram is an object detection-based audiogram classification system. the model use object detection to detect the audiological symbols form audiogram then use to classify the degree of hearing loss severity

## Get Start
1. Clone the repository: `git clone https://github.com/biodatlab/autoaudiogram.git`
2. Navigate into the project directory: `cd autoaudiogram`

## Graph and table training
1. navigate to train directory `cd src/data_extracter/graph_table`
2. organize folder in this directory 
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
3. training
```
python -m train
```

## Audiological symbol training
1. navigate to train directory `cd src/data_extracter/symbol_detection`
2. organize folder in this directory 
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
3. training
```
python -m train
```
## Severity classification
1. navigate to train directory `cd src/classification`
2. organize folder in this directory 
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
3. training
```
python -m train --path <path to train file>
```
  Example 
  ```
  python -m train --path "data/data_extract.csv"
  ```


# Evaluation
