# object_classification_package
Simple package to train and evaluate the Object Classification model on Oxford-IIIT Pet Dataset

Basic Requirements:
- Train and evaluate the model on Oxford-IIIT Pet Dataset
- And an API endpoint to predict the class of the image
- Log experiment details to MLflow
  
## Installation

Requirements:
- Python 3.9+
- pytorch
- torchvision
- mlflow
- fastapi
- uvicorn


## Usage

Store the dataset in the folder: my_package/data folder

To train the model:
```bash
python my_package/train.py
```

To run the API:
```bash
uvicorn api:app --reload
```

