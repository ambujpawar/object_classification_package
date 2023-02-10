"""
Script to train a model.

Input:
    -- data_path: path to the data
    -- backbone: backbone to use
    -- model_path: path where the model will be stored
"""

import argparse
import copy

# Third party imports
from loguru import logger
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision.datasets import OxfordIIITPet
# In module imports
from transforms import get_transforms
from model import load_model


# HYPERPARAMETERS
BATCH_SIZE = 8
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model, dataloaders, dataset_sizes):
    """
    Function to train a model.
    Params:
        model: model to train
        train_dataloader: train dataloader
        test_dataloader: test dataloader
    """
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(NUM_EPOCHS):
        logger.info('Epoch {}/{}'.format(epoch, NUM_EPOCHS - 1))
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            if phase == 'train':
                mlflow.log_metric("train_loss", running_loss / dataset_sizes[phase])
                mlflow.log_metric("train_acc", running_corrects.double() / dataset_sizes[phase])
            else:
                mlflow.log_metric("test_loss", running_loss / dataset_sizes[phase])
                mlflow.log_metric("test_acc", running_corrects.double() / dataset_sizes[phase])
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            logger.info('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def main(args):
    """
    Where the magic happens.
    """
    my_transforms = get_transforms()
    dataset = OxfordIIITPet(root=args.data_path, download=False, transform=my_transforms["train"])
    logger.info(f"Length of dataset: {len(dataset)}")
    class_names = dataset.classes
    logger.info(f"Class names: {class_names}")
    logger.info(f"Number of classes: {len(class_names)}")

    train_dataloader =torch.utils.data.DataLoader(dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
        )
    
    # Load the test dataset
    test_dataset = OxfordIIITPet(
        root=args.data_path,
        split="test",
        download=False,
        transform=my_transforms["test"]
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    logger.info(f"Length of test dataset: {len(test_dataset)}")
    my_model = load_model(args.backbone, num_classes=len(class_names))
    dataloaders = {"train": train_dataloader, "test": test_dataloader}
    dataset_sizes = {"train": len(dataset), "test": len(test_dataset)}

    mlflow.set_experiment("my_experiment")
    mlflow.log_artifact(args.data_path)

    with mlflow.start_run():
        mlflow.log_param("backbone", args.backbone)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("num_epochs", NUM_EPOCHS)

        best_model = train_model(my_model, dataloaders, dataset_sizes)
        torch.save(best_model.state_dict(), 'best_model.pt')
        mlflow.pytorch.log_model(best_model, "model")
        mlflow.end_run()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/Users/ambujpawar/Desktop/GithubProjects/object_classification_package/my_package/data")
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--model_path", type=str, default="model")
    args = parser.parse_args()
    main(args)
