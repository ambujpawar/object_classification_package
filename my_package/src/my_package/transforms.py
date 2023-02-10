"""
Script to perform transformation on the data before we:
    -train the model
    -validate the model
    -test the model
"""

from torchvision import transforms

# Define the transformations

def get_transforms()-> dict:
    """
    Function to define the transformations.
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop([224, 224]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
        ])
    }
    return data_transforms
