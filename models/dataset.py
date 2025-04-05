import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def create_dataloaders(train_dir, val_dir, test_dir, batch_size=32):
    """
    Create DataLoaders for train, validation, and test datasets.

    Args:
        train_dir (str): Path to the training directory with augmented images.
        val_dir (str): Path to the validation directory.
        test_dir (str): Path to the test directory.
        batch_size (int): Batch size for DataLoader.

    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    #transforms
    train_transforms = transforms.Compose([
        transforms.Resize((128, 128)),   
        transforms.ToTensor(),          
        transforms.Normalize([0.5], [0.5])  # normalize images to mean=0.5, std=0.5
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((128, 128)),  
        transforms.ToTensor(),          
        transforms.Normalize([0.5], [0.5])  
    ])

    # load datasets
    train_dataset = ImageFolder(root=train_dir, transform=train_transforms)
    val_dataset = ImageFolder(root=val_dir, transform=test_transforms)
    test_dataset = ImageFolder(root=test_dir, transform=test_transforms)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_dataset.classes