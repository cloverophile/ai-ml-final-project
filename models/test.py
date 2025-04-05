import torch
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

def test(model, test_data_path, device, batch_size=32, image_size=(128, 128), mean=[0.5], std=[0.5]):
    """
    Evaluates a trained model on a test dataset.

    Parameters:
    - model (torch.nn.Module): The model to evaluate.
    - test_data_path (str): Path to the test dataset.
    - device (torch.device): The device to run the evaluation on (CPU or CUDA).
    - batch_size (int, optional): The batch size for the DataLoader. Default is 32.
    - image_size (tuple, optional): The target image size for the input images. Default is (128, 128).
    - mean (list, optional): The mean for normalization. Default is [0.5].
    - std (list, optional): The std for normalization. Default is [0.5].

    Returns:
    - None: Prints the accuracy and F1 score of the model on the test set.
    """

    # Load test dataset with transformations
    test_transforms = transforms.Compose([
        transforms.Resize(image_size),  # Resize to match input size for the model
        transforms.ToTensor(),          # Convert PIL images to tensors
        transforms.Normalize(mean=mean, std=std)  # Normalize (same mean/std as training)
    ])

    test_dataset = ImageFolder(root=test_data_path, transform=test_transforms)
    class_names = test_dataset.classes  # Retrieve class names
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Set the model to evaluation mode
    model.eval()

    # Initialize variables to store true labels and predicted labels
    all_true_labels = []
    all_pred_labels = []

    # Evaluate the model
    with torch.no_grad():  # No need to compute gradients during inference
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            # Move the inputs and labels to the device (GPU or CPU)
            inputs, labels = inputs.to(device), labels.to(device)

            # Get model predictions
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)  # Get the predicted class (index of max logit)

            # Append true and predicted labels to the lists
            all_true_labels.extend(labels.cpu().numpy())
            all_pred_labels.extend(preds.cpu().numpy())

    # Compute accuracy and F1 score
    accuracy = accuracy_score(all_true_labels, all_pred_labels)
    f1 = f1_score(all_true_labels, all_pred_labels, average='macro')  # Weighted F1 score

    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (Macro): {f1:.4f}")