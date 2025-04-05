import torch
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from models import get_resnet18_pretrained  # Ensure this is your model's architecture
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

# Paths and Configurations
model_path = r"C:\Users\Lenovo\Desktop\AI_Project\checkpoints\resnet18_pretrained_adam.pth"  # Path to your saved .pth file
test_data_path = r"C:\Users\Lenovo\Desktop\AI_Project\dataset\TEST"  # Path to test dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load test dataset with transformations
test_transforms = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to match input size for the model
    transforms.ToTensor(),          # Convert PIL images to tensors
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize (same mean/std as training)
])

test_dataset = ImageFolder(root=test_data_path, transform=test_transforms)
class_names = test_dataset.classes  # Retrieve class names
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load the model (assuming ResNet18 architecture)
model = get_resnet18_pretrained(num_classes=len(class_names))  # Adjust for the correct number of classes
model.to(device)

# Load the checkpoint containing model weights, optimizer, and epoch
checkpoint = torch.load(model_path, map_location=device)

# Extract the model's state_dict from the checkpoint and load it into the model
model.load_state_dict(checkpoint['state_dict'])  # Only load the model weights

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