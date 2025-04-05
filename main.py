import torch
from src.dataset import create_dataloaders
from src.models import get_resnet18_pretrained, get_vgg16_pretrained, get_vgg16_simple,get_resnet18_simple
from src.utils import save_model
from src.train import train  
from src.train import val   
from src.test import test   

train_dir = r"C:\Users\Lenovo\Desktop\AI_Project\balanced_data\train"  
val_dir = r"C:\Users\Lenovo\Desktop\AI_Project\balanced_data\val"      
test_dir = r"C:\Users\Lenovo\Desktop\AI_Project\dataset\TEST"          
log_dir = "outputs/logs"        # TensorBoard logs
model_dir = "outputs/models"    # Model save path
batch_size = 32
num_epochs = 15
device = torch.device("cuda")

# directories for logs and models
import os
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)


train_loader, val_loader, test_loader, class_names = create_dataloaders(
    train_dir=train_dir,
    val_dir=val_dir,
    test_dir=test_dir,
    batch_size=batch_size
)

# ResNet18 Training
print("Training VGG16...")
#print(class_names)
vgg16 = get_vgg16_simple(num_classes=len(class_names)).to(device)
optimizer_sgd = torch.optim.SGD(vgg16.parameters(), lr=0.0005 , momentum=0.9)
optimizer_adam = torch.optim.Adam(vgg16.parameters(), lr=0.0001 )
criterion = torch.nn.CrossEntropyLoss()


train(
    model=vgg16,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer_adam,
    loss_fn=criterion,
    n_epochs=num_epochs,
    device=device
)

# saving the trained model
torch.save(vgg16.state_dict(), os.path.join(model_dir, "vgg16_not_pt_adam.pth"))
print(f"VGG16 model saved to {os.path.join(model_dir, 'vgg16_not_pt_adam.pth')}")


# test
test(vgg16, test_data_path=test_dir, device=device)