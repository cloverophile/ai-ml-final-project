import os
import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import F1Score

# Configuration
save_model_path = "checkpoints/"
pth_name = "saved_model.pth"
os.makedirs(save_model_path, exist_ok=True)

def val(model, val_loader, loss_function, writer, epoch, device):
    f1 = F1Score(num_classes=len(val_loader.dataset.classes),average='weighted', task='multiclass')
    val_iterator = enumerate(val_loader)
    f1_list, f1t_list = [], []
    total_loss = 0
    total_correct = 0
    total_samples = 0

    model.eval()
    tq = tqdm.tqdm(total=len(val_loader), desc="Validation")
    with torch.no_grad():
        for _, (images, labels) in val_iterator:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            predictions = model(images)
            loss = loss_function(predictions, labels.long())  # Change labels to .long()
            total_loss += loss.item()

            # Compute predictions
            #predictions = predictions.softmax(dim=1)
            predictions = torch.argmax(predictions, dim=1)
            f1_list.extend(predictions.cpu().numpy())
            f1t_list.extend(labels.cpu().numpy())
            # Calculate accuracy
            #_, predicted = torch.max(predictions, 1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)


            tq.update(1)

    tq.close()

    # Calculate F1 Score
    f1_score_value = f1(torch.tensor(f1_list), torch.tensor(f1t_list))
    val_accuracy = total_correct / total_samples

    writer.add_scalar("Validation F1", f1_score_value, epoch)
    writer.add_scalar("Validation Loss", total_loss / len(val_loader), epoch)
    writer.add_scalar("Validation Accuracy", val_accuracy, epoch)

    #print(f'list_pred {f1_list}')
    #print(f'list_pred {f1t_list}')
    print(f"Validation Loss: {total_loss / len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation F1 Score: {f1_score_value:.4f}")



def train(model, train_loader, val_loader, optimizer, loss_fn, n_epochs, device):
    """
    Train the model on the training dataset.
    Logs loss and saves the model at each epoch.

    Args:
        model: PyTorch model.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        optimizer: Optimizer for training.
        loss_fn: Loss function.
        n_epochs: Number of epochs.
        device: Device for computation (CPU/GPU).
    """
    writer = SummaryWriter()

    model.to(device)
    for epoch in range(n_epochs):
        model.train()  
        running_loss = 0.0
        total_correct = 0
        total_samples = 0
        f1 = F1Score(num_classes=len(train_loader.dataset.classes),average='macro', task='multiclass').to(device)  # Move F1 metric to device

        tq = tqdm.tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{n_epochs}")
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Zero the gradient
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # Backward pass and optimizer step
            loss.backward()
            optimizer.step()

            # Update running loss
            running_loss += loss.item()

            # Compute predictions
            predictions = torch.argmax(outputs, dim=1)

            # Update F1 score
            f1.update(predictions, labels)
            
            # Accuracy calculation
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

            tq.set_postfix(loss=loss.item())
            tq.update(1)

        tq.close()

        # Compute F1 score at the end of the epoch
        f1_score_value = f1.compute()

        # Calculate training accuracy
        train_accuracy = total_correct / total_samples
        epoch_loss = running_loss / len(train_loader)

        # Log training metrics
        writer.add_scalar("Training Loss", epoch_loss, epoch)
        writer.add_scalar("Training Accuracy", train_accuracy, epoch)
        writer.add_scalar("Training F1", f1_score_value, epoch)

        # Print training metrics
        print(f"Epoch [{epoch + 1}/{n_epochs}], Training Loss: {epoch_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, Training F1: {f1_score_value:.4f}")

        # Validate the model
        val(model, val_loader, loss_fn, writer, epoch, device)

        # Save the model checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(checkpoint, os.path.join(save_model_path, pth_name))
        print(f"Model saved at {save_model_path}{pth_name}")