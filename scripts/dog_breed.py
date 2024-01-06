import torch
from tqdm.notebook import tqdm
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

class DogBreedDataset(Dataset):
    
    def __init__(self, ds, transform=None):
        self.ds = ds
        self.transform = transform
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        img, label = self.ds[idx]
        if self.transform:
            img = self.transform(img)  
            return img, label

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    train_info = {}
    for inputs, labels in tqdm(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset)
    
    train_info["train_accuracy"] = epoch_acc
    train_info["train_loss"] = epoch_acc
    print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    return train_info

def test_model(model, test_loader, criterion, device):
    eval_info = {}
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
    epoch_loss = running_loss / len(test_loader.dataset)
    epoch_acc = running_corrects.double() / len(test_loader.dataset)
    
    eval_info["test_accuracy"] = epoch_acc
    eval_info["model_preds"] = running_corrects
    eval_info["true_labels"] = test_loader.dataset
    print(f'Test Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    return eval_info


def train_and_test(model, model_name, train_loader, test_loader, criterion, optimizer, device, num_epochs=10):
    train_history = []
    eval_history = []
    num_epochs = 15
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        train_info = train_one_epoch(model, train_loader, criterion, optimizer, device)
        eval_info = test_model(model, test_loader, criterion, device)
        
        train_history.append(train_info)
        eval_history.append(eval_info)
        
    torch.save(model.state_dict(), f"{model_name}_results.pt")
    
    return train_history, eval_history

    
def create_confusion_matrix(model, dataloader, device='cuda'):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    confusion = confusion_matrix(all_labels, all_predictions)
    return confusion

def plot_confusion_matrix(confusion_matrix, class_names):
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=75)
    plt.yticks(rotation=0)
    plt.show()
  
def plot_training_history(train_history, eval_history):
    train_losses = [info['train_loss'].cpu().detach().numpy() for info in train_history]
    train_accuracies = [info['train_accuracy'].cpu().detach().numpy() for info in train_history]
    test_accuracies = [info['test_accuracy'].cpu().detach().numpy() for info in eval_history]
    epochs = range(1, len(train_history) + 1)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train')
    plt.title('Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train')
    plt.plot(epochs, test_accuracies, label='Test')
    plt.title('Train and Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

  
class CatBreedDataset(Dataset):
    
    def __init__(self, ds, transform=None):
        self.ds = ds
        self.transform = transform
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        img, label = self.ds[idx]
        if self.transform:
            img = self.transform(img)  
            return img, label