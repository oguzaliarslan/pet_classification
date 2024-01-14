import torch
from tqdm.notebook import tqdm
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import torchvision.models as models


def get_pretrained_model(model_name, num_classes):
    if model_name == 'resnet18':
        model = models.resnet18(weights=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    
    elif model_name == 'alexnet':
        model = models.alexnet(weights=True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)
    
    elif model_name == 'vgg19bn':
        model = models.vgg19_bn(weights=True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)

    elif model_name == 'densenet':
        model = models.densenet121(weights=True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    
    elif model_name == 'mobilenetv3':
        model = models.mobilenet_v3_large(weights=True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
    
    else:
        raise ValueError("Invalid model name. Please choose from 'resnet18', 'resnet50', 'alexnet', 'vgg19bn', 'densenet', or 'mobilenetv3'")
    
    return model
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
    train_info["train_loss"] = epoch_loss
    print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    return train_info

def test_model(model, test_loader, criterion, device):
    eval_info = {}
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    model_preds = []
    true_labels = [] 
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            model_preds += preds.detach().cpu().numpy().astype(int).tolist()
            true_labels += labels.detach().cpu().numpy().tolist()

    epoch_loss = running_loss / len(test_loader.dataset)
    epoch_acc = running_corrects.double() / len(test_loader.dataset)
    
    eval_info["test_accuracy"] = epoch_acc
    eval_info["model_preds"] = model_preds
    eval_info["true_labels"] = true_labels
    print(f'Test Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    return eval_info


def train_and_test(model, model_name, train_loader, test_loader, criterion, optimizer, device, num_epochs=10, optimizer_name='none', type='none'):
    train_history = []
    eval_history = []
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        train_info = train_one_epoch(model, train_loader, criterion, optimizer, device)
        eval_info = test_model(model, test_loader, criterion, device)
        
        train_history.append(train_info)
        eval_history.append(eval_info)
    
    os.makedirs(f'breed_results/{type}/{optimizer_name}/', exist_ok=True)
    torch.save([train_history, eval_history], f"breed_results/{type}/{optimizer_name}/{model_name}_results.pt")
    
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

def plot_confusion_matrix(confusion_matrix, class_names, model_name):
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'Confusion Matrix for {model_name}')
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

def load_the_results(type='none'):
    results_directory = f'breed_results/{type}/adam'
    all_files = os.listdir(results_directory)
    model_files = [file for file in all_files if file.endswith('.pt')]
    model_names = [file.split('_')[0] for file in model_files]

    all_models_train_data = []
    for model_file in model_files:
        train_data = torch.load(os.path.join(results_directory, model_file))
        all_models_train_data.append(train_data)
        
    train_accuracies = [] 
    test_accuracies = []
    train_losses = []
    last_true_labels = []
    last_model_preds = []
    for idx, train_data in enumerate(all_models_train_data, start=1):
        model_train_accuracies = [] 
        model_test_accuracies = []
        model_train_losses = [] 
        last_true_label = None
        last_model_pred = None

        for epoch, epoch_data in enumerate(train_data, start=1):
            try:
                for data in epoch_data:
                    if 'train_accuracy' in data:
                        train_accuracy = data['train_accuracy'].item()
                        model_train_accuracies.append(train_accuracy)
                        train_loss = data['train_loss']
                        model_train_losses.append(train_loss)
                    elif 'test_accuracy' in data: 
                        test_accuracy = data['test_accuracy'].item()
                        model_test_accuracies.append(test_accuracy)
                        last_true_label = data['true_labels']
                        last_model_pred = data['model_preds']
            except KeyError:
                pass

        train_accuracies.append(model_train_accuracies)
        test_accuracies.append(model_test_accuracies)
        train_losses.append(model_train_losses)
        last_true_labels.append(last_true_label)
        last_model_preds.append(last_model_pred)

    model_accuracies_dict = {model_names[i]: train_accuracies[i] for i in range(len(model_names))}
    model_test_accuracies_dict = {model_names[i]: test_accuracies[i] for i in range(len(model_names))}
    model_losses_dict = {model_names[i]: train_losses[i] for i in range(len(model_names))}
    return model_accuracies_dict, model_test_accuracies_dict, model_losses_dict, last_true_labels, last_model_preds, model_names
  
  
def load_the_results_sgd(type='none'):
    results_directory = f'breed_results/{type}/sgd'
    all_files = os.listdir(results_directory)
    model_files = [file for file in all_files if file.endswith('.pt')]
    model_names = [file.split('_')[0] for file in model_files]

    all_models_train_data = []
    for model_file in model_files:
        train_data = torch.load(os.path.join(results_directory, model_file))
        all_models_train_data.append(train_data)
        
    train_accuracies = [] 
    test_accuracies = []
    train_losses = []
    last_true_labels = []
    last_model_preds = []
    for idx, train_data in enumerate(all_models_train_data, start=1):
        model_train_accuracies = [] 
        model_test_accuracies = []
        model_train_losses = [] 
        last_true_label = None
        last_model_pred = None

        for epoch, epoch_data in enumerate(train_data, start=1):
            try:
                for data in epoch_data:
                    if 'train_accuracy' in data:
                        train_accuracy = data['train_accuracy'].item()
                        model_train_accuracies.append(train_accuracy)
                        train_loss = data['train_loss']
                        model_train_losses.append(train_loss)
                    elif 'test_accuracy' in data: 
                        test_accuracy = data['test_accuracy'].item()
                        model_test_accuracies.append(test_accuracy)
                        last_true_label = data['true_labels']
                        last_model_pred = data['model_preds']
            except KeyError:
                pass

        train_accuracies.append(model_train_accuracies)
        test_accuracies.append(model_test_accuracies)
        train_losses.append(model_train_losses)
        last_true_labels.append(last_true_label)
        last_model_preds.append(last_model_pred)

    model_accuracies_dict = {model_names[i]: train_accuracies[i] for i in range(len(model_names))}
    model_test_accuracies_dict = {model_names[i]: test_accuracies[i] for i in range(len(model_names))}
    model_losses_dict = {model_names[i]: train_losses[i] for i in range(len(model_names))}
    return model_accuracies_dict, model_test_accuracies_dict, model_losses_dict, last_true_labels, last_model_preds, model_names

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