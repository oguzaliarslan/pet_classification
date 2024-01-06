import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import torch
from torch import nn, optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchvision import utils


# setting dataset for loader
class CatvsDogDataset(Dataset):
    def __init__(self, data, label, input_size):
        super().__init__()

        self.data = data / 255
        self.label = label
        self.input_size = input_size

    def __getitem__(self, index):
        # resize the image according to the models input size
        img_transform = transforms.Compose([transforms.Resize(self.input_size, antialias=True)
                                            #,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                            ])

        data_point = img_transform(self.data[index])
        data_label = self.label[index]

        return data_point, data_label

    def __len__(self):
        return self.data.shape[0]


def train(network, optimizer, loss_fn, train_loader, device):
    true_preds, num_preds = 0., 0.
    for train_inputs, train_labels in tqdm(train_loader):
        train_inputs = train_inputs.to(device)
        train_labels = train_labels.to(device)

        preds = network(train_inputs)

        loss = loss_fn(preds, train_labels.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # finding and storing the train predictions to find accuracy
        with torch.no_grad():
            pred_labels = F.sigmoid(preds) > 0.5

            true_preds += (pred_labels == train_labels).sum()
            num_preds += train_labels.shape[0]

    accuracy = ((true_preds / num_preds) * 100).item()

    return accuracy, loss.item()


def evalf(network, test_loader, device):
    model_preds = []
    true_labels = []
    eval_info = {}

    true_preds, num_preds = 0., 0.
    with torch.no_grad():
        for val_inputs, val_labels in tqdm(test_loader):
            val_inputs = val_inputs.to(device)
            val_labels = val_labels.to(device)

            preds = network(val_inputs)
            pred_labels = F.sigmoid(preds) > 0.5

            true_preds += (pred_labels == val_labels).sum()
            num_preds += val_labels.shape[0]

            model_preds += pred_labels.detach().cpu().numpy().astype(int).tolist()
            true_labels += val_labels.detach().cpu().numpy().tolist()

    accuracy = ((true_preds / num_preds) * 100).item()

    eval_info["accuracy"] = accuracy
    eval_info["model_preds"] = model_preds
    eval_info["true_labels"] = true_labels

    return eval_info


def experiment_pipeline(network, X_train, y_train, X_test, y_test, optimizer,
                        loss_fn, batch_size, epochs, input_size, device):
    # loaders
    train_loader = DataLoader(CatvsDogDataset(X_train, y_train, input_size), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(CatvsDogDataset(X_test, y_test, input_size), shuffle=False)

    train_accs = []
    train_losses = []

    # train model
    network.train()

    for e in range(epochs):
        acc, l = train(network, optimizer, loss_fn, train_loader, device)

        train_accs.append(acc)
        train_losses.append(l)

        print(f"Epoch-{e + 1}:")
        print(f"Train Accuracy: {acc}\nTrain Loss: {l}")

    # eval model
    network.eval()

    evals = evalf(network, test_loader, device)
    evals["train_accuracies"] = train_accs
    evals["train_losses"] = train_losses

    plot_results(evals)

    return evals


def plot_results(evals):
    print(f"Test Accuracy: {evals['accuracy']}")

    # confusion matrix
    plt.figure()
    sns.heatmap(confusion_matrix(evals["true_labels"], evals["model_preds"]),
                annot=True, fmt=".0f", cmap="Blues",
                annot_kws={"size": 15},
                xticklabels=["Dog", "Cat"],
                yticklabels=["Dog", "Cat"])
    plt.title("InceptionV3 Cat-vs-Dog Conf. Matrix")
    plt.xlabel("Preds")
    plt.ylabel("Labels")

    plt.show()