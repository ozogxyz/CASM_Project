from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os


class SimpleMLP(nn.Module):
    def __init__(self, n_in, n_classes):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(n_in, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, n_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.softmax(self.fc3(x), dim=1)
        return x


def train_mlp_model(model_name, model, optimizer, criterion, num_epochs, emb_tensor, labels_tensor):
    emb_train, emb_test, labels_train, labels_test = train_test_split(
        emb_tensor, labels_tensor, test_size=0.2, random_state=42
    )
    train_dataset = TensorDataset(emb_train, labels_train)
    test_dataset = TensorDataset(emb_test, labels_test)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)


    train_losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for mfcc_batch, labels_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(mfcc_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Embeddings Trainin (MLP)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{model_name}_embedings_training_loss_curve.png')
    plt.show()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for mfcc_batch, labels_batch in test_loader:
            outputs = model(mfcc_batch)
            _, predicted = torch.max(outputs, 1)
            total += labels_batch.size(0)
            correct += (predicted == labels_batch).sum().item()

    print(f"Accuracy: {100 * correct / total}%")


if __name__ == '__main__':
    dir_path = '/home/ics/Documents/Education/Media_engg/Sem_1/CASM/Project/fma_small_dataset_example'

    intermediate_df = pd.read_csv(os.path.join(dir_path, 'intermediate_df.csv'))
    labels = LabelEncoder().fit_transform(intermediate_df["genre_top"])
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    path_for_mfcc_plain = dir_path + '/mfcc_plain_embeddings_fma_5s.npy'
    mfcc_plain = np.load(path_for_mfcc_plain)
    print(mfcc_plain.shape)
    mfcc_plain_scaled = StandardScaler().fit_transform(mfcc_plain)
    mfcc_plain_tensor = torch.tensor(mfcc_plain_scaled, dtype=torch.float32)
    mfcc_plain_mlp_model = SimpleMLP(n_in=mfcc_plain_tensor.shape[1], n_classes=8)
    mfcc_plain_mlp_model_criterion = nn.CrossEntropyLoss()
    mfcc_plain_mlp_model_optimizer = torch.optim.Adam(mfcc_plain_mlp_model.parameters(), lr=0.001)
    model_name = 'mfcc_plain_mlp_model'
    train_mlp_model(model_name, mfcc_plain_mlp_model, mfcc_plain_mlp_model_optimizer, mfcc_plain_mlp_model_criterion, 20, mfcc_plain_tensor, labels_tensor)

    path_for_mfcc_sage = dir_path + '/mfcc_sage_embeddings_fma_5s.npy' 
    mfcc_sage = np.load(path_for_mfcc_sage)
    print(mfcc_sage.shape)
    mfcc_sage_scaled = StandardScaler().fit_transform(mfcc_sage)
    mfcc_sage_tensor = torch.tensor(mfcc_sage_scaled, dtype=torch.float32)
    mfcc_sage_mlp_model = SimpleMLP(n_in=mfcc_sage_tensor.shape[1], n_classes=8)
    mfcc_sage_mlp_model_criterion = nn.CrossEntropyLoss()
    mfcc_sage_mlp_model_optimizer = torch.optim.Adam(mfcc_sage_mlp_model.parameters(), lr=0.001)
    model_name = 'mfcc_sage_mlp_model'
    train_mlp_model(model_name, mfcc_sage_mlp_model, mfcc_sage_mlp_model_optimizer, mfcc_sage_mlp_model_criterion, 20, mfcc_sage_tensor, labels_tensor)

    path_for_mfcc_gcn = dir_path + '/mfcc_gcn_embeddings_fma_5s.npy'
    mfcc_gcn = np.load(path_for_mfcc_gcn)
    print(mfcc_gcn.shape)
    mfcc_gcn_scaled = StandardScaler().fit_transform(mfcc_gcn)
    mfcc_gcn_tensor = torch.tensor(mfcc_gcn_scaled, dtype=torch.float32)
    mfcc_gcn_mlp_model = SimpleMLP(n_in=mfcc_gcn_tensor.shape[1], n_classes=8)
    mfcc_gcn_mlp_model_criterion = nn.CrossEntropyLoss()
    mfcc_gcn_mlp_model_optimizer = torch.optim.Adam(mfcc_gcn_mlp_model.parameters(), lr=0.001)
    model_name = 'mfcc_gcn_mlp_model'
    train_mlp_model(model_name, mfcc_gcn_mlp_model, mfcc_gcn_mlp_model_optimizer, mfcc_gcn_mlp_model_criterion, 20, mfcc_gcn_tensor, labels_tensor)






