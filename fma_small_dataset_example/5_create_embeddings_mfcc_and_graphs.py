import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
import os
from prettytable import PrettyTable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import networkx as nx
from sklearn.preprocessing import LabelEncoder


def compute_mff_for_dataset(dir_path, emb_dim):
    intermediate_df = pd.read_csv(os.path.join(dir_path, 'intermediate_df.csv'))
    mfccs = []
    file_paths = intermediate_df['file_path_5s'].values
    for f in tqdm(file_paths):
        y, sr = librosa.load(f, sr=None)
        mfcc_unnorm = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=emb_dim)
        mfcc_mean = np.mean(mfcc_unnorm, axis=1)
        mfccs.append(mfcc_mean)
    mfcc_array = np.array(mfccs)
    np.save("mfcc_plain_embeddings_fma_5s.npy", mfcc_array)
    return mfcc_array


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, out_channels)
        self.conv2 = SAGEConv(out_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return x


def train_sage_model(model, optimizer, criterion, num_epochs, data_loader, labels_tensor):
    losses = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()
        for data in data_loader:
            out = model(data)
            loss = criterion(out, labels_tensor)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(data_loader)
        losses.append(epoch_loss)
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    plt.plot(range(1, len(losses)+1), losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Graph SAGE Train Loss (vs) Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('GraphSAGE_loss_plot.png')
    plt.show()

    model.eval()
    with torch.no_grad():
        sage_embeddings = model(data).detach().numpy()

    np.save("mfcc_sage_embeddings_fma_5s.npy", sage_embeddings)
    print(sage_embeddings.shape)

class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, out_channels)
        self.conv2 = GCNConv(out_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


def train_gcn_model(model, optimizer, criterion, num_epochs, data_loader, labels_tensor):
    losses = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()
        for data in data_loader:
            out = model(data)
            loss = criterion(out, labels_tensor)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(data_loader)
        losses.append(epoch_loss)
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    plt.plot(range(1, len(losses)+1), losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GCN Train Loss (vs) Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('GCN_loss_plot.png')
    plt.show()

    model.eval()
    with torch.no_grad():
        gcn_embeddings = model(data).detach().numpy()

    np.save("mfcc_gcn_embeddings_fma_5s.npy", gcn_embeddings)
    print(gcn_embeddings.shape)


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(f"Model: {model}")
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params



if __name__ == '__main__':
    dir_path = '/home/ics/Documents/Education/Media_engg/Sem_1/CASM/Project/fma_small_dataset_example'
    mfcc = compute_mff_for_dataset(dir_path, 100)
    G = nx.read_gml(os.path.join(dir_path, 'graph.gml'))
    node_to_index = {node: index for index, node in enumerate(G.nodes)}
    edge_index = (
        torch.tensor([[node_to_index[edge[0]], node_to_index[edge[1]]] for edge in G.edges])
        .t()
        .contiguous()
    )
    data = Data(edge_index=edge_index)
    data.x = torch.tensor(mfcc, dtype=torch.float32)
    data_loader = DataLoader([data], batch_size=16, shuffle=True)

    intermediate_df = pd.read_csv(os.path.join(dir_path, 'intermediate_df.csv'))
    labels = LabelEncoder().fit_transform(intermediate_df["genre_top"])
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    sage_model = GraphSAGE(in_channels=100, out_channels=100)
    sage_criterion = nn.CrossEntropyLoss()
    sage_optimizer = torch.optim.Adam(sage_model.parameters(), lr=0.01)
    count_parameters(sage_model)
    train_sage_model(sage_model, sage_optimizer, sage_criterion, 20, data_loader, labels_tensor)

    gcn_model = GCN(in_channels=100, out_channels=100)
    gcn_criterion = nn.CrossEntropyLoss()
    gcn_optimizer = torch.optim.Adam(gcn_model.parameters(), lr=0.01)
    count_parameters(gcn_model)
    train_gcn_model(sage_model, sage_optimizer, sage_criterion, 20, data_loader, labels_tensor)
