import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, GCNConv, TransformerConv
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import librosa


def load_data(path_for_csv):
    df = pd.read_csv(path_for_csv)
    unique_genres = sorted(list(set(df["genre"])))
    genre_to_id = {unique_genres[_]: _ for _ in range(len(unique_genres))}
    class_ids = [genre_to_id[_] for _ in df["genre"]]
    class_ids = np.array(class_ids)
    np.save("labels.npy", class_ids)
    return df, class_ids


def build_graph(df):
    G = nx.Graph()
    G.add_nodes_from(df["fn_wav"])
    for i, row1 in df.iterrows():
        for j, row2 in df.iterrows():
            if row1["fn_wav"] != row2["fn_wav"] and row1["genre"] == row2["genre"]:
                G.add_edge(row1["fn_wav"], row2["fn_wav"])
    node_to_index = {node: index for index, node in enumerate(G.nodes)}
    edge_index = (
        torch.tensor(
            [[node_to_index[edge[0]], node_to_index[edge[1]]] for edge in G.edges]
        )
        .t()
        .contiguous()
    )
    return edge_index, node_to_index


def compute_mfcc(fn_wav):
    y, sr = librosa.load(fn_wav)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfcc, axis=1)


def load_mfcc_data(df, dir_dataset):
    n_files = df.shape[0]
    mfcc = np.zeros((n_files, 40))
    for n in range(n_files):
        fn_wav = os.path.join(dir_dataset, df["fn_wav"][n])
        mfcc[n, :] = compute_mfcc(fn_wav)
    return mfcc


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


class GraphTransformer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphTransformer, self).__init__()
        self.conv1 = TransformerConv(in_channels, out_channels, heads=4, concat=True)
        self.conv2 = TransformerConv(
            out_channels * 4, out_channels, heads=4, concat=False
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


def train(model, data_loader, labels, criterion, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        for data in data_loader:
            out = model(data)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item()}")


def main(args):
    df, class_ids = load_data("songs_genre.csv")
    edge_index, node_to_index = build_graph(df)
    dir_dataset = "genres_mini"
    mfcc = load_mfcc_data(df, dir_dataset)

    data = Data(edge_index=edge_index)
    data.x = torch.tensor(mfcc, dtype=torch.float32)
    data_loader = DataLoader([data], batch_size=16, shuffle=True)

    labels = torch.tensor(class_ids, dtype=torch.long)

    if args.model == "GraphSAGE":
        model = GraphSAGE(in_channels=40, out_channels=args.embedding_size)
    elif args.model == "GraphCN":
        model = GCN(in_channels=40, out_channels=args.embedding_size)
    elif args.model == "GraphTransformer":
        model = GraphTransformer(in_channels=40, out_channels=args.embedding_size)
    else:
        raise ValueError(f"Unsupported model type: {args.model}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    train(model, data_loader, labels, criterion, optimizer, args.epochs)

    model.eval()
    with torch.no_grad():
        embeddings = model(data).detach().numpy()

    np.save(f"{args.model}_embeddings.npy", embeddings)

    print(embeddings.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embedding_size", type=int, default=40, help="Size of the embedding"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs to train"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["GraphSAGE", "GraphCN", "GraphTransformer"],
        default="GraphSAGE",
        help="Model type to use",
    )
    args = parser.parse_args()
    main(args)
