import argparse
import os
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import librosa
import IPython.display as ipd


def calculate_cosine_similarity(embeddings, random_id):
    embeddings = np.array(embeddings)
    reference_embedding = embeddings[random_id]
    dot_product = np.dot(embeddings, reference_embedding)
    norms = np.linalg.norm(embeddings, axis=1)
    reference_norm = np.linalg.norm(reference_embedding)
    cosine_similarity = dot_product / (norms * reference_norm)
    return cosine_similarity


# p = 1: Manhattan distance
# p = 2: Euclidean distance
# p = 2: Minkowski distance
# p = inf: Chebyshev distance


def calculate_minkowski_distance(embeddings, random_id, p):
    reference_embedding = embeddings[random_id]
    minkowski_distance = np.sum(
        np.abs(embeddings - reference_embedding) ** p, axis=1
    ) ** (1 / p)
    return minkowski_distance


# def calculate_manhattan_distance(embeddings, random_id):
#     reference_embedding = embeddings[random_id]
#     manhattan_distance = np.sum(np.abs(embeddings - reference_embedding), axis=1)
#     return manhattan_distance

# def calculate_euclidean_distance(embeddings, random_id):
#     embeddings = np.array(embeddings)
#     distances = np.sqrt(np.sum((embeddings - embeddings[random_id]) ** 2, axis=1))
#     return distances

# def calculate_chebyshev_distance(embeddings, random_id):
#     reference_embedding = embeddings[random_id]
#     chebyshev_distance = np.max(np.abs(embeddings - reference_embedding), axis=1)
#     return chebyshev_distance


def calculate_hamming_distance(embeddings, random_id):
    reference_embedding = embeddings[random_id]
    hamming_distance = np.sum(embeddings != reference_embedding, axis=1)
    return hamming_distance


def calculate_jaccard_distance(embeddings, random_id):
    reference_embedding = embeddings[random_id]
    intersection = np.sum(np.minimum(embeddings, reference_embedding), axis=1)
    union = np.sum(np.maximum(embeddings, reference_embedding), axis=1)
    jaccard_distance = np.abs(1 - intersection / union)
    return jaccard_distance


class SimpleMLP(nn.Module):
    def __init__(self, n_in, n_classes=10):
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


def load_data(embedding_path, labels_path):
    embeddings = np.load(embedding_path)
    labels = np.load(labels_path)
    return embeddings, labels


def train_and_evaluate(
    embeddings, labels, num_epochs=200, batch_size=16, learning_rate=0.001
):
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    embeddings_tensor = torch.tensor(embeddings_scaled, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    mfcc_train, mfcc_test, labels_train, labels_test = train_test_split(
        embeddings_tensor, labels_tensor, test_size=0.2, random_state=42
    )

    train_dataset = TensorDataset(mfcc_train, labels_train)
    test_dataset = TensorDataset(mfcc_test, labels_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = SimpleMLP(n_in=embeddings_tensor.shape[1], n_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for mfcc_batch, labels_batch in test_loader:
            outputs = model(mfcc_batch)
            _, predicted = torch.max(outputs, 1)
            total += labels_batch.size(0)
            correct += (predicted == labels_batch).sum().item()
    accuracy_of_model = 100 * correct / total
    return accuracy_of_model


def main(config):
    embedding_path = f"{config['embeddings']}_embeddings.npy"
    labels_path = "labels.npy"

    embeddings, labels = load_data(embedding_path, labels_path)
    accuracy_of_model = train_and_evaluate(
        embeddings, labels, config["epochs"], config["batch_size"], config["lr"]
    )
    print(f"Accuracy of model using {embedding_path} embeddings: {accuracy_of_model}")
    random_id = np.random.randint(0, 100)
    print(f"Random Song ID: {random_id}")
    df = pd.read_csv(config["path_for_csv"])
    dir_dataset = config["dataset_path"]

    fn_wav = os.path.join(dir_dataset, df["fn_wav"][random_id])
    genre = df["genre"][random_id]
    print(f"File: {fn_wav} - Music Genre: {genre}")
    x, fs = librosa.load(fn_wav, sr=44100)
    ipd.display(ipd.Audio(data=x, rate=fs))

    # dist_mfcc = np.sqrt(np.sum((embeddings - embeddings[random_id]) ** 2, axis=1))
    dist_mfcc = calculate_minkowski_distance(embeddings, random_id, 2)
    idx = np.argsort(dist_mfcc)

    x, fs = librosa.load(fn_wav, sr=44100)
    ipd.display(ipd.Audio(data=x, rate=fs))

    for i in range(7):
        curr_idx = idx[i + 1]
        fn_wav = os.path.join(dir_dataset, df["fn_wav"][curr_idx])
        genre = df["genre"][curr_idx]
        print(f"File: {fn_wav} - Music Genre: {genre} - Distance {dist_mfcc[idx[i+1]]}")
        x, fs = librosa.load(fn_wav, sr=44100)
        ipd.display(ipd.Audio(data=x, rate=fs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration file"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    main(config)
