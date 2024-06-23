import os
import pandas as pd
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt


def create_graph_and_visualize(dir_path):
    intermediate_df = pd.read_csv(os.path.join(dir_path, 'intermediate_df.csv'))
    G = nx.Graph()
    G.add_nodes_from(intermediate_df["track_id"])
    pos = nx.circular_layout(G)

    for i, row1 in tqdm(intermediate_df.iterrows()):
        for j, row2 in intermediate_df.iterrows():
            if row1["track_id"] != row2["track_id"] and row1["genre_top"] == row2["genre_top"]:
                G.add_edge(row1["track_id"], row2["track_id"])

    nx.write_gml(G, "graph.gml")
    nx.write_pajek(G, "graph.net")
    nx.write_edgelist(G, "graph.edgelist")
    nx.write_adjlist(G, "graph.adjlist")
    print("Graph created and saved")

    for_representation_df = intermediate_df.iloc[:15]
    H = nx.Graph()
    H.add_nodes_from(for_representation_df["track_id"])
    pos = nx.circular_layout(H)

    for i, row1 in tqdm(for_representation_df.iterrows()):
        for j, row2 in for_representation_df.iterrows():
            if row1["track_id"] != row2["track_id"] and row1["genre_top"] == row2["genre_top"]:
                H.add_edge(row1["track_id"], row2["track_id"])

    plt.figure(figsize=(15, 8))
    nx.draw(
        H,
        pos,
        with_labels=True,
        node_size=1500,
        node_color="skyblue",
        font_size=10,
        font_weight="bold",
        edge_color="gray",
    )

    plt.show()


if __name__ == '__main__':
    dir_path = '/home/ics/Documents/Education/Media_engg/Sem_1/CASM/Project/fma_small_dataset_example'
    create_graph_and_visualize(dir_path)