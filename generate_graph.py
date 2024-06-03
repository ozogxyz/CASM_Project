import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


def generate_graph(path_for_csv):
    df = pd.read_csv("songs_genre.csv")
    G = nx.Graph()
    G.add_nodes_from(df["fn_wav"])
    pos = nx.circular_layout(G)
    for i, row1 in df.iterrows():
        for j, row2 in df.iterrows():
            if row1["fn_wav"] != row2["fn_wav"] and row1["genre"] == row2["genre"]:
                G.add_edge(row1["fn_wav"], row2["fn_wav"])

    plt.figure(figsize=(15, 8))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=1500,
        node_color="skyblue",
        font_size=10,
        font_weight="bold",
        edge_color="gray",
    )
    plt.show()


if __name__ == "__main__":
    generate_graph("songs_genre.csv")
