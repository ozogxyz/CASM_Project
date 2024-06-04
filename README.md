# Generate MFCC embeddings
python generate_mfcc.py

# Generate graph for visualizing
python generate_graph.py

# Generate all graph embeddings
python train_graph_embeddings.py --embedding_size 40 --epochs 100 --model GraphCN

# Predict With argparse
python train_mlp_and_predict_with_args.py --embeddings GraphCN --epochs 200 --batch_size 16 --lr 0.001

# Predict With config.yaml
python train_mlp_and_predict_with_config.py --config config.yaml

