import pandas as pd
import os
import numpy as np

def get_recommendation(songs_for_inference, intermediate_df, embeddings):

    songs_recommendations = {}

    for k, v in songs_for_inference.items():

        for id in v:
            song = intermediate_df["file_path_5s"][id]
            genre = intermediate_df["genre_top"][id]

            if genre not in songs_recommendations:
                songs_recommendations[genre] = []

            dist_mfcc = np.sqrt(np.sum((embeddings - embeddings[id]) ** 2, axis=1))
            idx = np.argsort(dist_mfcc)

            recommendations = []
            for i in range(5):
                curr_idx = idx[i + 1]
                recommended_song = intermediate_df["file_path_5s"][curr_idx]
                recommended_genre = intermediate_df["genre_top"][curr_idx]
                recommendations.append(recommended_genre)

            songs_recommendations[genre].extend((id, recommendations))
    return songs_recommendations

if __name__ == '__main__':
    dir_path = '/home/ics/Documents/Education/Media_engg/Sem_1/CASM/Project/fma_small_dataset_example'
    intermediate_df = pd.read_csv(os.path.join(dir_path, 'intermediate_df.csv'))

    unique_genres = sorted(list(set(intermediate_df["genre_top"])))
    n_genres = len(unique_genres)
    genre_to_id = {unique_genres[_]: _ for _ in range(n_genres)}
    id_to_genre = {_: unique_genres[_] for _ in range(n_genres)}
    print(id_to_genre)
    songs_for_inference = {}
    for id, genre in id_to_genre.items():
        genre_df = intermediate_df[intermediate_df['genre_top'] == genre]
        if len(genre_df) >= 100:
            genre_indices = genre_df.sample(n=100, random_state=1).index.tolist()
        else:
            genre_indices = genre_df.sample(n=len(genre_df), random_state=1).index.tolist()
        songs_for_inference[genre] = genre_indices
    path_for_mfcc_plain = dir_path + '/mfcc_plain_embeddings_fma_5s.npy'
    mfcc_plain = np.load(path_for_mfcc_plain)
    path_for_mfcc_sage = dir_path + '/mfcc_sage_embeddings_fma_5s.npy' 
    mfcc_sage = np.load(path_for_mfcc_sage)
    path_for_mfcc_gcn = dir_path + '/mfcc_gcn_embeddings_fma_5s.npy'
    mfcc_gcn = np.load(path_for_mfcc_gcn)
    mfcc_plain_recommendations = get_recommendation(songs_for_inference, intermediate_df, mfcc_plain)
    mfcc_sage_recommendations = get_recommendation(songs_for_inference, intermediate_df, mfcc_sage)
    mfcc_gcn_recommendations = get_recommendation(songs_for_inference, intermediate_df, mfcc_gcn)
    all_recommendations = {
    'mfcc_rec': mfcc_plain_recommendations,
    'sage_rec': mfcc_sage_recommendations,
    'gcn_rec': mfcc_gcn_recommendations,
    }
    master_dict = {}
    for k, v in all_recommendations.items():
        master_dict[k] = {}
        for genre, rec in v.items():
            for_inf, recommendations = [], []
            for i, r in enumerate(rec):
                if i % 2 == 0:
                    for_inf.append(r)
                else:
                    recommendations.append(r)
            master_dict[k][genre] = recommendations

    accuracy_scores = {}
    for k, v in master_dict.items():
        accuracy_scores[k] = {}
        for i, j in v.items():
            accuracy_scores[k][i] = []
            for n in j:
                count_of_genres = n.count(i)
                accuracy_scores[k][i].append(count_of_genres / 10)

    final_dict = {}
    for emb, rec in accuracy_scores.items():
        final_dict[emb] = {}
        for genre, acc in rec.items():
            final_dict[emb][genre] = round(np.mean(acc) * 100, 3)

    final_df = pd.DataFrame(final_dict)
    final_df.to_csv('recommendations.csv', index=True)
    print(final_df)
