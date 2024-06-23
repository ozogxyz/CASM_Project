import os
import glob
import librosa
import soundfile as sf
import numpy as np
import pandas as pd

def cut_random_5s_from_songs(dir_path):
    intermediate_df = pd.read_csv(os.path.join(dir_path, 'intermediate_df.csv'))
    song_files_path = list(intermediate_df['file_path'])
    new_dir_path_5s = os.path.join(dir_path, 'fma_small_5s')
    os.makedirs(new_dir_path_5s, exist_ok=True)
    for i in song_files_path:
        x, fs = librosa.load(i)
        greater_than_5s = fs * 5
        if len(x) > greater_than_5s:
            start = np.random.randint(0, len(x) - greater_than_5s)
            x_5s = x[start:start + greater_than_5s]
            file_name = os.path.basename(i)
            file_path = os.path.join(new_dir_path_5s, file_name)
            print(file_name, file_path)
            sf.write(file_path, x_5s, fs)
    intermediate_df['file_path_5s'] = intermediate_df['file_path'].apply(lambda x: x.replace('fma_small_songs', 'fma_small_5s'))
    intermediate_df.drop('file_path', axis=1, inplace=True)
    intermediate_df.to_csv(os.path.join(dir_path, 'intermediate_df.csv'), index=False)
    print(intermediate_df['file_path_5s'].shape)


def check_missing_files(dir_path):
    intermediate_df = pd.read_csv(os.path.join(dir_path, 'intermediate_df.csv'))
    five_s_song_files_paths = glob.glob(os.path.join(dir_path, 'fma_small_5s', '**/*.mp3'), recursive=True)
    missing_tracks = []
    for i in intermediate_df['file_path_5s']:
        if i not in five_s_song_files_paths:
            missing_tracks.append(i)
    intermediate_df = intermediate_df[~intermediate_df['file_path_5s'].isin(missing_tracks)]
    intermediate_df.to_csv('intermediate_df.csv', index=False)
    print(f"Missing tracks: {len(missing_tracks)}")

if __name__ == '__main__':
    dir_path = '/home/ics/Documents/Education/Media_engg/Sem_1/CASM/Project/fma_small_dataset_example'
    cut_random_5s_from_songs(dir_path)
    check_missing_files(dir_path)
    