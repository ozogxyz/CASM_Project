import os
import glob
import pandas as pd

def move_files_to_folder_and_add_as_column(dataset_path, dir_path, df_path):
    intermediate_df = pd.read_csv(df_path)
    os.makedirs(os.path.join(dir_path, 'fma_small_songs'), exist_ok=True)    
    # move files to fma_small_songs folder from fma_small
    song_files = glob.glob(os.path.join(dataset_path, '**/*.mp3'), recursive=True)
    print(len(song_files))

    for song_file in song_files:
        os.rename(song_file, os.path.join(dir_path, 'fma_small_songs', os.path.basename(song_file)))
    
    new_dataset_path = os.path.join(dir_path, 'fma_small_songs')
    intermediate_df['track_id'] = intermediate_df['track_id'].apply(lambda x: str(x).zfill(6))
    song_files_paths = glob.glob(os.path.join(new_dataset_path, '**/*.mp3'), recursive=True)
    print(len(song_files_paths))

    for f in song_files_paths:
        track_id = os.path.basename(f).split('.')[0]
        if track_id in intermediate_df['track_id'].values:
            intermediate_df.loc[intermediate_df['track_id'] == track_id, 'file_path'] = f
    intermediate_df.dropna(inplace=True)
    intermediate_df.to_csv(df_path, index=False)
    print(intermediate_df.head())
    # remove fma_small folder
    os.system(f'rm -rf {dataset_path}')

if __name__ == '__main__':
    dir_path = '/home/ics/Documents/Education/Media_engg/Sem_1/CASM/Project/fma_small_dataset_example'
    dataset_path = os.path.join(dir_path, "fma_small")
    df_path = '/home/ics/Documents/Education/Media_engg/Sem_1/CASM/Project/fma_small_dataset_example/intermediate_df.csv'
    move_files_to_folder_and_add_as_column(dataset_path, dir_path, df_path)