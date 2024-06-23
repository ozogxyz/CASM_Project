import os
import glob

def move_files_to_folder(dataset_path, dir_path):
    os.makedirs(os.path.join(dir_path, 'fma_small_songs'), exist_ok=True)
    song_files = glob.glob(os.path.join(dataset_path, '**/*.mp3'), recursive=True)
    for song_file in song_files:
        os.rename(song_file, os.path.join(dir_path, 'fma_small_songs', os.path.basename(song_file)))
    
    new_dir_path = os.path.join(dir_path, 'fma_small_songs')
    track_id_col_val = track_id_col_val.apply(lambda x: str(x).zfill(6))
    song_files_paths = glob.glob(os.path.join(new_dir_path, '**/*.mp3'), recursive=True)
    print(len(song_files_paths))

    for f in song_files_paths:
        track_id = os.path.basename(f).split('.')[0]
        if track_id in track_id_col_val.values:
            track_id_col_val.loc[track_id_col_val == track_id, 'file_path'] = f
    track_id_col_val.dropna(inplace=True)
    print(track_id_col_val.shape)    
    folders = glob.glob(os.path.join(dataset_path, '**/'), recursive=True)
    for folder in folders[1:]:
        os.rmdir(folder)

if __name__ == '__main__':
    dataset_path = 'fma_small'
    dir_path = 'fma_small_dataset'
    move_files_to_folder(dataset_path, dir_path)