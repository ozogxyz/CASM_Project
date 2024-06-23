import pandas as pd
import os


class FMADataLoader:
    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.metadata_path = os.path.join(dir_path, "fma_metadata")
        self.tracks = pd.read_csv(os.path.join(self.metadata_path, "tracks.csv"), index_col=0, header=[0, 1])

    def make_dataframes(self):
        all_dfs_name = {}
        for col in self.tracks.columns:
            df_name = col[0]
            df_col_name = col[1]
            if df_name not in all_dfs_name:
                all_dfs_name[df_name] = {}
            all_dfs_name[df_name][df_col_name] = self.tracks[col].values
        print(all_dfs_name.keys())
        return all_dfs_name

    def get_intermediate_df(self, all_dfs_name):
        if 'track' in all_dfs_name.keys():
            track_df = pd.DataFrame(all_dfs_name['track'])
            self.trackid_list = track_df.index.tolist()
        self.intermediate_df = track_df[['title', 'duration', 'genre_top']]
        self.intermediate_df['track_id'] = self.trackid_list
        self.intermediate_df = self.intermediate_df.dropna()
        print(self.intermediate_df.shape)
        # save intermediate_df to csv
        self.intermediate_df.to_csv(os.path.join(self.metadata_path, "intermediate_df.csv"))
        return self.intermediate_df


if __name__ == "__main__":
    dir_path = "/home/ics/Documents/Education/Media_engg/Sem_1/CASM/Project/fma_small_dataset/"
    fma_data_loader = FMADataLoader(dir_path)
    all_dfs_name = fma_data_loader.make_dataframes()
    intermediate_df = fma_data_loader.get_intermediate_df(all_dfs_name)
    print(intermediate_df.head())