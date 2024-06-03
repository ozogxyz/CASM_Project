import os
import argparse
import numpy as np
import pandas as pd
import librosa
import IPython.display as ipd

def generate_mfcc(path_for_csv, emb_dim):
    df = pd.read_csv(path_for_csv)

    unique_genres = sorted(list(set(df["genre"])))
    n_genres = len(unique_genres)
    genre_to_id = {unique_genres[_]: _ for _ in range(n_genres)}
    id_to_genre = {_: unique_genres[_] for _ in range(n_genres)}

    print(unique_genres)
    print(genre_to_id)
    print(id_to_genre)

    class_ids = [genre_to_id[_] for _ in df["genre"]]
    class_ids = np.array(class_ids)
    print(class_ids)

    dir_dataset = "genres_mini"

    random_ids = [3, 45, 23, 67, 77]

    for i in random_ids:
        fn_wav = os.path.join(dir_dataset, df["fn_wav"][i])
        genre = df["genre"][i]
        print(f"File: {fn_wav} - Music Genre: {genre}")
        x, fs = librosa.load(fn_wav, sr=44100)
        ipd.display(ipd.Audio(data=x, rate=fs))

    def compute_mel_spec_for_audio_file(
        fn_wav, n_fft=1024, hop_length=441, fs=22050.0, n_mels=64
    ):
        x, fs = librosa.load(fn_wav, sr=fs, mono=True)
        if np.max(np.abs(x)) > 0:
            x = x / np.max(np.abs(x))
        X = librosa.feature.melspectrogram(
            y=x,
            sr=fs,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=0.0,
            fmax=fs / 2,
            power=1.0,
            htk=True,
            norm=None,
        )
        X = librosa.amplitude_to_db(X)
        return X

    files = df["fn_wav"]
    all_mel_specs = []
    for i, fn_wav in enumerate(files):
        full_path_of_wav = os.path.join(dir_dataset, fn_wav)
        if full_path_of_wav.endswith(".wav"):
            all_mel_specs.append(compute_mel_spec_for_audio_file(full_path_of_wav))

    print(
        "We have {} spectrograms of shape {}".format(
            len(all_mel_specs), all_mel_specs[0].shape
        )
    )

    all_mel_specs = np.array(all_mel_specs)
    print(f"Shape of our data tensor : {all_mel_specs.shape}")

    def compute_mfcc(fn_wav, emb_dim):
        y, sr = librosa.load(fn_wav)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=emb_dim)
        return np.mean(mfcc, axis=1)

    def compute_mff_for_dataset(df, dir_dataset, emb_dim):
        n_files = df.shape[0]
        mfcc = np.zeros((n_files, emb_dim))
        for n in range(n_files):
            if n % 10 == 0:
                print(f"{n+1}/{n_files}")
            fn_wav = os.path.join(dir_dataset, df["fn_wav"][n])
            mfcc[n, :] = compute_mfcc(fn_wav, emb_dim)
        return mfcc

    mfcc = compute_mff_for_dataset(df, dir_dataset, emb_dim)
    print(mfcc.shape)

    np.save("mfcc_embeddings.npy", mfcc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MFCC embeddings")
    parser.add_argument("--embedding_size", type=int, default=40, help="Size of MFCC embedding")
    path_for_csv = "songs_genre.csv"
    args = parser.parse_args()

    generate_mfcc(path_for_csv, emb_dim=args.embedding_size)
