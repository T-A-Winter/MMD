import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field

# l ... hash len
# n ... number of hash tables
# k ... k-neares neighbours

@dataclass
class Data:
    tracks_path: Path
    features_path: Path

    # gerners we should use as of PA1
    genres: list = field(default_factory=lambda: ["Hip-Hop", 
                                                  "Pop", 
                                                  "Folk", 
                                                  "Rock", 
                                                  "Experimental", 
                                                  "International", 
                                                  "Electronic", 
                                                  "Instrumental"])
    
    # dataframe for the tacks an featurs given from /data/tacks.csv and features.csv
    tracks: pd.DataFrame = field(init=False)
    features: pd.DataFrame = field(init=False)
    
    # split accoring to tracks split
    y_train: pd.Series = field(init=False)
    y_val: pd.Series = field(init=False)
    y_test: pd.Series = field(init=False)
    
    x_train: pd.DataFrame = field(init=False)
    x_val: pd.DataFrame = field(init=False)
    x_test: pd.DataFrame = field(init=False)

    def __post_init__(self):
        # df from given paths
        self.tracks = pd.read_csv(self.tracks_path, index_col=0, header=[0,1])
        self.features = pd.read_csv(self.features_path, index_col=0, header=[0,1,2])

        # we only need the medium dataset as of PA1
        medium_mask = self.tracks[('set', 'subset')] <= 'medium'
        self.tracks = self.tracks.loc[medium_mask]
        self.features = self.features.loc[self.tracks.index]

        # we only take the gernes from the top gernes that are in our gerne list above 
        genre_mask = self.tracks[('track', 'genre_top')].isin(self.genres)
        self.tracks = self.tracks.loc[genre_mask]
        self.features = self.features.loc[self.tracks.index]

        # getting the splits from tracks -> see how in notebook https://nbviewer.org/github/mdeff/fma/blob/outputs/usage.ipynb
        train_mask = self.tracks[('set', 'split')] == 'training'
        val_mask   = self.tracks[('set', 'split')] == 'validation'
        test_mask  = self.tracks[('set', 'split')] == 'test'
        
        # labels
        self.y_train = self.tracks.loc[train_mask, ('track', 'genre_top')]
        self.y_val   = self.tracks.loc[val_mask, ('track', 'genre_top')]
        self.y_test  = self.tracks.loc[test_mask, ('track', 'genre_top')]

        # features
        self.x_train = self.features.loc[train_mask]
        self.x_val   = self.features.loc[val_mask]
        self.x_test  = self.features.loc[test_mask]

def get_random_projection_matrix(input_dim, hash_length, seed=1):
    # so the matrix is reproducible
    np.random.seed(seed)
    
    scale = np.sqrt(3)
    values = np.array([1,0,-1])
    probabilitys = [1/6, 2/3, 1/6]

    R = np.random.choice(values, size=(input_dim, hash_length), p=probabilitys)
    R = scale * R
    return R

def compute_hashes(data, projection_matrix):
    dot_products = np.dot(data, projection_matrix)

    # TODO: I dont know if this is correct -> would create hashes that 
    # are strings of 1s and 0s as in the blog https://medium.com/data-science/locality-sensitive-hashing-for-music-search-f2f1940ace23
    hash_bits = (dot_products > 0).astype(int)
    hash_strings = ["".join(map(str, row)) for row in hash_bits]
    return hash_strings


if __name__ == "__main__":
    path_to_tracks = Path("data/fma_metadata/tracks.csv")
    path_to_features = Path("data/fma_metadata/features.csv")

    data = Data(path_to_tracks, path_to_features)
    # maybe we can start from here? 
    labels = data.x_train.values
    features = data.y_train

    input_dim = labels.shape[1]
    hash_length = 32

    R = get_random_projection_matrix(input_dim, hash_length)

    train_hashes = compute_hashes(labels, R)

    print(train_hashes)


