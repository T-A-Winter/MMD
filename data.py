import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field

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
        
        # Labels
        self.y_train = self.tracks.loc[train_mask, ('track', 'genre_top')]
        self.y_val   = self.tracks.loc[val_mask, ('track', 'genre_top')]
        self.y_test  = self.tracks.loc[test_mask, ('track', 'genre_top')]

        # features
        self.x_train = self.features.loc[train_mask]
        self.x_val   = self.features.loc[val_mask]
        self.x_test  = self.features.loc[test_mask]



if __name__ == "__main__":
    path_to_tracks = Path("data/fma_metadata/tracks.csv")
    path_to_features = Path("data/fma_metadata/features.csv")

    data = Data(path_to_tracks, path_to_features)

