import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
from collections import Counter

# l ... hash len
# n ... number of hash tables
# k ... k-neares neighbours

SEED = 1

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
    y_training: pd.Series = field(init=False)
    y_validation: pd.Series = field(init=False)
    y_test: pd.Series = field(init=False)
    
    x_training: pd.DataFrame = field(init=False)
    x_validation: pd.DataFrame = field(init=False)
    x_test: pd.DataFrame = field(init=False)

    def __post_init__(self):
        # df from given paths
        self.tracks = pd.read_csv(self.tracks_path, index_col=0, header=[0,1])
        self.features = pd.read_csv(self.features_path, index_col=0, header=[0,1,2])

        # we only need the medium dataset as of PA1
        medium_mask = self.tracks[('set', 'subset')] == 'medium'
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
        self.y_training = self.tracks.loc[train_mask, ('track', 'genre_top')]
        self.y_validation   = self.tracks.loc[val_mask, ('track', 'genre_top')]
        self.y_test  = self.tracks.loc[test_mask, ('track', 'genre_top')]

        # features
        self.x_training = self.features.loc[train_mask]
        self.x_validation   = self.features.loc[val_mask]
        self.x_test  = self.features.loc[test_mask]

class HashTable:
    def __init__(self, hash_size, input_dimension):
        self.hash_size = hash_size
        self.input_dimension = input_dimension
        self.projections = get_random_projection_matrix(input_dimension, hash_size)
        self.buckets = {}

    def generate_hash(self, vector):
        projection = np.dot(vector, self.projections)
        hash_bits = (projection > 0).astype(int)
        return "".join(map(str, hash_bits))
    
    def __setitem__(self, vector, label):
        hash_value = self.generate_hash(vector)
        self.buckets[hash_value] = self.buckets.get(hash_value, []) + [label]
    
    def __getitem__(self, vector):
        h = self.generate_hash(vector)
        return self.buckets.get(h, [])
    
class LSH:
    def __init__(self, num_tables, hash_size, input_dimension):
        self.num_tables = num_tables
        self.hash_tables = []
        for i in range(num_tables):
            self.hash_tables.append(HashTable(hash_size, input_dimension))
    
    def __setitem__(self, vector, label):
        for table in self.hash_tables:
            table[vector] = label
    
    def __getitem__(self, vector):
        results = []
        for table in self.hash_tables:
            results.extend(table[vector])
        return list(set(results))


def get_random_projection_matrix(input_dimension, hash_length):
    # so the matrix is reproducible
    np.random.seed(SEED)
    
    scale = np.sqrt(3)
    values = np.array([1,0,-1])
    probabilitys = [1/6, 2/3, 1/6]

    R = np.random.choice(values, size=(input_dimension, hash_length), p=probabilitys)
    R = scale * R
    return R

def compute_hashes(data, projection_matrix):
    dot_products = np.dot(data, projection_matrix)

    # TODO: I dont know if this is correct -> would create hashes that 
    # are strings of 1s and 0s as in the blog https://medium.com/data-science/locality-sensitive-hashing-for-music-search-f2f1940ace23
    hash_bits = (dot_products > 0).astype(int)
    hash_strings = ["".join(map(str, row)) for row in hash_bits]
    return hash_strings


def evaluate_knn_lsh(data, lsh, k=5, metric = "eucledian"):
    predictions = []

    for track_id, features in data.x_validation.iterrows():
        similar_tracks = lsh[features]  # tuples of (track_id, genre)
        distances = []  # tuples of (track_id, genre, distance)

        for track_id_candidate, candidate_label in similar_tracks: 
            candidate_feature_vector = data.x_training.loc[track_id_candidate].values
            
            if metric == "euclidean":
                euclidean_dist = euclidean(features, candidate_feature_vector)
                distances.append((track_id_candidate, candidate_label, euclidean_dist))
            if metric == "cosine":
               cos_sim = cosine_similarity(np.array(features).reshape(1,-1), np.array(candidate_feature_vector).reshape(1,-1))
               distances.append((track_id_candidate, candidate_label, cos_sim))

        distances.sort(key=lambda x: x[2])
        top_k = distances[:k] 

        if top_k:
            predicted_genre = Counter(genre for _, genre, _ in top_k).most_common(1)[0][0]
        else:
            predicted_genre = "unknown"

        predictions.append((track_id, predicted_genre))

    
    predicted_dict = dict(predictions)
    correct_pred = 0

    for track_id, true_genre in data.y_validation.items():
        if predicted_dict.get(track_id) == true_genre:
            correct_pred += 1

    accuracy = correct_pred / len(data.y_validation)

    return accuracy, predictions


if __name__ == "__main__":
    path_to_tracks = Path("data/fma_metadata/tracks.csv")
    path_to_features = Path("data/fma_metadata/features.csv")

    data = Data(path_to_tracks, path_to_features)
    # maybe we can start from here? 
    X_training = data.x_training.values 
    Y_training = data.y_training.values  # These are the genre labels.
    
    X_validation = data.x_validation.values
    Y_validation = data.y_validation.values 

    input_dimension = X_training.shape[1]
    hash_length = 64  # l: desired hash length
    num_tables = 30    # n: number of hash tables

    lsh = LSH(num_tables, hash_length, input_dimension)

    # hash each entry of the training set to its bucket. the buckets consist of tuples of (track_id, label)
    for track_id, features in data.x_training.iterrows():
        vector = features.values
        label = data.y_training.loc[track_id] 
        lsh[vector] = (track_id, label)

    k = 5
    accuracy, predictions = evaluate_knn_lsh(data, lsh, k, "cosine") # until now this only works for euclidean distance.
    print(f"Accuracy: {accuracy:.4f}")