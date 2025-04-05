from copy import copy, deepcopy
from time import monotonic

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean


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

@dataclass
class Bucket:
    genres: set[str] = field(default_factory=set)

class HashTable:
    def __init__(self, hash_size: int, input_dimension: int):
        self.hash_size: int = hash_size
        self.input_dimension: int = input_dimension
        self.projections: np.ndarray
        # {hash : (str)}
        self.buckets: defaultdict[str, Bucket] = defaultdict(Bucket)
        self.get_random_projection_matrix(input_dimension, hash_size)

    def get_random_projection_matrix(self, input_dimension, hash_length):
        # so the matrix is reproducible
        np.random.seed(SEED)
        
        scale = np.sqrt(3)
        values = np.array([1,0,-1])
        probabilitys = [1/6, 2/3, 1/6]

        R = np.random.choice(values, size=(input_dimension, hash_length), p=probabilitys)
        R = scale * R
        self.projections = R

    def generate_hash(self, vector) -> str:
        projection = np.dot(vector, self.projections)
        # TODO: I dont know if this is correct -> would create hashes that 
        # are strings of 1s and 0s as in the blog https://medium.com/data-science/locality-sensitive-hashing-for-music-search-f2f1940ace23
        hash_bits = (projection > 0).astype(int)
        return "".join(map(str, hash_bits))

    def set_item(self, vector: np.ndarray, label: str):
        hash_value = self.generate_hash(vector)
        self.buckets[hash_value].genres.add(label)

    def get_item(self, vector: np.ndarray) -> Bucket:
        hash_value = self.generate_hash(vector)
        return self.buckets[hash_value]

    
class LSH:
    def __init__(self, num_tables: int, hash_size: int, input_dimension: int):
        self.num_tables = num_tables
        self.hash_tables = []
        for i in range(num_tables):
            self.hash_tables.append(HashTable(hash_size, input_dimension))

    def set_item(self, vector: np.ndarray, label: str):
        for table in self.hash_tables:
            table.set_item(vector, label)

    def query(self, vector: np.ndarray):
        """getting back a bucket with all genres accords all tables and the hashes"""
        result_bucket = Bucket()
        result_hashes = []
        #create a bucket obj out of all buckets from all tables
        for table in self.hash_tables:
            bucket = table.get_item(vector)
            result_bucket.genres.update(bucket.genres)
            result_hashes.append(table.generate_hash(vector))

        return  result_hashes, result_bucket

def predict_genre_for_bucket(candidate_vectors: pd.DataFrame, candidate_labels: pd.DataFrame, k: int, metric: str = "euclidean"):
    predictions = {}

    features = candidate_vectors.values
    track_ids = candidate_vectors.index

    for i, track_id_i  in enumerate(track_ids):
        t_i_vector = features[i]

        similarities = []

        for j, track_id_j in enumerate(track_ids):
            if track_id_i == track_id_j:
                continue # we skip if we are the same item

            t_j_vector = features[j]

            if metric == "cosine":
                # higher means more similar
                similarity = cosine_similarity(t_i_vector.reshape(1, -1), t_j_vector.reshape(1, -1))[0,0]
            elif metric == "euclidean":
                # smaller means more similar
                similarity = -euclidean(t_i_vector, t_j_vector)
            else:
                raise ValueError("dont have that metric")

            similarities.append((similarity, track_id_j))

        # sort by similarity
        similarities.sort(reverse=True)

        k_nearest = similarities[:k]
        neighbours_ids = [track_id for _, track_id in k_nearest]
        neighbours_labels = candidate_labels.loc[neighbours_ids]

        most_common = Counter(neighbours_labels).most_common(1)
        predicted_genre = most_common[0][0] if most_common else None

        predictions[track_id_i] = predicted_genre

    return predictions

def runtime_estimate_exact_nn_search(x_train, x_val, metric="euclidean"):
    a = x_val.values[0]
    b = x_train.values[0]

    # measure computation time of one comparision
    start = monotonic()
    if metric == "cosine":
        _ = cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))
    elif metric == "euclidean":
        _ = euclidean(a, b)
    end = monotonic()
    
    # print(f"single comparision: {single_time}")
    single_time = end - start
    total_pairs = len(x_val) * len(x_train)
    est_total_time = single_time * total_pairs
    
    print(f"Estimated total time: {est_total_time:.2f} seconds")