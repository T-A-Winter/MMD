import sys
from collections import defaultdict
from pathlib import Path
from time import monotonic

from data_and_utilities import Data, LSH, predict_genre_for_bucket, runtime_estimate_exact_nn_search

# l ... hash len
# n ... number of hash tables
# k ... k-neares neighbours

def training_and_validation_merge(path_to_tracks: Path, path_to_features: Path, l: int, n: int, k: int, metric: str = "cosine",  verbose: bool = True):
    data = Data(path_to_tracks, path_to_features)

    X_training_vectors = data.x_training.values
    X_training_ids = data.x_training.index
    labels_training = data.y_training.values

    X_validation_vectors = data.x_validation.values
    X_validation_ids = data.x_validation.index
    labels_validation = data.y_validation.values

    X_test_vectors = data.x_test.values
    X_test_ids = data.x_test.index
    labels_test = data.y_test

    input_dimension = X_training_vectors.shape[1]
    hash_length = l
    num_tables = n

    lsh = LSH(num_tables, hash_length, input_dimension)

    # ------------------ TRAINING ------------------
    for vector, label in zip(X_training_vectors, labels_training):
        lsh.set_item(vector, label)
    for vector, label in zip(X_validation_vectors, labels_validation):
        lsh.set_item(vector, label)

    # ------------------ TESTING ------------------
    reverse_map = defaultdict(set)  # { (table_index, hash_value) : set of test track IDs }

    for index, vector in zip(X_test_ids, X_test_vectors):
        hashes, _ = lsh.query(vector)
        for table_index, hash_value in enumerate(hashes):
            key = (table_index, hash_value)
            reverse_map[key].add(index)

    # prediction
    correct = 0
    total = 0
    for (table_index, hash_value), track_ids in reverse_map.items():
        if len(track_ids) > 0:
            candidate_vectors = data.x_test.loc[list(track_ids)]
            candidate_labels = data.y_test.loc[list(track_ids)]
            predictions = predict_genre_for_bucket(candidate_vectors, candidate_labels, k=k, metric=metric)

            for track_id, predicted_genre in predictions.items():
                actual = data.y_test.loc[track_id]

                if verbose:
                    print(f"Track {track_id}: Predicted = {predicted_genre}, Actual = {actual}")

                assert predicted_genre is not None

                if predicted_genre is not None:
                    total += 1
                    if predicted_genre == actual:
                        correct += 1

    if total > 0:
        accuracy = correct / total * 100
        print(f"Accuracy: {accuracy:.2f}% predictions correct)")
    else:
        print("No predictions made (no collisions or empty candidate sets)")

def training_and_validation(path_to_tracks: Path, path_to_features: Path, l: int, n: int, k: int, metric: str = "cosine",  verbose: bool = True):
    data = Data(path_to_tracks, path_to_features)
    # maybe we can start from here?
    X_training_vectors = data.x_training.values  # These are the genre labels.
    X_training_ids = data.x_training.index
    labels_training = data.y_training.values
    X_validation = data.x_validation.values
    X_validation_ids = data.x_validation.index

    input_dimension = X_training_vectors.shape[1]
    hash_length = l  # l: desired hash length
    num_tables = n # n: number of hash tables

    lsh = LSH(num_tables, hash_length, input_dimension)
    # ------------------ TRAINING ------------------
    # putting each track into its hash table. "Training" the LSH
    for vector, track_id, label in zip(X_training_vectors, X_training_ids, labels_training):
        lsh.set_item(vector, label)

    # ------------------ VALIDATING ------------------
    reverse_map = defaultdict(set)  # { (table_index, hash_value) : set of validation track IDs }
    for index, vector in zip(X_validation_ids, X_validation):
        hashes, bucket = lsh.query(vector)
        for table_index, hash_value in enumerate(hashes):
            key = (table_index, hash_value)
            reverse_map[key].add(index)

    # prediction
    correct = 0
    total = 0
    for (table_index, hash_value), track_ids in reverse_map.items():
        # checking for collisions
        if len(track_ids) > 0:
            candidate_pairs_vectors = data.x_validation.loc[list(track_ids)]
            candidate_pairs_labels = data.y_validation.loc[list(track_ids)]
            predictions = predict_genre_for_bucket(candidate_pairs_vectors, candidate_pairs_labels, k=k, metric=metric)

            for track_id, predicted_genre in predictions.items():
                actual = data.y_validation.loc[track_id]

                if verbose:
                    print(f"Track {track_id}: Predicted = {predicted_genre}, Actual = {actual}")

                assert predicted_genre is not None

                if predicted_genre is not None:
                    total += 1
                    if predicted_genre == actual:
                        correct += 1

    if total > 0:
        accuracy = correct / total * 100
        print(f"Accuracy: {accuracy:.2f}% predictions correct)")
    else:
        print("No predictions made (no collisions or empty candidate sets)")

def main(l:int, n:int,k:int, path_to_tracks:Path, path_to_features:Path,  run_validation: bool = False, run_on_real_date: bool = False, run_estimate:bool = False, metric: str = "euclidean", verbose: bool = False):

    if run_validation:
        print("------------------------------------------------")
        configs = ((128, 15, 10, "cosine"),
                   (128, 15, 10, "euclidean"),
                   (128, 10, 5, "cosine"),
                   (128, 10, 5, "euclidean"),
                   (64, 10, 5, "cosine"),
                   (64, 10, 5, "euclidean"),
                   )
        for conf in configs:
            l_new = conf[0]
            n_new = conf[1]
            k_new = conf[2]
            metric_new = conf[3]
            print(f"Config: l: {l_new}, n: {n_new}, k: {k_new}, metric: {metric}")
            start = monotonic()
            training_and_validation(path_to_tracks, path_to_features, l_new, n_new, k_new, metric_new, verbose)
            end = monotonic()
            time_elapsed = end - start
            print(f"Time elapsed: {time_elapsed:.2f} seconds")
            print("------------------------------------------------")

    if run_on_real_date:
        training_and_validation_merge(path_to_tracks, path_to_features, l, n,k ,metric, verbose)

    if run_estimate:
        data = Data(path_to_tracks, path_to_features)
        runtime_estimate_exact_nn_search(data.x_training, data.x_validation)


if __name__ == "__main__":
    # l ... hash len
    # n ... number of hash tables
    # k ... k-neares neighbours
    l = 128
    n = 10
    k = 10
    metric = "cosine"
    path_to_tracks = Path("data/fma_metadata/tracks.csv")
    path_to_features = Path("data/fma_metadata/features.csv")
    verbose = False
    main(l, n, k, path_to_tracks=path_to_tracks, path_to_features=path_to_features, run_validation=True, run_on_real_date=True, run_estimate= True, metric=metric, verbose=verbose)


