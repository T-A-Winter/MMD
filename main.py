from collections import defaultdict
from pathlib import Path
from time import monotonic

from data_and_utilities import Data, LSH, predict_genre_for_bucket

# l ... hash len
# n ... number of hash tables
# k ... k-neares neighbours

def main(path_to_tracks: Path, path_to_features: Path, l: int, n: int, k: int, metric: str = "cosine",  verbose: bool = True):
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
        lsh.set_item(vector, label,)

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
        if len(track_ids) > 1:
            candidate_pairs_vectors = data.x_validation.loc[list(track_ids)]
            candidate_pairs_labels = data.y_validation.loc[list(track_ids)]
            predictions = predict_genre_for_bucket(candidate_pairs_vectors, candidate_pairs_labels, k=k, metric=metric)

            for track_id, predicted_genre in predictions.items():
                actual = data.y_validation.loc[track_id]

                if verbose:
                    print(f"Track {track_id}: Predicted = {predicted_genre}, Actual = {actual}")

                if predicted_genre is not None:
                    total += 1
                    if predicted_genre == actual:
                        correct += 1

    if total > 0:
        accuracy = correct / total * 100
        print(f"Accuracy: {accuracy:.2f}% ({correct}/{total} predictions correct)")
    else:
        print("No predictions made (no collisions or empty candidate sets)")

if __name__ == "__main__":
    # l ... hash len
    # n ... number of hash tables
    # k ... k-neares neighbours

    path_to_tracks = Path("data/fma_metadata/tracks.csv")
    path_to_features = Path("data/fma_metadata/features.csv")
    confics  = ((128, 15, 10, "cosine"),
                (128, 15, 10, "euclidean"),
                (128, 10, 5, "cosine"),
                (128, 10, 5, "euclidean"),
                (64, 10, 5, "cosine"),
                (64, 10, 5, "euclidean"),
                )
    print("------------------------------------------------")
    for conf in confics:
        l = conf[0]
        n = conf[1]
        k = conf[2]
        metric = conf[3]
        print(f"Config: l: {l}, n: {n}, k: {k}, metric: {metric}")
        start = monotonic()
        main(path_to_tracks, path_to_features, l, n, k, metric, verbose=False)
        end = monotonic()
        time_elapsed = end - start
        print(f"Time elapsed: {time_elapsed:.2f} seconds")
        print("------------------------------------------------")



