import os
import re
import pickle


def parse_filename(file_name):
    """Extract review ID and rating from the filename."""
    match = re.match(r'(\d+)_(\d+)\.txt', file_name)
    if match:
        return int(match.group(1)), int(match.group(2))
    raise ValueError(f"Invalid filename format: {file_name}")


def process_reviews(dir_path, label):
    """Process all reviews in a given directory."""
    reviews = []
    labels = []
    for file_name in os.listdir(dir_path):
        if file_name.endswith(".txt"):
            review_id, rating = parse_filename(file_name)
            with open(os.path.join(dir_path, file_name), 'r') as file:
                reviews.append(file.read())
                labels.append(label)
    return reviews, labels


def load_reviews():
    """Load reviews from pickle file or raw dataset."""
    data_dir = "./data/movRev_data"
    pickle_path = "./data/movie_review.pkl"

    if os.path.isfile(pickle_path):
        print("Pickled dataset is present. \nLoading pickle movie_review.pkl")
        with open(pickle_path, "rb") as f:
            data_pickle = pickle.load(f)
        if "reviews" in data_pickle and "labels" in data_pickle:
            return data_pickle["reviews"], data_pickle["labels"]
        raise KeyError("Pickle file is missing required keys.")

    reviews = []
    labels = []
    for p in ["train", "test"]:
        pos_dir = os.path.join(data_dir, p, 'pos')
        neg_dir = os.path.join(data_dir, p, 'neg')
        if os.path.isdir(pos_dir):
            pos_reviews, pos_labels = process_reviews(pos_dir, 1)
            reviews.extend(pos_reviews)
            labels.extend(pos_labels)
        if os.path.isdir(neg_dir):
            neg_reviews, neg_labels = process_reviews(neg_dir, 0)
            reviews.extend(neg_reviews)
            labels.extend(neg_labels)

    return reviews, labels
