import os, re

def parse_filename(file_name):
    match = re.match(r'(\d+)_(\d+)\.txt', file_name)

    if match:
        review_id = int(match.group(1))
        rating = int(match.group(2))
        return review_id, rating

    print("Error: Could not match ID and rating")

    return None, None




def load_reviews(data_dir):
    reviews = []
    labels = []
    ids = []
    ratings = []
    dict_pos = {}
    dict_neg = {}


    for p in ["train", "test"]:
        pos_dir = os.path.join(data_dir, p + '/pos')

        for file_name in os.listdir(pos_dir):

            if file_name.endswith(".txt"):

                review_id, rating = parse_filename(file_name)

                if review_id is not None and rating is not None:

                    with open(os.path.join(pos_dir, file_name), 'r') as file:
                        reviews.append(file.read())
                        labels.append(1)
                        ids.append(review_id)
                        ratings.append(rating)
                        dict_pos[review_id] = (file.read(), rating)

        neg_dir = os.path.join(data_dir, p + '/neg')

        for file_name in os.listdir(neg_dir):

            if file_name.endswith(".txt"):

                review_id, rating = parse_filename(file_name)

                if review_id is not None and rating is not None:

                    with open(os.path.join(neg_dir, file_name), 'r') as file:
                        reviews.append(file.read())
                        labels.append(0)
                        ids.append(review_id)
                        ratings.append(rating)
                        dict_neg[review_id] = (file.read(), rating)




    return reviews, labels, ids, ratings, dict_pos, dict_neg




