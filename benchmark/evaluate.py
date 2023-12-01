import surprise
from surprise import dump, accuracy
import pandas as pd
import argparse
from surprise.model_selection import train_test_split
import numpy as np
np.random.seed(42)
parser = argparse.ArgumentParser(description='Process user_id and top_k')
parser.add_argument('--user_id', type=int)
parser.add_argument('--top_k', type=int, default=5)
args = parser.parse_args()
_, algo = dump.load("../models/model.pickle")
dataset = pd.read_csv('./data/user_item_matrix.csv')
reader = surprise.Reader(rating_scale=(1, 5))
data = surprise.Dataset.load_from_df(dataset[['user_id', 'item_id', 'rating']], reader)
_, testset = train_test_split(data, test_size=0.25, random_state=42)
predictions = algo.test(testset)
print(f"Accuracy(RMSE) for rating prediction: {accuracy.rmse(predictions)}")
df_item = pd.read_csv('./data/u.item', sep='|', header=None, names=['movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'], encoding='latin-1')

def recommend_movies(algo, user_id, top_k):
    # First, the algorithm will predict ratings for each unseen movie, second, it will output the top k films with the highest (predicted) rating
    seen_movies = set(dataset[dataset['user_id']==user_id]['item_id'].tolist())
    all_movies = set(dataset['item_id'].tolist())
    unseen_movies = all_movies - seen_movies
    results = []
    for movie in unseen_movies:
        results.append((movie,algo.predict(user_id, movie).est))
    results.sort(key=lambda x: x[-1], reverse=True)
    # Names of movies
    results = [df_item[df_item['movie_id']==r[0]]['movie_title'].item() for r in results[:top_k]]
    return results
if args.user_id is None:
    user_id = np.random.choice(dataset['user_id'].tolist())
else:
    user_id = args.user_id
print(f"Movies recommendations for user {user_id}")
for idx, movie_title in enumerate(recommend_movies(algo, user_id, top_k=5)):
    print(f"{idx+1}. {movie_title}")