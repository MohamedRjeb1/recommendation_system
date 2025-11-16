# data_prep.py
# Chargement des données + précalculs (trending, content vectors, user-item matrix)
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from scipy.sparse import hstack

ANIME_PATH = r"C:\Users\moham\OneDrive\Desktop\Recommendation_system\anime_with_views.csv"
RATING_PATH = r"C:\Users\moham\OneDrive\Desktop\Recommendation_system\rating.csv"

def load_data(anime_path=ANIME_PATH, rating_path=RATING_PATH):
    animes = pd.read_csv(anime_path)
    # reduce sample size for faster testing
    animes = animes.head(3000)
    ratings = pd.read_csv(rating_path)
    ratings = ratings.head(8000)
    required_anime_cols = {"anime_id","name","genre","type","episodes","rating","members"}
    required_rating_cols = {"user_id","anime_id","rating"}
    if not required_anime_cols.issubset(set(animes.columns)):
        raise ValueError(f"anime.csv must contain columns: {required_anime_cols}")
    if not required_rating_cols.issubset(set(ratings.columns)):
        raise ValueError(f"rating.csv must contain columns: {required_rating_cols}")
    return animes, ratings

def compute_trending(animes_df):
    # If dataset does not provide views_last_3m or views_all_time, approximate from members
    if "views_last_3m" not in animes_df.columns:
        # heuristic: recent views = members * small fraction
        animes_df["views_last_3m"] = (animes_df["members"] * 0.05).astype(int)
    if "views_all_time" not in animes_df.columns:
        animes_df["views_all_time"] = (animes_df["members"] * 1.0).astype(int)
    return animes_df

def build_content_embeddings(animes_df):
    # Transform 'genre' into TF-IDF, scale numeric, encode type
    tf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    # Replace commas with space so each genre token is captured
    genre_matrix = tf.fit_transform(animes_df["genre"].fillna("").str.replace(",", " "))
    scaler = MinMaxScaler()
    numeric = scaler.fit_transform(animes_df[["episodes","rating"]].fillna(animes_df[["episodes","rating"]].mean()))
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    type_ohe = ohe.fit_transform(animes_df[["type"]].fillna("Unknown"))
    content_matrix = hstack([genre_matrix, numeric, type_ohe]).tocsr()
    meta = {"tf": tf, "scaler": scaler, "ohe": ohe}
    return content_matrix, meta

def build_user_item_matrix(ratings_df, animes_df):
    # Construct users list and items list (items = anime_id from animes_df)
    users = np.unique(ratings_df["user_id"].values)
    items = animes_df["anime_id"].unique()
    user_index = {u:i for i,u in enumerate(users)}
    item_index = {it:j for j,it in enumerate(items)}
    mat = np.zeros((len(users), len(items)), dtype=float)
    for _, row in ratings_df.iterrows():
        u = user_index[row["user_id"]]
        it = item_index.get(row["anime_id"])
        if it is None:
            continue
        # rating == -1 => implicit watch
        if row["rating"] == -1:
            mat[u, it] = 1.0
        else:
            # explicit ratings scaled to give slightly higher weight
            mat[u, it] = 1.0 + (float(row["rating"]) / 10.0)
    return mat, users, items, user_index, item_index

# Utility exports
__all__ = ["load_data","compute_trending","build_content_embeddings","build_user_item_matrix"]
