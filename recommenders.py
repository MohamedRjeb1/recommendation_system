# recommenders.py
# Fonctions de recommandation : content shortlist, user-user, hybrid scoring et update weights
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Content shortlist ----------
def content_shortlist(profile_vector, content_matrix, top_n=200):
    """Retourne indices (0-based) des top_n items par similarité cosine."""
    sims = cosine_similarity(profile_vector, content_matrix).flatten()
    idxs = np.argsort(-sims)[:top_n]
    return idxs, sims[idxs]

# ---------- User-user collaborative ----------
def predict_useruser_score(target_user_id, target_item_id, ui_mat, users, items, user_index, item_index, k=100):
    """
    Prédit un score implicite user-user.
    - ui_mat: numpy array (n_users x n_items)
    - users/items: arrays listing user_ids and item_ids (in same index order as matrix)
    """
    if target_user_id not in user_index or target_item_id not in item_index:
        return 0.0
    uidx = user_index[target_user_id]
    iidx = item_index[target_item_id]
    # similarities between target user and all users
    sims = cosine_similarity(ui_mat)[uidx]
    sims[uidx] = 0.0
    neigh_idx = np.argsort(-sims)[:min(k, len(sims))]
    weights = []
    scores = []
    for n in neigh_idx:
        if ui_mat[n, iidx] > 0:
            weights.append(sims[n])
            scores.append(ui_mat[n, iidx])
    if not weights:
        return 0.0
    weights = np.array(weights)
    scores = np.array(scores)
    pred = np.dot(weights, scores) / (weights.sum() + 1e-9)
    # normalize by global max in matrix to keep approx 0..1
    return pred / (ui_mat.max() + 1e-9)

# ---------- Hybrid scoring ----------
def hybrid_score_for_user_item(user_weights, user_id, anime_idx, profile_vec, content_matrix, animes_df,
                               ui_mat, users, items, user_index, item_index):
    """
    anime_idx: integer position in animes_df (0..len-1)
    user_weights: dict {user_id_str: [w_c, w_cf, w_pop]}
    """
    # content score: cosine between user profile and item row
    cont_sim = float(cosine_similarity(profile_vec, content_matrix.getrow(anime_idx)).flatten()[0])
    # collab score: need anime_id
    anime_id = int(animes_df.iloc[anime_idx]["anime_id"])
    collab = predict_useruser_score(user_id, anime_id, ui_mat, users, items, user_index, item_index)
    pop = float(animes_df.iloc[anime_idx]["views_all_time"] / animes_df["views_all_time"].max())

    # --- IMPORTANT: use str(user_id) to read from dict (weights_store uses string keys) ---
    key = str(user_id)
    w_c, w_cf, w_pop = user_weights.get(key, (0.7, 0.2, 0.1))
    score = w_c * cont_sim + w_cf * collab + w_pop * pop
    return {"score": score, "content": cont_sim, "collab": collab, "pop": pop}

# ---------- Weight update ----------
def update_weights_on_watch(user_weights, user_id, anime_idx, profile_vec, content_matrix, animes_df,
                            ui_mat, users, items, user_index, item_index,
                            alpha=0.02, delta=0.05, min_w=0.15, max_w=0.85):
    """
    Met à jour les poids de l'utilisateur après un watch event.
    Small step alpha, threshold delta pour attribuer la cause.
    """
    # Use same key normalization
    key = str(user_id)

    # compute current signals
    res = hybrid_score_for_user_item(user_weights, user_id, anime_idx, profile_vec, content_matrix, animes_df,
                                     ui_mat, users, items, user_index, item_index)
    z_c = res["content"]
    z_cf = res["collab"]

    # read existing weights (default if missing)
    w_c, w_cf, w_pop = user_weights.get(key, (0.7, 0.2, 0.1))

    # small update logic
    if z_c - z_cf > delta:
        w_c += alpha
        w_cf -= alpha * 0.8
    elif z_cf - z_c > delta:
        w_cf += alpha
        w_c -= alpha * 0.8

    # clamp
    w_c = max(min(w_c, max_w), min_w)
    w_cf = max(min(w_cf, max_w), min_w)
    w_pop = 1.0 - (w_c + w_cf)

    if w_pop < 0.0:
        # renormalize proportionally
        s = w_c + w_cf
        if s > 0:
            w_c /= s
            w_cf /= s
        w_pop = 0.0

    # write back using string key
    user_weights[key] = [w_c, w_cf, w_pop]
    return user_weights[key]

# exports
__all__ = ["content_shortlist", "predict_useruser_score", "hybrid_score_for_user_item", "update_weights_on_watch"]
