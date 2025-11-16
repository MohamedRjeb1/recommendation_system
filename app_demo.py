# app_demo.py
# Script principal à lancer pour tester le pipeline end-to-end avec gestion JSON des poids
import numpy as np
from data_prep import load_data, compute_trending, build_content_embeddings, build_user_item_matrix
from recommenders import content_shortlist, hybrid_score_for_user_item, update_weights_on_watch
from weights_store import WeightsStore  

def build_profile_from_onboarding(onboarding, animes_df, meta):
    """Crée un vecteur profile sparse compatible avec content_matrix"""
    tf = meta["tf"]
    scaler = meta["scaler"]
    ohe = meta["ohe"]
    genres_text = " ".join(onboarding.get("genres", []))
    vec_genre = tf.transform([genres_text])

    length = onboarding.get("length","both")
    if length == "short":
        mask = animes_df["episodes"] <= 12
    elif length == "long":
        mask = animes_df["episodes"] > 24
    else:
        mask = np.array([True]*len(animes_df))

    if mask.sum() == 0:
        num = np.array([0.5,0.5]).reshape(1,-1)
    else:
        num = scaler.transform(animes_df.loc[mask, ["episodes","rating"]]).mean(axis=0).reshape(1,-1)

    t = onboarding.get("type", ["TV"])
    try:
        t_ohe = ohe.transform([[t[0]]])
    except Exception:
        t_ohe = np.zeros((1, len(ohe.categories_[0])))

    from scipy.sparse import hstack as sh
    profile = sh([vec_genre, num, t_ohe])
    return profile

def main():
    animes, ratings = load_data()
    animes = compute_trending(animes)
    content_matrix, meta = build_content_embeddings(animes)
    ui_mat, users, items, user_index, item_index = build_user_item_matrix(ratings, animes)

    # --- UI Mixte step: trending + all-time ---
    print("Top trending (3 months):")
    print(animes.sort_values("views_last_3m", ascending=False).head(5)[["anime_id","name","views_last_3m"]])
    print("\nTop all-time:")
    print(animes.sort_values("views_all_time", ascending=False).head(5)[["anime_id","name","views_all_time"]])

    # --- Onboarding example (simulate checkboxes) ---
    onboarding = {"genres":["Action","Fantasy"], "type":["TV"], "length":"both"}
    profile_vec = build_profile_from_onboarding(onboarding, animes, meta)

    # content shortlist
    shortlist_idxs, _ = content_shortlist(profile_vec, content_matrix, top_n=200)
    print("\nContent shortlist (top 10):")
    print(animes.iloc[shortlist_idxs[:10]][["anime_id","name","genre"]])

    # prepare user state
    user_id = 10000  # nouveau user de test

    # --- Gestion des poids avec JSON ---
    weights_store = WeightsStore("user_weights.json")

    # Initialiser si utilisateur jamais vu
    if not weights_store.get_user_weights(user_id):
        weights_store.set_user_weights(user_id,  [0.7, 0.2, 0.1])

    user_weights = weights_store.get_user_weights(user_id)

    # --- initial hybrid re-rank over a smaller final pool (top 50 of shortlist) ---
    final_pool = shortlist_idxs[:50]
    scored = []
    for idx in final_pool:
        res = hybrid_score_for_user_item(weights_store.weights, user_id, idx, profile_vec, content_matrix, animes,
                                         ui_mat, users, items, user_index, item_index)
        scored.append((idx, res["score"], res["content"], res["collab"], res["pop"]))
    scored_sorted = sorted(scored, key=lambda x: -x[1])
    print("\nInitial top recommendations (hybrid):")
    for idx,score,zc,zcf,pop in scored_sorted[:10]:
        print(animes.iloc[idx]["anime_id"], animes.iloc[idx]["name"], f"score={score:.3f}")

    # --- Simuler watch d'un anime ---
    watched_idx = final_pool[0]  
    print("\nSimulate watch of:", animes.iloc[watched_idx][["anime_id","name"]].to_dict())

    # --- Mettre à jour les poids ---
    # update_weights_on_watch modifie le dictionnaire user_weights et retourne la nouvelle liste
    updated_weights_list = update_weights_on_watch(
        weights_store.weights, user_id, watched_idx,
        profile_vec, content_matrix, animes,
        ui_mat, users, items, user_index, item_index
    )

    # Écraser complètement les poids existants de l'utilisateur
    weights_store.set_user_weights(user_id, updated_weights_list)
    weights_store.save_weights()

    print("Updated weights:", weights_store.get_user_weights(user_id))

    # --- Re-rank après update ---
    scored_after = []
    for idx in final_pool:
        res = hybrid_score_for_user_item(weights_store.weights, user_id, idx, profile_vec, content_matrix, animes,
                                         ui_mat, users, items, user_index, item_index)
        scored_after.append((idx, res["score"]))
    scored_after_sorted = sorted(scored_after, key=lambda x: -x[1])
    print("\nTop recommendations after watch and weight update:")
    for idx,score in scored_after_sorted[:10]:
        print(animes.iloc[idx]["anime_id"], animes.iloc[idx]["name"], f"score={score:.3f}")

if __name__ == "__main__":
    main()
