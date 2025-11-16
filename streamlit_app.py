# streamlit_app.py
"""
Interface Streamlit pour le système de recommandation hybride.
Usage:
    pip install streamlit
    streamlit run streamlit_app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
from recommenders import content_shortlist, hybrid_score_for_user_item, update_weights_on_watch
from data_prep import load_data, compute_trending, build_content_embeddings, build_user_item_matrix
from weights_store import WeightsStore
from evaluation import evaluate_precision_recall_f1


st.set_page_config(page_title="Hybrid Recommender Demo", layout="wide", initial_sidebar_state="expanded")

# --------------------------
# Helpers
# --------------------------
@st.cache_resource
def load_all():
    animes, ratings = load_data()
    animes = compute_trending(animes)
    content_matrix, meta = build_content_embeddings(animes)
    ui_mat, users, items, user_index, item_index = build_user_item_matrix(ratings, animes)
    return animes, ratings, content_matrix, meta, ui_mat, users, items, user_index, item_index

def ensure_user_key_normalization(weights_store):
    normalized = {}
    for k, v in weights_store.weights.items():
        normalized[str(k)] = v
    weights_store.weights = normalized

def build_profile_from_onboarding_local(onboarding, animes_df, meta):
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

def pretty_weights(w):
    return [round(float(x), 4) for x in w]

# Nouvelle fonction pour afficher les cartes anime simplifiées (moderne)
def display_anime_cards(df, title):
    st.subheader(title)
    for _, row in df.iterrows():
        with st.container():
            cols = st.columns([5,1])
            with cols[0]:
                st.markdown(f"**{row['name']}**")
                st.caption(f"{row['genre']}")
            with cols[1]:
                if st.button("Watch", key=f"watch_{row['index']}_{title}"):
                    watched_idx = int(row['index'])
                    before = weights_store.get_user_weights(user_id)
                    st.info(f"Avant update: {pretty_weights(before)}")
                    updated = update_weights_on_watch(
                        weights_store.weights, user_id, watched_idx,
                        content_matrix.getrow(0),
                        content_matrix, animes,
                        ui_mat, users, items, user_index, item_index
                    )
                    weights_store.set_user_weights(user_id, updated)
                    weights_store.save_weights()
                    st.success(f"Poids mis à jour: {pretty_weights(updated)}")
                    st.rerun()

# --------------------------
# Load data
# --------------------------
with st.spinner("Chargement des données..."):
    animes, ratings, content_matrix, meta, ui_mat, users, items, user_index, item_index = load_all()

# --------------------------
# Sidebar controls
# --------------------------
st.sidebar.title("Session")
user_id = st.sidebar.number_input("User id (test)", value=10000, step=1)
weights_store = WeightsStore("user_weights.json")
ensure_user_key_normalization(weights_store)
if weights_store.get_user_weights(user_id) is None:
    weights_store.set_user_weights(user_id, [0.7, 0.2, 0.1])
    weights_store.save_weights()
st.sidebar.write("User internal weights:", pretty_weights(weights_store.get_user_weights(user_id)))

# --------------------------
# Page title
# --------------------------
st.title("Hybrid Recommendation — Demo")
st.markdown("**Cold-start → Onboarding → Shortlist → Watch → Hybrid rerank & update weights**")

# --------------------------
# Trending & All-time panels côte à côte (simplifiées)
# --------------------------
col1, col2 = st.columns(2, gap="medium")
top_trend = animes.sort_values("views_last_3m", ascending=False).head(10)[["name","genre"]].reset_index()
top_all = animes.sort_values("views_all_time", ascending=False).head(10)[["name","genre"]].reset_index()

with col1:
    display_anime_cards(top_trend, "Trending (last 3 months)")
with col2:
    display_anime_cards(top_all, "All-time most viewed")

st.markdown("---")

# --------------------------
# Onboarding
# --------------------------
st.subheader("Onboarding (sélectionne tes préférences)")
genres_all = sorted({g.strip() for row in animes["genre"].dropna().astype(str) for g in row.split(",")})
selected_genres = st.multiselect("Choisis tes genres préférés", options=genres_all, default=["Action","Fantasy"])
type_choice = st.radio("Type", options=["TV", "Movie"], index=0)
length_choice = st.radio("Longueur", options=["both","short","long"], index=0)

do_profile = st.button("Générer shortlist contenu")

if "profile_vec" not in st.session_state:
    st.session_state["profile_vec"] = None
if "shortlist_idxs" not in st.session_state:
    st.session_state["shortlist_idxs"] = []

if do_profile:
    onboarding = {"genres": selected_genres, "type":[type_choice], "length": length_choice}
    profile_vec = build_profile_from_onboarding_local(onboarding, animes, meta)
    st.session_state["profile_vec"] = profile_vec
    shortlist_idxs, sims = content_shortlist(profile_vec, content_matrix, top_n=200)
    st.session_state["shortlist_idxs"] = shortlist_idxs
    st.success(f"Shortlist contenu générée ({len(shortlist_idxs)} items)")

# --------------------------
# Display shortlist (top 20)
# --------------------------
profile_vec = st.session_state.get("profile_vec")
if len(st.session_state.get("shortlist_idxs", []))>0:
    shortlist = animes.iloc[st.session_state["shortlist_idxs"][:20]][["name","genre","views_last_3m"]].reset_index()
    display_anime_cards(shortlist, "Shortlist basée sur ton profil")

# --------------------------
# Hybrid recommendations rerank
# --------------------------
st.subheader("Recommandations hybrides (rerank avec poids utilisateur)")
final_pool = st.session_state["shortlist_idxs"][:50] if len(st.session_state.get("shortlist_idxs", []))>0 else list(top_all.index[:50])
scored = []
for idx in final_pool:
    profile_vec_safe = profile_vec if profile_vec is not None else content_matrix.getrow(0)
    res = hybrid_score_for_user_item(weights_store.weights, user_id, int(idx),
                                     profile_vec_safe,
                                     content_matrix, animes,
                                     ui_mat, users, items, user_index, item_index)
    scored.append((idx, res["score"]))
scored_sorted = sorted(scored, key=lambda x: -x[1])
top_rec = [(int(x[0]), x[1]) for x in scored_sorted[:10]]
rec_df = pd.DataFrame([{"name": animes.iloc[i]["name"], "score": round(s,4)} for i,s in top_rec])
st.table(rec_df)

# --------------------------
# Show current weights
# --------------------------
st.subheader("Poids utilisateur (current)")
curr = weights_store.get_user_weights(user_id)
st.write("Interne:", pretty_weights(curr))
st.bar_chart(pd.DataFrame({"weights": curr}, index=["content","collab","pop"]))



if st.button("Évaluer Precision, Recall & F1"):
    with st.spinner("Évaluation en cours..."):
        precision, recall, f1 = evaluate_precision_recall_f1(
            ui_mat, users, items, user_index, item_index,
            animes, content_matrix, weights_store, top_k=10
        )
        st.success(f"Precision@10: {precision:.4f} | Recall@10: {recall:.4f} | F1@10: {f1:.4f}")



