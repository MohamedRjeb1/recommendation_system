# evaluation.py
import numpy as np
from recommenders import hybrid_score_for_user_item

def evaluate_precision_recall_f1(ui_mat, users, items, user_index, item_index,
                                 animes, content_matrix, weights_store,
                                 top_k=10, test_ratio=0.2, random_seed=42):
    """
    Évaluation du système de recommandation hybride.
    Calcule Precision@K, Recall@K et F1@K.
    """
    np.random.seed(random_seed)
    precisions = []
    recalls = []

    for user_id in users:
        user_idx = user_index[user_id]
        # Suppression de .toarray(), ui_mat est un ndarray
        rated_items = np.where(ui_mat[user_idx].flatten() > 0)[0]

        if len(rated_items) < 2:
            continue  # Ignorer utilisateurs avec trop peu de notes

        test_size = max(1, int(len(rated_items) * test_ratio))
        test_items = np.random.choice(rated_items, size=test_size, replace=False)
        train_items = np.setdiff1d(rated_items, test_items)

        profile_vec = content_matrix.getrow(0)  # Profil neutre pour simplifier

        scored = []
        for idx in range(len(items)):
            if idx in train_items:
                continue
            res = hybrid_score_for_user_item(weights_store.weights, user_id, idx,
                                             profile_vec, content_matrix,
                                             animes, ui_mat, users, items,
                                             user_index, item_index)
            scored.append((idx, res["score"]))

        topk_idx = [x[0] for x in sorted(scored, key=lambda x: -x[1])[:top_k]]

        # Calcul Precision et Recall
        hits = sum([1 for t in test_items if t in topk_idx])
        precisions.append(hits / top_k)
        recalls.append(hits / len(test_items))

    precision_avg = np.mean(precisions) if precisions else 0.0
    recall_avg = np.mean(recalls) if recalls else 0.0
    if precision_avg + recall_avg > 0:
        f1_avg = 2 * (precision_avg * recall_avg) / (precision_avg + recall_avg)
    else:
        f1_avg = 0.0

    print(f"Precision@{top_k}: {precision_avg:.4f}")
    print(f"Recall@{top_k}: {recall_avg:.4f}")
    print(f"F1@{top_k}: {f1_avg:.4f}")

    return precision_avg, recall_avg, f1_avg
