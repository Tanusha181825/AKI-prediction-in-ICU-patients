import numpy as np
from sklearn.cluster import KMeans
from config import RANDOM_SEED


def badge_gradient_embeddings(model, X_pool):
    """
    Approximate BADGE gradient embeddings: uncertainty-weighted feature vectors.
    True BADGE uses the gradient of the loss w.r.t. the last layer; for tree
    models we approximate with prediction uncertainty p*(1-p) as the weight,
    which peaks at p=0.5 (maximum uncertainty) and vanishes at confident predictions.
    """
    probs = model.predict_proba(X_pool)[:, 1]
    uncertainty = probs * (1 - probs)
    return X_pool.values * uncertainty.reshape(-1, 1)


def _kmeans_select(X_group, embeddings, k, seed=RANDOM_SEED):
    """
    Run k-means++ on the embeddings, then for each cluster pick the single
    point closest to its centroid. This is the 'diverse + representative'
    part of BADGE — one pick per cluster, not just the k most uncertain points.
    """
    km = KMeans(n_clusters=k, init="k-means++", n_init=1, random_state=seed)
    km.fit(embeddings)

    selected = []
    for cluster in range(k):
        mask_c = km.labels_ == cluster
        pts = embeddings[mask_c]
        centroid = km.cluster_centers_[cluster]
        dist = np.linalg.norm(pts - centroid, axis=1)
        local_idx = np.where(mask_c)[0][np.argmin(dist)]
        selected.append(X_group.index[local_idx])
    return selected


def badge_query(model, X_pool, query_size=100):
    """Plain BADGE — single pool, no class stratification."""
    k = min(query_size, len(X_pool))
    embeddings = badge_gradient_embeddings(model, X_pool)
    return _kmeans_select(X_pool, embeddings, k)


def stratified_badge_query(model, X_pool, current_prevalence, query_size=100, prob_threshold=None):
    """
    Prevalence-stratified BADGE.

    Splits the unlabeled pool by predicted probability, runs BADGE
    independently within each group, and combines results proportional
    to the current labeled pool's AKI prevalence — preventing early
    iterations from over-selecting majority-class patients near a
    boundary that is itself skewed by the imbalance.

    prob_threshold: split point for pos/neg groups. If None, defaults
    to current_prevalence itself rather than a fixed 0.5 — this keeps
    the split meaningful even when the model's predicted probabilities
    are not yet well-calibrated around 0.5 in early iterations.
    """
    if prob_threshold is None:
        prob_threshold = current_prevalence

    probs = model.predict_proba(X_pool)[:, 1]
    pos_mask = probs >= prob_threshold
    neg_mask = ~pos_mask

    X_pos = X_pool[pos_mask]
    X_neg = X_pool[neg_mask]

    k_pos = max(1, int(round(query_size * current_prevalence)))
    k_neg = query_size - k_pos

    k_pos = min(k_pos, len(X_pos))
    k_neg = min(k_neg, len(X_neg))

    queried = []

    if k_pos > 0 and len(X_pos) >= k_pos:
        emb_pos = badge_gradient_embeddings(model, X_pos)
        queried += _kmeans_select(X_pos, emb_pos, k_pos)

    if k_neg > 0 and len(X_neg) >= k_neg:
        emb_neg = badge_gradient_embeddings(model, X_neg)
        queried += _kmeans_select(X_neg, emb_neg, k_neg)

    # Fallback: if stratification couldn't fill the full query_size
    # (e.g. one group too small), top up from the larger group with plain BADGE
    shortfall = query_size - len(queried)
    if shortfall > 0:
        remaining_pool = X_pool.drop(index=queried, errors="ignore")
        if len(remaining_pool) > 0:
            top_up = badge_query(model, remaining_pool, query_size=min(shortfall, len(remaining_pool)))
            queried += top_up

    return queried


def jaccard_similarity(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 1.0