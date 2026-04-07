from sklearn.cluster import KMeans

def cluster_topics(vectors, k=8):
    if len(vectors) < k:
        k = max(1, len(vectors))

    model = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = model.fit_predict(vectors)
    return labels