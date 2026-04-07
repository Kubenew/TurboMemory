def merge_max(results_a, results_b, top_k=10):
    merged={}
    for cid, score, via in results_a + results_b:
        if cid not in merged or score > merged[cid]["score"]:
            merged[cid]={"score":score, "via":via}
    items=sorted(merged.items(), key=lambda x: -x[1]["score"])[:top_k]
    return [(cid, v["score"], v["via"]) for cid, v in items]