def detect_contradictions(texts):
    flags = []
    for t in texts:
        s = t.lower()
        score = 0.0

        if "paid" in s and "unpaid" in s:
            score = 0.9
        elif "approved" in s and "rejected" in s:
            score = 0.9
        elif "yes" in s and "no" in s:
            score = 0.5

        flags.append(score)
    return flags