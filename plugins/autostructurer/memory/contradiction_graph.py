def contradiction_score(text: str) -> float:
    t=text.lower()
    if "paid" in t and "unpaid" in t:
        return 0.9
    if "approved" in t and "rejected" in t:
        return 0.9
    return 0.0