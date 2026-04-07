def detect_schema(text: str) -> str:
    t=text.lower()
    if "invoice" in t or "amount due" in t or "iban" in t:
        return "invoice"
    if "contract" in t or "agreement" in t:
        return "contract"
    if "meeting" in t or "agenda" in t:
        return "meeting_notes"
    if "error" in t or "exception" in t:
        return "log"
    return "generic_text"