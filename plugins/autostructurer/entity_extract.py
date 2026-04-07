import re
from dateutil.parser import parse as parse_date

money_re = re.compile(r"(\d+(?:\.\d+)?)\s?(usd|eur|czk|gbp|\$|€|kč)", re.IGNORECASE)

def extract_entities(text: str):
    ent={"dates":[], "money":[], "emails":[], "phones":[], "urls":[]}
    ent["emails"]=list(set(re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)))
    ent["urls"]=list(set(re.findall(r"https?://[^\s]+", text)))
    phones=re.findall(r"\+?\d[\d\s\-]{7,}\d", text)
    ent["phones"]=list(set([p.strip() for p in phones]))
    for m in money_re.findall(text):
        ent["money"].append({"amount": float(m[0]), "currency": m[1].upper()})
    for token in re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text):
        try:
            ent["dates"].append(parse_date(token).date().isoformat())
        except:
            pass
    ent["dates"]=list(set(ent["dates"]))
    return ent