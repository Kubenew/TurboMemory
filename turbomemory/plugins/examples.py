"""Example: Custom quality scorer using keyword density."""

from turbomemory.plugins import QualityScorer


class KeywordDensityScorer(QualityScorer):
    """Quality scorer based on keyword density and specificity."""

    name = "keyword_density_scorer"
    version = "0.1.0"
    description = "Scores memory quality based on keyword density and specificity"

    def compute_score(self, chunk: dict) -> float:
        text = chunk.get("text", "")
        confidence = chunk.get("confidence", 0.5)
        staleness = chunk.get("staleness", 0.0)

        words = text.split()
        if not words:
            return 0.0

        unique_words = set(w.lower() for w in words)
        lexical_diversity = len(unique_words) / len(words)

        has_numbers = any(c.isdigit() for c in text)
        has_dates = any(len(w) == 10 and '-' in w for w in words)

        specificity = 0.0
        if lexical_diversity > 0.5:
            specificity += 0.3
        if has_numbers:
            specificity += 0.2
        if has_dates:
            specificity += 0.2
        if len(words) >= 10:
            specificity += 0.3

        freshness = 1.0 - staleness

        score = confidence * 0.3 + freshness * 0.3 + specificity * 0.4
        return min(1.0, max(0.0, score))
