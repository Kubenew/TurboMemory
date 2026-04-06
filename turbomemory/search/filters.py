"""Metadata filters for search."""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum


class FilterOperator(Enum):
    """Filter operators."""
    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GE = "ge"
    LT = "lt"
    LE = "le"
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    BETWEEN = "between"


@dataclass
class Filter:
    """A single filter condition."""
    field: str
    operator: FilterOperator
    value: Any
    
    def matches(self, doc: Dict[str, Any]) -> bool:
        """Check if document matches this filter."""
        doc_value = doc.get(self.field)
        
        if self.operator == FilterOperator.EQ:
            return doc_value == self.value
        elif self.operator == FilterOperator.NE:
            return doc_value != self.value
        elif self.operator == FilterOperator.GT:
            return doc_value is not None and doc_value > self.value
        elif self.operator == FilterOperator.GE:
            return doc_value is not None and doc_value >= self.value
        elif self.operator == FilterOperator.LT:
            return doc_value is not None and doc_value < self.value
        elif self.operator == FilterOperator.LE:
            return doc_value is not None and doc_value <= self.value
        elif self.operator == FilterOperator.IN:
            return doc_value in self.value if doc_value else False
        elif self.operator == FilterOperator.NOT_IN:
            return doc_value not in self.value if doc_value else True
        elif self.operator == FilterOperator.CONTAINS:
            if isinstance(doc_value, str):
                return self.value in doc_value
            return False
        elif self.operator == FilterOperator.BETWEEN:
            if isinstance(self.value, (list, tuple)) and len(self.value) == 2:
                return doc_value is not None and self.value[0] <= doc_value <= self.value[1]
            return False
        
        return False


class MetadataFilter:
    """Filter documents by metadata fields."""
    
    def __init__(self):
        self._filters: List[Filter] = []
    
    def add_filter(
        self,
        field: str,
        operator: FilterOperator,
        value: Any,
    ) -> "MetadataFilter":
        """Add a filter. Returns self for chaining."""
        self._filters.append(Filter(field=field, operator=operator, value=value))
        return self
    
    def topic_eq(self, topic: str) -> "MetadataFilter":
        """Filter by exact topic match."""
        return self.add_filter("topic", FilterOperator.EQ, topic)
    
    def topic_in(self, topics: List[str]) -> "MetadataFilter":
        """Filter by topic in list."""
        return self.add_filter("topic", FilterOperator.IN, topics)
    
    def confidence_ge(self, min_confidence: float) -> "MetadataFilter":
        """Filter by minimum confidence."""
        return self.add_filter("confidence", FilterOperator.GE, min_confidence)
    
    def staleness_le(self, max_staleness: float) -> "MetadataFilter":
        """Filter by maximum staleness."""
        return self.add_filter("staleness", FilterOperator.LE, max_staleness)
    
    def quality_ge(self, min_quality: float) -> "MetadataFilter":
        """Filter by minimum quality score."""
        return self.add_filter("quality_score", FilterOperator.GE, min_quality)
    
    def verified_only(self) -> "MetadataFilter":
        """Filter to only verified chunks."""
        return self.add_filter("verified", FilterOperator.EQ, True)
    
    def created_after(self, ts: str) -> "MetadataFilter":
        """Filter by creation timestamp after."""
        return self.add_filter("timestamp", FilterOperator.GE, ts)
    
    def created_before(self, ts: str) -> "MetadataFilter":
        """Filter by creation timestamp before."""
        return self.add_filter("timestamp", FilterOperator.LE, ts)
    
    def created_between(self, start: str, end: str) -> "MetadataFilter":
        """Filter by creation timestamp between."""
        return self.add_filter("timestamp", FilterOperator.BETWEEN, [start, end])
    
    def has_ttl(self) -> "MetadataFilter":
        """Filter to only chunks with TTL."""
        return self.add_filter("ttl_ts", FilterOperator.NE, None)
    
    def expired(self) -> "MetadataFilter":
        """Filter to only expired chunks."""
        now = datetime.now(timezone.utc).isoformat()
        return self.add_filter("ttl_ts", FilterOperator.LT, now)
    
    def clear(self) -> "MetadataFilter":
        """Clear all filters."""
        self._filters.clear()
        return self
    
    def filter(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter a list of documents."""
        if not self._filters:
            return documents
        
        return [doc for doc in documents if self._matches_all(doc)]
    
    def _matches_all(self, doc: Dict[str, Any]) -> bool:
        """Check if document matches all filters."""
        return all(f.matches(doc) for f in self._filters)
    
    def __len__(self) -> int:
        return len(self._filters)
    
    def __repr__(self) -> str:
        return f"MetadataFilter(filters={len(self._filters)})"


# Convenience function for creating filters
def create_filter(
    field: str,
    operator: str,
    value: Any,
) -> Filter:
    """Create a filter from string operator."""
    op = FilterOperator(operator)
    return Filter(field=field, operator=op, value=value)


# Query string parser for filters
def parse_filter_string(filter_str: str) -> List[Filter]:
    """Parse filter string like 'topic=python,confidence>0.5'."""
    filters = []
    
    for part in filter_str.split(","):
        part = part.strip()
        if not part:
            continue
        
        # Find operator
        for op_str in ["!=", ">=", "<=", "==", ">", "<", "=", " in ", " not in "]:
            if op_str in part:
                field, value_str = part.split(op_str, 1)
                field = field.strip()
                value_str = value_str.strip()
                
                # Determine operator
                if op_str == "==" or op_str == "=":
                    op = FilterOperator.EQ
                elif op_str == "!=":
                    op = FilterOperator.NE
                elif op_str == ">":
                    op = FilterOperator.GT
                elif op_str == ">=":
                    op = FilterOperator.GE
                elif op_str == "<":
                    op = FilterOperator.LT
                elif op_str == "<=":
                    op = FilterOperator.LE
                elif op_str == " in ":
                    op = FilterOperator.IN
                    value_str = [v.strip() for v in value_str.split(",")]
                elif op_str == " not in ":
                    op = FilterOperator.NOT_IN
                    value_str = [v.strip() for v in value_str.split(",")]
                else:
                    continue
                
                # Try to convert value to appropriate type
                value = value_str
                if value_str.lower() == "true":
                    value = True
                elif value_str.lower() == "false":
                    value = False
                elif value_str.isdigit():
                    value = int(value_str)
                elif value_str.replace(".", "", 1).isdigit():
                    value = float(value_str)
                
                filters.append(Filter(field=field, operator=op, value=value))
                break
    
    return filters
