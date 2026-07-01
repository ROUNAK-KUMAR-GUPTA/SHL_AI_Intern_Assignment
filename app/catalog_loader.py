"""
Catalog loader: parses and indexes the SHL product catalog.
Provides search functionality using TF-IDF + FAISS for semantic retrieval.
"""
import json
import re
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import faiss

CATALOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "catalog_clean.json")

# Key category to test_type code mapping
KEY_TO_TYPE = {
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Simulations": "S",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
}

# Test type code to full name (for display)
TYPE_TO_NAME = {v: k for k, v in KEY_TO_TYPE.items()}
TYPE_TO_NAME.update({
    "A,S": "Ability & Aptitude, Simulations",
    "K,S": "Knowledge & Skills, Simulations",
    "B,S": "Biodata & Situational Judgment, Simulations",
    "P,C": "Personality & Behavior, Competencies",
    "C,K": "Competencies, Knowledge & Skills",
})


def _parse_duration(raw: str) -> str:
    """Extract clean duration string from raw duration field."""
    if not raw or not raw.strip():
        return "—"
    raw = raw.strip()
    # Try to extract minutes
    m = re.search(r'(\d+)\s*min', raw, re.IGNORECASE)
    if m:
        return f"{m.group(1)} minutes"
    # Try to extract "Approximate Completion Time in minutes = X"
    m = re.search(r'=\s*(\d+)', raw)
    if m:
        return f"{m.group(1)} minutes"
    # Try "Untimed"
    if 'untimed' in raw.lower():
        # Check if there's an approximate time after
        m2 = re.search(r'approx\.?\s*(\d+)', raw, re.IGNORECASE)
        if m2:
            return f"Untimed (approx. {m2.group(1)} minutes)"
        return "Untimed"
    # Try "Variable"
    if 'variable' in raw.lower():
        return "Variable"
    return raw.strip()


def _parse_languages(raw: str) -> list:
    """Parse languages_raw into a list of languages."""
    if not raw or not raw.strip():
        return []
    langs = [l.strip() for l in raw.split(',') if l.strip()]
    return langs


def _format_languages(langs: list, raw: str) -> str:
    """Format languages for display. Show first few + count."""
    if not langs:
        return "—"
    if len(langs) <= 4:
        return ", ".join(langs)
    return f"{', '.join(langs[:4])} _(+{len(langs)-4} more)_"


class Catalog:
    """SHL Assessment Catalog with search capabilities."""

    def __init__(self, path: str = CATALOG_PATH):
        self.path = path
        self.items = []
        self._load()
        self._build_index()

    def _load(self):
        """Load and parse catalog JSON."""
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.items = []
        for item in data:
            # Parse duration
            duration = _parse_duration(item.get("duration_raw", ""))
            # Parse languages
            langs = _parse_languages(item.get("languages_raw", ""))
            lang_display = _format_languages(langs, item.get("languages_raw", ""))

            # Derive test_type from keys
            keys = item.get("keys", [])
            type_codes = []
            for k in keys:
                if k in KEY_TO_TYPE:
                    type_codes.append(KEY_TO_TYPE[k])

            self.items.append({
                "entity_id": item.get("entity_id", ""),
                "name": item.get("name", "").strip(),
                "url": item.get("link", ""),
                "test_type": ",".join(type_codes),
                "keys": keys,
                "duration": duration,
                "duration_raw": item.get("duration_raw", ""),
                "languages": langs,
                "languages_display": lang_display,
                "languages_raw": item.get("languages_raw", ""),
                "job_levels": item.get("job_levels", []),
                "job_levels_raw": item.get("job_levels_raw", ""),
                "remote": item.get("remote", ""),
                "adaptive": item.get("adaptive", ""),
                "description": item.get("description", ""),
                "search_text": self._build_search_text(item),
            })

    def _build_search_text(self, item: dict) -> str:
        """Build a rich search text for TF-IDF indexing."""
        parts = [
            item.get("name", ""),
            item.get("description", ""),
            " ".join(item.get("keys", [])),
            item.get("job_levels_raw", ""),
            item.get("languages_raw", ""),
            item.get("duration_raw", ""),
        ]
        # Add common aliases for better search
        name_lower = item.get("name", "").lower()
        # Add expanded test type descriptions
        for k in item.get("keys", []):
            parts.append(k)
        # Add specific keywords from name
        if "opq" in name_lower:
            parts.append("personality questionnaire occupational behavioural style")
        if "verify" in name_lower and "g+" in name_lower:
            parts.append("cognitive ability reasoning inductive numerical deductive general")
        if "numerical" in name_lower:
            parts.append("math quantitative numbers calculation")
        if "verbal" in name_lower:
            parts.append("reading comprehension language vocabulary")
        if "safety" in name_lower or "dependability" in name_lower:
            parts.append("safety reliability compliance procedure integrity")
        if "java" in name_lower:
            parts.append("programming developer software")
        if "excel" in name_lower or "word" in name_lower or "microsoft" in name_lower:
            parts.append("office administrative tools software")
        if "svar" in name_lower:
            parts.append("spoken language verbal communication accent")
        if "simulation" in name_lower or "contact center" in name_lower:
            parts.append("simulation scenario role-play practical")
        if "graduate" in name_lower:
            parts.append("graduate entry campus university student")
        if "sales" in name_lower:
            parts.append("selling revenue quota business development")
        if "leadership" in name_lower:
            parts.append("executive leader management strategic cxo director")
        if "smart interview" in name_lower:
            parts.append("live coding interview technical assessment")
        if "docker" in name_lower:
            parts.append("containerization devops deployment")
        if "aws" in name_lower or "amazon" in name_lower:
            parts.append("cloud computing devops infrastructure")
        if "spring" in name_lower:
            parts.append("java framework backend microservice")
        if "sql" in name_lower:
            parts.append("database query relational")
        if "hipaa" in name_lower:
            parts.append("healthcare compliance privacy security patient")
        if "medical" in name_lower:
            parts.append("healthcare clinical terminology")
        if "networking" in name_lower:
            parts.append("network infrastructure tcp/ip protocol")
        if "linux" in name_lower:
            parts.append("operating system unix systems administration")
        if "financial" in name_lower or "accounting" in name_lower:
            parts.append("finance accounting bookkeeping")
        if "statistics" in name_lower:
            parts.append("data analysis statistical")
        if "motivation" in name_lower:
            parts.append("motivation drives engagement energy")
        if "gsa" in name_lower or "global skills" in name_lower:
            parts.append("skills assessment competency self-report development")

        return " ".join(parts)

    def _build_index(self):
        """Build TF-IDF + FAISS index for search."""
        texts = [item["search_text"] for item in self.items]
        # Filter out empty texts
        valid_indices = [i for i, t in enumerate(texts) if t.strip()]
        self.valid_indices = valid_indices

        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2),
            lowercase=True,
            sublinear_tf=True,
        )

        valid_texts = [texts[i] for i in valid_indices]
        if not valid_texts:
            raise ValueError("No valid texts to index")

        tfidf_matrix = self.vectorizer.fit_transform(valid_texts)
        tfidf_dense = tfidf_matrix.toarray().astype(np.float32)
        tfidf_norm = normalize(tfidf_dense, norm='l2')

        # Build FAISS index (inner product = cosine similarity for normalized vectors)
        dim = tfidf_norm.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(tfidf_norm)

    def search(self, query: str, top_k: int = 20) -> list:
        """Search catalog using TF-IDF + FAISS. Returns top-k items with scores."""
        if not query.strip():
            return []

        query_vec = self.vectorizer.transform([query]).toarray().astype(np.float32)
        query_norm = normalize(query_vec, norm='l2')

        k = min(top_k, len(self.valid_indices))
        scores, indices = self.index.search(query_norm, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            real_idx = self.valid_indices[idx]
            item = self.items[real_idx].copy()
            item["score"] = float(score)
            results.append(item)

        return results

    def search_by_filters(
        self,
        query: str = "",
        test_types: list = None,
        job_levels: list = None,
        languages: list = None,
        top_k: int = 20,
    ) -> list:
        """Search with optional filters applied on top of text search."""
        results = self.search(query, top_k=min(top_k * 3, 100)) if query else []

        # If no query, start with all items
        if not query:
            results = [item.copy() for item in self.items]
            for item in results:
                item["score"] = 1.0

        # Apply filters
        filtered = []
        for item in results:
            # Filter by test types
            if test_types:
                item_types = set(item["test_type"].split(",")) if item["test_type"] else set()
                if not any(t in item_types for t in test_types):
                    continue

            # Filter by job levels
            if job_levels:
                item_levels = set(item.get("job_levels", []))
                if not any(l in item_levels for l in job_levels):
                    continue

            # Filter by languages
            if languages:
                item_langs = set(item.get("languages", []))
                if not any(l in item_langs for l in languages):
                    continue

            filtered.append(item)

        # Sort by score descending
        filtered.sort(key=lambda x: x.get("score", 0), reverse=True)
        return filtered[:top_k]

    def get_by_name(self, name: str) -> dict:
        """Find an assessment by exact or fuzzy name match."""
        name_lower = name.lower().strip()
        # Exact match
        for item in self.items:
            if item["name"].lower() == name_lower:
                return item
        # Partial match
        for item in self.items:
            if name_lower in item["name"].lower() or item["name"].lower() in name_lower:
                return item
        # Keyword match
        keywords = name_lower.split()
        best_match = None
        best_score = 0
        for item in self.items:
            item_lower = item["name"].lower()
            score = sum(1 for kw in keywords if kw in item_lower)
            if score > best_score:
                best_score = score
                best_match = item
        return best_match

    def get_by_url(self, url: str) -> dict:
        """Find an assessment by URL."""
        for item in self.items:
            if item["url"] == url:
                return item
        return None

    def format_recommendation(self, item: dict) -> dict:
        """Format a catalog item as a recommendation for the API response."""
        return {
            "name": item["name"],
            "url": item["url"],
            "test_type": item["test_type"],
        }

    def format_table_row(self, item: dict, index: int) -> str:
        """Format a catalog item as a markdown table row."""
        return (
            f"| {index} | {item['name']} | {item['test_type']} | "
            f"{', '.join(item['keys'])} | {item['duration']} | "
            f"{item['languages_display']} | {item['url']} |"
        )

    def format_table(self, items: list) -> str:
        """Format a list of items as a markdown table."""
        if not items:
            return ""
        header = "| # | Name | Test Type | Keys | Duration | Languages | URL |\n|---|------|-----------|------|----------|-----------|-----|"
        rows = [self.format_table_row(item, i+1) for i, item in enumerate(items)]
        return header + "\n" + "\n".join(rows)

    def compare_items(self, name1: str, name2: str) -> str:
        """Compare two assessments and return a grounded comparison."""
        item1 = self.get_by_name(name1)
        item2 = self.get_by_name(name2)
        if not item1 or not item2:
            missing = []
            if not item1:
                missing.append(name1)
            if not item2:
                missing.append(name2)
            return f"I couldn't find {' and '.join(missing)} in the catalog. Could you clarify which assessments you'd like to compare?"

        # Build comparison from catalog data
        comparison_parts = []
        comparison_parts.append(f"**{item1['name']}** vs **{item2['name']}**:")

        # Test types
        if item1["test_type"] != item2["test_type"]:
            comparison_parts.append(
                f"- Test type: {item1['name']} is {item1['test_type']} ({', '.join(item1['keys'])}), "
                f"while {item2['name']} is {item2['test_type']} ({', '.join(item2['keys'])})."
            )
        else:
            comparison_parts.append(
                f"- Both are {item1['test_type']} type ({', '.join(item1['keys'])})."
            )

        # Duration
        if item1["duration"] != item2["duration"]:
            comparison_parts.append(
                f"- Duration: {item1['name']} takes {item1['duration']}, "
                f"while {item2['name']} takes {item2['duration']}."
            )

        # Job levels
        levels1 = set(item1.get("job_levels", []))
        levels2 = set(item2.get("job_levels", []))
        if levels1 != levels2:
            only1 = levels1 - levels2
            only2 = levels2 - levels1
            if only1 or only2:
                parts = []
                if only1:
                    parts.append(f"{item1['name']} additionally covers {', '.join(sorted(only1))}")
                if only2:
                    parts.append(f"{item2['name']} additionally covers {', '.join(sorted(only2))}")
                comparison_parts.append(f"- Job levels: {'; '.join(parts)}.")

        # Languages
        langs1 = set(item1.get("languages", []))
        langs2 = set(item2.get("languages", []))
        if langs1 and langs2 and langs1 != langs2:
            only1 = langs1 - langs2
            only2 = langs2 - langs1
            if only1 or only2:
                parts = []
                if only1:
                    parts.append(f"{item1['name']} supports {len(only1)} additional languages")
                if only2:
                    parts.append(f"{item2['name']} supports {len(only2)} additional languages")
                comparison_parts.append(f"- Languages: {'; '.join(parts)}.")

        # Remote/Adaptive
        if item1.get("remote") != item2.get("remote"):
            comparison_parts.append(
                f"- Remote testing: {item1['name']} is {'remote' if item1.get('remote') == 'yes' else 'not remote'}, "
                f"while {item2['name']} is {'remote' if item2.get('remote') == 'yes' else 'not remote'}."
            )
        if item1.get("adaptive") != item2.get("adaptive"):
            comparison_parts.append(
                f"- Adaptive: {item1['name']} is {'adaptive' if item1.get('adaptive') == 'yes' else 'not adaptive'}, "
                f"while {item2['name']} is {'adaptive' if item2.get('adaptive') == 'yes' else 'not adaptive'}."
            )

        # Description comparison
        if item1.get("description") and item2.get("description"):
            comparison_parts.append(
                f"- {item1['name']}: {item1['description'][:200]}"
            )
            comparison_parts.append(
                f"- {item2['name']}: {item2['description'][:200]}"
            )

        return "\n".join(comparison_parts)

    def get_all_urls(self) -> set:
        """Get all valid catalog URLs for validation."""
        return {item["url"] for item in self.items if item["url"]}

    def is_valid_url(self, url: str) -> bool:
        """Check if a URL is from the catalog."""
        return url in self.get_all_urls()


# Singleton instance
_catalog = None

def get_catalog() -> Catalog:
    """Get or create the catalog singleton."""
    global _catalog
    if _catalog is None:
        _catalog = Catalog()
    return _catalog
