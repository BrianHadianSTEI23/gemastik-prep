"""
Preprocessing and Feature Engineering module.
Implements rule-based, non-destructive, and leak-free text transformations.
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

INDONESIAN_NEGATIONS = {"tidak", "bukan", "jangan", "tanpa", "belum", "tak", "tiada"}
INDONESIAN_STOPWORDS = {
    "yang", "di", "dan", "dari", "ke", "ini", "itu", "untuk", "pada", "adalah",
    "sebagai", "dengan", "akan", "juga", "ada", "mereka", "sudah", "saya", "kami",
    "kita", "dia", "oleh", "saat", "dalam", "bisa", "karena", "atau", "pun", "saja"
} - INDONESIAN_NEGATIONS

ENGLISH_NEGATIONS = {"not", "no", "never", "neither", "nor", "without", "hardly", "barely"}

@dataclass
class PreprocessingRecommendation:
    lowercase: bool
    unicode_normalize: bool
    remove_html: bool
    handle_urls: bool
    normalize_whitespace: bool
    remove_stopwords: bool
    preserve_negations: bool
    stemming: bool
    ngram_range: Tuple[int, int]
    sublinear_tf: bool
    min_df: int
    max_features: int
    include_metadata_features: bool
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return vars(self)


def recommend_preprocessing(
    task: str,
    model_family: str = "classical",
    language: str = "indonesian"
) -> PreprocessingRecommendation:
    """
    Expert rule-based recommendation system:
    - Never remove stopwords or stem text for Transformer tokenizers.
    - Never lowercase for cased models or NER tasks.
    - Always preserve negations in classification / sentiment tasks.
    - Enable sublinear_tf and bigrams for classical TF-IDF.
    """
    task = task.lower().strip()
    fam = model_family.lower().strip()
    is_transformer = any(k in fam for k in ["transformer", "bert", "roberta", "t5", "gpt", "qwen", "llama"])

    if is_transformer or task in ["ner", "qa", "generation", "summarization", "translation"]:
        return PreprocessingRecommendation(
            lowercase=False,
            unicode_normalize=True,
            remove_html=True,
            handle_urls=True,
            normalize_whitespace=True,
            remove_stopwords=False,  # Transformers rely on full grammatical syntax and position embeddings
            preserve_negations=True,
            stemming=False,          # Transformers use subword tokenization (WordPiece/BPE)
            ngram_range=(1, 1),
            sublinear_tf=False,
            min_df=1,
            max_features=0,
            include_metadata_features=False,
            rationale=(
                "Transformer models utilize subword tokenizers (WordPiece/BPE) and position embeddings. "
                "Casing, punctuation, and stopwords are preserved to retain semantic and grammatical cues."
            )
        )
    else:
        # Classical TF-IDF model
        return PreprocessingRecommendation(
            lowercase=True,
            unicode_normalize=True,
            remove_html=True,
            handle_urls=True,
            normalize_whitespace=True,
            remove_stopwords=False,  # Keep negations and semantic markers
            preserve_negations=True,
            stemming=False,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
            max_features=15000,
            include_metadata_features=True,
            rationale=(
                "Classical linear/Bayesian models benefit from unigrams + bigrams and sublinear term frequency. "
                "Negations are explicitly preserved to maintain sentiment polarity."
            )
        )


class TextCleaner:
    def __init__(
        self,
        remove_html: bool = True,
        handle_urls: bool = True,
        normalize_whitespace: bool = True,
        unicode_normalize: bool = True,
        lowercase: bool = False,
        language: str = "indonesian",
        remove_stopwords: bool = False,
        preserve_negations: bool = True,
    ):
        self.remove_html = remove_html
        self.handle_urls = handle_urls
        self.normalize_whitespace = normalize_whitespace
        self.unicode_normalize = unicode_normalize
        self.lowercase = lowercase
        self.language = language.lower()
        self.remove_stopwords = remove_stopwords
        self.preserve_negations = preserve_negations

    def clean_text(self, text: Any) -> str:
        """Sanitizes a single text string."""
        if not isinstance(text, str):
            return ""
        # 1. Unicode normalization (NFKD)
        if self.unicode_normalize:
            text = unicodedata.normalize("NFKD", text)
        # 2. HTML stripping
        if self.remove_html:
            text = re.sub(r"<[^>]+>", " ", text)
        # 3. URL normalization
        if self.handle_urls:
            text = re.sub(r"https?://\S+|www\.\S+", " [URL] ", text)
        # 4. Whitespace standardization
        if self.normalize_whitespace:
            text = re.sub(r"\s+", " ", text).strip()
        # 5. Lowercase if explicitly requested
        if self.lowercase:
            text = text.lower()
        # 6. Stopword removal (preserving negations)
        if self.remove_stopwords:
            words = text.split()
            if self.language == "indonesian":
                filtered = [w for w in words if w.lower() not in INDONESIAN_STOPWORDS]
            elif self.language == "english":
                eng_stops = {"the", "is", "at", "which", "on", "and", "a", "an", "in", "to", "for", "of"}
                filtered = [w for w in words if w.lower() not in eng_stops]
            else:
                filtered = words
            text = " ".join(filtered)
        return text

    def clean_series(self, series: pd.Series) -> pd.Series:
        """Cleans an entire pandas Series."""
        return series.apply(self.clean_text)

    def combine_and_clean(self, df: pd.DataFrame, text_columns: List[str]) -> pd.Series:
        """Combines multiple text columns with a [SEP] token and cleans the resulting text."""
        valid_cols = [c for c in text_columns if c in df.columns]
        if not valid_cols:
            raise KeyError(f"None of {text_columns} exist in the DataFrame.")
        combined = df[valid_cols].fillna("").astype(str).agg(" [SEP] ".join, axis=1)
        return self.clean_series(combined)


def extract_lexical_meta_features(text_series: pd.Series) -> np.ndarray:
    """Extracts complementary lexical statistics without data leakage."""
    char_len = text_series.str.len().values[:, None]
    word_count = text_series.apply(lambda x: len(x.split())).values[:, None]
    digit_ratio = text_series.apply(lambda x: sum(c.isdigit() for c in x) / max(1, len(x))).values[:, None]
    punct_ratio = text_series.apply(lambda x: sum(c in '!?.,:;' for c in x) / max(1, len(x))).values[:, None]
    return np.hstack([char_len, word_count, digit_ratio, punct_ratio])
