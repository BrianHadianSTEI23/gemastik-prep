"""
Task adapters package exposing standardized model adapters for all 14 NLP categories.
"""

from .base import BaseTaskAdapter
from .classification import ClassicalClassificationAdapter, TransformerClassificationAdapter
from .zeroshot import ZeroShotClassificationAdapter
from .multilabel import MultilabelClassificationAdapter
from .regression import TextRegressionAdapter
from .ner import NERAdapter
from .qa import QuestionAnsweringAdapter
from .summarization import SummarizationAdapter
from .translation import TranslationAdapter
from .generation import GenerationAdapter
from .similarity import SemanticSimilarityAdapter
from .retrieval import RetrievalAdapter
from .topic_modeling import TopicModelingAdapter
from .pair_classification import TextPairClassificationAdapter

__all__ = [
    "BaseTaskAdapter",
    "ClassicalClassificationAdapter",
    "TransformerClassificationAdapter",
    "ZeroShotClassificationAdapter",
    "MultilabelClassificationAdapter",
    "TextRegressionAdapter",
    "NERAdapter",
    "QuestionAnsweringAdapter",
    "SummarizationAdapter",
    "TranslationAdapter",
    "GenerationAdapter",
    "SemanticSimilarityAdapter",
    "RetrievalAdapter",
    "TopicModelingAdapter",
    "TextPairClassificationAdapter",
]
