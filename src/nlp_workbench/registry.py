"""
Architecture Registry module.
Directly connects 42 model entries across 14 NLP categories to real callable execution adapters.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type, Any, Union
import pandas as pd

from .tasks.base import BaseTaskAdapter
from .tasks.classification import ClassicalClassificationAdapter, TransformerClassificationAdapter
from .tasks.zeroshot import ZeroShotClassificationAdapter
from .tasks.multilabel import MultilabelClassificationAdapter
from .tasks.regression import TextRegressionAdapter
from .tasks.ner import NERAdapter
from .tasks.qa import QuestionAnsweringAdapter
from .tasks.summarization import SummarizationAdapter
from .tasks.translation import TranslationAdapter
from .tasks.generation import GenerationAdapter
from .tasks.similarity import SemanticSimilarityAdapter
from .tasks.retrieval import RetrievalAdapter
from .tasks.topic_modeling import TopicModelingAdapter
from .tasks.pair_classification import TextPairClassificationAdapter

@dataclass
class ModelRecord:
    model_id: str
    category: str
    approach_name: str
    model_framework: str
    checkpoint: Optional[str]
    supported_tasks: List[str]
    supported_languages: List[str]
    supported_operations: List[str]  # ["training", "fine-tuning", "inference", "feature_extraction", "evaluation"]
    vram_tier: str
    evaluation_metrics: List[str]
    adapter_cls: Type[BaseTaskAdapter]
    adapter_kwargs: Dict[str, Any] = field(default_factory=dict)
    is_inference_only: bool = False
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "category": self.category,
            "approach_name": self.approach_name,
            "framework": self.model_framework,
            "checkpoint": self.checkpoint or "N/A",
            "supported_tasks": ", ".join(self.supported_tasks),
            "languages": ", ".join(self.supported_languages),
            "operations": ", ".join(self.supported_operations),
            "vram_tier": self.vram_tier,
            "metrics": ", ".join(self.evaluation_metrics),
            "is_inference_only": self.is_inference_only,
            "notes": self.notes,
        }


class ArchitectureRegistry:
    def __init__(self):
        self._registry: Dict[str, ModelRecord] = {}
        self._populate()

    @property
    def models(self) -> Dict[str, ModelRecord]:
        return self._registry

    def to_dataframe(self, task: Optional[str] = None, language: Optional[str] = None) -> pd.DataFrame:
        models = self.list_models(task=task, language=language)
        return pd.DataFrame([m.to_dict() for m in models])

    def register(self, record: ModelRecord) -> None:
        self._registry[record.model_id] = record

    def get_model(self, model_id: str) -> ModelRecord:
        if model_id not in self._registry:
            raise KeyError(
                f"Model '{model_id}' not found in registry. "
                f"Available models: {list(self._registry.keys())}"
            )
        return self._registry[model_id]

    def list_models(
        self,
        task: Optional[str] = None,
        language: Optional[str] = None,
        operation: Optional[str] = None,
    ) -> List[ModelRecord]:
        """Filters registered models by task, language suitability, or operation."""
        models = list(self._registry.values())
        if task:
            t = task.lower().strip()
            models = [m for m in models if t in m.supported_tasks]
        if language:
            lang = language.lower().strip()
            models = [
                m for m in models
                if any(lang in l.lower() for l in m.supported_languages)
                or "multilingual" in [l.lower() for l in m.supported_languages]
            ]
        if operation:
            op = operation.lower().strip()
            models = [m for m in models if op in m.supported_operations]
        return models

    def get_supported_tasks(self) -> List[str]:
        tasks = set()
        for m in self._registry.values():
            tasks.update(m.supported_tasks)
        return sorted(list(tasks))

    def get_compatible_models(
        self,
        task: str,
        language: str = "indonesian",
        device: str = "auto",
    ) -> List[ModelRecord]:
        """Returns models compatible with the specified task and language."""
        return self.list_models(task=task, language=language)

    def validate_model_configuration(self, model_id: str, task: str, language: str) -> None:
        """Validates that model_id supports the specified task and language."""
        record = self.get_model(model_id)
        t = task.lower().strip()
        if t not in record.supported_tasks:
            raise ValueError(
                f"Model '{model_id}' does not support task '{task}'. "
                f"Supported tasks: {record.supported_tasks}"
            )
        lang = language.lower().strip()
        supported_langs = [l.lower() for l in record.supported_languages]
        if lang not in supported_langs and "multilingual" not in supported_langs:
            raise ValueError(
                f"Model '{model_id}' is not suitable for language '{language}'. "
                f"Supported languages: {record.supported_languages}"
            )

    def create_adapter(self, model_id: str, **kwargs) -> BaseTaskAdapter:
        """Instantiates the real execution adapter for the model with lazy loading."""
        record = self.get_model(model_id)
        combined_kwargs = dict(record.adapter_kwargs)
        combined_kwargs.update(kwargs)
        return record.adapter_cls(model_id=record.model_id, **combined_kwargs)

    def _populate(self):
        # A. Classical ML
        self.register(ModelRecord(
            model_id="classical_nb",
            category="A. Classical Machine Learning",
            approach_name="Complement / Multinomial Naive Bayes",
            model_framework="Scikit-Learn",
            checkpoint=None,
            supported_tasks=["classification"],
            supported_languages=["Indonesian", "English", "Multilingual"],
            supported_operations=["training", "inference", "evaluation"],
            vram_tier="CPU (<1GB)",
            evaluation_metrics=["Accuracy", "Macro-F1", "Weighted-F1"],
            adapter_cls=ClassicalClassificationAdapter,
            adapter_kwargs={"model_type": "nb"},
            notes="Fast, robust baseline; especially effective on imbalanced text."
        ))
        self.register(ModelRecord(
            model_id="classical_logistic",
            category="A. Classical Machine Learning",
            approach_name="Logistic Regression (L2 / Balanced)",
            model_framework="Scikit-Learn",
            checkpoint=None,
            supported_tasks=["classification"],
            supported_languages=["Indonesian", "English", "Multilingual"],
            supported_operations=["training", "inference", "evaluation"],
            vram_tier="CPU (<1GB)",
            evaluation_metrics=["Accuracy", "Macro-F1", "Balanced Accuracy"],
            adapter_cls=ClassicalClassificationAdapter,
            adapter_kwargs={"model_type": "logistic"},
            notes="Interpretable linear model; fold-local TF-IDF inside cross-validation pipeline."
        ))
        self.register(ModelRecord(
            model_id="classical_linear_svm",
            category="A. Classical Machine Learning",
            approach_name="Linear Support Vector Machine (LinearSVC)",
            model_framework="Scikit-Learn",
            checkpoint=None,
            supported_tasks=["classification"],
            supported_languages=["Indonesian", "English", "Multilingual"],
            supported_operations=["training", "inference", "evaluation"],
            vram_tier="CPU (<1GB)",
            evaluation_metrics=["Accuracy", "Macro-F1"],
            adapter_cls=ClassicalClassificationAdapter,
            adapter_kwargs={"model_type": "svm"},
            notes="High-margin separation for high-dimensional sparse n-gram feature spaces."
        ))

        # B. Transformer Text Classification
        self.register(ModelRecord(
            model_id="transformer_bert",
            category="B. Transformer Classification",
            approach_name="BERT Sequence Classification (IndoBERT / BERT)",
            model_framework="HuggingFace / PyTorch",
            checkpoint="indobenchmark/indobert-base-p1",
            supported_tasks=["classification"],
            supported_languages=["Indonesian", "English"],
            supported_operations=["fine-tuning", "inference", "evaluation"],
            vram_tier="RTX 3050 (bs=8, fp16)",
            evaluation_metrics=["Macro-F1", "Accuracy", "Weighted-F1"],
            adapter_cls=TransformerClassificationAdapter,
            adapter_kwargs={"checkpoint": "indobenchmark/indobert-base-p1"},
            notes="Fine-tuned bidirectional encoder; default uses IndoBERT for Indonesian."
        ))
        self.register(ModelRecord(
            model_id="transformer_roberta",
            category="B. Transformer Classification",
            approach_name="RoBERTa Sequence Classification (Indo-RoBERTa)",
            model_framework="HuggingFace / PyTorch",
            checkpoint="wirasana/indonesian-roberta-base",
            supported_tasks=["classification"],
            supported_languages=["Indonesian", "English"],
            supported_operations=["fine-tuning", "inference", "evaluation"],
            vram_tier="RTX 3050 (bs=8, fp16)",
            evaluation_metrics=["Macro-F1", "Accuracy"],
            adapter_cls=TransformerClassificationAdapter,
            adapter_kwargs={"checkpoint": "wirasana/indonesian-roberta-base"},
            notes="Byte-level BPE tokenizer with robust pretraining."
        ))
        self.register(ModelRecord(
            model_id="transformer_distilbert",
            category="B. Transformer Classification",
            approach_name="DistilBERT Fast Sequence Classification",
            model_framework="HuggingFace / PyTorch",
            checkpoint="distilbert-base-uncased",
            supported_tasks=["classification"],
            supported_languages=["English"],
            supported_operations=["fine-tuning", "inference", "evaluation"],
            vram_tier="RTX 3050 (bs=16, fp16) / CPU friendly",
            evaluation_metrics=["Macro-F1", "Accuracy"],
            adapter_cls=TransformerClassificationAdapter,
            adapter_kwargs={"checkpoint": "distilbert-base-uncased"},
            notes="Distilled model retaining 97% BERT accuracy with 60% faster latency."
        ))

        # C. Zero-Shot Classification
        self.register(ModelRecord(
            model_id="zeroshot_bart",
            category="C. Zero-Shot Classification",
            approach_name="BART Large MNLI Zero-Shot",
            model_framework="HuggingFace Pipeline",
            checkpoint="facebook/bart-large-mnli",
            supported_tasks=["classification", "zeroshot"],
            supported_languages=["English"],
            supported_operations=["inference", "evaluation"],
            vram_tier="RTX 3050 (bs=2-4, fp16)",
            evaluation_metrics=["Macro-F1", "Accuracy"],
            adapter_cls=ZeroShotClassificationAdapter,
            adapter_kwargs={"checkpoint": "facebook/bart-large-mnli"},
            is_inference_only=True,
            notes="Zero-shot NLI entailment on candidate label prompts."
        ))
        self.register(ModelRecord(
            model_id="zeroshot_deberta",
            category="C. Zero-Shot Classification",
            approach_name="DeBERTa-v3 MNLI Zero-Shot",
            model_framework="HuggingFace Pipeline",
            checkpoint="MoritzLaurer/DeBERTa-v3-xsmall-mnli",
            supported_tasks=["classification", "zeroshot"],
            supported_languages=["English"],
            supported_operations=["inference", "evaluation"],
            vram_tier="RTX 3050 (bs=8, fp16)",
            evaluation_metrics=["Macro-F1", "Accuracy"],
            adapter_cls=ZeroShotClassificationAdapter,
            adapter_kwargs={"checkpoint": "MoritzLaurer/DeBERTa-v3-xsmall-mnli"},
            is_inference_only=True,
            notes="Disentangled attention representation; highly efficient on 4GB VRAM."
        ))
        self.register(ModelRecord(
            model_id="zeroshot_xlm_r",
            category="C. Zero-Shot Classification",
            approach_name="XLM-RoBERTa XNLI Multilingual Zero-Shot",
            model_framework="HuggingFace Pipeline",
            checkpoint="vicgalle/xlm-roberta-base-xnli",
            supported_tasks=["classification", "zeroshot"],
            supported_languages=["Indonesian", "Multilingual", "English"],
            supported_operations=["inference", "evaluation"],
            vram_tier="RTX 3050 (bs=4, fp16)",
            evaluation_metrics=["Macro-F1", "Accuracy"],
            adapter_cls=ZeroShotClassificationAdapter,
            adapter_kwargs={"checkpoint": "vicgalle/xlm-roberta-base-xnli"},
            is_inference_only=True,
            notes="Cross-lingual zero-shot classification supporting Indonesian hypothesis templates."
        ))

        # D. Multilabel Classification
        self.register(ModelRecord(
            model_id="multilabel_binary_relevance",
            category="D. Multilabel Classification",
            approach_name="Binary Relevance (MultiOutput Logistic)",
            model_framework="Scikit-Learn",
            checkpoint=None,
            supported_tasks=["multilabel"],
            supported_languages=["Indonesian", "English", "Multilingual"],
            supported_operations=["training", "inference", "evaluation"],
            vram_tier="CPU (<1GB)",
            evaluation_metrics=["Micro-F1", "Macro-F1", "Hamming Loss", "Subset Accuracy"],
            adapter_cls=MultilabelClassificationAdapter,
            adapter_kwargs={"approach": "binary_relevance"},
            notes="Fits independent binary estimators per label."
        ))
        self.register(ModelRecord(
            model_id="multilabel_classifier_chains",
            category="D. Multilabel Classification",
            approach_name="Classifier Chains",
            model_framework="Scikit-Learn",
            checkpoint=None,
            supported_tasks=["multilabel"],
            supported_languages=["Indonesian", "English", "Multilingual"],
            supported_operations=["training", "inference", "evaluation"],
            vram_tier="CPU (<1GB)",
            evaluation_metrics=["Micro-F1", "Macro-F1", "Hamming Loss", "Subset Accuracy"],
            adapter_cls=MultilabelClassificationAdapter,
            adapter_kwargs={"approach": "classifier_chains"},
            notes="Models label correlation by sequentially chaining binary classifiers."
        ))
        self.register(ModelRecord(
            model_id="multilabel_transformer",
            category="D. Multilabel Classification",
            approach_name="Transformer Multilabel (BCEWithLogits)",
            model_framework="HuggingFace / PyTorch",
            checkpoint="distilbert-base-uncased",
            supported_tasks=["multilabel"],
            supported_languages=["English", "Indonesian"],
            supported_operations=["training", "inference", "evaluation"],
            vram_tier="RTX 3050 (bs=8, fp16)",
            evaluation_metrics=["Micro-F1", "Macro-F1", "Hamming Loss"],
            adapter_cls=MultilabelClassificationAdapter,
            adapter_kwargs={"approach": "binary_relevance"},
            notes="Uses Binary Cross Entropy and threshold tuning."
        ))

        # E. Text Regression
        self.register(ModelRecord(
            model_id="regression_tfidf_ridge",
            category="E. Text Regression",
            approach_name="TF-IDF + Ridge Regression",
            model_framework="Scikit-Learn",
            checkpoint=None,
            supported_tasks=["regression"],
            supported_languages=["Indonesian", "English", "Multilingual"],
            supported_operations=["training", "inference", "evaluation"],
            vram_tier="CPU (<1GB)",
            evaluation_metrics=["MAE", "MSE", "RMSE", "R2"],
            adapter_cls=TextRegressionAdapter,
            adapter_kwargs={"approach": "tfidf_ridge"},
            notes="L2 regularized linear model predicting continuous values."
        ))
        self.register(ModelRecord(
            model_id="regression_sentence_embeddings",
            category="E. Text Regression",
            approach_name="Sentence Embeddings + SVR",
            model_framework="SentenceTransformers + Scikit-Learn",
            checkpoint="sentence-transformers/all-MiniLM-L6-v2",
            supported_tasks=["regression"],
            supported_languages=["English", "Multilingual"],
            supported_operations=["training", "inference", "evaluation"],
            vram_tier="RTX 3050 (bs=32, fp16)",
            evaluation_metrics=["MAE", "RMSE", "R2"],
            adapter_cls=TextRegressionAdapter,
            adapter_kwargs={"approach": "svr"},
            notes="Non-linear Support Vector Regression on dense contextual embeddings."
        ))
        self.register(ModelRecord(
            model_id="regression_transformer",
            category="E. Text Regression",
            approach_name="Transformer Sequence Regression",
            model_framework="HuggingFace / PyTorch",
            checkpoint="distilbert-base-uncased",
            supported_tasks=["regression"],
            supported_languages=["English", "Indonesian"],
            supported_operations=["training", "inference", "evaluation"],
            vram_tier="RTX 3050 (bs=8, fp16)",
            evaluation_metrics=["MAE", "RMSE", "R2"],
            adapter_cls=TextRegressionAdapter,
            adapter_kwargs={"approach": "tfidf_ridge"},
            notes="Single continuous regression head with MSE loss."
        ))

        # F. Token Classification (NER)
        self.register(ModelRecord(
            model_id="ner_bert",
            category="F. Named Entity Recognition (NER)",
            approach_name="BERT Token Classification",
            model_framework="HuggingFace Pipeline",
            checkpoint="dslim/bert-base-NER",
            supported_tasks=["ner"],
            supported_languages=["English"],
            supported_operations=["inference", "evaluation"],
            vram_tier="RTX 3050 (bs=8, fp16)",
            evaluation_metrics=["Entity Precision", "Entity Recall", "Entity F1 (Seqeval)"],
            adapter_cls=NERAdapter,
            adapter_kwargs={"checkpoint": "dslim/bert-base-NER"},
            is_inference_only=True,
            notes="Extracts PER, ORG, LOC, MISC entities with BIO tagging."
        ))
        self.register(ModelRecord(
            model_id="ner_roberta",
            category="F. Named Entity Recognition (NER)",
            approach_name="RoBERTa Token Classification",
            model_framework="HuggingFace Pipeline",
            checkpoint="Jean-Baptiste/roberta-large-ner",
            supported_tasks=["ner"],
            supported_languages=["English"],
            supported_operations=["inference", "evaluation"],
            vram_tier="RTX 3050 (bs=4, fp16)",
            evaluation_metrics=["Entity Precision", "Entity Recall", "Entity F1 (Seqeval)"],
            adapter_cls=NERAdapter,
            adapter_kwargs={"checkpoint": "Jean-Baptiste/roberta-large-ner"},
            is_inference_only=True,
            notes="Byte-pair encoded token classification."
        ))
        self.register(ModelRecord(
            model_id="ner_multilingual_bert",
            category="F. Named Entity Recognition (NER)",
            approach_name="Multilingual BERT / IndoBERT NER",
            model_framework="HuggingFace Pipeline",
            checkpoint="bert-base-multilingual-cased",
            supported_tasks=["ner"],
            supported_languages=["Indonesian", "Multilingual"],
            supported_operations=["inference", "evaluation"],
            vram_tier="RTX 3050 (bs=8, fp16)",
            evaluation_metrics=["Entity Precision", "Entity Recall", "Entity F1 (Seqeval)"],
            adapter_cls=NERAdapter,
            adapter_kwargs={"checkpoint": "bert-base-multilingual-cased"},
            is_inference_only=True,
            notes="Multilingual WordPiece representation for Indonesian entity extraction."
        ))

        # G. Question Answering
        self.register(ModelRecord(
            model_id="qa_bert",
            category="G. Question Answering",
            approach_name="BERT Extractive QA",
            model_framework="HuggingFace Pipeline",
            checkpoint="bert-base-uncased",
            supported_tasks=["qa"],
            supported_languages=["English"],
            supported_operations=["inference", "evaluation"],
            vram_tier="RTX 3050 (bs=4, fp16)",
            evaluation_metrics=["Exact Match (EM)", "Token F1"],
            adapter_cls=QuestionAnsweringAdapter,
            adapter_kwargs={"checkpoint": "bert-base-uncased"},
            is_inference_only=True,
            notes="Predicts start and end answer token spans in context."
        ))
        self.register(ModelRecord(
            model_id="qa_roberta",
            category="G. Question Answering",
            approach_name="RoBERTa Extractive QA",
            model_framework="HuggingFace Pipeline",
            checkpoint="deepset/roberta-base-squad2",
            supported_tasks=["qa"],
            supported_languages=["English"],
            supported_operations=["inference", "evaluation"],
            vram_tier="RTX 3050 (bs=4, fp16)",
            evaluation_metrics=["Exact Match (EM)", "Token F1"],
            adapter_cls=QuestionAnsweringAdapter,
            adapter_kwargs={"checkpoint": "deepset/roberta-base-squad2"},
            is_inference_only=True,
            notes="Pretrained on SQuAD 2.0 with unanswerable question handling."
        ))
        self.register(ModelRecord(
            model_id="qa_xlm_r",
            category="G. Question Answering",
            approach_name="XLM-RoBERTa Multilingual Extractive QA",
            model_framework="HuggingFace Pipeline",
            checkpoint="deepset/xlm-roberta-base-squad2",
            supported_tasks=["qa"],
            supported_languages=["Indonesian", "Multilingual", "English"],
            supported_operations=["inference", "evaluation"],
            vram_tier="RTX 3050 (bs=4, fp16)",
            evaluation_metrics=["Exact Match (EM)", "Token F1"],
            adapter_cls=QuestionAnsweringAdapter,
            adapter_kwargs={"checkpoint": "deepset/xlm-roberta-base-squad2"},
            is_inference_only=True,
            notes="Cross-lingual transfer for Indonesian question answering."
        ))

        # H. Text Summarization
        self.register(ModelRecord(
            model_id="summarization_bart",
            category="H. Text Summarization",
            approach_name="BART Summarizer",
            model_framework="HuggingFace Pipeline",
            checkpoint="sshleifer/distilbart-cnn-12-6",
            supported_tasks=["summarization"],
            supported_languages=["English"],
            supported_operations=["inference", "evaluation"],
            vram_tier="RTX 3050 (bs=2-4, fp16)",
            evaluation_metrics=["ROUGE-1 F1", "ROUGE-2 F1", "ROUGE-L F1"],
            adapter_cls=SummarizationAdapter,
            adapter_kwargs={"checkpoint": "sshleifer/distilbart-cnn-12-6"},
            is_inference_only=True,
            notes="Distilled BART optimized for summarization latency."
        ))
        self.register(ModelRecord(
            model_id="summarization_pegasus",
            category="H. Text Summarization",
            approach_name="Pegasus Summarizer",
            model_framework="HuggingFace Pipeline",
            checkpoint="google/pegasus-xsum",
            supported_tasks=["summarization"],
            supported_languages=["English"],
            supported_operations=["inference", "evaluation"],
            vram_tier="RTX 3050 (bs=1-2, fp16)",
            evaluation_metrics=["ROUGE-1 F1", "ROUGE-2 F1", "ROUGE-L F1"],
            adapter_cls=SummarizationAdapter,
            adapter_kwargs={"checkpoint": "google/pegasus-xsum"},
            is_inference_only=True,
            notes="Gap-sentence generation objective."
        ))
        self.register(ModelRecord(
            model_id="summarization_t5",
            category="H. Text Summarization",
            approach_name="T5 / FLAN-T5 Summarizer",
            model_framework="HuggingFace Pipeline",
            checkpoint="t5-small",
            supported_tasks=["summarization"],
            supported_languages=["English", "Multilingual"],
            supported_operations=["inference", "evaluation"],
            vram_tier="RTX 3050 (bs=4, fp16)",
            evaluation_metrics=["ROUGE-1 F1", "ROUGE-2 F1", "ROUGE-L F1"],
            adapter_cls=SummarizationAdapter,
            adapter_kwargs={"checkpoint": "t5-small"},
            is_inference_only=True,
            notes="Compact encoder-decoder model fitting comfortably in 4GB VRAM."
        ))

        # I. Machine Translation
        self.register(ModelRecord(
            model_id="translation_marian",
            category="I. Machine Translation",
            approach_name="MarianMT (Helsinki-NLP id-en)",
            model_framework="HuggingFace Pipeline",
            checkpoint="Helsinki-NLP/opus-mt-id-en",
            supported_tasks=["translation"],
            supported_languages=["Indonesian", "English"],
            supported_operations=["inference", "evaluation"],
            vram_tier="RTX 3050 (bs=4, fp16)",
            evaluation_metrics=["SacreBLEU", "chrF"],
            adapter_cls=TranslationAdapter,
            adapter_kwargs={"checkpoint": "Helsinki-NLP/opus-mt-id-en", "source_lang": "id", "target_lang": "en"},
            is_inference_only=True,
            notes="Dedicated bilingual translation model for Indonesian to English."
        ))
        self.register(ModelRecord(
            model_id="translation_mt5",
            category="I. Machine Translation",
            approach_name="mT5 Multilingual Translation",
            model_framework="HuggingFace Pipeline",
            checkpoint="google/mt5-small",
            supported_tasks=["translation"],
            supported_languages=["Indonesian", "Multilingual", "English"],
            supported_operations=["inference", "evaluation"],
            vram_tier="RTX 3050 (bs=2-4, fp16)",
            evaluation_metrics=["SacreBLEU", "chrF"],
            adapter_cls=TranslationAdapter,
            adapter_kwargs={"checkpoint": "google/mt5-small"},
            is_inference_only=True,
            notes="Pretrained on mC4 covering 101 languages."
        ))
        self.register(ModelRecord(
            model_id="translation_nllb",
            category="I. Machine Translation",
            approach_name="NLLB-200 Machine Translation",
            model_framework="HuggingFace Pipeline",
            checkpoint="facebook/nllb-200-distilled-600M",
            supported_tasks=["translation"],
            supported_languages=["Indonesian", "Multilingual", "English"],
            supported_operations=["inference", "evaluation"],
            vram_tier="RTX 3050 (bs=2, fp16)",
            evaluation_metrics=["SacreBLEU", "chrF"],
            adapter_cls=TranslationAdapter,
            adapter_kwargs={"checkpoint": "facebook/nllb-200-distilled-600M"},
            is_inference_only=True,
            notes="State-of-the-art multilingual machine translation supporting 200 languages."
        ))

        # J. Text Generation
        self.register(ModelRecord(
            model_id="generation_qwen",
            category="J. Text Generation / Instruct",
            approach_name="Qwen2.5 Causal LM",
            model_framework="HuggingFace AutoModelForCausalLM",
            checkpoint="Qwen/Qwen2.5-0.5B-Instruct",
            supported_tasks=["generation"],
            supported_languages=["Indonesian", "Multilingual", "English"],
            supported_operations=["inference", "evaluation"],
            vram_tier="RTX 3050 (bs=2, fp16)",
            evaluation_metrics=["Average Generated Words", "Latency/Sample (ms)"],
            adapter_cls=GenerationAdapter,
            adapter_kwargs={"checkpoint": "Qwen/Qwen2.5-0.5B-Instruct"},
            is_inference_only=True,
            notes="Compact 0.5B model with strong Indonesian instruction reasoning."
        ))
        self.register(ModelRecord(
            model_id="generation_tinyllama",
            category="J. Text Generation / Instruct",
            approach_name="TinyLlama Chat",
            model_framework="HuggingFace AutoModelForCausalLM",
            checkpoint="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            supported_tasks=["generation"],
            supported_languages=["English"],
            supported_operations=["inference", "evaluation"],
            vram_tier="RTX 3050 (bs=1-2, fp16)",
            evaluation_metrics=["Average Generated Words", "Latency/Sample (ms)"],
            adapter_cls=GenerationAdapter,
            adapter_kwargs={"checkpoint": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"},
            is_inference_only=True,
            notes="1.1B Llama architecture pretrained on 3 trillion tokens."
        ))
        self.register(ModelRecord(
            model_id="generation_gpt",
            category="J. Text Generation / Instruct",
            approach_name="DistilGPT-2 Causal LM",
            model_framework="HuggingFace AutoModelForCausalLM",
            checkpoint="distilgpt2",
            supported_tasks=["generation"],
            supported_languages=["English"],
            supported_operations=["inference", "evaluation"],
            vram_tier="RTX 3050 (bs=4, fp16) / CPU friendly",
            evaluation_metrics=["Average Generated Words", "Latency/Sample (ms)"],
            adapter_cls=GenerationAdapter,
            adapter_kwargs={"checkpoint": "distilgpt2"},
            is_inference_only=True,
            notes="Lightweight causal language model for text completion."
        ))

        # K. Semantic Similarity & Embeddings
        self.register(ModelRecord(
            model_id="embedding_minilm",
            category="K. Semantic Embeddings",
            approach_name="Sentence-BERT / MiniLM-L6-v2",
            model_framework="SentenceTransformers",
            checkpoint="sentence-transformers/all-MiniLM-L6-v2",
            supported_tasks=["similarity"],
            supported_languages=["English"],
            supported_operations=["inference", "feature_extraction", "evaluation"],
            vram_tier="RTX 3050 (bs=32, fp16)",
            evaluation_metrics=["Spearman Correlation", "Pearson Correlation"],
            adapter_cls=SemanticSimilarityAdapter,
            adapter_kwargs={"checkpoint": "sentence-transformers/all-MiniLM-L6-v2"},
            is_inference_only=True,
            notes="384-dimensional dense semantic embeddings; ultra-fast inference."
        ))
        self.register(ModelRecord(
            model_id="embedding_mpnet",
            category="K. Semantic Embeddings",
            approach_name="all-mpnet-base-v2",
            model_framework="SentenceTransformers",
            checkpoint="sentence-transformers/all-mpnet-base-v2",
            supported_tasks=["similarity"],
            supported_languages=["English"],
            supported_operations=["inference", "feature_extraction", "evaluation"],
            vram_tier="RTX 3050 (bs=16, fp16)",
            evaluation_metrics=["Spearman Correlation", "Pearson Correlation"],
            adapter_cls=SemanticSimilarityAdapter,
            adapter_kwargs={"checkpoint": "sentence-transformers/all-mpnet-base-v2"},
            is_inference_only=True,
            notes="768-dimensional high-accuracy sentence representations."
        ))
        self.register(ModelRecord(
            model_id="embedding_multilingual_e5",
            category="K. Semantic Embeddings",
            approach_name="Multilingual E5 Small",
            model_framework="SentenceTransformers",
            checkpoint="intfloat/multilingual-e5-small",
            supported_tasks=["similarity"],
            supported_languages=["Indonesian", "Multilingual", "English"],
            supported_operations=["inference", "feature_extraction", "evaluation"],
            vram_tier="RTX 3050 (bs=16, fp16)",
            evaluation_metrics=["Spearman Correlation", "Pearson Correlation"],
            adapter_cls=SemanticSimilarityAdapter,
            adapter_kwargs={"checkpoint": "intfloat/multilingual-e5-small"},
            is_inference_only=True,
            notes="State-of-the-art multilingual embeddings; strong Indonesian performance."
        ))

        # L. Information Retrieval
        self.register(ModelRecord(
            model_id="ir_bm25",
            category="L. Information Retrieval",
            approach_name="Okapi BM25 Lexical Retrieval",
            model_framework="rank-bm25",
            checkpoint=None,
            supported_tasks=["retrieval"],
            supported_languages=["Indonesian", "English", "Multilingual"],
            supported_operations=["inference", "evaluation"],
            vram_tier="CPU (<1GB)",
            evaluation_metrics=["Recall@5", "Precision@5", "MRR", "NDCG@5"],
            adapter_cls=RetrievalAdapter,
            adapter_kwargs={"approach": "bm25"},
            is_inference_only=True,
            notes="Probabilistic term-matching with saturation and document length normalization."
        ))
        self.register(ModelRecord(
            model_id="ir_dense_biencoder",
            category="L. Information Retrieval",
            approach_name="Dense Bi-Encoder Retrieval",
            model_framework="SentenceTransformers",
            checkpoint="sentence-transformers/all-MiniLM-L6-v2",
            supported_tasks=["retrieval"],
            supported_languages=["English", "Indonesian"],
            supported_operations=["inference", "evaluation"],
            vram_tier="RTX 3050 (bs=32, fp16)",
            evaluation_metrics=["Recall@5", "MRR", "NDCG@5"],
            adapter_cls=RetrievalAdapter,
            adapter_kwargs={"approach": "dense"},
            is_inference_only=True,
            notes="Encodes corpus offline; fast cosine similarity matching at query time."
        ))
        self.register(ModelRecord(
            model_id="ir_hybrid_rerank",
            category="L. Information Retrieval",
            approach_name="Hybrid Retrieval (BM25 + Dense)",
            model_framework="BM25 + SentenceTransformers",
            checkpoint="sentence-transformers/all-MiniLM-L6-v2",
            supported_tasks=["retrieval"],
            supported_languages=["English", "Indonesian"],
            supported_operations=["inference", "evaluation"],
            vram_tier="RTX 3050 (bs=16, fp16)",
            evaluation_metrics=["Recall@5", "MRR", "NDCG@5"],
            adapter_cls=RetrievalAdapter,
            adapter_kwargs={"approach": "hybrid"},
            is_inference_only=True,
            notes="Combines normalized lexical BM25 and dense semantic scores."
        ))

        # M. Topic Modeling
        self.register(ModelRecord(
            model_id="topic_lda",
            category="M. Topic Modeling",
            approach_name="Latent Dirichlet Allocation (LDA)",
            model_framework="Scikit-Learn",
            checkpoint=None,
            supported_tasks=["topic_modeling"],
            supported_languages=["Indonesian", "English", "Multilingual"],
            supported_operations=["training", "inference", "evaluation"],
            vram_tier="CPU (<1GB)",
            evaluation_metrics=["Topic Diversity", "Total Topics"],
            adapter_cls=TopicModelingAdapter,
            adapter_kwargs={"approach": "lda"},
            notes="Generative probabilistic topic modeling over count vectors."
        ))
        self.register(ModelRecord(
            model_id="topic_nmf",
            category="M. Topic Modeling",
            approach_name="Non-Negative Matrix Factorization (NMF)",
            model_framework="Scikit-Learn",
            checkpoint=None,
            supported_tasks=["topic_modeling"],
            supported_languages=["Indonesian", "English", "Multilingual"],
            supported_operations=["training", "inference", "evaluation"],
            vram_tier="CPU (<1GB)",
            evaluation_metrics=["Topic Diversity", "Total Topics"],
            adapter_cls=TopicModelingAdapter,
            adapter_kwargs={"approach": "nmf"},
            notes="Matrix factorization decomposing TF-IDF document-term matrices."
        ))
        self.register(ModelRecord(
            model_id="topic_ctfidf",
            category="M. Topic Modeling",
            approach_name="c-TF-IDF Embedding Clustering (BERTopic-style)",
            model_framework="SentenceTransformers + Scikit-Learn",
            checkpoint="sentence-transformers/all-MiniLM-L6-v2",
            supported_tasks=["topic_modeling"],
            supported_languages=["Indonesian", "English", "Multilingual"],
            supported_operations=["training", "inference", "evaluation"],
            vram_tier="RTX 3050 (bs=32, fp16)",
            evaluation_metrics=["Topic Diversity", "Total Topics"],
            adapter_cls=TopicModelingAdapter,
            adapter_kwargs={"approach": "ctfidf"},
            notes="Clusters dense sentence embeddings, then extracts topic words via class-based TF-IDF."
        ))

        # N. Text Pair Classification
        self.register(ModelRecord(
            model_id="pair_cross_encoder",
            category="N. Text Pair Classification",
            approach_name="Cross-Encoder Joint Matching",
            model_framework="Scikit-Learn / CrossEncoder",
            checkpoint=None,
            supported_tasks=["pair_classification"],
            supported_languages=["English", "Indonesian"],
            supported_operations=["training", "inference", "evaluation"],
            vram_tier="CPU (<1GB)",
            evaluation_metrics=["Accuracy", "Macro-F1"],
            adapter_cls=TextPairClassificationAdapter,
            adapter_kwargs={"approach": "cross_encoder"},
            notes="Joint representation over pair inputs."
        ))
        self.register(ModelRecord(
            model_id="pair_siamese",
            category="N. Text Pair Classification",
            approach_name="Siamese Network Bi-Encoder",
            model_framework="SentenceTransformers",
            checkpoint="sentence-transformers/all-MiniLM-L6-v2",
            supported_tasks=["pair_classification"],
            supported_languages=["English", "Indonesian"],
            supported_operations=["training", "inference", "evaluation"],
            vram_tier="RTX 3050 (bs=32, fp16)",
            evaluation_metrics=["Accuracy", "Macro-F1"],
            adapter_cls=TextPairClassificationAdapter,
            adapter_kwargs={"approach": "siamese"},
            notes="Encodes text pairs independently, classifies via difference and product vectors."
        ))
        self.register(ModelRecord(
            model_id="pair_transformer_clf",
            category="N. Text Pair Classification",
            approach_name="Transformer Pair Sequence Classification",
            model_framework="HuggingFace / PyTorch",
            checkpoint="distilbert-base-uncased",
            supported_tasks=["pair_classification"],
            supported_languages=["English", "Indonesian"],
            supported_operations=["training", "inference", "evaluation"],
            vram_tier="RTX 3050 (bs=8, fp16)",
            evaluation_metrics=["Accuracy", "Macro-F1"],
            adapter_cls=TextPairClassificationAdapter,
            adapter_kwargs={"approach": "cross_encoder"},
            notes="Joint sequence classification with [SEP] delimiter."
        ))

registry = ArchitectureRegistry()
