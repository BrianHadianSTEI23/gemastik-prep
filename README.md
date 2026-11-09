# Configurable NLP Experimentation Workbench (RTX 3050 & CPU Ready)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x%20(CUDA%20%2B%20CPU)-EE4C2C.svg)](https://pytorch.org/)
[![Hardware](https://img.shields.io/badge/Hardware-NVIDIA%20RTX%203050%20(4GB%2F6GB%2F8GB)-76B900.svg)](https://www.nvidia.com/)
[![Focus](https://img.shields.io/badge/Focus-Indonesian%20%26%20Multilingual%20NLP-orange.svg)]()
[![Competition](https://img.shields.io/badge/Competition-Gemastik%20Data%20Mining%20Ready-green.svg)]()
[![Tests](https://img.shields.io/badge/pytest-51%20passed%20(100%25)-brightgreen.svg)]()

A modular, hardware-aware, and explainable **Natural Language Processing (NLP) Experimentation Workbench** in Jupyter Notebook format (`nlp_workbench.ipynb`) backed by a clean Python package (`nlp_workbench`). Engineered specifically for resource-conscious machine learning practitioners, researchers, and data science competitors working on **NVIDIA GeForce RTX 3050 GPUs (4GB, 6GB, or 8GB VRAM)** as well as **CPU-only environments**.

---

## 1. System Architecture & Workflow Pipeline

The workbench decouples reusable engineering logic into the `nlp_workbench` package while providing a user-friendly, sequential Jupyter Notebook interface:

```
[01. Configuration (WorkbenchConfig)]
       │
       ▼
[02. Hardware Diagnostics (HardwareManager: RTX 3050 / CPU Detection & VRAM Profiling)]
       │
       ▼
[03. Data Ingestion (DataLoader: CSV / TSV / Parquet / JSONL / Synthetic Benchmarks)]
       │
       ▼
[04. Dataset Profiling & Data Quality Audit (Missing, Cardinality, Text Length, IDs)]
       │
       ▼
[05. Actionable EDA & 5-Point Explainable Decision Support (DataQualityAuditor)]
       │
       ▼
[06. Task Identification & Target Schema Validation (validate_dataset_schema)]
       │
       ▼
[07. Non-Destructive Cleaning (TextCleaner: Negation Preservation, HTML, URLs)]
       │
       ▼
[08. Leakage-Safe Splitting (DatasetSplitter: Stratified / Group / Time / K-Fold + LeakageAuditor)]
       │
       ▼
[09. Preprocessing Recommendation System (recommend_preprocessing)]
       │
       ▼
[10. Architecture Registry (ArchitectureRegistry: 14 Categories × 3 Usable Approaches = 42 Models)]
       │
       ├──► [11. Classical Machine Learning Experiments (Fold-Local Preprocessing)]
       ├──► [12. Transformer Fine-Tuning (FP16 Mixed Precision & Dynamic Padding)]
       └──► [13. Specialized NLP Tasks (Zero-Shot, Multilabel, NER, QA, Summary, Translation, IR, Topics)]
       │
       ▼
[14. Cross-Validation & Model Selection (select_best_model strictly on CV scores)]
       │
       ▼
[15. Unified Leaderboard Comparison & Held-Out Test Evaluation (Evaluated Strictly Once)]
       │
       ▼
[16. Error Analysis & Diagnostics (Confusion Matrix, Misclassified Analysis)]
       │
       ▼
[17. Model Packaging & Export (ModelExporter: Self-Contained Artifact Bundle)]
       │
       ▼
[18. Production Inference (`predict_new()` API on Raw Texts)]
       │
       ▼
[19. Gemastik Data Mining Competition Mode (GemastikRunner: Test Ingestion & Strict submission.csv Export)]
       │
       ▼
[20. Experiment Summary & Reproducibility Audit (experiment_audit.json with Hashes & Environment)]
```

---

## 2. Core Highlights

1. **Hardware-Intelligent for RTX 3050 (4GB VRAM) & CPU-Only Systems**:
   - Detects exact available VRAM and compute capability (Ampere SM 8.6).
   - Recommends memory-safe batch sizes (e.g., batch size 8 with 4 gradient accumulation steps for base transformers in 4GB VRAM).
   - Dynamic padding to avoid wasting memory on padding tokens.
   - Mixed precision training (`torch.amp.autocast('cuda')`).
   - OOM recovery utilities (`HardwareManager.clear_cuda_cache()`) and seamless CPU fallback.

2. **No Silent Destructive Preprocessing**:
   - Explicit 5-point explainable decision support: **Issue $\rightarrow$ Evidence $\rightarrow$ Why It Matters $\rightarrow$ Recommendation $\rightarrow$ Drawbacks & Controlling Config Variable**.
   - Preserves negations (*"tidak", "bukan", "jangan", "tanpa", "belum"*) for sentiment classification.
   - Preserves casing and punctuation for transformer subword tokenizers.
   - Prevents stopword removal or stemming when fine-tuning pretrained language models.

3. **Strict Experimental Methodology & Zero Leakage**:
   - Investigates duplicate texts across train, validation, and test partitions (`LeakageAuditor`).
   - Checks ID overlap and target alignment.
   - Enforces the hard rule: **All encoders, TF-IDF vectorizers, and scalers are fitted exclusively within each cross-validation fold**.
   - Model selection is performed strictly using validation / cross-validation scores. The held-out test partition is evaluated **exactly once**.

4. **Indonesian & Multilingual NLP**:
   - First-class support for Indonesian datasets (IndoBERT, Indo-RoBERTa, Multilingual E5, XLM-RoBERTa, MarianMT id-en).
   - Indonesian-aware text normalization and negation preservation.

5. **Gemastik Data Mining Competition Mode**:
   - Dedicated runner (`GemastikRunner`).
   - Handles unlabelled competition test files with row count and ID ordering verification.
   - Out-of-fold cross-validation ensembling.
   - Export to `submission.csv` with strict verification: no row drops, no nulls, and guaranteed ID alignment.

---

## 3. Architecture Registry: 14 Categories × 3 Usable Approaches (42 Models)

Every registered architecture is connected to a genuine callable adapter with lazy loading:

| Category | Approach 1 | Approach 2 | Approach 3 | Primary Metrics |
| :--- | :--- | :--- | :--- | :--- |
| **A. Classical ML** | `classical_nb` (Complement / Multinomial Naive Bayes) | `classical_logistic` (Logistic Regression L2 / Balanced) | `classical_linear_svm` (Linear Support Vector Machine) | Accuracy, Balanced Acc, Macro-F1 |
| **B. Transformer Classification** | `transformer_bert` (BERT / IndoBERT) | `transformer_roberta` (RoBERTa / Indo-RoBERTa) | `transformer_distilbert` (DistilBERT Sequence Classifier) | Macro-F1, Weighted-F1, Accuracy |
| **C. Zero-Shot Classification** | `zeroshot_bart` (BART-MNLI) | `zeroshot_deberta` (DeBERTa-v3 MNLI) | `zeroshot_xlm_r` (XLM-RoBERTa XNLI Multilingual) | Macro-F1, Accuracy |
| **D. Multilabel Classification** | `multilabel_binary_relevance` (Binary Relevance) | `multilabel_classifier_chains` (Classifier Chains) | `multilabel_transformer` (Transformer BCEWithLogits + Threshold Tuning) | Micro-F1, Macro-F1, Hamming Loss |
| **E. Text Regression** | `regression_tfidf_ridge` (TF-IDF + Ridge) | `regression_sentence_embeddings` (MiniLM + SVR / Ridge) | `regression_transformer` (Transformer Sequence Regression) | MAE, MSE, RMSE, $R^2$ |
| **F. Token Classification (NER)** | `ner_bert` (BERT Token Classification / Fine-tuning) | `ner_roberta` (RoBERTa Token Classification / Fine-tuning) | `ner_multilingual_bert` (Multilingual BERT NER / Fine-tuning) | Entity-level Precision, Recall, F1 (`seqeval`) |
| **G. Question Answering** | `qa_bert` (BERT Extractive SQuAD2) | `qa_roberta` (RoBERTa Extractive SQuAD2) | `qa_xlm_r` (XLM-RoBERTa Multilingual QA) | Exact Match (EM), Token-level F1 (Multiset) |
| **H. Text Summarization** | `summarization_bart` (DistilBART) | `summarization_pegasus` (Pegasus XSum) | `summarization_t5` (FLAN-T5 Small) | ROUGE-1, ROUGE-2, ROUGE-L |
| **I. Machine Translation** | `translation_marian` (MarianMT id-en) | `translation_mt5` (mT5 Multilingual Prompt) | `translation_nllb` (NLLB-200 Distilled FLORES) | SacreBLEU, chrF |
| **J. Text Generation / Instruct** | `generation_qwen` (Qwen2.5 0.5B Instruct) | `generation_tinyllama` (TinyLlama 1.1B Chat) | `generation_gpt` (DistilGPT-2 Causal LM) | Perplexity, Output Completeness |
| **K. Semantic Embeddings** | `embedding_minilm` (all-MiniLM-L6-v2) | `embedding_mpnet` (all-mpnet-base-v2) | `embedding_multilingual_e5` (multilingual-e5-small) | Spearman $\rho$, Pearson Correlation |
| **L. Information Retrieval** | `ir_bm25` (Okapi BM25) | `ir_dense_biencoder` (Dense Bi-Encoder Dot Product) | `ir_hybrid_rerank` (Hybrid Retrieval + Cross-Encoder Reranking) | Recall@K, Precision@K, MRR, NDCG@K |
| **M. Topic Modeling** | `topic_lda` (Latent Dirichlet Allocation) | `topic_nmf` (Non-Negative Matrix Factorization) | `topic_ctfidf` (Multilingual Embedding Clustering + c-TF-IDF) | Topic Diversity, Topic Keywords |
| **N. Text Pair Classification** | `pair_cross_encoder` (Cross-Encoder Joint Matching) | `pair_siamese` (Siamese Bi-Encoder Feature Interaction) | `pair_transformer_clf` (Joint Sequence `[SEP]` Classifier) | Accuracy, Macro-F1, ROC-AUC |

---

## 4. Repository Structure

```
gemastik-prep/
├── src/
│   └── nlp_workbench/
│       ├── __init__.py           # Package entry point
│       ├── config.py             # WorkbenchConfig with validation & defaults
│       ├── hardware.py           # HardwareManager (RTX 3050 VRAM profiling)
│       ├── data.py               # DataLoader & task synthetic generators
│       ├── eda.py                # DataQualityAuditor, ExploratoryDataAnalysis, 5-point decision support
│       ├── preprocessing.py      # TextCleaner (negations preserved) & recommender
│       ├── splitting.py          # DatasetSplitter & LeakageAuditor
│       ├── evaluation.py         # Direction-aware standard metrics for all 14 task categories
│       ├── registry.py           # ArchitectureRegistry (42 real model records)
│       ├── training.py           # run_cross_validation & select_best_model
│       ├── export.py             # ModelExporter & LoadedModelBundle (file & directory support)
│       ├── inference.py          # predict_new production API
│       ├── competition.py        # GemastikRunner for competition submissions
│       ├── orchestration.py      # Central ExperimentRunner & ExperimentResult
│       └── tasks/                # Real callable task adapters
│           ├── base.py
│           ├── classification.py
│           ├── regression.py
│           ├── multilabel.py
│           ├── ner.py
│           ├── qa.py
│           ├── summarization.py
│           ├── translation.py
│           ├── generation.py
│           ├── similarity.py
│           ├── retrieval.py
│           ├── topic_modeling.py
│           ├── pair_classification.py
│           └── zeroshot.py
├── tests/                        # Automated Pytest suite (51 tests)
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_data.py
│   ├── test_splitting.py
│   ├── test_registry.py
│   ├── test_metrics.py
│   ├── test_tasks.py
│   ├── test_inference_export.py
│   ├── test_competition.py
│   ├── test_training_eda.py
│   └── test_adv_metrics_ner_runner.py
├── .github/
│   └── workflows/
│       └── tests.yml             # GitHub Actions CI workflow
├── nlp_workbench.ipynb           # User-facing experimentation notebook
├── pyproject.toml                # Package configuration & pytest markers
├── requirements.txt              # Grouped project dependencies
├── constraints.txt               # Pinned exact verified dependency versions
└── README.md
```

---

## 5. Setup and Quickstart

### Option A: Using the Configured Workstation Environment
```bash
# Path to environment:
/home/hadynata/vs-code-data/hdl/project-haphaestus/.venv/bin/python

# Register kernel for Jupyter / VS Code:
/home/hadynata/vs-code-data/hdl/project-haphaestus/.venv/bin/python -m ipykernel install --user --name=nlp-workbench --display-name="Python (NLP Workbench RTX 3050)"
```

### Option B: Local or Cloud Installation
```bash
git clone https://github.com/BrianHadianSTEI23/gemastik-prep.git
cd gemastik-prep
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
```

### Running Tests
Execute the automated test suite:
```bash
pytest tests/ -v
```

---

## 6. Configuration Quickstart (Notebook Section 01)

To run an experiment, open `nlp_workbench.ipynb` and configure the control center:

```python
from nlp_workbench import get_default_config

config = get_default_config(task="classification", language="indonesian")
config.dataset_path = "path/to/dataset.csv"  # or "synthetic"
config.text_column = "text"
config.target_column = "target"
config.device = "auto"
config.gemastik_mode = True
```

Execute cells sequentially from top to bottom.

---

## 7. License & Credits

Built for Gemastik Data Mining preparation and academic NLP research. All code is distributed under the MIT License.
