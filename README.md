# Configurable NLP Experimentation Workbench (RTX 3050 & CPU Ready)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x%20(CUDA%20%2B%20CPU)-EE4C2C.svg)](https://pytorch.org/)
[![Hardware](https://img.shields.io/badge/Hardware-NVIDIA%20RTX%203050%20(4GB%2F6GB%2F8GB)-76B900.svg)](https://www.nvidia.com/)
[![Focus](https://img.shields.io/badge/Focus-Indonesian%20%26%20Multilingual%20NLP-orange.svg)]()
[![Competition](https://img.shields.io/badge/Competition-Gemastik%20Data%20Mining%20Ready-green.svg)]()

A modular, end-to-end, and explainable **Natural Language Processing (NLP) Experimentation Workbench** in Jupyter Notebook format (`nlp_workbench.ipynb`). Engineered specifically for resource-conscious machine learning practitioners, researchers, and data science competitors working on **NVIDIA RTX 3050 GPUs (4GB, 6GB, or 8GB VRAM)** as well as **CPU-only environments**.

---

## 1. System Architecture & Workflow Pipeline

The workbench follows a strict, non-destructive, and leak-free machine learning lifecycle:

```
[01. Configuration]
       │
       ▼
[02. Hardware Diagnostics (RTX 3050 / CPU Detection & VRAM Profiling)]
       │
       ▼
[03. Data Ingestion (CSV / Parquet / JSONL / Synthetic Fallback)]
       │
       ▼
[04. Dataset Profiling & Data Quality Audit (Missing, Cardinality, Text Length, IDs)]
       │
       ▼
[05. Actionable EDA & Decision Support Board (Problems → Suggestions → Accepted Actions)]
       │
       ▼
[06. Task Identification & Target Schema Validation]
       │
       ▼
[07. Non-Destructive Cleaning (HTML, URLs, Whitespace, Unicode Normalization)]
       │
       ▼
[08. Leakage-Safe Splitting (Stratified / Group / Time / K-Fold + Leakage Audit)]
       │
       ▼
[09. Preprocessing Recommendation System (Rule-based TF-IDF vs Transformer tokenizers)]
       │
       ▼
[10. Architecture Registry (14 Categories × 3 Usable Approaches = 42 Models)]
       │
       ├──► [11. Classical Machine Learning Experiments (NB, Logistic, SVM, Ridge)]
       ├──► [12. Transformer Fine-Tuning (BERT, RoBERTa, DistilBERT with FP16 & GradAccum)]
       └──► [13. Specialized NLP Tasks (Zero-Shot, Multilabel, NER, QA, Summary, Translation, IR, Topics)]
       │
       ▼
[14. Hyperparameter Optimization (Optuna Study)]
       │
       ▼
[15. Unified Model Comparison Leaderboard (Val/Test Metric, Latency, VRAM Footprint)]
       │
       ▼
[16. Error Analysis & Interpretability (Confusion Matrix, Residuals, Top Misclassifications)]
       │
       ▼
[17. Final Model Selection & Best Checkpoint Pick]
       │
       ▼
[18. Model Export & Packaging (.joblib pipeline & HuggingFace weights)]
       │
       ▼
[19. Production Inference on New Raw Text (`predict_new()` API)]
       │
       ▼
[20. Gemastik Data Mining Competition Mode (OOF Ensembling & Strict Submission CSV Export)]
       │
       ▼
[21. Experiment Summary & Reproducibility Audit (Seeds, Hashes, Environment Snapshot)]
```

---

## 2. Core Highlights

1. **Hardware-Intelligent for RTX 3050 (4GB VRAM) & CPU-Only Systems**:
   - Detects exact available VRAM and compute capability (e.g., Ampere SM 8.6).
   - Recommends memory-safe batch sizes (e.g., batch size 8 with 4 gradient accumulation steps for base transformers in 4GB VRAM).
   - Dynamic padding to avoid wasting memory on padding tokens.
   - Mixed precision training (`torch.cuda.amp.autocast(dtype=torch.float16)`).
   - OOM recovery utilities (`HardwareManager.clear_cuda_cache()`) and seamless CPU fallback.

2. **No Silent Destructive Preprocessing**:
   - Explicit 3-tier decision board: **Detected Findings** $\rightarrow$ **Suggested Recommendations** $\rightarrow$ **User-Accepted Actions**.
   - Preserves negations (*"tidak", "bukan", "jangan", "tanpa", "belum"*) for sentiment classification.
   - Preserves casing and punctuation for transformer subword tokenizers.
   - Prevents stopword removal or stemming when fine-tuning pretrained language models.

3. **Strict Data Leakage Prevention**:
   - Investigates duplicate texts across train, validation, and test partitions.
   - Checks ID overlap and target alignment.
   - Enforces the hard rule: **All encoders, TF-IDF vectorizers, and scalers are fitted exclusively on the training split**.

4. **Indonesian & Multilingual NLP**:
   - First-class support for Indonesian datasets (e.g., IndoBERT, Indo-RoBERTa, Multilingual E5, XLM-RoBERTa).
   - Indonesian-aware text normalization and negation preservation.

5. **Gemastik Data Mining Competition Mode**:
   - Dedicated workflow flag (`GEMASTIK_MODE = True`).
   - Handles unlabelled competition test files (`COMPETITION_TEST_PATH`).
   - Out-of-fold cross-validation ensembling.
   - Export to `submission.csv` with strict verification of row count, ordering, and ID alignment.

---

## 3. Architecture Registry: 14 Categories × 3 Approaches (42 Models)

| Category | Approach 1 | Approach 2 | Approach 3 | Primary Metrics |
| :--- | :--- | :--- | :--- | :--- |
| **A. Classical ML** | Multinomial / Complement Naive Bayes | Logistic Regression (L2 / Balanced) | Linear SVM (`LinearSVC`) | Accuracy, Macro-F1, ROC-AUC |
| **B. Transformer Classification** | BERT (`bert-base-uncased` / `indobert-base-p1`) | RoBERTa (`roberta-base` / `indonesian-roberta-base`) | DistilBERT (`distilbert-base-uncased`) | Macro-F1, Weighted-F1, Accuracy |
| **C. Zero-Shot Classification** | BART-MNLI (`facebook/bart-large-mnli`) | DeBERTa-v3 (`MoritzLaurer/DeBERTa-v3-xsmall-mnli`) | XLM-RoBERTa XNLI (`vicgalle/xlm-roberta-base-xnli`) | Zero-shot Macro-F1, Accuracy |
| **D. Multilabel Classification** | Binary Relevance (`MultiOutputClassifier`) | Classifier Chains (`ClassifierChain`) | Transformer Multilabel (`BCEWithLogitsLoss`) | Micro-F1, Macro-F1, Hamming Loss |
| **E. Text Regression** | TF-IDF + Ridge Regression | Sentence Embeddings + SVR / Ridge | Transformer Sequence Regression (`MSELoss`) | MAE, MSE, RMSE, $R^2$ |
| **F. Token Classification (NER)** | BERT Token Classification (`AutoModelForTokenClassification`) | RoBERTa Token Classification | Multilingual BERT / IndoBERT NER | Entity-level Precision, Recall, F1 (`seqeval`) |
| **G. Question Answering** | BERT Extractive QA (`AutoModelForQuestionAnswering`) | RoBERTa Extractive QA (`roberta-base-squad2`) | XLM-R Multilingual QA (`xlm-roberta-base-squad2`) | Exact Match (EM), Token-level F1 |
| **H. Text Summarization** | BART (`sshleifer/distilbart-cnn-12-6`) | Pegasus (`google/pegasus-xsum`) | T5 / FLAN-T5 (`t5-small` / `flan-t5-small`) | ROUGE-1, ROUGE-2, ROUGE-L |
| **I. Machine Translation** | MarianMT (`Helsinki-NLP/opus-mt-id-en`) | mT5 (`google/mt5-small`) | NLLB (`facebook/nllb-200-distilled-600M`) | SacreBLEU, chrF |
| **J. Text Generation / Instruct** | Qwen2.5 (`Qwen/Qwen2.5-0.5B-Instruct`) | TinyLlama (`TinyLlama/TinyLlama-1.1B-Chat-v1.0`) | GPT-style Causal LM (`distilgpt2` / `gpt2`) | Perplexity, Repetition Penalty |
| **K. Semantic Embeddings** | Sentence-BERT (`all-MiniLM-L6-v2`) | MPNet (`all-mpnet-base-v2`) | Multilingual E5 (`multilingual-e5-small`) | Cosine Similarity, Spearman $\rho$ |
| **L. Information Retrieval** | Okapi BM25 (`rank-bm25`) | Dense Bi-Encoder Retrieval (Dot Product) | Hybrid Retrieval + Cross-Encoder Reranker | Recall@K, Precision@K, MRR, NDCG |
| **M. Topic Modeling** | Latent Dirichlet Allocation (LDA) | Non-Negative Matrix Factorization (NMF) | BERTopic-Style Embedding Clustering + c-TF-IDF | Topic Coherence, Topic Diversity |
| **N. Text Pair Classification** | Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) | Siamese Network Bi-Encoder with Distance Metric | Transformer Pair Classification (`[SEP]` delimiter) | Accuracy, Macro-F1, ROC-AUC |

---

## 4. Setup and Installation

### Option A: Using the Configured Virtual Environment (Recommended on this workstation)
This machine is configured with a dedicated Python environment containing PyTorch (with CUDA 12.6 support for the RTX 3050):

```bash
# Path to environment:
/home/hadynata/vs-code-data/hdl/project-haphaestus/.venv/bin/python
```

To register the kernel for Jupyter Notebook / VS Code:
```bash
/home/hadynata/vs-code-data/hdl/project-haphaestus/.venv/bin/python -m ipykernel install --user --name=nlp-workbench --display-name="Python (NLP Workbench RTX 3050)"
```

### Option B: Fresh Local / Colab / Kaggle Installation
```bash
git clone https://github.com/hadynata/gemastik-prep.git
cd gemastik-prep
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 5. Configuration Quickstart (Section 01 of Notebook)

To start an experiment, simply open `nlp_workbench.ipynb` and configure the top variables in **Section 01**:

```python
# ==========================================
# 1. CORE DATASET & TASK CONFIGURATION
# ==========================================
DATA_PATH = "your_dataset.csv"       # Path to CSV, Parquet, JSONL or "synthetic" for demo
TEXT_COLUMNS = ["text"]              # Single column ["text"] or multi-column ["title", "content"]
TARGET_COLUMN = "label"              # Name of target column (or None for unsupervised tasks)
TASK = "classification"              # "classification", "regression", "multilabel", "ner", "retrieval", etc.
LANGUAGE = "indonesian"              # "indonesian", "english", or "multilingual"

# ==========================================
# 2. COMPETITION SETTINGS (GEMASTIK MODE)
# ==========================================
GEMASTIK_MODE = False                # Set True to enable competition pipeline & submission export
COMPETITION_TEST_PATH = "test.csv"   # Separate unlabelled test dataset
ID_COLUMN = "id"                     # Unique ID column preserved in final submission

# ==========================================
# 3. REPRODUCIBILITY & HARDWARE
# ==========================================
RANDOM_STATE = 42                    # Global random seed
DEVICE = "auto"                      # "auto" (selects CUDA if available), "cuda", or "cpu"
```

Then execute cells sequentially from top to bottom.

---

## 6. RTX 3050 Memory Optimization Tips

1. **Keep Batch Sizes Small**: On 4GB VRAM, set `batch_size = 8` for DistilBERT/MiniLM, and `batch_size = 4` for BERT/RoBERTa base.
2. **Use Gradient Accumulation**: To simulate an effective batch size of 32 without memory spikes, set `gradient_accumulation_steps = 4` (with batch size 8).
3. **Always Enable FP16**: `torch.cuda.amp.autocast(dtype=torch.float16)` halves activations memory and accelerates Tensor Core execution on Ampere GPUs.
4. **Free Memory Between Model Runs**: The notebook includes `HardwareManager.clear_cuda_cache()` to clean up PyTorch tensors and avoid VRAM fragmentation before loading new architectures.

---

## 7. Deliverables Summary

- `nlp_workbench.ipynb`: Complete self-contained experimentation notebook.
- `requirements.txt`: Pinned package dependencies.
- `README.md`: System documentation, task guide, and hardware setup instructions.
