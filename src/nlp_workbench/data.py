"""
Data loading and dataset validation module.
Supports CSV, TSV, Parquet, JSONL formats with strict schema verification and explicit error handling.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd

from .config import WorkbenchConfig

class DataLoader:
    @classmethod
    def load(cls, path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Alias for load_data, supporting class or instance call."""
        return cls.load_data(path, **kwargs)

    @staticmethod
    def load_data(
        path: Union[str, Path],
        file_format: Optional[str] = None,
        sep: str = ",",
        encoding: str = "utf-8",
        max_rows: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Loads dataset from disk. Never silently generates synthetic data if path is missing.
        Raises FileNotFoundError if the file does not exist.
        """
        if str(path).lower().strip() == "synthetic":
            raise ValueError(
                "Use 'DataLoader.generate_synthetic_dataset(task=...)' when synthetic data is needed."
            )

        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found at '{file_path.resolve()}'. "
                f"Please ensure the file exists or explicitly set data_path='synthetic' in configuration."
            )

        # Detect format
        suffix = file_path.suffix.lower()
        fmt = (file_format or suffix.lstrip(".")).lower()

        if fmt in ["parquet", "pq"]:
            df = pd.read_parquet(file_path, engine="pyarrow")
        elif fmt == "jsonl":
            df = pd.read_json(file_path, lines=True)
        elif fmt == "json":
            df = pd.read_json(file_path)
        elif fmt in ["tsv", "tab"]:
            df = pd.read_csv(file_path, sep="\t", encoding=encoding)
        else:  # Default to CSV
            try:
                df = pd.read_csv(file_path, sep=sep, encoding=encoding)
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, sep=sep, encoding="latin-1")

        if max_rows is not None and len(df) > max_rows:
            df = df.iloc[:max_rows].copy()

        return df

    @staticmethod
    def generate_synthetic_dataset(
        task: str = "classification",
        n_samples: int = 150,
        language: str = "indonesian",
        random_state: int = 42,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generates task-specific synthetic datasets with authentic schemas and realistic content.
        Used strictly when explicitly requested.
        """
        np.random.seed(random_state)
        task = task.lower().strip()

        if task in ["classification", "zeroshot"]:
            categories = ["Teknologi", "Ekonomi", "Kesehatan", "Pendidikan", "Olahraga"]
            p_weights = [0.40, 0.25, 0.15, 0.10, 0.10]
            titles = {
                "Teknologi": [
                    "Inovasi Kecerdasan Buatan dan Deep Learning di Era Modern",
                    "Peluncuran Startup Baru dengan Solusi Komputasi Cloud",
                    "Gemastik Kompetisi Data Mining: Mengembangkan Model Klasifikasi Teks",
                ],
                "Ekonomi": [
                    "Bank Sentral Menaikkan Suku Bunga untuk Mengendalikan Laju Inflasi",
                    "Pasar Modal dan Investasi Saham Menguat pada Kuartal Ketiga",
                ],
                "Kesehatan": [
                    "Kementerian Kesehatan Menggalakkan Program Vaksinasi Nasional",
                    "Pola Makan Bergizi dan Olahraga Teratur Meningkatkan Imunitas Tubuh",
                ],
                "Pendidikan": [
                    "Transformasi Kurikulum Merdeka di Tingkat Perguruan Tinggi",
                    "Penerapan Pembelajaran Interaktif Menggunakan Media Digital",
                ],
                "Olahraga": [
                    "Tim Nasional Indonesia Meraih Kemenangan Dramatis di Laga Final",
                    "Pertandingan Sepak Bola Liga Utama Berlangsung dengan Sengit",
                ],
            }
            contents = {
                "Teknologi": "Pengembangan teknologi terkini memberikan dampak signifikan pada efisiensi komputasi cloud.",
                "Ekonomi": "Stabilitas moneter sangat dipengaruhi oleh kebijakan fiskal dan tingkat belanja konsumen di pasar modal.",
                "Kesehatan": "Para dokter spesialis mengimbau masyarakat untuk menjaga kebersihan dan vaksinasi berkala.",
                "Pendidikan": "Pendidikan merupakan fondasi utama karakter bangsa dengan metode pedagogi penalaran kritis.",
                "Olahraga": "Dukungan penuh suporter memberikan motivasi ekstra bagi para pemain di lapangan hijau.",
            }
            rows = []
            for i in range(n_samples):
                cat = np.random.choice(categories, p=p_weights)
                t = np.random.choice(titles[cat]) + f" #{np.random.randint(100, 999)}"
                c = contents[cat]
                if i % 8 == 0:
                    c = f"<p>{c} Kunjungi https://portal-berita.id/artikel-{i} untuk update.</p>"
                rows.append({"id": f"ID_{i:05d}", "title": t, "content": c, "category": cat})
            return pd.DataFrame(rows)

        elif task == "regression":
            base_df = DataLoader.generate_synthetic_dataset("classification", n_samples, language, random_state)
            cat_scores = {"Teknologi": 88.0, "Ekonomi": 75.0, "Kesehatan": 68.0, "Pendidikan": 60.0, "Olahraga": 80.0}
            scores = [
                float(np.round(cat_scores[row["category"]] + np.random.normal(0, 7.5), 2))
                for _, row in base_df.iterrows()
            ]
            base_df["engagement_score"] = scores
            return base_df

        elif task == "multilabel":
            base_df = DataLoader.generate_synthetic_dataset("classification", n_samples, language, random_state)
            all_tags = ["teknologi", "ekonomi", "kesehatan", "pendidikan", "olahraga", "nasional", "populer", "berita"]
            tags = []
            for _, row in base_df.iterrows():
                assigned = [row["category"].lower()]
                extra = np.random.choice(all_tags, size=np.random.randint(1, 3), replace=False).tolist()
                tags.append("|".join(sorted(list(set(assigned + extra)))))
            base_df["tags"] = tags
            return base_df

        elif task == "ner":
            # Token and BIO NER schema
            sample_sentences = [
                (["Presiden", "Joko", "Widodo", "mengunjungi", "Ibu", "Kota", "Nusantara", "."],
                 ["B-PER", "I-PER", "I-PER", "O", "B-LOC", "I-LOC", "I-LOC", "O"]),
                (["Bank", "Indonesia", "menaikkan", "suku", "bunga", "acuan", "di", "Jakarta", "."],
                 ["B-ORG", "I-ORG", "O", "O", "O", "O", "O", "B-LOC", "O"]),
                (["Tim", "nasional", "sepak", "bola", "berlatih", "di", "Stadion", "Gelora", "Bung", "Karno", "."],
                 ["O", "O", "O", "O", "O", "O", "B-LOC", "I-LOC", "I-LOC", "I-LOC", "O"]),
                (["Perusahaan", "GoTo", "dan", "Telkomsel", "mengumumkan", "kolaborasi", "digital", "."],
                 ["O", "B-ORG", "O", "B-ORG", "O", "O", "O", "O"]),
            ]
            rows = []
            for i in range(n_samples):
                tokens, tags = sample_sentences[i % len(sample_sentences)]
                rows.append({"id": f"NER_{i:04d}", "tokens": tokens, "ner_tags": tags})
            return pd.DataFrame(rows)

        elif task == "qa":
            contexts = [
                ("Gemastik adalah Pagelaran Mahasiswa Nasional Bidang Teknologi Informasi dan Komunikasi yang diselenggarakan oleh Balai Pengembangan Talenta Indonesia.",
                 "Apa kepanjangan dari Gemastik?",
                 "Pagelaran Mahasiswa Nasional Bidang Teknologi Informasi dan Komunikasi", 16),
                ("Kecerdasan buatan berkembang pesat di Indonesia dengan penerapan model bahasa besar IndoBERT untuk klasifikasi teks.",
                 "Model bahasa apa yang diterapkan untuk teks bahasa Indonesia?",
                 "IndoBERT", 74),
                ("Jakarta adalah pusat perekonomian Indonesia sebelum pemindahan bertahap ke Nusantara.",
                 "Kota apa yang menjadi pusat perekonomian Indonesia?",
                 "Jakarta", 0),
            ]
            rows = []
            for i in range(n_samples):
                ctx, q, ans, start = contexts[i % len(contexts)]
                rows.append({
                    "id": f"QA_{i:04d}",
                    "context": ctx,
                    "question": q,
                    "answer_text": ans,
                    "answer_start": start,
                })
            return pd.DataFrame(rows)

        elif task == "summarization":
            pairs = [
                ("Pengembangan model kecerdasan buatan di Indonesia kini semakin pesat. Para peneliti merancang arsitektur transformer lokal yang dilatih pada miliaran korpus teks bahasa Indonesia untuk berbagai aplikasi.",
                 "Kecerdasan buatan di Indonesia berkembang pesat dengan arsitektur transformer lokal."),
                ("Kementerian Kesehatan mengumumkan pencapaian target vaksinasi nasional setelah mendistribusikan puluhan juta dosis ke berbagai pelosok daerah terpencil.",
                 "Kemenkes berhasil mencapai target vaksinasi nasional hingga daerah pelosok."),
            ]
            rows = []
            for i in range(n_samples):
                doc, summ = pairs[i % len(pairs)]
                rows.append({"id": f"SUM_{i:04d}", "document": doc, "summary": summ})
            return pd.DataFrame(rows)

        elif task == "translation":
            pairs = [
                ("Model pembelajaran mesin ini memiliki performa yang sangat tinggi.", "This machine learning model has very high performance."),
                ("Kompetisi data mining Gemastik diadakan setiap tahun.", "The Gemastik data mining competition is held annually."),
                ("Teknologi informasi memfasilitasi transformasi digital nasional.", "Information technology facilitates national digital transformation."),
            ]
            rows = []
            for i in range(n_samples):
                src, tgt = pairs[i % len(pairs)]
                rows.append({"id": f"TR_{i:04d}", "source_text": src, "target_text": tgt})
            return pd.DataFrame(rows)

        elif task == "generation":
            prompts = [
                ("Jelaskan manfaat machine learning dalam analisis teks:", "Machine learning memungkinkan analisis sentimen dan klasifikasi teks secara otomatis dan efisien."),
                ("Tuliskan definisi dari natural language processing:", "Natural Language Processing adalah cabang AI yang berfokus pada interaksi antara komputer dan bahasa manusia."),
            ]
            rows = []
            for i in range(n_samples):
                p, r = prompts[i % len(prompts)]
                rows.append({"id": f"GEN_{i:04d}", "prompt": p, "reference_output": r})
            return pd.DataFrame(rows)

        elif task in ["pair_classification", "similarity"]:
            pairs_data = [
                ("Machine learning digunakan untuk klasifikasi teks.", "Deep learning diterapkan untuk pengenalan teks.", 1, 0.85),
                ("Kucing peliharaan sedang tidur di atas sofa.", "Saham perbankan mengalami kenaikan signifikan.", 0, 0.08),
                ("Dokter memberikan resep obat demam untuk pasien.", "Tenaga medis menangani pasien flu di klinik.", 1, 0.90),
                ("Pelari cepat memecahkan rekor dunia atletik.", "Programmer menulis algoritma komputasi cloud.", 0, 0.12),
            ]
            rows = []
            for i in range(n_samples):
                t_a, t_b, label, sim = pairs_data[i % len(pairs_data)]
                rows.append({
                    "id": f"PAIR_{i:04d}",
                    "text_a": t_a,
                    "text_b": t_b,
                    "label": label,
                    "similarity_score": sim,
                })
            return pd.DataFrame(rows)

        elif task == "retrieval":
            # Corpus and query-relevance judgements
            corpus_docs = [
                ("D_01", "Model machine learning dan transformer untuk klasifikasi teks bahasa Indonesia."),
                ("D_02", "Pasar saham dan laporan keuangan perbankan kuartal ketiga."),
                ("D_03", "Panduan kesehatan: menjaga pola makan bergizi dan vaksinasi flu."),
                ("D_04", "Algoritma pencarian informasi dan temu balik dokumen menggunakan BM25."),
                ("D_05", "Pelatihan kurikulum merdeka bagi tenaga pengajar di universitas."),
            ]
            queries = [
                ("Q_01", "metode temu balik dokumen BM25", "D_04", 1),
                ("Q_02", "transformer teks bahasa indonesia", "D_01", 1),
                ("Q_03", "investasi saham dan perbankan", "D_02", 1),
                ("Q_04", "pola makan sehat dan imun tubuh", "D_03", 1),
            ]
            corpus_df = pd.DataFrame([{"doc_id": d[0], "document": d[1]} for d in corpus_docs])
            q_rows = []
            for i in range(n_samples):
                qid, qtext, docid, rel = queries[i % len(queries)]
                q_rows.append({"query_id": f"{qid}_{i}", "query": qtext, "doc_id": docid, "relevance": rel})
            query_df = pd.DataFrame(q_rows)
            return corpus_df, query_df

        elif task == "topic_modeling":
            texts = [
                "Inovasi kecerdasan buatan, algoritma deep learning, dan komputasi awan.",
                "Pertumbuhan pasar modal, investasi saham reksadana, dan inflasi perbankan.",
                "Layanan rumah sakit, dokter spesialis, pola makan sehat, dan vaksin imunisasi.",
                "Kurikulum merdeka perguruan tinggi, beasiswa mahasiswa berprestasi, dan pedagogi.",
                "Kemenangan turnamen sepak bola, latihan atlet bulu tangkis, dan kejuaraan liga.",
            ]
            rows = [{"id": f"TM_{i:04d}", "content": texts[i % len(texts)]} for i in range(n_samples)]
            return pd.DataFrame(rows)

        else:
            raise ValueError(f"Unknown task '{task}' for synthetic data generation.")


def validate_dataset_schema(
    df: pd.DataFrame,
    config: Optional[WorkbenchConfig] = None,
    task: Optional[str] = None,
    text_col: Optional[str] = None,
    target_col: Optional[str] = None,
    raise_on_error: Optional[bool] = None,
) -> Tuple[bool, List[str]]:
    """Validates that the dataset DataFrame matches the required schema for the task."""
    errors = []
    should_raise = raise_on_error if raise_on_error is not None else (config is None)

    if df is None or len(df) == 0:
        if should_raise:
            raise ValueError("Dataset is empty (0 rows).")
        return False, ["Dataset is empty (0 rows)."]

    # Resolve parameters from config or kwargs
    effective_task = (config.task if config else (task or "classification")).lower().strip()
    effective_text_cols = list(config.text_columns) if config else ([text_col] if text_col else ["text"])
    effective_target_col = config.target_column if config else target_col

    # 1. Text columns
    for col in effective_text_cols:
        if col not in df.columns:
            msg = f"Text column '{col}' not found in dataset."
            if should_raise:
                raise KeyError(msg)
            errors.append(msg)

    # 2. Target column
    if effective_target_col and effective_task not in ["topic_modeling", "retrieval", "clustering"]:
        if effective_target_col not in df.columns:
            msg = f"Target column '{effective_target_col}' not found in dataset."
            if should_raise:
                raise KeyError(msg)
            errors.append(msg)
        else:
            null_targets = df[effective_target_col].isna().sum()
            if null_targets > 0:
                msg = (
                    f"Target column '{effective_target_col}' has {null_targets} missing values. "
                    f"Supervised tasks require explicit labels (do not convert nulls to strings)."
                )
                if should_raise:
                    raise ValueError(msg)
                errors.append(msg)

    # 3. Task-specific columns
    if config:
        if effective_task in ["pair_classification", "similarity"]:
            t_b = config.task_params.get("text_b_column")
            if t_b and t_b not in df.columns:
                errors.append(f"Secondary text column '{t_b}' is missing from dataset.")

        elif effective_task == "qa":
            ctx = config.task_params.get("context_column", "context")
            ans = config.task_params.get("answer_column", "answer_text")
            if ctx not in df.columns:
                errors.append(f"QA context column '{ctx}' is missing.")
            if ans not in df.columns:
                errors.append(f"QA answer column '{ans}' is missing.")

        elif effective_task == "ner":
            if "tokens" in config.text_columns and "tokens" in df.columns:
                first_val = df["tokens"].iloc[0]
                if not isinstance(first_val, (list, np.ndarray)):
                    errors.append("NER 'tokens' column must contain sequences of tokens (list of strings).")

    return (len(errors) == 0), errors


# Standalone convenience generators
def generate_synthetic_dataset(task: str = "classification", n_samples: int = 150, language: str = "indonesian", random_state: int = 42):
    return DataLoader.generate_synthetic_dataset(task=task, n_samples=n_samples, language=language, random_state=random_state)

def generate_synthetic_classification_dataset(n_samples: int = 150, random_state: int = 42) -> pd.DataFrame:
    df = DataLoader.generate_synthetic_dataset("classification", n_samples=n_samples, random_state=random_state)
    return df.rename(columns={"content": "text", "category": "target"})

def generate_synthetic_regression_dataset(n_samples: int = 150, random_state: int = 42) -> pd.DataFrame:
    df = DataLoader.generate_synthetic_dataset("regression", n_samples=n_samples, random_state=random_state)
    return df.rename(columns={"content": "text", "engagement_score": "target"})

def generate_synthetic_multilabel_dataset(n_samples: int = 150, random_state: int = 42) -> pd.DataFrame:
    df = DataLoader.generate_synthetic_dataset("multilabel", n_samples=n_samples, random_state=random_state)
    df = df.rename(columns={"content": "text"})
    tags_split = df["tags"].str.get_dummies(sep="|")
    tags_split.columns = [f"label_{c}" for c in tags_split.columns]
    return pd.concat([df[["id", "text"]], tags_split], axis=1)

def generate_synthetic_ner_dataset(n_samples: int = 100, random_state: int = 42) -> pd.DataFrame:
    return DataLoader.generate_synthetic_dataset("ner", n_samples=n_samples, random_state=random_state)

def generate_synthetic_qa_dataset(n_samples: int = 100, random_state: int = 42) -> pd.DataFrame:
    df = DataLoader.generate_synthetic_dataset("qa", n_samples=n_samples, random_state=random_state)
    return df.rename(columns={"answer_text": "answer"})

def generate_synthetic_summarization_dataset(n_samples: int = 100, random_state: int = 42) -> pd.DataFrame:
    df = DataLoader.generate_synthetic_dataset("summarization", n_samples=n_samples, random_state=random_state)
    return df.rename(columns={"document": "text"})

def generate_synthetic_translation_dataset(n_samples: int = 100, random_state: int = 42) -> pd.DataFrame:
    return DataLoader.generate_synthetic_dataset("translation", n_samples=n_samples, random_state=random_state)

def generate_synthetic_pair_classification_dataset(n_samples: int = 100, random_state: int = 42) -> pd.DataFrame:
    return DataLoader.generate_synthetic_dataset("pair_classification", n_samples=n_samples, random_state=random_state)

def generate_synthetic_retrieval_dataset(n_samples: int = 100, random_state: int = 42) -> pd.DataFrame:
    corpus_df, query_df = DataLoader.generate_synthetic_dataset("retrieval", n_samples=n_samples, random_state=random_state)
    merged = query_df.merge(corpus_df, on="doc_id", how="left")
    return merged.rename(columns={"document": "doc_text"})
