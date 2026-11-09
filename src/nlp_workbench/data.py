"""
Data loading and dataset validation module.
Supports CSV, TSV, Parquet, JSONL formats with strict schema verification and explicit error handling.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd

from .config import WorkbenchConfig


@dataclass
class RetrievalDatasetBundle:
    """
    Dedicated Information Retrieval data bundle preserving separate corpus, query collection, and qrels.
    """
    corpus_df: pd.DataFrame
    queries_df: pd.DataFrame
    qrels: Optional[Dict[str, Dict[str, int]]] = None
    doc_id_col: str = "doc_id"
    doc_text_col: str = "document"
    query_id_col: str = "query_id"
    query_text_col: str = "query"
    relevance_col: str = "relevance"

    def __post_init__(self):
        if self.doc_id_col not in self.corpus_df.columns:
            raise KeyError(f"Corpus is missing doc_id_col '{self.doc_id_col}'.")
        if self.doc_text_col not in self.corpus_df.columns:
            raise KeyError(f"Corpus is missing doc_text_col '{self.doc_text_col}'.")
        if self.query_id_col not in self.queries_df.columns:
            raise KeyError(f"Queries dataset is missing query_id_col '{self.query_id_col}'.")
        if self.query_text_col not in self.queries_df.columns:
            raise KeyError(f"Queries dataset is missing query_text_col '{self.query_text_col}'.")

        # Build qrels if relevance column is present
        if self.qrels is None and self.relevance_col in self.queries_df.columns and self.doc_id_col in self.queries_df.columns:
            self.qrels = {}
            for _, row in self.queries_df.iterrows():
                qid = str(row[self.query_id_col])
                did = str(row[self.doc_id_col])
                rel = int(row.get(self.relevance_col, 1))
                if qid not in self.qrels:
                    self.qrels[qid] = {}
                self.qrels[qid][did] = rel

    def __iter__(self):
        """Allows backwards-compatible tuple unpacking: corpus_df, queries_df = bundle"""
        return iter((self.corpus_df, self.queries_df))

    def __getitem__(self, index):
        if index == 0:
            return self.corpus_df
        elif index == 1:
            return self.queries_df
        raise IndexError("RetrievalDatasetBundle index out of range (0=corpus_df, 1=queries_df).")

    def __len__(self):
        return len(self.queries_df)

    @property
    def shape(self) -> Tuple[int, int]:
        return (len(self.queries_df), len(self.corpus_df))

    def to_dataframe(self) -> pd.DataFrame:
        """Merges queries and corpus into a tabular DataFrame."""
        return self.queries_df.merge(self.corpus_df, on=self.doc_id_col, how="left")


class DatasetValidator:
    """Class wrapper for dataset schema validation."""
    def __init__(
        self,
        config: Optional[WorkbenchConfig] = None,
        task: Optional[str] = None,
        text_columns: Optional[List[str]] = None,
        target_column: Optional[str] = None,
        id_column: Optional[str] = None,
    ):
        self.config = config
        self.task = task
        self.text_columns = text_columns
        self.target_column = target_column
        self.id_column = id_column

    def validate(self, df: Union[pd.DataFrame, RetrievalDatasetBundle], raise_on_error: bool = True) -> Tuple[bool, List[str]]:
        return validate_dataset_schema(
            df=df,
            config=self.config,
            task=self.task,
            text_col=self.text_columns[0] if self.text_columns else None,
            target_col=self.target_column,
            raise_on_error=raise_on_error,
        )


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
    ) -> Union[pd.DataFrame, RetrievalDatasetBundle]:
        """
        Generates task-specific synthetic datasets with authentic schemas, realistic content,
        and guaranteed sample-level variation so no duplicate text causes synthetic leakage.
        Provides distinct, non-overlapping column aliases ensuring compatibility with both
        WorkbenchConfig defaults and get_default_config.
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
                t = np.random.choice(titles[cat]) + f" #{np.random.randint(100, 999)} (Dokumen {i:04d})"
                c = f"{contents[cat]} Dokumen riset dan analisis sektor {cat.lower()} indeks {i:04d}."
                if i % 8 == 0:
                    c = f"<p>{c} Kunjungi https://portal-berita.id/artikel-{i} untuk update informasi.</p>"
                rows.append({
                    "id": f"ID_{i:05d}",
                    "title": t,
                    "text": c,
                    "content": c,
                    "target": cat,
                    "category": cat,
                })
            return pd.DataFrame(rows)

        elif task == "regression":
            categories = ["Teknologi", "Ekonomi", "Kesehatan", "Pendidikan", "Olahraga"]
            cat_scores = {"Teknologi": 88.0, "Ekonomi": 75.0, "Kesehatan": 68.0, "Pendidikan": 60.0, "Olahraga": 80.0}
            contents = {
                "Teknologi": "Pengembangan teknologi terkini memberikan dampak signifikan pada efisiensi komputasi cloud.",
                "Ekonomi": "Stabilitas moneter sangat dipengaruhi oleh kebijakan fiskal dan tingkat belanja konsumen di pasar modal.",
                "Kesehatan": "Para dokter spesialis mengimbau masyarakat untuk menjaga kebersihan dan vaksinasi berkala.",
                "Pendidikan": "Pendidikan merupakan fondasi utama karakter bangsa dengan metode pedagogi penalaran kritis.",
                "Olahraga": "Dukungan penuh suporter memberikan motivasi ekstra bagi para pemain di lapangan hijau.",
            }
            rows = []
            for i in range(n_samples):
                cat = categories[i % len(categories)]
                c = f"{contents[cat]} Laporan performa sektor {cat.lower()} indeks {i:04d}."
                score = float(np.round(cat_scores[cat] + np.random.normal(0, 7.5), 2))
                rows.append({
                    "id": f"ID_{i:05d}",
                    "title": f"Laporan Finansial #{i:04d}",
                    "text": c,
                    "content": c,
                    "target": score,
                    "engagement_score": score,
                })
            return pd.DataFrame(rows)

        elif task == "multilabel":
            categories = ["Teknologi", "Ekonomi", "Kesehatan", "Pendidikan", "Olahraga"]
            all_tags = ["teknologi", "ekonomi", "kesehatan", "pendidikan", "olahraga", "nasional", "populer", "berita"]
            contents = {
                "Teknologi": "Pengembangan teknologi terkini memberikan dampak signifikan pada efisiensi komputasi cloud.",
                "Ekonomi": "Stabilitas moneter sangat dipengaruhi oleh kebijakan fiskal dan tingkat belanja konsumen di pasar modal.",
                "Kesehatan": "Para dokter spesialis mengimbau masyarakat untuk menjaga kebersihan dan vaksinasi berkala.",
                "Pendidikan": "Pendidikan merupakan fondasi utama karakter bangsa dengan metode pedagogi penalaran kritis.",
                "Olahraga": "Dukungan penuh suporter memberikan motivasi ekstra bagi para pemain di lapangan hijau.",
            }
            rows = []
            for i in range(n_samples):
                cat = categories[i % len(categories)]
                assigned = [cat.lower()]
                extra = np.random.choice(all_tags, size=np.random.randint(1, 3), replace=False).tolist()
                tags = sorted(list(set(assigned + extra)))
                c = f"{contents[cat]} Berita multi-topik nomor {i:04d}."
                rows.append({
                    "id": f"ID_{i:05d}",
                    "title": f"Artikel #{i:04d}",
                    "text": c,
                    "content": c,
                    "target": "|".join(tags),
                    "tags": "|".join(tags),
                    "label_tech": int("teknologi" in tags),
                    "label_biz": int("ekonomi" in tags),
                    "label_health": int("kesehatan" in tags),
                })
            return pd.DataFrame(rows)

        elif task == "ner":
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
                toks, tags = sample_sentences[i % len(sample_sentences)]
                var_toks = list(toks) + [f"({i:04d})"]
                var_tags = list(tags) + ["O"]
                rows.append({
                    "id": f"NER_{i:04d}",
                    "tokens": var_toks,
                    "ner_tags": var_tags,
                    "text": " ".join(var_toks),
                    "target": var_tags,
                })
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
                var_ctx = f"{ctx} [Konteks referensi #{i:04d}]"
                var_q = f"{q} #{i:04d}"
                rows.append({
                    "id": f"QA_{i:04d}",
                    "context": var_ctx,
                    "question": var_q,
                    "answer_text": ans,
                    "answer": ans,
                    "answer_start": start,
                    "text": var_q,
                    "target": ans,
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
                var_doc = f"{doc} Laporan rilis informasi versi #{i:04d}."
                var_summ = f"{summ} (Rangkuman #{i:04d})"
                rows.append({
                    "id": f"SUM_{i:04d}",
                    "document": var_doc,
                    "summary": var_summ,
                    "text": var_doc,
                    "target": var_summ,
                })
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
                var_src = f"{src} [Kalimat {i:04d}]"
                var_tgt = f"{tgt} [Sentence {i:04d}]"
                rows.append({
                    "id": f"TR_{i:04d}",
                    "source_text": var_src,
                    "target_text": var_tgt,
                    "text": var_src,
                    "target": var_tgt,
                })
            return pd.DataFrame(rows)

        elif task == "generation":
            prompts = [
                ("Jelaskan manfaat machine learning dalam analisis teks:", "Machine learning memungkinkan analisis sentimen dan klasifikasi teks secara otomatis dan efisien."),
                ("Tuliskan definisi dari natural language processing:", "Natural Language Processing adalah cabang AI yang berfokus pada interaksi antara komputer dan bahasa manusia."),
            ]
            rows = []
            for i in range(n_samples):
                p, r = prompts[i % len(prompts)]
                var_p = f"{p} (Kasus {i:04d})"
                var_r = f"{r} Catatan evaluasi {i:04d}."
                rows.append({
                    "id": f"GEN_{i:04d}",
                    "prompt": var_p,
                    "reference_output": var_r,
                    "text": var_p,
                    "target": var_r,
                })
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
                var_a = f"{t_a} #{i:04d}"
                var_b = f"{t_b} #{i:04d}"
                rows.append({
                    "id": f"PAIR_{i:04d}",
                    "text_a": var_a,
                    "text_b": var_b,
                    "text": var_a,
                    "label": label,
                    "similarity_score": sim,
                    "target": label if task == "pair_classification" else sim,
                })
            return pd.DataFrame(rows)

        elif task == "retrieval":
            corpus_docs = [
                ("D_01", "Model machine learning dan transformer untuk klasifikasi teks bahasa Indonesia."),
                ("D_02", "Pasar saham dan laporan keuangan perbankan kuartal ketiga."),
                ("D_03", "Panduan kesehatan: menjaga pola makan bergizi dan vaksinasi flu."),
                ("D_04", "Algoritma pencarian informasi dan temu balik dokumen menggunakan BM25."),
                ("D_05", "Pelatihan kurikulum merdeka bagi tenaga pengajar di universitas."),
                ("D_06", "Strategi mitigasi perubahan iklim dan konservasi hutan lindung nasional."),
                ("D_07", "Pengembangan ekosistem semikonduktor dan komputasi mikro modern."),
            ]
            queries = [
                ("Q_01", "metode temu balik dokumen BM25", "D_04", 1),
                ("Q_02", "transformer teks bahasa indonesia", "D_01", 1),
                ("Q_03", "investasi saham dan perbankan", "D_02", 1),
                ("Q_04", "pola makan sehat dan imun tubuh", "D_03", 1),
                ("Q_05", "kurikulum merdeka universitas", "D_05", 1),
                ("Q_06", "konservasi lingkungan dan iklim", "D_06", 1),
            ]
            corpus_df = pd.DataFrame([{"doc_id": d[0], "document": d[1]} for d in corpus_docs])
            q_rows = []
            for i in range(n_samples):
                qid, qtext, docid, rel = queries[i % len(queries)]
                q_rows.append({
                    "query_id": f"{qid}_{i:04d}",
                    "query": f"{qtext} [Varian {i:04d}]",
                    "doc_id": docid,
                    "relevance": rel,
                })
            query_df = pd.DataFrame(q_rows)
            return RetrievalDatasetBundle(corpus_df=corpus_df, queries_df=query_df)

        elif task == "topic_modeling":
            texts = [
                "Inovasi kecerdasan buatan, algoritma deep learning, dan komputasi awan.",
                "Pertumbuhan pasar modal, investasi saham reksadana, dan inflasi perbankan.",
                "Layanan rumah sakit, dokter spesialis, pola makan sehat, dan vaksin imunisasi.",
                "Kurikulum merdeka perguruan tinggi, beasiswa mahasiswa berprestasi, dan pedagogi.",
                "Kemenangan turnamen sepak bola, latihan atlet bulu tangkis, dan kejuaraan liga.",
            ]
            rows = []
            for i in range(n_samples):
                base_txt = texts[i % len(texts)]
                rows.append({
                    "id": f"TM_{i:04d}",
                    "content": f"{base_txt} Korpus topik nomor {i:04d}.",
                    "text": f"{base_txt} Korpus topik nomor {i:04d}.",
                })
            return pd.DataFrame(rows)

        else:
            raise ValueError(f"Unknown task '{task}' for synthetic data generation.")


def validate_dataset_schema(
    df: Union[pd.DataFrame, RetrievalDatasetBundle],
    config: Optional[WorkbenchConfig] = None,
    task: Optional[str] = None,
    text_col: Optional[str] = None,
    target_col: Optional[str] = None,
    raise_on_error: Optional[bool] = None,
) -> Tuple[bool, List[str]]:
    """Validates that the dataset DataFrame or RetrievalDatasetBundle matches the required schema for the task."""
    errors = []
    should_raise = raise_on_error if raise_on_error is not None else (config is None)

    # 1. RetrievalDatasetBundle Validation
    if isinstance(df, RetrievalDatasetBundle):
        if df.corpus_df is None or len(df.corpus_df) == 0:
            msg = "Retrieval corpus_df is empty (0 rows)."
            if should_raise:
                raise ValueError(msg)
            errors.append(msg)
        if df.queries_df is None or len(df.queries_df) == 0:
            msg = "Retrieval queries_df is empty (0 rows)."
            if should_raise:
                raise ValueError(msg)
            errors.append(msg)
        if df.doc_text_col not in df.corpus_df.columns:
            msg = f"Retrieval corpus missing document column '{df.doc_text_col}'."
            if should_raise:
                raise KeyError(msg)
            errors.append(msg)
        if df.query_text_col not in df.queries_df.columns:
            msg = f"Retrieval queries missing query column '{df.query_text_col}'."
            if should_raise:
                raise KeyError(msg)
            errors.append(msg)
        return (len(errors) == 0), errors

    # 2. DataFrame Validation
    if df is None or len(df) == 0:
        if should_raise:
            raise ValueError("Dataset is empty (0 rows).")
        return False, ["Dataset is empty (0 rows)."]

    # Resolve parameters from config or kwargs
    effective_task = (config.task if config else (task or "classification")).lower().strip()
    effective_text_cols = list(config.text_columns) if config else ([text_col] if text_col else ["text"])
    effective_target_col = config.target_column if config else target_col

    # Text columns
    for col in effective_text_cols:
        if col not in df.columns:
            msg = f"Text column '{col}' not found in dataset."
            if should_raise:
                raise KeyError(msg)
            errors.append(msg)

    # Target column
    if effective_target_col and effective_task not in ["topic_modeling", "retrieval"]:
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

    # Task-specific validations
    if effective_task == "regression" and effective_target_col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[effective_target_col]):
            try:
                pd.to_numeric(df[effective_target_col])
            except Exception:
                msg = f"Target column '{effective_target_col}' must be numeric for regression task."
                if should_raise:
                    raise ValueError(msg)
                errors.append(msg)

    elif effective_task == "multilabel":
        target_cols = config.task_params.get("target_columns") if config else None
        if target_cols:
            missing_cols = [c for c in target_cols if c not in df.columns]
            if missing_cols:
                msg = f"Multilabel target columns missing from dataset: {missing_cols}"
                if should_raise:
                    raise KeyError(msg)
                errors.append(msg)
        elif not effective_target_col or effective_target_col not in df.columns:
            msg = "Multilabel classification requires target_column or task_params['target_columns']."
            if should_raise:
                raise KeyError(msg)
            errors.append(msg)

    elif effective_task in ["pair_classification", "similarity"]:
        t_b = (config.task_params.get("text_b_column") if config else None) or "text_b"
        if t_b not in df.columns:
            msg = f"Secondary text column '{t_b}' is missing from dataset."
            if should_raise:
                raise KeyError(msg)
            errors.append(msg)

    elif effective_task == "qa":
        ctx = (config.task_params.get("context_column") if config else None) or "context"
        ans = (config.task_params.get("answer_column") if config else None) or "answer_text"
        if ctx not in df.columns:
            msg = f"QA context column '{ctx}' is missing."
            if should_raise:
                raise KeyError(msg)
            errors.append(msg)
        if ans not in df.columns and effective_target_col not in df.columns:
            msg = f"QA answer column '{ans}' is missing."
            if should_raise:
                raise KeyError(msg)
            errors.append(msg)

    elif effective_task == "ner":
        tok_col = effective_text_cols[0] if effective_text_cols else "tokens"
        if tok_col in df.columns:
            first_val = df[tok_col].iloc[0]
            if not isinstance(first_val, (list, np.ndarray, str)):
                msg = f"NER tokens column '{tok_col}' must contain sequence of tokens or strings."
                if should_raise:
                    raise ValueError(msg)
                errors.append(msg)

    return (len(errors) == 0), errors


# Standalone convenience generators
def generate_synthetic_dataset(task: str = "classification", n_samples: int = 150, language: str = "indonesian", random_state: int = 42):
    return DataLoader.generate_synthetic_dataset(task=task, n_samples=n_samples, language=language, random_state=random_state)

def generate_synthetic_classification_dataset(n_samples: int = 150, random_state: int = 42) -> pd.DataFrame:
    df = DataLoader.generate_synthetic_dataset("classification", n_samples=n_samples, random_state=random_state)
    return df[["id", "title", "text", "target"]].copy()

def generate_synthetic_regression_dataset(n_samples: int = 150, random_state: int = 42) -> pd.DataFrame:
    df = DataLoader.generate_synthetic_dataset("regression", n_samples=n_samples, random_state=random_state)
    return df[["id", "title", "text", "target"]].copy()

def generate_synthetic_multilabel_dataset(n_samples: int = 150, random_state: int = 42) -> pd.DataFrame:
    df = DataLoader.generate_synthetic_dataset("multilabel", n_samples=n_samples, random_state=random_state)
    return df[["id", "title", "text", "target", "label_tech", "label_biz", "label_health"]].copy()

def generate_synthetic_ner_dataset(n_samples: int = 100, random_state: int = 42) -> pd.DataFrame:
    df = DataLoader.generate_synthetic_dataset("ner", n_samples=n_samples, random_state=random_state)
    return df[["id", "tokens", "ner_tags"]].copy()

def generate_synthetic_qa_dataset(n_samples: int = 100, random_state: int = 42) -> pd.DataFrame:
    df = DataLoader.generate_synthetic_dataset("qa", n_samples=n_samples, random_state=random_state)
    return df[["id", "context", "question", "answer", "answer_text", "answer_start"]].copy()

def generate_synthetic_summarization_dataset(n_samples: int = 100, random_state: int = 42) -> pd.DataFrame:
    df = DataLoader.generate_synthetic_dataset("summarization", n_samples=n_samples, random_state=random_state)
    return df[["id", "document", "summary", "text", "target"]].copy()

def generate_synthetic_translation_dataset(n_samples: int = 100, random_state: int = 42) -> pd.DataFrame:
    df = DataLoader.generate_synthetic_dataset("translation", n_samples=n_samples, random_state=random_state)
    return df[["id", "source_text", "target_text", "text", "target"]].copy()

def generate_synthetic_pair_classification_dataset(n_samples: int = 100, random_state: int = 42) -> pd.DataFrame:
    df = DataLoader.generate_synthetic_dataset("pair_classification", n_samples=n_samples, random_state=random_state)
    return df[["id", "text_a", "text_b", "label", "similarity_score", "text", "target"]].copy()

def generate_synthetic_retrieval_dataset(n_samples: int = 100, random_state: int = 42) -> pd.DataFrame:
    bundle = DataLoader.generate_synthetic_dataset("retrieval", n_samples=n_samples, random_state=random_state)
    corpus_df, query_df = bundle
    merged = query_df.merge(corpus_df, on="doc_id", how="left")
    return merged.rename(columns={"document": "doc_text"})
