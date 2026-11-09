"""
Exploratory Data Analysis (EDA) and Decision Support module.
Provides automated data profiling, leakage detection, and 5-point explainable decision support.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

class AuditReport(dict):
    """Container allowing both dictionary access and tuple unpacking for audit results."""
    def __init__(self, summary_df: pd.DataFrame, findings: List[Dict[str, Any]]):
        super().__init__(summary_stats=summary_df, findings=findings)
        self.summary_df = summary_df
        self.findings = findings

    def __iter__(self):
        return iter((self.summary_df, self.findings))


class ExploratoryDataAnalysis:
    """
    Profiles text statistics: character lengths, word counts, vocabulary, and target distributions.
    """
    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        text_columns: Optional[List[str]] = None,
        target_column: Optional[str] = None,
    ):
        self.df = df
        self.text_columns = text_columns or []
        self.target_column = target_column

    def profile(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        data = df if df is not None else self.df
        if data is None:
            raise ValueError("No DataFrame provided for EDA profiling.")

        cols = [c for c in self.text_columns if c in data.columns]
        if not cols:
            cols = [c for c in ["title", "content", "document", "text", "query", "source_text"] if c in data.columns]

        stats = {}
        for c in cols:
            series = data[c].fillna("").astype(str)
            char_lens = series.str.len()
            word_counts = series.str.split().str.len()
            stats[c] = {
                "sample_count": len(series),
                "char_len_mean": round(float(char_lens.mean()), 1),
                "char_len_median": float(char_lens.median()),
                "char_len_max": int(char_lens.max()) if len(char_lens) else 0,
                "word_count_mean": round(float(word_counts.mean()), 1),
                "word_count_median": float(word_counts.median()),
                "word_count_max": int(word_counts.max()) if len(word_counts) else 0,
            }

        target_stats = {}
        if self.target_column and self.target_column in data.columns:
            vc = data[self.target_column].value_counts()
            target_stats = {
                "unique_values": int(data[self.target_column].nunique()),
                "distribution": vc.to_dict(),
            }

        return {
            "total_rows": len(data),
            "columns": list(data.columns),
            "text_stats": stats,
            "target_stats": target_stats,
        }


class DataQualityAuditor:
    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        text_columns: Optional[List[str]] = None,
        target_column: Optional[str] = None,
        id_column: Optional[str] = None,
        task: str = "classification",
    ):
        self.df = df
        self.text_columns = [c for c in text_columns if c in df.columns] if (df is not None and text_columns) else (text_columns or [])
        self.target_column = target_column if (df is not None and target_column in df.columns) else target_column
        self.id_column = id_column if (df is not None and id_column in df.columns) else id_column
        self.task = task.lower().strip()
        self.findings: List[Dict[str, Any]] = []

    def audit(
        self,
        df: Optional[pd.DataFrame] = None,
        text_columns: Optional[List[str]] = None,
        target_column: Optional[str] = None,
        id_column: Optional[str] = None,
    ) -> AuditReport:
        """Performs comprehensive data quality audit and returns an AuditReport."""
        if df is not None:
            self.df = df
        if text_columns is not None:
            self.text_columns = [c for c in text_columns if self.df is not None and c in self.df.columns]
        if target_column is not None:
            self.target_column = target_column
        if id_column is not None:
            self.id_column = id_column

        if self.df is None:
            raise ValueError("No DataFrame provided for auditing.")

        self.findings = []
        n_rows = len(self.df)

        # 1. Column-by-column profiling
        summary_rows = []
        for col in self.df.columns:
            s = self.df[col]
            missing_count = int(s.isna().sum())
            missing_pct = (missing_count / n_rows) * 100 if n_rows > 0 else 0.0
            n_unique = int(s.nunique(dropna=True))
            cardinality = (n_unique / n_rows) if n_rows > 0 else 0.0
            dtype_str = str(s.dtype)

            # Check potential identifier
            is_potential_id = (
                cardinality > 0.85
                and n_rows >= 50
                and col != self.target_column
                and col not in self.text_columns
            )
            if is_potential_id and col != self.id_column:
                self.findings.append({
                    "column": col,
                    "issue_type": "Potential Identifier Column",
                    "severity": "WARNING",
                    "observed": f"{cardinality*100:.1f}% unique values ({n_unique}/{n_rows})",
                    "why_it_matters": "Identifier columns lead to artificial memorization and severe test overfitting.",
                    "recommendation": f"Exclude '{col}' from model feature inputs.",
                    "drawbacks": "None for modeling; make sure to retain ID separately if needed for submission.",
                    "action_key": f"drop_id_{col}",
                })

            # Check potential target leakage
            if self.target_column and col != self.target_column:
                if s.dtype == object and self.df[self.target_column].dtype == object:
                    match_rate = float((s.astype(str) == self.df[self.target_column].astype(str)).mean())
                    if match_rate > 0.80:
                        self.findings.append({
                            "column": col,
                            "issue_type": "Potential Target Leakage",
                            "severity": "CRITICAL",
                            "observed": f"Matches target '{self.target_column}' in {match_rate*100:.1f}% of rows",
                            "why_it_matters": "Features that duplicate the target invalidate offline evaluation and will fail at test time.",
                            "recommendation": f"Drop feature '{col}' before training.",
                            "drawbacks": "None; removing leaked features reflects real-world inference conditions.",
                            "action_key": f"drop_leakage_{col}",
                        })

            if missing_pct > 0:
                severity = "HIGH" if missing_pct > 25 else "INFO"
                self.findings.append({
                    "column": col,
                    "issue_type": "Missing Values Detected",
                    "severity": severity,
                    "observed": f"{missing_count} missing rows ({missing_pct:.2f}%)",
                    "why_it_matters": "Missing text inputs break tokenization pipelines; missing targets corrupt training.",
                    "recommendation": (
                        f"Impute missing text in '{col}' with empty strings ('') or drop rows if target column has nulls."
                    ),
                    "drawbacks": "Imputing empty text may slightly dilute document length statistics.",
                    "action_key": f"impute_{col}",
                })

            role = (
                "Target" if col == self.target_column
                else ("Text Feature" if col in self.text_columns
                else ("ID" if col == self.id_column else "Tabular Feature"))
            )
            summary_rows.append({
                "Column": col,
                "Dtype": dtype_str,
                "Missing": missing_count,
                "Missing%": round(missing_pct, 2),
                "Unique": n_unique,
                "Cardinality": round(cardinality, 4),
                "Role": role,
            })

        summary_df = pd.DataFrame(summary_rows)

        # 2. Combined Text Diagnostics
        if self.text_columns:
            combined = self.df[self.text_columns].fillna("").astype(str).agg(" ".join, axis=1).str.strip()
            empty_count = int((combined == "").sum())
            if empty_count > 0:
                self.findings.append({
                    "column": "Combined Text",
                    "issue_type": "Empty Text Entries",
                    "severity": "WARNING",
                    "observed": f"{empty_count} rows have empty or whitespace-only text",
                    "why_it_matters": "Empty strings provide zero semantic features and degrade loss gradients.",
                    "recommendation": "Impute placeholder token ('[EMPTY]') or drop empty rows from training split.",
                    "drawbacks": "Dropping rows slightly reduces training sample size.",
                    "action_key": "handle_empty_text",
                })

            exact_dupes = int(combined.duplicated(keep=False).sum())
            if exact_dupes > 0:
                self.findings.append({
                    "column": "Combined Text",
                    "issue_type": "Duplicate Text Entries",
                    "severity": "WARNING",
                    "observed": f"{exact_dupes} rows share identical text strings",
                    "why_it_matters": "Duplicate text across training and validation splits inflates performance artificially.",
                    "recommendation": "Perform group-based splitting or deduplicate training records.",
                    "drawbacks": "Legitimate duplicate queries in production might have distinct context.",
                    "action_key": "deduplicate_texts",
                })

            word_counts = combined.apply(lambda x: len(x.split()))
            q75, q25 = word_counts.quantile(0.75), word_counts.quantile(0.25)
            iqr = q75 - q25
            outliers = int((word_counts > (q75 + 1.5 * iqr)).sum())
            if outliers > 0:
                rec_len = min(256, int(word_counts.quantile(0.95)))
                self.findings.append({
                    "column": "Combined Text",
                    "issue_type": "Document Length Outliers",
                    "severity": "INFO",
                    "observed": f"{outliers} documents exceed length threshold ({q75 + 1.5 * iqr:.0f} words; max={word_counts.max()})",
                    "why_it_matters": "Long documents exhaust RTX 3050 VRAM quadratic in attention sequence length.",
                    "recommendation": f"Set max_seq_length={rec_len} to balance information retention and 4GB VRAM safety.",
                    "drawbacks": "Text beyond max_seq_length will be truncated.",
                    "action_key": "set_max_seq_len",
                })

            has_html = int(combined.str.contains(r"<[^>]+>", regex=True).sum())
            if has_html > 0:
                self.findings.append({
                    "column": "Combined Text",
                    "issue_type": "HTML Tags Detected",
                    "severity": "INFO",
                    "observed": f"{has_html} rows contain HTML markup tags",
                    "why_it_matters": "HTML boilerplate introduces noise tokens that clutter vocabulary.",
                    "recommendation": "Enable clean_remove_html to sanitize markup artifacts.",
                    "drawbacks": "Stripping HTML might discard structure if markup tags carry semantic headers.",
                    "action_key": "enable_html_cleaning",
                })

            has_urls = int(combined.str.contains(r"https?://\S+|www\.\S+", regex=True).sum())
            if has_urls > 0:
                self.findings.append({
                    "column": "Combined Text",
                    "issue_type": "Web URLs Detected",
                    "severity": "INFO",
                    "observed": f"{has_urls} rows contain web URLs",
                    "why_it_matters": "Unique URLs increase sparse dimensionality and cause vocabulary fragmentation without contributing semantic intent.",
                    "recommendation": "Normalize URLs to a standard token '[URL]'.",
                    "drawbacks": "Domain names that carry entity information (e.g. news sources) are generalized.",
                    "action_key": "handle_urls",
                })

        # 3. Supervised Target Diagnostics
        if self.target_column:
            t = self.df[self.target_column].dropna()
            if self.task == "regression" or (pd.api.types.is_numeric_dtype(t) and t.nunique() > 20):
                skew = float(t.skew())
                if abs(skew) > 1.2:
                    self.findings.append({
                        "column": self.target_column,
                        "issue_type": "High Target Skewness",
                        "severity": "WARNING",
                        "observed": f"Target skewness = {skew:.2f} (substantially non-normal)",
                        "why_it_matters": "Skewed regression targets bias linear and neural losses toward tail outliers.",
                        "recommendation": "Apply np.log1p transform or use Huber / MAE loss.",
                        "drawbacks": "Predictions must be inverted with np.expm1 before final evaluation.",
                        "action_key": "log_transform_target",
                    })
            elif self.task == "classification":
                vc = t.value_counts()
                imbalance_ratio = float(vc.iloc[0] / max(1, vc.iloc[-1]))
                if imbalance_ratio > 3.0:
                    self.findings.append({
                        "column": self.target_column,
                        "issue_type": "Class Imbalance Detected",
                        "severity": "WARNING",
                        "observed": f"Imbalance ratio {imbalance_ratio:.1f}:1 (Majority={vc.iloc[0]}, Minority={vc.iloc[-1]})",
                        "why_it_matters": "Standard models collapse toward majority class, yielding misleading accuracy.",
                        "recommendation": "Use StratifiedKFold, evaluate Macro-F1 / Balanced Accuracy, and enable balanced class weights.",
                        "drawbacks": "Class weighting slightly increases false positive rate for minority classes.",
                        "action_key": "enable_class_weights",
                    })
                rare = vc[vc < 5]
                if len(rare) > 0:
                    self.findings.append({
                        "column": self.target_column,
                        "issue_type": "Rare Classes (< 5 samples)",
                        "severity": "CRITICAL",
                        "observed": f"{len(rare)} classes have fewer than 5 instances: {dict(rare)}",
                        "why_it_matters": "Cannot reliably perform 5-fold cross validation with fewer than 5 instances per class.",
                        "recommendation": "Filter or consolidate rare classes into an 'Other' category.",
                        "drawbacks": "Consolidating classes alters the original taxonomy.",
                        "action_key": "consolidate_rare_classes",
                    })

        # Ensure dual keys for all findings for compatibility
        for f in self.findings:
            f.setdefault("issue", f.get("issue_type", ""))
            f.setdefault("evidence", f.get("observed", ""))
            f.setdefault("recommendations", f.get("recommendation", ""))
            f.setdefault("potential_drawbacks", f.get("drawbacks", ""))
            f.setdefault("action_keys", f.get("action_key", ""))

        return AuditReport(summary_df, self.findings)

    def to_dataframe(self, report: Optional[Any] = None) -> pd.DataFrame:
        """Converts audit findings to a DataFrame."""
        if report is not None:
            if isinstance(report, dict) and "findings" in report:
                findings = report["findings"]
            elif isinstance(report, list):
                findings = report
            else:
                findings = self.findings
        else:
            findings = self.findings
        return pd.DataFrame(findings)

    def decision_board(self) -> pd.DataFrame:
        """Returns structured DataFrame of actionable recommendations with 5-point explanations."""
        if not self.findings:
            return pd.DataFrame()
        return pd.DataFrame([
            {
                "Severity": f.get("severity", "INFO"),
                "Column": f.get("column", "All"),
                "Finding": f.get("issue_type", f.get("issue", "Quality Observation")),
                "Observed Metric": f.get("observed", f.get("evidence", "")),
                "Why It Matters": f.get("why_it_matters", ""),
                "Recommended Action": f.get("recommendation", f.get("recommendations", "")),
                "Potential Drawback": f.get("drawbacks", f.get("potential_drawbacks", "")),
                "Action Key": f.get("action_key", f.get("action_keys", "")),
            }
            for f in self.findings
        ])
