"""
================================================================================
  DATA PIPELINE: Ingestion → Cleaning → Transformation → Enrichment
================================================================================
  Supports CSV, JSON, Excel, and SQLite sources.
  Run:  python data_pipeline.py
================================================================================
"""

import os
import json
import logging
import sqlite3
import hashlib
from io import StringIO
from datetime import datetime
from typing import Any

import pandas as pd
import numpy as np

url = "amazon_reviews.csv"

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log"),
    ],
)
log = logging.getLogger("DataPipeline")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STAGE 1 – INGESTION                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class DataIngestion:
    """Load data from CSV, JSON, Excel, SQLite, or a raw dict/list."""

    SUPPORTED = (".csv", ".json", ".xlsx", ".xls", ".db", ".sqlite")

    def ingest(self, source: Any, **kwargs) -> pd.DataFrame:
        """
        Parameters
        ----------
        source : str | dict | list
            File path, SQLite connection string, or in-memory data.
        kwargs :
            Extra arguments forwarded to the reader.
            For SQLite pass  query="SELECT …"
        """
        if isinstance(source, (dict, list)):
            log.info("Ingesting from in-memory object.")
            return pd.DataFrame(source)

        if not isinstance(source, str):
            raise TypeError(f"Unsupported source type: {type(source)}")

        ext = os.path.splitext(source)[-1].lower()

        if ext == ".csv":
            return self._from_csv(source, **kwargs)
        elif ext == ".json":
            return self._from_json(source, **kwargs)
        elif ext in (".xlsx", ".xls"):
            return self._from_excel(source, **kwargs)
        elif ext in (".db", ".sqlite"):
            return self._from_sqlite(source, **kwargs)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    # ── readers ──────────────────────────────────────────────────────────────

    def _from_csv(self, path: str, **kw) -> pd.DataFrame:
        log.info(f"Reading CSV: {path}")
        df = pd.read_csv(path, **kw)
        log.info(f"  → {len(df):,} rows × {len(df.columns)} cols")
        return df

    def _from_json(self, path: str, **kw) -> pd.DataFrame:
        log.info(f"Reading JSON: {path}")
        with open(path) as f:
            data = json.load(f)
        df = pd.DataFrame(data if isinstance(data, list) else [data])
        log.info(f"  → {len(df):,} rows × {len(df.columns)} cols")
        return df

    def _from_excel(self, path: str, **kw) -> pd.DataFrame:
        log.info(f"Reading Excel: {path}")
        df = pd.read_excel(path, **kw)
        log.info(f"  → {len(df):,} rows × {len(df.columns)} cols")
        return df

    def _from_sqlite(self, path: str, query: str = "SELECT * FROM data", **kw) -> pd.DataFrame:
        log.info(f"Reading SQLite: {path}  |  query: {query}")
        with sqlite3.connect(path) as conn:
            df = pd.read_sql_query(query, conn)
        log.info(f"  → {len(df):,} rows × {len(df.columns)} cols")
        return df

    # ── demo helper ──────────────────────────────────────────────────────────

    @staticmethod
    def generate_sample() -> pd.DataFrame:
        """Return a realistic messy demo dataset (no file needed)."""
        np.random.seed(42)
        n = 200
        cities = ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", None]
        categories = ["Electronics", "Clothing", "Food", "Books", "Sports", "ELECTRONICS", "clothing"]

        return pd.DataFrame({
            "customer_id":  [f"C{i:04d}" for i in range(n)],
            "name":         np.random.choice(
                ["Alice Johnson", "  bob smith  ", "CAROL WHITE", "david LEE", None], n),
            "email":        [
                f"user{i}@example.com" if i % 10 != 0 else "BAD_EMAIL" for i in range(n)],
            "age":          np.where(
                np.random.rand(n) < 0.05, -1,
                np.random.randint(18, 80, n)),
            "city":         np.random.choice(cities, n),
            "category":     np.random.choice(categories, n),
            "purchase_amt": np.where(
                np.random.rand(n) < 0.05, None,
                np.round(np.random.exponential(500, n), 2)),
            "purchase_date": pd.date_range("2023-01-01", periods=n, freq="D").astype(str),
            "is_premium":   np.random.choice([0, 1, "yes", "no", None], n),
        })


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STAGE 2 – CLEANING                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class DataCleaning:
    """Fix common quality problems: nulls, duplicates, casing, outliers, types."""

    def __init__(self, null_threshold: float = 0.5):
        self.null_threshold = null_threshold   # drop column if > X% null

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        log.info("── STAGE 2: CLEANING ─────────────────────────────────────")
        df = df.copy()
        df = self._report_and_drop_high_null_cols(df)
        df = self._standardise_text(df)
        df = self._fix_types(df)
        df = self._handle_nulls(df)
        df = self._remove_duplicates(df)
        df = self._fix_outliers(df)
        df = self._validate_email(df)
        log.info(f"  Cleaned shape: {df.shape}")
        return df

    # ── steps ────────────────────────────────────────────────────────────────

    def _report_and_drop_high_null_cols(self, df):
        null_pct = df.isnull().mean()
        drop_cols = null_pct[null_pct > self.null_threshold].index.tolist()
        if drop_cols:
            log.warning(f"  Dropping cols with >{self.null_threshold*100:.0f}% nulls: {drop_cols}")
            df.drop(columns=drop_cols, inplace=True)
        return df

    def _standardise_text(self, df):
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].astype(str).str.strip()
            # title-case columns that look like names/categories
            if col.lower() in ("name", "city", "category"):
                df[col] = df[col].str.title()
            df[col].replace("None", np.nan, inplace=True)
            df[col].replace("nan", np.nan, inplace=True)
        return df

    def _fix_types(self, df):
        # dates
        for col in df.columns:
            if "date" in col.lower():
                df[col] = pd.to_datetime(df[col], errors="coerce")
        # boolean-ish columns
        for col in df.columns:
            if df[col].dropna().isin([0, 1, "0", "1", "yes", "no", "Yes", "No"]).all():
                df[col] = df[col].map(
                    {1: True, 0: False, "1": True, "0": False,
                     "yes": True, "no": False, "Yes": True, "No": False})
        return df

    def _handle_nulls(self, df):
        # numeric → median; object → mode
        for col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count == 0:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                fill = df[col].median()
                df[col].fillna(fill, inplace=True)
                log.info(f"  Filled {null_count} nulls in '{col}' with median={fill:.2f}")
            else:
                mode_vals = df[col].mode()
                if not mode_vals.empty:
                    df[col].fillna(mode_vals[0], inplace=True)
                    log.info(f"  Filled {null_count} nulls in '{col}' with mode='{mode_vals[0]}'")
        return df

    def _remove_duplicates(self, df):
        before = len(df)
        df.drop_duplicates(inplace=True)
        removed = before - len(df)
        if removed:
            log.info(f"  Removed {removed} duplicate rows.")
        return df

    def _fix_outliers(self, df):
        for col in df.select_dtypes(include=[np.number]).columns:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
            mask = (df[col] < lower) | (df[col] > upper)
            count = mask.sum()
            if count:
                df.loc[mask, col] = df[col].clip(lower, upper)
                log.info(f"  Clipped {count} outliers in '{col}'  [{lower:.2f}, {upper:.2f}]")
        return df

    def _validate_email(self, df):
        if "email" in df.columns:
            pattern = r"^[\w\.\+\-]+@[\w\-]+\.[a-zA-Z]{2,}$"
            invalid = ~df["email"].str.match(pattern, na=False)
            log.info(f"  Invalid emails flagged: {invalid.sum()}")
            df.loc[invalid, "email"] = np.nan
        return df


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STAGE 3 – TRANSFORMATION                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class DataTransformation:
    """Reshape, encode, scale, and derive new features."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        log.info("── STAGE 3: TRANSFORMATION ───────────────────────────────")
        df = df.copy()
        df = self._normalise_numeric(df)
        df = self._encode_categoricals(df)
        df = self._extract_datetime_features(df)
        df = self._create_derived_features(df)
        df = self._rename_columns(df)
        log.info(f"  Transformed shape: {df.shape}")
        return df

    # ── steps ────────────────────────────────────────────────────────────────

    def _normalise_numeric(self, df):
        skip = {"age"}   # keep age human-readable
        for col in df.select_dtypes(include=[np.number]).columns:
            if col in skip:
                continue
            mn, mx = df[col].min(), df[col].max()
            if mx != mn:
                df[f"{col}_norm"] = (df[col] - mn) / (mx - mn)
                log.info(f"  Min-max normalised: '{col}' → '{col}_norm'")
        return df

    def _encode_categoricals(self, df):
        low_card = [c for c in df.select_dtypes("object").columns
                    if df[c].nunique() <= 10]
        if low_card:
            df = pd.get_dummies(df, columns=low_card, drop_first=False, dtype=int)
            log.info(f"  One-hot encoded: {low_card}")
        return df

    def _extract_datetime_features(self, df):
        for col in df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns:
            df[f"{col}_year"]    = df[col].dt.year
            df[f"{col}_month"]   = df[col].dt.month
            df[f"{col}_day"]     = df[col].dt.day
            df[f"{col}_weekday"] = df[col].dt.day_name()
            df[f"{col}_quarter"] = df[col].dt.quarter
            log.info(f"  Extracted datetime features from '{col}'")
        return df

    def _create_derived_features(self, df):
        if "purchase_amt" in df.columns:
            # Safely convert to numeric by stripping currency symbols or commas if it's a string
            if df["purchase_amt"].dtype == object:
                df["purchase_amt"] = pd.to_numeric(
                    df["purchase_amt"].astype(str).str.replace(r'[^\d.-]', '', regex=True),
                    errors="coerce"
                )
            else:
                df["purchase_amt"] = pd.to_numeric(df["purchase_amt"], errors="coerce")
            
            # Ensure no negative values are passed to log1p
            valid_amt = np.where(df["purchase_amt"] >= 0, df["purchase_amt"], np.nan)
            df["log_purchase_amt"] = np.log1p(valid_amt)
            
            df["amt_bin"] = pd.cut(
                df["purchase_amt"],
                bins=[0, 100, 500, 1000, np.inf],
                labels=["Low", "Medium", "High", "VeryHigh"],
            )
            log.info("  Derived: log_purchase_amt, amt_bin")
        if "age" in df.columns:
            df["age_group"] = pd.cut(
                df["age"],
                bins=[0, 25, 40, 60, np.inf],
                labels=["Youth", "Adult", "MiddleAge", "Senior"],
            )
            log.info("  Derived: age_group")
        return df

    def _rename_columns(self, df):
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        return df


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STAGE 4 – ENRICHMENT                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class DataEnrichment:
    """Add metadata, scoring, hashing, and statistical summaries."""

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        log.info("── STAGE 4: ENRICHMENT ───────────────────────────────────")
        df = df.copy()
        df = self._add_record_id(df)
        df = self._add_customer_score(df)
        df = self._add_data_quality_flag(df)
        df = self._add_pipeline_metadata(df)
        log.info(f"  Enriched shape: {df.shape}")
        return df

    # ── steps ────────────────────────────────────────────────────────────────

    def _add_record_id(self, df):
        df.insert(0, "record_id",
                  [hashlib.md5(str(row).encode()).hexdigest()[:12]
                   for row in df.itertuples(index=False)])
        log.info("  Added: record_id (MD5 hash)")
        return df

    def _add_customer_score(self, df):
        score = pd.Series(50.0, index=df.index)
        if "purchase_amt" in df.columns:
            norm_amt = (df["purchase_amt"] - df["purchase_amt"].min()) / \
                       (df["purchase_amt"].max() - df["purchase_amt"].min() + 1e-9)
            score += norm_amt * 30
        if "is_premium" in df.columns:
            score += df["is_premium"].map({True: 20, False: 0}).fillna(0)
        if "age" in df.columns:
            score += np.where((df["age"] >= 30) & (df["age"] <= 55), 5, 0)
        df["customer_score"] = score.clip(0, 100).round(2)
        log.info("  Added: customer_score (0–100)")
        return df

    def _add_data_quality_flag(self, df):
        issues = pd.Series(0, index=df.index)
        for col in df.columns:
            if col.startswith("record_id") or col.startswith("pipeline_"):
                continue
            issues = issues + pd.Series(df[col].isnull(), dtype=int)  # type: ignore[operator]
        df["data_quality_flag"] = np.select(
            [issues == 0, issues <= 2, issues > 2],
            ["GOOD",       "WARN",     "POOR"],
            default="UNKNOWN"
        )
        log.info("  Added: data_quality_flag (GOOD / WARN / POOR)")
        return df

    def _add_pipeline_metadata(self, df):
        df["pipeline_version"]    = "1.0.0"
        df["pipeline_run_utc"]    = datetime.utcnow().isoformat(timespec="seconds")
        df["pipeline_row_count"]  = len(df)
        log.info("  Added: pipeline_version, pipeline_run_utc, pipeline_row_count")
        return df


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PIPELINE ORCHESTRATOR                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class DataPipeline:
    """
    Orchestrates all four stages and saves the result.

    Usage
    -----
    pipeline = DataPipeline()
    result   = pipeline.run(source="data.csv")
    # or
    result   = pipeline.run(source=my_dict_list)
    """

    def __init__(self, output_path: str = "pipeline_output.csv"):
        self.ingestion      = DataIngestion()
        self.cleaning       = DataCleaning()
        self.transformation = DataTransformation()
        self.enrichment     = DataEnrichment()
        self.output_path    = output_path

    def run(self, source: Any = None, **ingest_kwargs) -> pd.DataFrame:
        log.info("════════════════════════════════════════════════════════")
        log.info("              DATA PIPELINE  –  STARTING               ")
        log.info("════════════════════════════════════════════════════════")

        # ── 1. Ingest ─────────────────────────────────────────────────────
        log.info("── STAGE 1: INGESTION ────────────────────────────────────")
        if source is None:
            log.info("  No source provided → using built-in demo dataset.")
            raw = self.ingestion.generate_sample()
        else:
            raw = self.ingestion.ingest(source, **ingest_kwargs)
        self._snapshot("After Ingestion", raw)

        # ── 2. Clean ──────────────────────────────────────────────────────
        cleaned = self.cleaning.clean(raw)
        self._snapshot("After Cleaning", cleaned)

        # ── 3. Transform ──────────────────────────────────────────────────
        transformed = self.transformation.transform(cleaned)
        self._snapshot("After Transformation", transformed)

        # ── 4. Enrich ─────────────────────────────────────────────────────
        final = self.enrichment.enrich(transformed)
        self._snapshot("After Enrichment", final)

        # ── Save ──────────────────────────────────────────────────────────
        final.to_csv(self.output_path, index=False)
        log.info(f"\n✅  Pipeline complete. Output saved → {self.output_path}")
        log.info("════════════════════════════════════════════════════════\n")

        self._print_summary(final)
        return final

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _snapshot(label: str, df: pd.DataFrame):
        log.info(f"  [{label}] rows={len(df):,}  cols={len(df.columns)}")

    @staticmethod
    def _print_summary(df: pd.DataFrame):
        print("\n" + "─" * 60)
        print("  PIPELINE SUMMARY")
        print("─" * 60)
        print(f"  Total records   : {len(df):,}")
        print(f"  Total columns   : {len(df.columns)}")
        if "data_quality_flag" in df.columns:
            dist = df["data_quality_flag"].value_counts().to_dict()
            print(f"  Quality flags   : {dist}")
        if "customer_score" in df.columns:
            print(f"  Customer score  : "
                  f"min={df['customer_score'].min():.1f}  "
                  f"mean={df['customer_score'].mean():.1f}  "
                  f"max={df['customer_score'].max():.1f}")
        print("─" * 60 + "\n")
        print("  SAMPLE OUTPUT (first 5 rows):")
        preview_cols = ["record_id", "customer_id", "age",
                        "purchase_amt", "customer_score", "data_quality_flag"]
        show = [c for c in preview_cols if c in df.columns]
        with pd.option_context("display.max_columns", None, "display.width", 120):
            print(df[show].head().to_string(index=False))
        print()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ENTRY POINT                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    pipeline = DataPipeline(output_path="pipeline_output.csv")

    # ── Option A: built-in demo (no file needed) ──────────────────────────
    result = pipeline.run()

    # ── Option B: load from a file (uncomment one) ────────────────────────
    # result = pipeline.run("your_data.csv")
    # result = pipeline.run("your_data.json")
    # result = pipeline.run("your_data.xlsx")
    # result = pipeline.run("your_data.db", query="SELECT * FROM customers")
