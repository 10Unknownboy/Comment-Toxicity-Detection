"""
Data Insight Report Generator for Comment Toxicity Detection.

This script loads the training dataset, performs comprehensive analysis
across multiple dimensions, and writes a detailed, well-formatted report
to datainsight.txt in the project root.

Requires: pandas, numpy
"""

import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
LABEL_COLUMNS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]
SEPARATOR_WIDTH = 80
SECTION_SEP = "=" * SEPARATOR_WIDTH
SUBSECTION_SEP = "-" * SEPARATOR_WIDTH

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "train.csv"
OUTPUT_PATH = PROJECT_ROOT / "datainsight.txt"


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def progress(message):
    """Print a timestamped progress message to the console."""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")


def section_header(title, order):
    """Return a formatted section header string."""
    lines = [
        "",
        SECTION_SEP,
        f"  {order}. {title}",
        SECTION_SEP,
        "",
    ]
    return "\n".join(lines)


def subsection_header(title):
    """Return a formatted subsection header string."""
    lines = [
        "",
        SUBSECTION_SEP,
        f"  {title}",
        SUBSECTION_SEP,
        "",
    ]
    return "\n".join(lines)


def truncate_text(text, max_length=100):
    """Truncate text to a maximum length, appending an ellipsis if needed."""
    if not isinstance(text, str):
        return str(text)
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------


def build_header():
    """Build the report header with title and timestamp."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        SECTION_SEP,
        "  Data Insight Report - Comment Toxicity Detection",
        SECTION_SEP,
        "",
        f"  Generated on : {now}",
        f"  Data source  : {DATA_PATH}",
        "",
        SECTION_SEP,
    ]
    return "\n".join(lines)


def analyse_overview(df):
    """Section 1 -- Dataset Overview: shape, columns, memory usage."""
    progress("Analysing dataset overview ...")
    rows, cols = df.shape
    mem_usage = df.memory_usage(deep=True)
    total_mem_mb = mem_usage.sum() / (1024 * 1024)

    lines = [section_header("Dataset Overview", 1)]
    lines.append(f"  Number of rows    : {rows:,}")
    lines.append(f"  Number of columns : {cols}")
    lines.append("")
    lines.append("  Columns:")
    for idx, col in enumerate(df.columns, start=1):
        lines.append(f"    {idx:>3}. {col}")
    lines.append("")
    lines.append(f"  Total memory usage : {total_mem_mb:.2f} MB")
    lines.append("")
    lines.append("  Memory per column:")
    for col in df.columns:
        col_mem = mem_usage[col] / 1024
        lines.append(f"    {col:<20s} : {col_mem:>10.2f} KB")
    return "\n".join(lines)


def analyse_dtypes(df):
    """Section 2 -- Data Types for each column."""
    progress("Analysing data types ...")
    lines = [section_header("Data Types", 2)]
    for col in df.columns:
        lines.append(f"    {col:<20s} : {df[col].dtype}")
    return "\n".join(lines)


def analyse_missing(df):
    """Section 3 -- Missing Values: count and percentage per column."""
    progress("Analysing missing values ...")
    lines = [section_header("Missing Values", 3)]
    total_cells = df.shape[0] * df.shape[1]
    total_missing = 0

    lines.append(f"  {'Column':<20s}  {'Missing':>10s}  {'Percentage':>10s}")
    lines.append(f"  {'-' * 20}  {'-' * 10}  {'-' * 10}")

    for col in df.columns:
        missing = df[col].isna().sum()
        pct = (missing / df.shape[0]) * 100
        total_missing += missing
        lines.append(f"  {col:<20s}  {missing:>10,}  {pct:>9.2f}%")

    lines.append("")
    lines.append(f"  Total cells          : {total_cells:,}")
    lines.append(f"  Total missing cells  : {total_missing:,}")
    pct_total = (total_missing / total_cells) * 100 if total_cells else 0
    lines.append(f"  Overall missing rate : {pct_total:.4f}%")
    return "\n".join(lines)


def analyse_duplicates(df):
    """Section 4 -- Duplicate Rows."""
    progress("Analysing duplicate rows ...")
    dup_count = df.duplicated().sum()
    dup_pct = (dup_count / df.shape[0]) * 100

    lines = [section_header("Duplicate Rows", 4)]
    lines.append(f"  Exact duplicate rows : {dup_count:,}")
    lines.append(f"  Percentage           : {dup_pct:.4f}%")
    return "\n".join(lines)


def analyse_class_distribution(df):
    """Section 5 -- Class Distribution for each toxicity label."""
    progress("Analysing class distribution ...")
    lines = [section_header("Class Distribution", 5)]

    for label in LABEL_COLUMNS:
        if label not in df.columns:
            lines.append(f"  [WARNING] Column '{label}' not found in dataset.")
            continue

        pos = int(df[label].sum())
        neg = int(df.shape[0] - pos)
        pct_pos = (pos / df.shape[0]) * 100
        ratio = f"{neg}:{pos}" if pos > 0 else "N/A (no positives)"

        lines.append(subsection_header(label))
        lines.append(f"    Positive (1) : {pos:>10,}  ({pct_pos:.2f}%)")
        lines.append(f"    Negative (0) : {neg:>10,}  ({100 - pct_pos:.2f}%)")
        lines.append(f"    Neg:Pos ratio: {ratio}")

    return "\n".join(lines)


def analyse_multilabel(df):
    """Section 6 -- Multi-Label Statistics."""
    progress("Analysing multi-label statistics ...")
    available = [c for c in LABEL_COLUMNS if c in df.columns]
    label_counts = df[available].sum(axis=1)

    lines = [section_header("Multi-Label Statistics", 6)]
    lines.append("  Distribution of label counts per comment:")
    lines.append("")
    lines.append(f"  {'Labels':>8s}  {'Count':>10s}  {'Percentage':>10s}")
    lines.append(f"  {'-' * 8}  {'-' * 10}  {'-' * 10}")

    for n_labels in range(len(available) + 1):
        count = int((label_counts == n_labels).sum())
        pct = (count / df.shape[0]) * 100
        lines.append(f"  {n_labels:>8d}  {count:>10,}  {pct:>9.2f}%")

    clean = int((label_counts == 0).sum())
    toxic = int((label_counts > 0).sum())
    lines.append("")
    lines.append(f"  Completely clean (0 labels) : {clean:>10,}  ({clean / df.shape[0] * 100:.2f}%)")
    lines.append(f"  At least one toxic label    : {toxic:>10,}  ({toxic / df.shape[0] * 100:.2f}%)")
    return "\n".join(lines)


def analyse_comment_length(df):
    """Section 7 -- Comment Length Analysis."""
    progress("Analysing comment lengths ...")
    if "comment_text" not in df.columns:
        return section_header("Comment Length Analysis", 7) + "\n  [WARNING] 'comment_text' column not found."

    lengths = df["comment_text"].fillna("").str.len()

    lines = [section_header("Comment Length Analysis", 7)]
    lines.append(f"  Minimum length       : {int(lengths.min()):,} chars")
    lines.append(f"  Maximum length       : {int(lengths.max()):,} chars")
    lines.append(f"  Mean length          : {lengths.mean():,.2f} chars")
    lines.append(f"  Median length        : {lengths.median():,.2f} chars")
    lines.append(f"  Standard deviation   : {lengths.std():,.2f} chars")

    lines.append("")
    lines.append("  Percentiles:")
    for p in [25, 50, 75, 90, 95, 99]:
        val = lengths.quantile(p / 100)
        lines.append(f"    {p:>3d}th percentile : {val:>10,.0f} chars")

    short = int((lengths < 10).sum())
    long_ = int((lengths > 5000).sum())
    lines.append("")
    lines.append(f"  Very short comments (< 10 chars)   : {short:,}")
    lines.append(f"  Very long comments  (> 5000 chars)  : {long_:,}")
    return "\n".join(lines)


def analyse_statistics(df):
    """Section 8 -- Basic Statistical Summary via describe()."""
    progress("Generating basic statistical summary ...")
    numeric_desc = df.describe().to_string()

    lines = [section_header("Basic Statistical Summary", 8)]
    lines.append(numeric_desc)
    return "\n".join(lines)


def analyse_correlation(df):
    """Section 9 -- Label Correlation Matrix (Pearson)."""
    progress("Computing label correlation matrix ...")
    available = [c for c in LABEL_COLUMNS if c in df.columns]

    lines = [section_header("Label Correlation Matrix", 9)]

    if len(available) < 2:
        lines.append("  Not enough label columns to compute correlations.")
        return "\n".join(lines)

    corr = df[available].corr(method="pearson")

    # Header row
    header = "  " + " " * 16
    for col in available:
        header += f"{col:>14s}"
    lines.append(header)
    lines.append("  " + "-" * (16 + 14 * len(available)))

    # Data rows
    for row_label in available:
        row_str = f"  {row_label:<16s}"
        for col_label in available:
            row_str += f"{corr.loc[row_label, col_label]:>14.4f}"
        lines.append(row_str)

    return "\n".join(lines)


def analyse_samples(df):
    """Section 10 -- Sample Records: first rows, toxic examples, clean examples."""
    progress("Selecting sample records ...")
    rng = np.random.RandomState(RANDOM_SEED)
    available = [c for c in LABEL_COLUMNS if c in df.columns]

    lines = [section_header("Sample Records", 10)]

    # --- First 3 rows ---
    lines.append(subsection_header("First 3 Rows"))
    for idx in range(min(3, len(df))):
        row = df.iloc[idx]
        lines.append(f"  Row {idx}:")
        for col in df.columns:
            val = row[col]
            if col == "comment_text":
                val = truncate_text(val, 100)
            lines.append(f"    {col:<20s} : {val}")
        lines.append("")

    # --- 3 random toxic examples ---
    lines.append(subsection_header("3 Random Toxic Examples (toxic=1)"))
    if "toxic" in df.columns:
        toxic_df = df[df["toxic"] == 1]
        if len(toxic_df) >= 3:
            sample_idx = rng.choice(toxic_df.index, size=3, replace=False)
            for i, idx in enumerate(sample_idx, start=1):
                row = df.loc[idx]
                lines.append(f"  Example {i} (index={idx}):")
                for col in df.columns:
                    val = row[col]
                    if col == "comment_text":
                        val = truncate_text(val, 100)
                    lines.append(f"    {col:<20s} : {val}")
                lines.append("")
        else:
            lines.append(f"  Only {len(toxic_df)} toxic rows found; fewer than 3.")
    else:
        lines.append("  [WARNING] 'toxic' column not found.")

    # --- 3 random clean examples ---
    lines.append(subsection_header("3 Random Clean Examples (all labels=0)"))
    if available:
        clean_mask = df[available].sum(axis=1) == 0
        clean_df = df[clean_mask]
        if len(clean_df) >= 3:
            sample_idx = rng.choice(clean_df.index, size=3, replace=False)
            for i, idx in enumerate(sample_idx, start=1):
                row = df.loc[idx]
                lines.append(f"  Example {i} (index={idx}):")
                for col in df.columns:
                    val = row[col]
                    if col == "comment_text":
                        val = truncate_text(val, 100)
                    lines.append(f"    {col:<20s} : {val}")
                lines.append("")
        else:
            lines.append(f"  Only {len(clean_df)} clean rows found; fewer than 3.")
    else:
        lines.append("  [WARNING] No label columns found.")

    return "\n".join(lines)


def analyse_anomalies(df):
    """Section 11 -- Anomalies and Patterns."""
    progress("Detecting anomalies and patterns ...")
    available = [c for c in LABEL_COLUMNS if c in df.columns]

    lines = [section_header("Anomalies and Patterns", 11)]

    # --- Comments with ALL 6 labels positive ---
    lines.append(subsection_header("Comments with ALL labels positive"))
    if len(available) == len(LABEL_COLUMNS):
        all_pos = df[available].sum(axis=1) == len(available)
        count_all = int(all_pos.sum())
        lines.append(f"  Count : {count_all:,}")
        if count_all > 0:
            lines.append("")
            examples = df[all_pos].head(3)
            for idx, row in examples.iterrows():
                text = truncate_text(row.get("comment_text", ""), 100)
                lines.append(f"  index={idx} : {text}")
    else:
        lines.append("  [WARNING] Not all 6 label columns are present.")

    # --- Empty or NaN comment texts ---
    lines.append(subsection_header("Empty or NaN Comment Texts"))
    if "comment_text" in df.columns:
        nan_comments = df["comment_text"].isna().sum()
        empty_comments = int((df["comment_text"].fillna("").str.strip() == "").sum())
        lines.append(f"  NaN comment_text   : {nan_comments:,}")
        lines.append(f"  Empty (blank) text : {empty_comments:,}")
    else:
        lines.append("  [WARNING] 'comment_text' column not found.")

    # --- Extremely long comments (top 5) ---
    lines.append(subsection_header("Top 5 Longest Comments"))
    if "comment_text" in df.columns:
        lengths = df["comment_text"].fillna("").str.len()
        top5_idx = lengths.nlargest(5).index
        for rank, idx in enumerate(top5_idx, start=1):
            length = int(lengths.loc[idx])
            text = truncate_text(df.loc[idx, "comment_text"], 80)
            lines.append(f"  #{rank}  index={idx}  length={length:,} chars")
            lines.append(f"       preview: {text}")
            lines.append("")
    else:
        lines.append("  [WARNING] 'comment_text' column not found.")

    # --- Unusual patterns ---
    lines.append(subsection_header("Unusual Patterns"))
    notes = []

    if available:
        label_sums = df[available].sum()
        min_label = label_sums.idxmin()
        max_label = label_sums.idxmax()
        notes.append(
            f"Most frequent label  : '{max_label}' with {int(label_sums[max_label]):,} positives"
        )
        notes.append(
            f"Least frequent label : '{min_label}' with {int(label_sums[min_label]):,} positives"
        )

        # Check for severe_toxic without toxic
        if "toxic" in df.columns and "severe_toxic" in df.columns:
            severe_not_toxic = int(
                ((df["severe_toxic"] == 1) & (df["toxic"] == 0)).sum()
            )
            notes.append(
                f"severe_toxic=1 but toxic=0 : {severe_not_toxic:,} rows"
            )

        # Check for single-label rows
        single_label = int((df[available].sum(axis=1) == 1).sum())
        notes.append(f"Rows with exactly 1 label  : {single_label:,}")

    if "comment_text" in df.columns:
        lengths = df["comment_text"].fillna("").str.len()
        one_char = int((lengths == 1).sum())
        if one_char > 0:
            notes.append(f"Comments with exactly 1 character : {one_char:,}")

    if not notes:
        notes.append("No unusual patterns detected.")

    for note in notes:
        lines.append(f"  - {note}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def generate_report():
    """Load data, run all analyses, and write the report file."""
    progress("Starting Data Insight Report generation ...")

    # --- Check data file exists ---
    if not DATA_PATH.exists():
        print(f"[ERROR] Data file not found: {DATA_PATH}")
        print("Please ensure 'data/train.csv' exists relative to the project root.")
        sys.exit(1)

    # --- Load data ---
    progress(f"Loading data from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    progress(f"Loaded {len(df):,} rows and {len(df.columns)} columns.")

    # --- Run all analysis sections ---
    sections = [
        build_header(),
        analyse_overview(df),
        analyse_dtypes(df),
        analyse_missing(df),
        analyse_duplicates(df),
        analyse_class_distribution(df),
        analyse_multilabel(df),
        analyse_comment_length(df),
        analyse_statistics(df),
        analyse_correlation(df),
        analyse_samples(df),
        analyse_anomalies(df),
    ]

    # --- Footer ---
    footer_lines = [
        "",
        SECTION_SEP,
        "  End of Report",
        SECTION_SEP,
        "",
    ]
    sections.append("\n".join(footer_lines))

    # --- Write report ---
    report = "\n".join(sections)
    OUTPUT_PATH.write_text(report, encoding="utf-8")
    progress(f"Report written to {OUTPUT_PATH}")
    progress("Done.")


if __name__ == "__main__":
    generate_report()
















