"""Create an ICD-O mapping file with cleaned prompt text."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path("/home/alic/RAG")
DATA_DIR = PROJECT_ROOT / "data"

IN_CSV = DATA_DIR / "ICDO3_LE_dataset.csv"
OUT_PATH = DATA_DIR / "ICDO_mapping.csv"

SEP_IN = ";"
SEP_OUT = ";"


def _normalize_for_contains(text: str) -> str:
    """
    Normalize text for robust substring checks.

    The normalization steps are:
    - casefold text
    - replace punctuation and separators with spaces
    - replace multiple spaces with one space

    Args:
        text (str): Input text.

    Returns:
        str: Normalized text.
    """
    if text is None:
        return ""

    normalized = str(text).casefold()
    normalized = re.sub(r"[(){}\[\],;:–—\-_/\\|]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    return normalized


def _contains_topography(base_text: str, topo_text: str) -> bool:
    """
    Check whether the topography text already occurs in the base text.

    Args:
        base_text (str): Main diagnosis text.
        topo_text (str): Topography text.

    Returns:
        bool: True if the topography text is already contained in the base text.
    """
    normalized_base = _normalize_for_contains(base_text)
    normalized_topography = _normalize_for_contains(topo_text)

    return bool(normalized_topography) and (
        normalized_topography in normalized_base
    )


def _remove_sonstige_lokalisationen(text: str) -> str:
    """
    Remove 'sonstige Lokalisation' or 'sonstige Lokalisationen' from text.

    The phrase is removed if it appears at the end of the text or directly
    before an opening parenthesis.

    Args:
        text (str): Input text.

    Returns:
        str: Cleaned text.
    """
    if text is None:
        return ""

    cleaned_text = str(text)

    cleaned_text = re.sub(
        r"[\s,;:–—\-]*\bsonstige\s+lokalisation(?:en)?\b\s*(?=\()",
        "",
        cleaned_text,
        flags=re.IGNORECASE,
    )

    cleaned_text = re.sub(
        r"[\s,;:–—\-]*\bsonstige\s+lokalisation(?:en)?\b\s*$",
        "",
        cleaned_text,
        flags=re.IGNORECASE,
    )

    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

    return cleaned_text


def _replace_ona(text: str) -> str:
    """
    Replace variants of 'o.n.A.' with 'ohne nähere Angabe'.

    Args:
        text (str): Input text.

    Returns:
        str: Text with expanded 'o.n.A.' abbreviation.
    """
    if text is None:
        return ""

    cleaned_text = str(text)
    cleaned_text = re.sub(
        r"\bo\.?\s*n\.?\s*a\.?\b\.?",
        "ohne nähere Angabe",
        cleaned_text,
        flags=re.IGNORECASE,
    )

    return cleaned_text


def format_text_with_topo(base_text: str, topo_text: str) -> str:
    """
    Combine base text with optional topography text.

    The same cleaning rules are applied as in the previous ICD-O dataset
    creation script:
    - remove 'sonstige Lokalisation(en)'
    - replace 'o.n.A.' with 'ohne nähere Angabe'
    - append the topography text only if it is not already contained in the
      base text

    Args:
        base_text (str): Main diagnosis text.
        topo_text (str): Topography text.

    Returns:
        str: Cleaned and combined prompt text.
    """
    base = _remove_sonstige_lokalisationen(base_text)
    base = _replace_ona(base)

    topo = "" if topo_text is None else str(topo_text).strip()

    if topo and not _contains_topography(base, topo):
        output_text = f"{base} ({topo})".strip()
    else:
        output_text = base.strip()

    output_text = _remove_sonstige_lokalisationen(output_text)
    output_text = _replace_ona(output_text)

    return output_text


def main() -> None:
    """Load the ICD-O dataset, create prompt text, and save the mapping file."""
    df = pd.read_csv(IN_CSV, sep=SEP_IN, dtype=str, engine="python")

    required_columns = ["ICD-10 Text", "Topographie Text"]
    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )

    df["ICD-10 Text"] = (
        df["ICD-10 Text"].fillna("").astype(str).str.strip()
    )
    df["Topographie Text"] = (
        df["Topographie Text"].fillna("").astype(str).str.strip()
    )

    df["prompttext"] = df.apply(
        lambda row: format_text_with_topo(
            row["ICD-10 Text"],
            row["Topographie Text"],
        ),
        axis=1,
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, sep=SEP_OUT, index=False)

    print(f"Saved: {OUT_PATH}")
    print(f"Rows: {len(df)}")


if __name__ == "__main__":
    main()