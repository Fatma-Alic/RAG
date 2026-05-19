"""Create top-20 ICD-10 and ICD-O distribution plots."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

DPI = 300

# Keep text editable in SVG/PDF.
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

# Keep previous font-size settings.
mpl.rcParams["font.size"] = 10
mpl.rcParams["axes.titlesize"] = 10
mpl.rcParams["axes.labelsize"] = 10
mpl.rcParams["xtick.labelsize"] = 8
mpl.rcParams["ytick.labelsize"] = 8


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

DATA_DIR = Path("/home/alic/RAG/data")
OUTPUT_DIR = Path("/home/alic/RAG/results/figures/plots_top20")

ALPHA_PATH = DATA_DIR / "Alpha_ID_dataset.csv"
ICDO_PATH = DATA_DIR / "ICDO3_LE_dataset.csv"
GTDS_PATH = DATA_DIR / "fully_cleaned_gtds_just_2023_2024_AcronymsExtended.csv"

OUTPUT_PNG = OUTPUT_DIR / "top20_4panel.png"
OUTPUT_SVG = OUTPUT_DIR / "top20_4panel.svg"


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def read_csv_auto(path: Path) -> pd.DataFrame:
    """
    Read a CSV file with automatic delimiter detection.

    This works for semicolon-separated and comma-separated CSV files.
    All columns are read as strings to avoid unwanted type conversions.

    Args:
        path (Path): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded CSV file.
    """
    return pd.read_csv(
        path,
        sep=None,
        engine="python",
        dtype=str,
        keep_default_na=False,
    )


def detect_code_column(df: pd.DataFrame) -> str:
    """
    Detect an ICD-10 or ICD-O code column.

    The function checks common column-name variants and is case-insensitive.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        str: Detected code column name.

    Raises:
        ValueError: If no ICD or ICD-O code column is found.
    """
    candidates = [
        "ICD-10-Code",
        "ICD-10",
        "ICD10",
        "ICD-O-Code",
        "ICDO",
    ]

    lower_map = {column.lower(): column for column in df.columns}

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

        candidate_lower = candidate.lower()
        if candidate_lower in lower_map:
            return lower_map[candidate_lower]

    raise ValueError(
        "No ICD/ICD-O code column found. "
        f"Available columns: {list(df.columns)}"
    )


def normalize_codes(series: pd.Series) -> pd.Series:
    """
    Normalize code values.

    The normalization steps are:
    - convert values to strings
    - trim whitespace
    - replace duplicated whitespace
    - convert values to uppercase
    - remove empty values

    Args:
        series (pd.Series): Series containing code values.

    Returns:
        pd.Series: Normalized code values.
    """
    normalized = (
        series.astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.upper()
    )

    return normalized[normalized != ""]


def load_top20(path: Path, code_column: str | None = None) -> pd.Series:
    """
    Load a dataset and return the 20 most frequent code values.

    Args:
        path (Path): Path to the dataset.
        code_column (str | None): Optional code column name.

    Returns:
        pd.Series: Top 20 code counts.
    """
    df = read_csv_auto(path)

    if code_column is None:
        code_column = detect_code_column(df)

    counts = normalize_codes(df[code_column]).value_counts()

    return counts.head(20)


def detect_gtds_column(df: pd.DataFrame, code_type: str) -> str:
    """
    Detect either the ICD-10 or ICD-O column in the GTDS dataset.

    Args:
        df (pd.DataFrame): GTDS DataFrame.
        code_type (str): Either "icd10" or "icdo".

    Returns:
        str: Detected GTDS column name.

    Raises:
        ValueError: If the code type is invalid or no matching column is found.
    """
    keyword_map = {
        "icd10": ["ICD-10", "ICD10"],
        "icdo": ["ICD-O", "ICDO"],
    }

    if code_type not in keyword_map:
        raise ValueError("code_type must be either 'icd10' or 'icdo'.")

    for column in df.columns:
        column_upper = column.upper()
        if any(keyword in column_upper for keyword in keyword_map[code_type]):
            return column

    raise ValueError(
        f"No {code_type} column found in GTDS. "
        f"Available columns: {list(df.columns)}"
    )


def plot_horizontal_top20(
    ax: plt.Axes,
    series: pd.Series,
    title: str,
    ylabel: str,
) -> None:
    """
    Plot a horizontal top-20 bar chart.

    The largest values appear at the top.

    Args:
        ax (plt.Axes): Matplotlib axes object.
        series (pd.Series): Series containing code counts.
        title (str): Plot title.
        ylabel (str): Label for the y-axis.
    """
    sorted_series = series.sort_values(ascending=True)

    ax.barh(
        sorted_series.index,
        sorted_series.values,
        height=0.6,
        edgecolor="black",
    )

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Frequency")
    ax.set_ylabel(ylabel)
    ax.set_box_aspect(1)

    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)


# -----------------------------------------------------------------------------
# Main figure function
# -----------------------------------------------------------------------------

def make_four_panel_top20(
    alpha_path: Path,
    icdo_catalog_path: Path,
    gtds_path: Path,
    output_png: Path,
    output_svg: Path | None = None,
    gtds_icd10_col: str | None = None,
    gtds_icdo_col: str | None = None,
    dpi: int = DPI,
) -> None:
    """
    Create one 2x2 panel figure.

    Panels:
    - Training dataset ICD-10
    - Test dataset ICD-10
    - Training dataset ICD-O
    - Test dataset ICD-O

    Outputs:
    - PNG with selected DPI
    - SVG as vector graphic, if requested

    Args:
        alpha_path (Path): Path to the Alpha-ID dataset.
        icdo_catalog_path (Path): Path to the ICD-O catalog dataset.
        gtds_path (Path): Path to the GTDS test dataset.
        output_png (Path): Output path for the PNG file.
        output_svg (Path | None): Optional output path for the SVG file.
        gtds_icd10_col (str | None): Optional ICD-10 column name in GTDS.
        gtds_icdo_col (str | None): Optional ICD-O column name in GTDS.
        dpi (int): Resolution of the PNG output.
    """
    top20_alpha_icd10 = load_top20(alpha_path)
    top20_icdo_catalog = load_top20(icdo_catalog_path)

    df_gtds = read_csv_auto(gtds_path)

    if gtds_icd10_col is not None:
        if gtds_icd10_col not in df_gtds.columns:
            raise ValueError(f"Column '{gtds_icd10_col}' not found in GTDS.")
        col_icd10 = gtds_icd10_col
    else:
        col_icd10 = detect_gtds_column(df_gtds, "icd10")

    if gtds_icdo_col is not None:
        if gtds_icdo_col not in df_gtds.columns:
            raise ValueError(f"Column '{gtds_icdo_col}' not found in GTDS.")
        col_icdo = gtds_icdo_col
    else:
        col_icdo = detect_gtds_column(df_gtds, "icdo")

    top20_gtds_icd10 = (
        normalize_codes(df_gtds[col_icd10])
        .value_counts()
        .head(20)
    )

    top20_gtds_icdo = (
        normalize_codes(df_gtds[col_icdo])
        .value_counts()
        .head(20)
    )

    # Journal-size figure.
    fig_width_cm = 17.0
    fig_height_cm = 15.9

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(fig_width_cm / 2.54, fig_height_cm / 2.54),
        constrained_layout=True,
    )

    fig.suptitle(
        "Top-20 ICD-10 and ICD-O code distributions in "
        "training and test datasets",
        fontsize=12,
        fontweight="bold",
    )

    plot_horizontal_top20(
        axes[0, 0],
        top20_alpha_icd10,
        "Training dataset (ICD-10)",
        "ICD-10 codes",
    )

    plot_horizontal_top20(
        axes[0, 1],
        top20_gtds_icd10,
        "Test dataset (ICD-10)",
        "ICD-10 codes",
    )

    plot_horizontal_top20(
        axes[1, 0],
        top20_icdo_catalog,
        "Training dataset (ICD-O)",
        "ICD-O codes",
    )

    plot_horizontal_top20(
        axes[1, 1],
        top20_gtds_icdo,
        "Test dataset (ICD-O)",
        "ICD-O codes",
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")

    if output_svg is not None:
        fig.savefig(output_svg, format="svg", bbox_inches="tight")

    plt.close(fig)

    print(f"PNG saved: {output_png}")

    if output_svg is not None:
        print(f"SVG saved: {output_svg}")


def main() -> None:
    """Run figure generation."""
    make_four_panel_top20(
        alpha_path=ALPHA_PATH,
        icdo_catalog_path=ICDO_PATH,
        gtds_path=GTDS_PATH,
        output_png=OUTPUT_PNG,
        output_svg=OUTPUT_SVG,
        dpi=DPI,
    )


if __name__ == "__main__":
    main()