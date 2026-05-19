"""Prepare ICD-10 datasets and ICD-10 code mappings for model training."""

import csv
from pathlib import Path

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "preprocessing"

GPU_DEVICE = torch.device("cuda:0")


def check_duplicates_in_df(
    df: pd.DataFrame,
    column_name: str,
) -> pd.DataFrame:
    """
    Check a DataFrame for duplicate entries in a selected column.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column_name (str): Name of the column to check.

    Returns:
        pd.DataFrame: DataFrame containing all duplicate rows.
    """
    return df[df.duplicated(subset=column_name, keep=False)]


def save_icd_mapping(
    mapping: dict,
    filename: str | Path,
    code_col_name: str = "ICD-10-Code",
) -> None:
    """
    Save an ICD-10-to-number mapping dictionary as a CSV file.

    Args:
        mapping (dict): Mapping from ICD-10 code to numeric ID.
        filename (str | Path): Output file path.
        code_col_name (str): Name of the ICD-10 code column.
    """
    df = pd.DataFrame(
        list(mapping.items()),
        columns=[code_col_name, "ICD-10-Num"],
    )

    df[code_col_name] = df[code_col_name].astype(str)
    df = df.sort_values(by="ICD-10-Num")
    df.to_csv(filename, sep=";", index=False)


def create_icd_dataframe(
    data: pd.DataFrame,
    label_column: str,
    code_column: str,
    mapping_file: str | Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Create a DataFrame with labels and numeric ICD-10 code IDs.

    Args:
        data (pd.DataFrame): Input data.
        label_column (str): Column containing text descriptions.
        code_column (str): Column containing ICD-10 codes.
        mapping_file (str | Path | None): Optional output path for the
            ICD-10-to-number mapping file.

    Returns:
        tuple[pd.DataFrame, dict]: Prepared DataFrame and mapping from
        numeric IDs to ICD-10 codes.
    """
    icd_data = (
        data[[label_column, code_column]]
        .rename(
            columns={
                label_column: "Label",
                code_column: "ICD-10-Code",
            }
        )
        .reset_index(drop=True)
    )

    icd_unique = icd_data["ICD-10-Code"].unique()
    icd_to_number = {code: idx for idx, code in enumerate(icd_unique)}

    icd_data["ICD-10-Num"] = icd_data["ICD-10-Code"].map(icd_to_number)
    icd_data = icd_data.sort_values(by="ICD-10-Num").reset_index(drop=True)

    number_to_icd = {value: key for key, value in icd_to_number.items()}

    if mapping_file is not None:
        save_icd_mapping(icd_to_number, mapping_file)

    return icd_data, number_to_icd


def prep_alpha_icd10_dataset(
    output_dir: str | Path,
    file_name: str,
    icd_raw_data: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Prepare the Alpha-ID ICD-10 dataset for training.

    Args:
        output_dir (str | Path): Output directory.
        file_name (str): Dataset name used for unique output file names.
        icd_raw_data (pd.DataFrame): Input data with text and ICD-10 columns.

    Returns:
        tuple[pd.DataFrame, dict]: Prepared DataFrame and mapping from
        numeric IDs to ICD-10 codes.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = Path(file_name).stem

    df = pd.DataFrame(
        {
            "Label": icd_raw_data["Label"],
            "ICD-10-Code": icd_raw_data["ICD-10-Code"],
        }
    )
    df["Label"] = df["Label"].str.strip('"')

    icd_df, mapping = create_icd_dataframe(
        df,
        "Label",
        "ICD-10-Code",
        output_dir / f"{dataset_name}_icd10_mapping.csv",
    )

    icd_df.to_csv(
        output_dir / f"{dataset_name}_icd10_prepared.csv",
        sep=";",
        index=False,
    )

    return icd_df, mapping


def main() -> None:
    """Read the Alpha-ID ICD-10 file, prepare the dataset, and save outputs."""
    input_file = DATA_DIR / "Alpha_ID_dataset.csv"

    try:
        icd_raw_data = pd.read_csv(
            input_file,
            encoding="utf-8",
            sep=";",
            engine="python",
        )
    except Exception as error:
        print(f"Error while reading the file: {error}")
        return

    icd_df, mapping = prep_alpha_icd10_dataset(
        RESULTS_DIR,
        input_file.name,
        icd_raw_data,
    )

    print("ICD-10 dataset was created successfully.")
    print(f"Input file: {input_file}")
    print(f"Output directory: {RESULTS_DIR}")
    print(f"Number of entries: {len(icd_df)}")
    print(f"Example ICD-10 codes: {list(mapping.items())[:5]}")


if __name__ == "__main__":
    main()