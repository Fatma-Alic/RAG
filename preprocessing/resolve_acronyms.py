"""Expand medical acronyms in GTDS text data and save the extended CSV file."""

import re

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# Download the stopword lists required by NLTK.
nltk.download("stopwords")

# Load German and English stopwords.
GERMAN_STOPS = set(stopwords.words("german"))
ENGLISH_STOPS = set(stopwords.words("english"))
COMBINED_STOPS = GERMAN_STOPS.union(ENGLISH_STOPS)

# Define additional medically relevant stopwords.
CUSTOM_STOPWORDS = {
    "nhl", "ia", "std", "arm", "tag", "iva", "ivb", "b", "zns", "nase",
    "ol", "ml", "a", "e", "ul", "grade", "g2", "re", "li", "who", "bds",
    "nos", "nst", "v", "iv", "-ca", "p", "p16", "aeg", "iia", "u", "d3",
    "gl", "nnh", "met", "ii", "os", "us", "ed", "gi", "lk", "l", "kl",
    "crc",
}
STOPS = COMBINED_STOPS.union(CUSTOM_STOPWORDS)

# Define custom medical abbreviations and their meanings.
CUSTOM_REPLACEMENTS = {
    "OL": "Oberlappen",
    "UL": "Unterlappen",
    "ML": "Mittellappen",
    "OA": "Oberarm",
    "UA": "Unterarm",
    "NNH": "Nasennebenhöhlen",
    "li": "links",
    "re": "rechts",
    "met": "metastasiert",
    "OS": "Oberschenkel",
    "US": "Unterschenkel",
    "ED": "Erstdiagnose",
    "LK": "Lungenkarzinom",
    "kl": "klein",
    "CRC": "colorectal cancer",
    "BWS": "Brustwirbelsäule",
    "ZNS": "Zentralnervensystem",
    "HPA": "hypothalamic pituitary adrenal",
    "IDH": "isocitratdehydrogenase",
    "LWS": "Lendenwirbelsäule",
    "CS": "clinical stage",
    "o.n.A": "ohne nähere Angabe",
    "a.n.k": "anderweitig nicht klassifiziert",
}

# Load the embedding model.
model = SentenceTransformer(
    "jinaai/jina-embeddings-v2-base-de",
    trust_remote_code=True,
)

# Load the GTDS dataset.
query_data = pd.read_csv(
    (
        "/srv/llms/llit/GTDS-Dateien/"
        "fully_cleaned_gtds_just_2023_2024.csv"
    ),
    sep=";",
)

# Load the acronym inventory.
acronym_data = pd.read_csv(
    "/home/alic/ChromaDB/Metainventory_Version1.0.0.csv",
    sep="|",
    usecols=["RecordID", "SF", "LF", "NormLF"],
)
acronym_data["SF"] = acronym_data["SF"].astype(str)


def find_best_match(query: str, corpus: list[str]) -> str:
    """
    Find the best semantic match from a list of possible long forms.

    Parameters:
        query (str): The original text.
        corpus (list[str]): List of possible long forms for one abbreviation.

    Returns:
        str: The most suitable long form.
    """
    query_embedding = model.encode(f"Tumordiagnose: {query}")
    corpus_embeddings = model.encode(corpus)
    similarities = util.cos_sim(query_embedding, corpus_embeddings)
    best_index = np.argmax(similarities.numpy())

    return corpus[best_index]


def process_acronyms(row: pd.Series) -> str:
    """
    Replace abbreviations in the text with known long forms.

    Parameters:
        row (pd.Series): One DataFrame row with the column 'Text'.

    Returns:
        str: Text with expanded abbreviations.
    """
    text = row["Text"]

    # Apply custom abbreviation replacements.
    for short_form, full_form in CUSTOM_REPLACEMENTS.items():
        if re.search(rf"\b{re.escape(short_form)}\b", text):
            text = re.sub(
                rf"\b{re.escape(short_form)}\b",
                f"{short_form} ({full_form})",
                text,
            )

    # Search for remaining untreated short forms.
    matches = [
        sf
        for sf in acronym_data["SF"]
        if (
            re.search(rf"\b{re.escape(sf)}\b", text)
            and sf.lower() not in STOPS
        )
    ]

    if not matches:
        return text

    # Create a mapping from short forms to possible long forms.
    match_dict = {
        match: (
            acronym_data[acronym_data["SF"] == match]["LF"]
            .dropna()
            .tolist()
        )
        for match in matches
    }

    # Process each abbreviation.
    for match, lf_list in match_dict.items():
        best_match = (
            lf_list[0]
            if len(lf_list) == 1
            else find_best_match(text, lf_list)
        )
        text = re.sub(
            rf"\b{re.escape(match)}\b",
            f"{match} ({best_match})",
            text,
        )

    return text


def process_acronyms_with_progress(df: pd.DataFrame) -> list[str]:
    """
    Apply acronym processing to the complete DataFrame and show progress.

    Parameters:
        df (pd.DataFrame): DataFrame with the column 'Text'.

    Returns:
        list[str]: List of texts with expanded abbreviations.
    """
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        results.append(process_acronyms(row))

    return results


# Process the abbreviations and save the extended file.
query_data["Text extended"] = process_acronyms_with_progress(query_data)
query_data.to_csv(
    "fully_cleaned_gtds_just_2023_2024_AcronymsExtended.csv",
    sep=";",
    index=False,
)