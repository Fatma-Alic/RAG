# Datasets
**Alpha-ID_dataset**
**fully_cleaned_gtds_just_2023_2024_AcronymsExtended**
**ICD-O_ICD-10_Überleitung_Solide_Tumoren**
**ICD10_GM**
**ICDO3_LE_dataset**
**Metainventory_Version1.0.0**

# Characteristics of the ICD-O-3 LE dataset

**Version:** [v1.0]
**Short description:** This dataset contains 8,981 tumor-related entries in CSV format for tumor coding and code-mapping tasks.

The dataset was created by mapping Alpha-ID diagnosis descriptions to corresponding ICD-O-3 topography codes and ICD-10-GM codes using the conversion table for solid tumors.
---

## Dataset Contents

- Number of entries: 8,981
- Number of unique classes: 615 ICD-10 codes and 326 ICD-O-3 topography codes
- Data format: CSV

The main features included are:

- `ID`: unique identifier
- `ICD-10-Code`: ICD-10 code
- `ICD-O-Code`: ICD-O-3 topography code
- `ICD-10 Text`: ICD-10-GM diagnosis description (from ICD-O_ICD-10_Überleitung_Solide_Tumoren.CSV*)
- `Label`: Alpha-ID diagnosis description
- `Topography Text`: tumor localisation
- `Extended Label`: Alpha-ID diagnosis description + (topography text)
- `Prompt Text`: ICD-10-GM diagnosis description (from ICD-O_ICD-10_Überleitung_Solide_Tumoren.CSV*) + (topography text)

* The ICD-10 text was derived from the ICD-O to ICD-10 conversion table for solid tumors, which defines rules for mapping valid ICD-O-3 codes to corresponding ICD-10 codes.
## Source Resource
Umsetzungsleitfaden - Umsetzungsleitfaden - Plattform § 65c - Confluence Instanz [Internet]. [cited 2025 Nov 26].
Available from: https://plattform65c.atlassian.net/wiki/spaces/UMK/overview?homepageId=15532036

---

## Preprocessing

The following preprocessing steps were applied:

- Alpha-ID diagnosis descriptions were linked to ICD-O-3 topography codes through shared ICD-10 codes.
- Duplicate rows were removed.
- For `Extended Label`, the `Label` was used as the main text when the ICD-10 code mapped to only one ICD-O code. If an ICD-10 code mapped to multiple ICD-O codes, the `ICD-10 Text` was used instead to avoid ambiguity.
- `Topography Text` was added in brackets only when it was not already included in the text.
- Non-informative phrases such as `sonstige Lokalisation(en)` were removed.
- Abbreviations such as `o.n.A.` were replaced with `ohne nähere Angabe`.

---
## Data Availability and Privacy

The clinical tumor diagnosis texts used for evaluation cannot be publicly released due to data protection restrictions.

---

## Citation

**To RAG, or Not to RAG? A Comparative Evaluation of Retrieval-Augmented Generation for ICD Coding of German Tumor Diagnoses**
F. Alickovic, S. Lenz, A. Ustjanzew, L. O. Rosario, G. Vollmar, T. Kindler, T. Panholzer

