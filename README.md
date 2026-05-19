# RAG Pipeline for Automated ICD-10 and ICD-O Coding of German Tumor Diagnoses
This repository contains a Retrieval-Augmented Generation (RAG) pipeline for the automated ICD coding of German tumor diagnoses using embedding-based retrieval and base and fine-tuned Large Language Models (LLMs).

---

## RAG Pipeline

The RAG pipeline consists of the following main steps:
![RAG Pipeline](results/figures/RAG_pipeline.png)


1. **Input diagnosis**
   A German tumor diagnosis is provided as free text.

2. **Preprocessing**
   The diagnosis text is cleaned and normalized.

3. **Embedding retrieval**
   The input diagnosis is converted into an embedding and compared with indexed reference examples.

4. **Context construction**
   The most relevant retrieved examples are added to the prompt as context.

5. **LLM-based coding**
   The language model predicts ICD-10 and/or ICD-O topography codes based on the diagnosis and retrieved context.

6. **Evaluation**
   The predicted codes are compared with reference codes using evaluation metrics such as exact match, precision, recall, and F1 score.

---

## Repository Structure

| Folder | Description |
|---|---|
| `preprocessing/` | Scripts for cleaning and normalizing German diagnosis texts. |
| `classifier/` | Classifier-based ICD prediction baseline. |
| `similarity_search/` | Embedding-based retrieval and nearest-neighbor search. |
| `word2vec/` | Word2Vec-based retrieval baseline. |
| `prompts/` | Prompt templates and scripts for constructing LLM inputs. |
| `rag/` | RAG-based ICD coding and evaluation pipeline. |
| `configs/` | Experiment configuration files. |
| `results/` | Evaluation plots. |

---

## Citation

**To RAG, or Not to RAG? A Comparative Evaluation of Retrieval-Augmented Generation for ICD Coding of German Tumor Diagnoses**
F. Alickovic, S. Lenz, A. Ustjanzew, L. O. Rosario, G. Vollmar, T. Kindler, T. Panholzer


