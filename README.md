# LENR Topic Modeling

## Overview
This project performs topic discovery and clustering on a collection of LENR (Low Energy Nuclear Reactions) papers. The workflow (implemented in `topic_modeling.ipynb`) covers:

- PDF ingestion from Google Drive
- Text extraction using PyMuPDF (primary) and Tesseract OCR (fallback/comparison)
- Paragraph/sentence chunking (newline- and spaCy-based)
- Text normalization and lemmatization
- TF–IDF (1–2 grams) vectorization and TruncatedSVD dimensionality reduction
- Visualization with PCA and t-SNE
- K-Means sweep with elbow/silhouette to select k and final clustering
- Cluster keyword extraction (class-based TF–IDF) and representative paragraphs
- BERT embeddings for semantic analysis and 2D visualization

The notebook is designed to run in Google Colab and saves intermediate artifacts to Google Drive.

## Data
- Place a zip named `Documents.zip` in your Google Drive at `MyDrive/Documents.zip`.
- The zip should contain PDF files of LENR papers under the `Documents/` folder.

Example (inside the zip):
```
Documents/
  StormsEanexplanat.pdf
  Scaramuzzitenyearsof.pdf
  OgawaHcorrelatio.pdf
  ... (total ~15 PDFs)
```

## Environment and Dependencies
The notebook installs and/or uses the following:

- Core: `pandas`, `numpy`, `matplotlib`, `seaborn`, `tqdm`, `pickle`, `pathlib`
- PDF/Text: `pymupdf` (PyMuPDF), `pytesseract`, `Pillow`
- NLP: `spacy` with model `en_core_web_sm`
- ML: `scikit-learn` (TF–IDF, SVD, PCA, t-SNE, KMeans, metrics)
- Transformers: `transformers`, `torch` (BERT embeddings)

If running locally, install requirements and the spaCy model:
```bash
pip install pymupdf pytesseract Pillow spacy scikit-learn transformers torch tqdm matplotlib seaborn pandas numpy
python -m spacy download en_core_web_sm
```


## Outputs (written to Google Drive)
- `Topic_Modeling/pymupdf_output.json` — Concatenated text per document from PyMuPDF.
- `Topic_Modeling/tesseract_output.json` — Concatenated text per document from OCR.
- `Topic_Modeling/cleaned_doc_chunks.json` — Cleaned chunks per document.
- `Topic_Modeling/bert_embeddings.pkl` — Array of BERT embeddings and metadata.
- `Topic_Modeling/bert_embeddings_pca_2d.pkl` — PCA 2D reduction + metadata.
- `Topic_Modeling/bert_embeddings_tsne_2d.pkl` — t-SNE 2D reduction + metadata.

## Key Findings (from the notebook)
- For digitally-created PDFs, PyMuPDF extraction produces systematic, repairable spacing/formatting artifacts, while Tesseract can introduce irrecoverable character errors; thus PyMuPDF is preferred.
- Example scale: ~510 cleaned paragraph chunks across ~15 documents, TF–IDF→SVD of shape `(510, 128)`.
- Model exploration suggests higher k may be viable on TF–IDF/SVD; downstream visualizations and cluster summaries help interpret topics (e.g., electrodes/cathodes, isotopic/transmutation, heat/power, etc.).

## License
If you plan to share or publish, add a license file (e.g., MIT) and update this section.