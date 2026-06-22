# 📂 data/ — Put Your Documents Here

Place your **PDF** and **Word (.docx)** files in this folder.

## Supported File Types
| Format | Extension | Library Used |
|--------|-----------|--------------|
| PDF    | `.pdf`    | `pypdf`      |
| Word   | `.docx`   | `python-docx`|

## Example Files
```
data/
├── annual_report.pdf
├── user_manual.pdf
└── product_overview.docx
```

## After Adding Files

Run the ingestion pipeline to process and index the documents:

```bash
python src/ingest.py
```

This reads all files, breaks them into chunks, embeds them with Gemini,
and saves everything to the `db/` folder.

You only need to run this **once per set of documents**.
