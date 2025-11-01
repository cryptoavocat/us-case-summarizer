# US Caselaw Pinecone Search Toolkit

This repository contains a lightweight Python toolkit for searching your US caselaw
Pinecone index and optionally generating short AI summaries of the returned
opinions. It includes a reusable `PineconeSearcher` class, a CLI entry point, and
setup documentation to help you bring your own embeddings and sparse BM25 model.

## Features

- 🔍 Hybrid dense + sparse (BM25) retrieval against an existing Pinecone index.
- 🧠 Optional OpenAI-powered summaries for each result.
- 🧰 Configurable through environment variables so it fits into your existing
  Colab or local workflow.
- 🗂️ Metadata filtering to narrow results by court, date, jurisdiction, or any
  other attributes stored alongside your vectors.

## Repository Layout

```
.
├── README.md                ← You're here
├── requirements.txt         ← Python dependencies
└── case_summarizer/
    ├── __init__.py
    └── search.py            ← Pinecone search + summarization utilities
```

## Prerequisites

1. **Python environment** – The CLI and examples assume Python 3.9+.
2. **Dependencies** – Install packages with:
   ```bash
   pip install -r requirements.txt
   ```
3. **API keys** – Store these as environment variables before running any
   commands:
   - `OPENAI_API_KEY`
   - `US_PINECONE_KEY`
4. **Pinecone index** – A populated Pinecone index that already contains your
   caselaw embeddings. Set `PINECONE_INDEX` (defaults to `caselaw`) and, if you
   are on a pod-based deployment, `PINECONE_HOST`.
5. **BM25 encoder (optional but recommended)** – A pickled
   `pinecone_text.sparse.BM25Encoder` trained on the caselaw corpus. Upload the
   pickle to your environment (for example, Google Drive) and set
   `BM25_ENCODER_PATH` to its location. If omitted or missing, the tool falls
   back to dense-only retrieval.

## Running a Search

Execute the search CLI with your query:

```bash
python -m case_summarizer.search --query "Miranda warnings" --top-k 5
```

You will receive a JSON response containing the matched vectors, their scores,
IDs, and metadata (unless you pass `--no-metadata`).

### Using Metadata Filters

Metadata filters let you narrow the results to specific courts, jurisdictions,
or time periods. Provide filters as JSON via the `--filter` flag. Here are a few
examples:

```bash
# Limit to Supreme Court cases
python -m case_summarizer.search \
    --query "qualified immunity" \
    --filter '{"court": "Supreme Court of the United States"}'

# Restrict to decisions issued after 2000 (assuming you stored year metadata)
python -m case_summarizer.search \
    --query "habeas corpus" \
    --filter '{"year": {"$gte": 2000}}'

# Combine multiple constraints
python -m case_summarizer.search \
    --query "environmental regulation" \
    --filter '{"jurisdiction": "Federal", "year": {"$between": [1990, 2010]}}'
```

> **Tip:** Pinecone filter syntax mirrors Mongo-style operators. Supported
> operators include `$in`, `$gte`, `$lte`, `$between`, `$exists`, `$ne`, and
> more. Consult the [Pinecone filter documentation](https://docs.pinecone.io/docs/metadata-filtering)
> for a full reference.

### Summarizing Retrieved Cases

To summarize a case body returned from the search results, pass the opinion text
to `case_summarizer.summarize_case`:

```python
from case_summarizer import PineconeSearcher, summarize_case

searcher = PineconeSearcher()
response = searcher.search("Miranda rights", top_k=3)
for match in response["matches"]:
    summary = summarize_case(match["metadata"].get("opinion_text", ""))
    print(match["id"], summary)
```

The summary helper calls the OpenAI Chat Completions API using the model defined
in the `OPENAI_SUMMARY_MODEL` environment variable (default `gpt-4o-mini`).

## Configuration Reference

The `SearchConfig` dataclass reads the following environment variables. All are
optional unless noted:

| Variable | Description | Default |
| --- | --- | --- |
| `OPENAI_API_KEY` | **Required.** OpenAI API key for embeddings and summaries. | – |
| `US_PINECONE_KEY` | **Required.** Pinecone API key for index access. | – |
| `PINECONE_INDEX` | Name of the index containing caselaw vectors. | `caselaw` |
| `PINECONE_HOST` | Host URL for pod-based indexes. | Resolved by SDK |
| `PINECONE_NAMESPACE` | Optional namespace to scope all queries. | `None` |
| `BM25_ENCODER_PATH` | Filesystem path to BM25 encoder pickle. | `/content/drive/MyDrive/caselaw_pipeline/frozen_bm25.pkl` |
| `OPENAI_EMBEDDING_MODEL` | Embedding model used for queries. | `text-embedding-3-large` |
| `OPENAI_SUMMARY_MODEL` | Model used for summaries. | `gpt-4o-mini` |
| `DEFAULT_TOP_K` | Default number of matches when `--top-k` is omitted. | `5` |

Adjust these variables to match your environment (for example, when running in
Google Colab with Google Drive mounted).

## Uploading Your BM25 Encoder

Because the pickled encoder can be large, it is not stored in this repository.
Upload it separately (e.g., to Google Drive) and set `BM25_ENCODER_PATH` to the
full path, such as:

```python
os.environ["BM25_ENCODER_PATH"] = "/content/drive/MyDrive/caselaw_pipeline/frozen_bm25.pkl"
```

When the file is present, the CLI automatically performs hybrid dense+sparse
retrieval. If the file is missing, you will see a helpful error message.

## Next Steps

- Integrate the CLI into a larger pipeline or notebook for caselaw research.
- Extend the metadata filters to support your custom schema.
- Use the returned metadata to link back to the full-text opinions or docket
  information stored elsewhere.

Happy searching!
