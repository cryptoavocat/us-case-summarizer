# How to Search the Caselaw Dataset

This document explains how to use the provided Python script to search the caselaw dataset stored in Pinecone.

## Prerequisites

Before running the search script, ensure you have the following:

1.  **Google Colab Environment:** The script is designed to run in a Google Colab environment.
2.  **Required Libraries:** The script uses `openai`, `pinecone`, `pinecone-text`, `tiktoken`, `tqdm`, `tenacity`, `orjson`, and `nltk`. These should be installed in your Colab environment.
3.  **Google Drive Mounted:** The script expects the BM25 encoder model to be stored in your Google Drive. Ensure your Drive is mounted in Colab.
4.  **API Keys:**
    *   **OpenAI API Key:** For dense embeddings. Store this in Colab secrets named `OPENAI_API_KEY`.
    *   **Pinecone API Key:** For accessing your Pinecone index. Store this in Colab secrets named `US_PINECONE_KEY`.
5.  **BM25 Encoder Model:** The script requires a pre-trained and saved BM25 encoder model. The default path is set in the `CONFIG`. Ensure this file exists at the specified location (`/content/drive/MyDrive/caselaw_pipeline/frozen_bm25.pkl`). If not, you'll need to run the BM25 fitting step (usually a previous step in the overall pipeline) to generate this file.
6.  **Pinecone Index:** A Pinecone index named `caselaw` (as per `CONFIG`) should exist and be populated with your caselaw data. The script connects to the index host specified in the `CONFIG`.

## How to Use the Script

1.  **Open the Notebook:** Open the Google Colab notebook containing the search script.
2.  **Ensure Prerequisites are Met:** Verify that all prerequisites listed above are satisfied (libraries installed, Drive mounted, API keys set in secrets, BM25 model available, Pinecone index exists).
3.  **Locate the Search Script Cell:** Identify the code cell containing the comprehensive search script (the one with imports, configuration, functions, and the `if __name__ == "__main__":` block).
4.  **Define Your Query:** In the `if __name__ == "__main__":` block, modify the `query` variable to the text you want to search for.

## Using Metadata Filters

You can refine your search results by applying metadata filters. The `index.query` method accepts a `filter` argument, which is a dictionary specifying the filtering conditions.

Here's how you can add a metadata filter to your search:
