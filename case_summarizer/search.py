"""Search utilities for the caselaw Pinecone index.

This module provides a :class:`PineconeSearcher` class that wraps common search
operations over a Pinecone index populated with caselaw vectors.  It supports
hybrid dense+sparse retrieval when a BM25 encoder pickle is supplied and offers a
helper to generate short summaries of returned cases by calling the OpenAI
Completions API.

Example
-------
The module can be executed as a script:

.. code-block:: bash

    python -m case_summarizer.search --query "Miranda rights" --top-k 3 \\
        --filter '{"court": "Supreme Court"}'

See the project README for end-to-end instructions.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import orjson
from openai import OpenAI
from pinecone import Pinecone
from pinecone.core.client.model.sparse_values import SparseValues

try:
    from pinecone_text.sparse import BM25Encoder
except ImportError:  # pragma: no cover - optional dependency
    BM25Encoder = None  # type: ignore[assignment]


_DEFAULT_EMBED_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
_DEFAULT_SUMMARY_MODEL = os.getenv("OPENAI_SUMMARY_MODEL", "gpt-4o-mini")


@dataclass
class SearchConfig:
    """Runtime configuration for :class:`PineconeSearcher`.

    Attributes
    ----------
    index_name:
        Name of the Pinecone index containing caselaw embeddings.
    host:
        Fully-qualified host for the Pinecone index. Required for Pod-based
        indexes. If omitted the SDK will use the control plane to resolve the
        host (recommended for serverless indexes).
    namespace:
        Optional namespace to scope queries to.
    bm25_encoder_path:
        Filesystem path to a pickled :class:`~pinecone_text.sparse.BM25Encoder`.
        Leave empty to disable sparse retrieval.
    top_k:
        Default number of results to return when none is specified.
    """

    index_name: str = os.getenv("PINECONE_INDEX", "caselaw")
    host: Optional[str] = os.getenv("PINECONE_HOST")
    namespace: Optional[str] = os.getenv("PINECONE_NAMESPACE")
    bm25_encoder_path: Optional[str] = os.getenv(
        "BM25_ENCODER_PATH", "/content/drive/MyDrive/caselaw_pipeline/frozen_bm25.pkl"
    )
    top_k: int = int(os.getenv("DEFAULT_TOP_K", "5"))


class PineconeSearcher:
    """Convenience wrapper around a Pinecone index."""

    def __init__(
        self,
        config: Optional[SearchConfig] = None,
        *,
        openai_client: Optional[OpenAI] = None,
    ) -> None:
        self.config = config or SearchConfig()
        self._pc = Pinecone(api_key=_get_env_or_error("US_PINECONE_KEY"))
        self._index = self._pc.Index(self.config.index_name, host=self.config.host)
        self._openai_client = openai_client or OpenAI(api_key=_get_env_or_error("OPENAI_API_KEY"))
        self._bm25_encoder: Any = None

    @property
    def bm25_encoder(self) -> Optional[BM25Encoder]:
        """Lazily load and cache the BM25 encoder if a path is configured."""

        if BM25Encoder is None:
            raise RuntimeError(
                "pinecone-text is required for sparse retrieval. Install it with 'pip "
                "install pinecone-text'."
            )

        if self._bm25_encoder is None and self.config.bm25_encoder_path:
            path = os.fspath(self.config.bm25_encoder_path)
            if os.path.exists(path):
                with open(path, "rb") as handle:
                    self._bm25_encoder = pickle.load(handle)
            else:
                raise FileNotFoundError(
                    f"BM25 encoder file not found at '{path}'. Upload the pickle to this "
                    "location or set BM25_ENCODER_PATH to the correct path."
                )
        return self._bm25_encoder

    def embed_query(self, query: str, *, model: Optional[str] = None) -> List[float]:
        """Create a dense embedding for ``query`` using the OpenAI embeddings API."""

        response = self._openai_client.embeddings.create(
            model=model or _DEFAULT_EMBED_MODEL,
            input=query,
        )
        return response.data[0].embedding

    def encode_sparse(self, query: str) -> SparseValues:
        """Encode a query string into sparse BM25 weights."""

        encoder = self.bm25_encoder
        if encoder is None:
            raise RuntimeError(
                "BM25 encoder not configured. Ensure BM25_ENCODER_PATH points to a "
                "pickled pinecone_text.sparse.BM25Encoder."
            )
        sparse_vector = encoder.encode_queries(query)
        if isinstance(sparse_vector, list):
            sparse_vector = sparse_vector[0]

        if hasattr(sparse_vector, "indices"):
            indices = getattr(sparse_vector, "indices")
            values = getattr(sparse_vector, "values")
        else:
            indices = sparse_vector["indices"]
            values = sparse_vector["values"]

        return SparseValues(indices=indices, values=values)

    def search(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        filter_: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
        include_metadata: bool = True,
    ) -> Dict[str, Any]:
        """Execute a Pinecone query using dense + sparse (if available) retrieval."""

        dense_vector = self.embed_query(query)
        sparse_vector: Optional[SparseValues] = None
        if self.config.bm25_encoder_path:
            try:
                sparse_vector = self.encode_sparse(query)
            except FileNotFoundError:
                # Surface clearer error message when file is missing.
                raise
            except RuntimeError:
                # BM25 not configured; skip sparse component.
                sparse_vector = None

        kwargs: Dict[str, Any] = {
            "vector": dense_vector,
            "top_k": top_k or self.config.top_k,
            "include_metadata": include_metadata,
        }
        if sparse_vector is not None:
            kwargs["sparse_vector"] = sparse_vector
        if filter_:
            kwargs["filter"] = filter_
        if namespace or self.config.namespace:
            kwargs["namespace"] = namespace or self.config.namespace

        return self._index.query(**kwargs)


def summarize_case(case_text: str, *, model: Optional[str] = None) -> str:
    """Generate a brief summary for ``case_text`` using an OpenAI model."""

    client = OpenAI(api_key=_get_env_or_error("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model or _DEFAULT_SUMMARY_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You summarize United States legal case documents. Produce a concise "
                    "summary with the holding and key context."
                ),
            },
            {"role": "user", "content": case_text},
        ],
        temperature=0.2,
        max_tokens=256,
    )
    return response.choices[0].message["content"].strip()


def _get_env_or_error(name: str) -> str:
    try:
        return os.environ[name]
    except KeyError as exc:  # pragma: no cover - configuration guardrail
        raise EnvironmentError(
            f"Environment variable '{name}' is required but not set."
        ) from exc


def _parse_filter(filter_arg: Optional[str]) -> Optional[Dict[str, Any]]:
    if not filter_arg:
        return None
    try:
        return json.loads(filter_arg)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Metadata filter must be valid JSON, e.g. '{\"court\": \"Supreme Court\"}'."
        ) from exc


def _format_response(response: Dict[str, Any]) -> str:
    """Return a pretty-printed representation of query results."""

    return orjson.dumps(response, option=orjson.OPT_INDENT_2).decode("utf-8")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search the caselaw Pinecone index")
    parser.add_argument("--query", required=True, help="Natural language query text")
    parser.add_argument("--top-k", type=int, default=None, help="Number of results to return")
    parser.add_argument(
        "--filter",
        dest="filter_",
        help="Metadata filter expressed as JSON (e.g. '{\"court\": \"Supreme Court\"}')",
    )
    parser.add_argument(
        "--namespace",
        help="Override the configured namespace for this search only",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Do not include metadata payloads in the response",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    searcher = PineconeSearcher()
    response = searcher.search(
        args.query,
        top_k=args.top_k,
        filter_=_parse_filter(args.filter_),
        namespace=args.namespace,
        include_metadata=not args.no_metadata,
    )
    print(_format_response(response))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
