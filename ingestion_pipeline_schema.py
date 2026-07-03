"""
Ingestion pipeline schema for VeraDoc.

Routes files landed in MinIO into one of three tiers based on format,
each with its own storage target and retrieval strategy:

  UNSTRUCTURED     -> chunk + embed -> LanceDB              -> vector RAG agent
  SEMI_STRUCTURED   -> normalize     -> DuckDB (queryable)   -> structured-query agent
                                      -> LanceDB (schema/summary only) -> routing signal
  STRUCTURED        -> no ingestion  -> external SQL DB      -> text-to-SQL agent
                                      -> LanceDB (schema/description) -> routing signal

This is a design skeleton, not production code: fill in the TODOs with
your existing LangGraph nodes, LanceDB client, embedding model, etc.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


# ---------------------------------------------------------------------------
# 1. Format classification
# ---------------------------------------------------------------------------

class FormatTier(str, enum.Enum):
    UNSTRUCTURED = "unstructured"
    SEMI_STRUCTURED = "semi_structured"
    STRUCTURED = "structured"          # no file ingestion; DB connections registered separately
    SKIP = "skip"                      # unsupported / non-content (e.g. no-text images)


FORMAT_TIER_MAP: dict[str, FormatTier] = {
    # Unstructured -> chunk, embed, LanceDB
    ".pdf": FormatTier.UNSTRUCTURED,
    ".doc": FormatTier.UNSTRUCTURED,
    ".docx": FormatTier.UNSTRUCTURED,
    ".ppt": FormatTier.UNSTRUCTURED,
    ".pptx": FormatTier.UNSTRUCTURED,
    ".pptm": FormatTier.UNSTRUCTURED,
    ".txt": FormatTier.UNSTRUCTURED,
    ".md": FormatTier.UNSTRUCTURED,
    ".rtf": FormatTier.UNSTRUCTURED,
    ".epub": FormatTier.UNSTRUCTURED,
    # OCR images: unstructured IF OCR yields real text (checked at runtime, see OCRQualityGate)
    ".png": FormatTier.UNSTRUCTURED,
    ".jpg": FormatTier.UNSTRUCTURED,
    ".jpeg": FormatTier.UNSTRUCTURED,
    ".tiff": FormatTier.UNSTRUCTURED,

    # Semi-structured -> normalize into DuckDB, index schema-only summary in LanceDB
    ".csv": FormatTier.SEMI_STRUCTURED,
    ".xls": FormatTier.SEMI_STRUCTURED,
    ".xlsx": FormatTier.SEMI_STRUCTURED,
    ".xml": FormatTier.SEMI_STRUCTURED,
    ".json": FormatTier.SEMI_STRUCTURED,  # only if record-shaped; see note in SemiStructuredIngestor
}


def classify(path: str | Path) -> FormatTier:
    ext = Path(path).suffix.lower()
    return FORMAT_TIER_MAP.get(ext, FormatTier.SKIP)


# ---------------------------------------------------------------------------
# 2. Shared types
# ---------------------------------------------------------------------------

@dataclass
class SourceObject:
    """A file that just landed in MinIO, pre-ingestion."""
    bucket: str
    key: str
    tags: list[str] = field(default_factory=list)   # your existing MinIO TagSet pattern
    tenant_id: str | None = None                     # multi-tenant scoping if applicable


@dataclass
class RoutingEntry:
    """
    Metadata row written to LanceDB for EVERY source, regardless of tier.
    For unstructured docs this coexists with the real chunk embeddings.
    For semi/structured sources, this IS the only LanceDB footprint —
    it's what lets the router agent decide which tool to call next.
    """
    source_id: str
    tier: FormatTier
    title: str
    description: str          # short LLM-generated or heuristic summary
    schema_preview: str | None = None   # column names / table structure, semi/structured only
    backend_ref: str | None = None      # e.g. "duckdb://sales.parquet" or "postgres://.../orders"


# ---------------------------------------------------------------------------
# 3. Ingestors (one per tier)
# ---------------------------------------------------------------------------

class Ingestor(Protocol):
    def ingest(self, source: SourceObject) -> RoutingEntry: ...


class UnstructuredIngestor:
    """doc/docx/ppt/pptx/txt/md/pdf + OCR'd images -> chunk -> embed -> LanceDB."""

    def __init__(self, file_extractors: dict, embedder, lancedb_chunks_table, ocr_quality_gate):
        self.file_extractors = file_extractors     # your get_llamaindex_file_extractor() dict
        self.embedder = embedder                   # e.g. multilingual-e5-large-instruct client
        self.chunks_table = lancedb_chunks_table    # id, doc_id, vector, text, metadata
        self.ocr_quality_gate = ocr_quality_gate    # callable(text) -> bool

    def ingest(self, source: SourceObject) -> RoutingEntry:
        # TODO: download from MinIO, run through LlamaIndex SimpleDirectoryReader
        #       with your file_extractor dict (from earlier in this conversation)
        raw_docs = self._load(source)

        # OCR-specific gate: images with no meaningful extracted text get skipped
        # or rerouted to a captioning path instead of polluting the vector index.
        if source.key.lower().endswith((".png", ".jpg", ".jpeg", ".tiff")):
            text = "\n".join(d.text for d in raw_docs)
            if not self.ocr_quality_gate(text):
                return RoutingEntry(
                    source_id=source.key,
                    tier=FormatTier.SKIP,
                    title=source.key,
                    description="Image with no extractable text; skipped or sent to captioning.",
                )

        # TODO: chunk (SentenceSplitter or your current strategy),
        #       embed with correct e5 prefix convention (query vs document),
        #       upsert into self.chunks_table with source.tags propagated.
        chunks = self._chunk(raw_docs)
        self._embed_and_write(source, chunks)

        return RoutingEntry(
            source_id=source.key,
            tier=FormatTier.UNSTRUCTURED,
            title=source.key,
            description=self._summarize(raw_docs),
            backend_ref=f"lancedb://{self.chunks_table}",
        )

    def _load(self, source: SourceObject): ...
    def _chunk(self, docs): ...
    def _embed_and_write(self, source, chunks): ...
    def _summarize(self, docs) -> str: ...


class SemiStructuredIngestor:
    """
    csv/xls/xlsx/xml -> normalize into DuckDB (as a table/view) -> index
    schema-only summary in LanceDB for routing.

    NOTE on .json: only route here if it's record-shaped (array of flat
    objects). Narrative/nested JSON (exports, configs) belongs in the
    unstructured tier instead — decide per-source, not purely by extension.
    """

    def __init__(self, duckdb_conn, lancedb_routing_table, summarizer):
        self.duckdb_conn = duckdb_conn
        self.routing_table = lancedb_routing_table
        self.summarizer = summarizer   # LLM call to describe the table in 1-2 sentences

    def ingest(self, source: SourceObject) -> RoutingEntry:
        # TODO: download from MinIO
        local_path = self._download(source)
        table_name = self._safe_table_name(source.key)

        # DuckDB reads csv/xlsx/parquet natively; xls needs pandas+xlrd first (see earlier table)
        ext = Path(source.key).suffix.lower()
        if ext == ".csv":
            self.duckdb_conn.execute(
                f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_csv_auto('{local_path}')"
            )
        elif ext in (".xlsx", ".xls"):
            df = self._read_excel(local_path)  # pandas + openpyxl/xlrd per earlier table
            self.duckdb_conn.register(f"{table_name}_df", df)
            self.duckdb_conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM {table_name}_df")
        elif ext == ".xml":
            df = self._flatten_xml(local_path)  # your own flattening logic
            self.duckdb_conn.register(f"{table_name}_df", df)
            self.duckdb_conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM {table_name}_df")
        else:
            raise ValueError(f"Unsupported semi-structured format: {ext}")

        schema = self.duckdb_conn.execute(f"DESCRIBE {table_name}").fetchall()
        schema_preview = ", ".join(f"{col}:{dtype}" for col, dtype, *_ in schema)

        entry = RoutingEntry(
            source_id=source.key,
            tier=FormatTier.SEMI_STRUCTURED,
            title=table_name,
            description=self.summarizer(table_name, schema_preview),
            schema_preview=schema_preview,
            backend_ref=f"duckdb://{table_name}",
        )
        self._write_routing_entry(entry)
        return entry

    def _download(self, source: SourceObject) -> str: ...
    def _safe_table_name(self, key: str) -> str: ...
    def _read_excel(self, path: str): ...
    def _flatten_xml(self, path: str): ...
    def _write_routing_entry(self, entry: RoutingEntry) -> None: ...


class StructuredConnector:
    """
    SQL databases: nothing to ingest. Register the connection + expose
    table schemas as a routing signal so the agent knows this data source
    exists and what it contains.
    """

    def __init__(self, lancedb_routing_table, summarizer):
        self.routing_table = lancedb_routing_table
        self.summarizer = summarizer

    def register(self, connection_uri: str, tables: list[str], sql_inspector) -> list[RoutingEntry]:
        entries = []
        for table in tables:
            schema_preview = sql_inspector.describe(connection_uri, table)
            entry = RoutingEntry(
                source_id=f"{connection_uri}#{table}",
                tier=FormatTier.STRUCTURED,
                title=table,
                description=self.summarizer(table, schema_preview),
                schema_preview=schema_preview,
                backend_ref=connection_uri,
            )
            self._write_routing_entry(entry)
            entries.append(entry)
        return entries

    def _write_routing_entry(self, entry: RoutingEntry) -> None: ...


# ---------------------------------------------------------------------------
# 4. Dispatcher — wire this to your MinIO event trigger
# ---------------------------------------------------------------------------

class IngestionDispatcher:
    def __init__(
        self,
        unstructured: UnstructuredIngestor,
        semi_structured: SemiStructuredIngestor,
    ):
        self.unstructured = unstructured
        self.semi_structured = semi_structured

    def dispatch(self, source: SourceObject) -> RoutingEntry | None:
        tier = classify(source.key)
        if tier is FormatTier.UNSTRUCTURED:
            return self.unstructured.ingest(source)
        if tier is FormatTier.SEMI_STRUCTURED:
            return self.semi_structured.ingest(source)
        if tier is FormatTier.SKIP:
            return None  # log and move on
        raise ValueError(f"Unhandled tier for {source.key}: {tier}")


# ---------------------------------------------------------------------------
# 5. Retrieval side — LangGraph tool stubs
# ---------------------------------------------------------------------------
#
# Your existing LangGraph agent gains three tools instead of one:
#
#   vector_search_tool(query)       -> hybrid search over LanceDB chunks
#                                       (unstructured tier)
#   structured_query_tool(query)    -> text-to-SQL over DuckDB, using
#                                       RoutingEntry.schema_preview to pick
#                                       the right table(s) first
#   sql_db_tool(query)              -> text-to-SQL over registered external
#                                       databases (structured tier)
#
# A lightweight router node (or the agent's own tool-selection step) first
# queries the RoutingEntry table in LanceDB — schema/description only, cheap
# semantic match — to decide WHICH backend(s) are relevant before calling
# the corresponding tool. This keeps "what exists" (routing) separate from
# "how to answer" (retrieval), which also gives you a natural place to log
# provenance for citations.
