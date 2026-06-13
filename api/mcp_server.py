import asyncio
import json
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from config.settings import get_settings
from observability.logger import setup_logger

logger = setup_logger(__name__)
settings = get_settings()

# Initialize MCP server
mcp = Server("lexrag")


@mcp.list_tools()
async def list_tools() -> list[Tool]:
    """Expose LexRAG capabilities as MCP tools."""
    return [
        Tool(
            name="query_legal",
            description=(
                "Query the LexRAG legal knowledge base. "
                "Ask questions about Indian law — IPC, CrPC, Constitution, judgments. "
                "Returns a cited answer grounded in ingested legal documents."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The legal question to answer",
                    },
                    "doc_type": {
                        "type": "string",
                        "description": "Optional filter: act, judgment, constitution, regulation",
                        "enum": ["act", "judgment", "constitution", "regulation", "generic"],
                    },
                    "rewrite": {
                        "type": "boolean",
                        "description": "Whether to rewrite query for better retrieval. Default true.",
                        "default": True,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="ingest_document",
            description=(
                "Ingest a legal document into LexRAG from a local file path. "
                "Supports PDF, TXT, and DOCX formats. "
                "Use this to add new legal documents to the knowledge base."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the document file",
                    },
                    "doc_type": {
                        "type": "string",
                        "description": "Type of legal document",
                        "enum": ["act", "judgment", "constitution", "regulation", "generic"],
                    },
                },
                "required": ["file_path", "doc_type"],
            },
        ),
        Tool(
            name="list_ingested_documents",
            description="List all documents currently in the LexRAG knowledge base.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]


@mcp.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls from MCP clients."""

    # Import here to avoid circular imports at module load time
    from embeddings.embedder import embedder
    from vectorstore.store import vector_store
    from retrieval.retriever import retriever
    from retrieval.reranker import reranker
    from generation.query_rewriter import query_rewriter
    from generation.generator import generator
    from ingestion.pipeline import ingest_document
    from config.exceptions import (
        BelowConfidenceThresholdError,
        DuplicateDocumentError,
        LexRAGError,
    )

    # Ensure components are initialized
    embedder.load()
    vector_store.connect()
    reranker.load()

    if name == "query_legal":
        query = arguments["query"]
        doc_type = arguments.get("doc_type")
        rewrite = arguments.get("rewrite", True)

        try:
            # Rewrite
            rewritten = query_rewriter.rewrite(query) if rewrite else query

            # Retrieve
            filters = {"doc_type": doc_type} if doc_type else None
            chunks = retriever.retrieve(rewritten, filters=filters)

            # Rerank
            reranked = reranker.rerank(rewritten, chunks)

            # Generate
            result = generator.generate(rewritten, reranked)

            output = {
                "query": query,
                "rewritten_query": rewritten,
                "answer": result["answer"],
                "sources": result["sources"],
                "model": result["model"],
            }

            return [TextContent(type="text", text=json.dumps(output, indent=2))]

        except BelowConfidenceThresholdError as e:
            return [TextContent(type="text", text=json.dumps({
                "answer": str(e),
                "sources": [],
            }))]

        except LexRAGError as e:
            return [TextContent(type="text", text=json.dumps({
                "error": type(e).__name__,
                "detail": str(e),
            }))]

    elif name == "ingest_document":
        file_path = arguments["file_path"]
        doc_type = arguments["doc_type"]

        try:
            result = ingest_document(file_path, doc_type)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except DuplicateDocumentError as e:
            return [TextContent(type="text", text=json.dumps({
                "error": "DuplicateDocument",
                "detail": str(e),
            }))]

        except LexRAGError as e:
            return [TextContent(type="text", text=json.dumps({
                "error": type(e).__name__,
                "detail": str(e),
            }))]

    elif name == "list_ingested_documents":
        try:
            results, _ = vector_store._client.scroll(
                collection_name=settings.qdrant_collection_name,
                with_payload=True,
                with_vectors=False,
                limit=1000,
            )

            # Deduplicate by file hash
            seen = {}
            for r in results:
                fh = r.payload.get("file_hash", "unknown")
                if fh not in seen:
                    seen[fh] = {
                        "source": r.payload.get("source"),
                        "doc_type": r.payload.get("doc_type"),
                        "ingested_at": r.payload.get("ingested_at"),
                        "file_hash": fh,
                    }

            documents = list(seen.values())
            return [TextContent(type="text", text=json.dumps({
                "total_documents": len(documents),
                "documents": documents,
            }, indent=2))]

        except Exception as e:
            return [TextContent(type="text", text=json.dumps({
                "error": str(e)
            }))]

    else:
        return [TextContent(type="text", text=json.dumps({
            "error": f"Unknown tool: {name}"
        }))]


async def run():
    """Run the MCP server over stdio."""
    logger.info("Starting LexRAG MCP server")
    async with stdio_server() as (read_stream, write_stream):
        await mcp.run(
            read_stream,
            write_stream,
            mcp.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(run())