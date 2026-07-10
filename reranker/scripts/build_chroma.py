"""Build ChromaDB dense index from clauses.jsonl. Run from implementation/ folder."""
import json
import logging
import sys
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("build_chroma")

CHROMA_DIR = "data/chroma_db"
CLAUSES_PATH = Path("data/clauses.jsonl")

clauses = [
    json.loads(line)
    for line in CLAUSES_PATH.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
logger.info("Loaded %d clauses", len(clauses))

chroma_path = Path(CHROMA_DIR)
if chroma_path.exists() and any(chroma_path.iterdir()):
    logger.info("ChromaDB already exists at %s: loading to verify", CHROMA_DIR)
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name="aml_clauses",
    )
    logger.info("Collection has %d documents", vectorstore._collection.count())
else:
    logger.info("Embedding %d clauses with all-MiniLM-L6-v2...", len(clauses))
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    documents = [
        Document(
            page_content=c["text"],
            metadata={
                "clause_id": c["clause_id"],
                "source": c["source"],
                "marker": c["marker"],
            },
        )
        for c in clauses
    ]
    logger.info("Building ChromaDB collection...")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name="aml_clauses",
    )
    logger.info("Done: %d documents indexed at %s", vectorstore._collection.count(), CHROMA_DIR)
