from pathlib import Path
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticSearch,
    SemanticPrioritizedFields,
    SemanticField,
)
from azure.core.credentials import AzureKeyCredential
from rich.console import Console
from rich.table import Table
import sys
import time

sys.path.append(str(Path(__file__).parent.parent))

from ingestion.document_loader import load_all_documents
from ingestion.chunker import build_child_chunks, build_parent_chunks, link_children_to_parents
from ingestion.embedder import embed_child_chunks
import config

console = Console()

CHILD_INDEX_NAME  = config.AZURE_SEARCH_INDEX_NAME
PARENT_INDEX_NAME = config.AZURE_SEARCH_INDEX_NAME + "-parents"

credential = AzureKeyCredential(config.AZURE_SEARCH_API_KEY)
index_client = SearchIndexClient(endpoint=config.AZURE_SEARCH_ENDPOINT, credential=credential)


# ─────────────────────────────────────────────────────────────
# DELETE INDEX
# ─────────────────────────────────────────────────────────────
def delete_index_if_exists(name: str):
    try:
        index_client.delete_index(name)
        console.print(f"  🗑️ Deleted existing index: [yellow]{name}[/yellow]")
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────
# CREATE CHILD INDEX
# ─────────────────────────────────────────────────────────────
def create_child_index():

    console.print(f"\n[bold yellow]🏗️ Creating Child Index: [cyan]{CHILD_INDEX_NAME}[/cyan][/bold yellow]\n")

    fields = [
        SimpleField(name="chunk_id", type=SearchFieldDataType.String, key=True, filterable=True),

        SearchableField(
            name="text",
            type=SearchFieldDataType.String,
            analyzer_name="en.microsoft"
        ),

        SimpleField(name="parent_id", type=SearchFieldDataType.String, filterable=True, retrievable=True),
        SimpleField(name="doc_name", type=SearchFieldDataType.String, filterable=True, retrievable=True),
        SimpleField(name="token_count", type=SearchFieldDataType.Int32, retrievable=True),

        # ✅ embedding used ONLY for vector search (not retrievable)
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,
            vector_search_profile_name="rag-vector-profile"
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="rag-hnsw",
                parameters={
                    "m": 4,
                    "efConstruction": 400,
                    "efSearch": 500,
                    "metric": "cosine"
                }
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="rag-vector-profile",
                algorithm_configuration_name="rag-hnsw"
            )
        ]
    )

    semantic_config = SemanticConfiguration(
        name="rag-semantic-config",
        prioritized_fields=SemanticPrioritizedFields(
            content_fields=[SemanticField(field_name="text")]
        )
    )

    semantic_search = SemanticSearch(configurations=[semantic_config])

    index = SearchIndex(
        name=CHILD_INDEX_NAME,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_search
    )

    index_client.create_or_update_index(index)

    console.print(f"  ✅ Child index created")
    console.print(f"     Fields     : chunk_id, text, parent_id, doc_name, token_count, embedding")
    console.print(f"     Vector     : 1536 dims, HNSW (cosine)")
    console.print(f"     Semantic   : enabled")
    console.print(f"     Keyword    : BM25")


# ─────────────────────────────────────────────────────────────
# CREATE PARENT INDEX
# ─────────────────────────────────────────────────────────────
def create_parent_index():

    console.print(f"\n[bold yellow]🏗️ Creating Parent Index: [cyan]{PARENT_INDEX_NAME}[/cyan][/bold yellow]\n")

    fields = [
        SimpleField(name="parent_id", type=SearchFieldDataType.String, key=True, filterable=True),
        SearchableField(name="text", type=SearchFieldDataType.String),
        SimpleField(name="doc_name", type=SearchFieldDataType.String, filterable=True, retrievable=True),
        SimpleField(name="token_count", type=SearchFieldDataType.Int32, retrievable=True),
    ]

    index = SearchIndex(name=PARENT_INDEX_NAME, fields=fields)
    index_client.create_or_update_index(index)

    console.print(f"  ✅ Parent index created")


# ─────────────────────────────────────────────────────────────
# UPLOAD CHILD
# ─────────────────────────────────────────────────────────────
def upload_child_chunks(child_chunks):

    console.print(f"\n📤 Uploading {len(child_chunks)} child chunks...\n")

    client = SearchClient(
        endpoint=config.AZURE_SEARCH_ENDPOINT,
        index_name=CHILD_INDEX_NAME,
        credential=credential
    )

    docs = [{
        "chunk_id": c["chunk_id"],
        "text": c["text"],
        "parent_id": c["parent_id"],
        "doc_name": c["doc_name"],
        "token_count": c["token_count"],
        "embedding": c["embedding"]
    } for c in child_chunks]

    result = client.upload_documents(documents=docs)

    success = sum(1 for r in result if r.succeeded)

    console.print(f"  ✅ Uploaded {success}/{len(docs)}")

    return success


# ─────────────────────────────────────────────────────────────
# UPLOAD PARENT
# ─────────────────────────────────────────────────────────────
def upload_parent_chunks(parent_chunks):

    console.print(f"\n📤 Uploading {len(parent_chunks)} parent chunks...\n")

    client = SearchClient(
        endpoint=config.AZURE_SEARCH_ENDPOINT,
        index_name=PARENT_INDEX_NAME,
        credential=credential
    )

    docs = [{
        "parent_id": p["parent_id"],
        "text": p["text"],
        "doc_name": p["doc_name"],
        "token_count": p["token_count"]
    } for p in parent_chunks]

    result = client.upload_documents(documents=docs)

    success = sum(1 for r in result if r.succeeded)

    console.print(f"  ✅ Uploaded {success}/{len(docs)}")

    return success


# ─────────────────────────────────────────────────────────────
# VERIFY INDEX
# ─────────────────────────────────────────────────────────────
def verify_index(child_uploaded, parent_uploaded):

    console.print(f"\n🔍 Verifying Index...\n")

    time.sleep(2)

    child_client = SearchClient(
        endpoint=config.AZURE_SEARCH_ENDPOINT,
        index_name=CHILD_INDEX_NAME,
        credential=credential
    )

    count = child_client.get_document_count()
    console.print(f"  Child docs: {count}")

    sample = child_client.get_document(
        key="child_0000",
        selected_fields=["chunk_id", "doc_name", "parent_id", "token_count", "text"]
    )

    console.print("\n  Sample Document:")
    console.print(f"    chunk_id   : {sample['chunk_id']}")
    console.print(f"    doc_name   : {sample['doc_name']}")
    console.print(f"    parent_id  : {sample['parent_id']}")
    console.print(f"    token_count: {sample['token_count']}")
    console.print(f"    embedding  : [dim]stored (vector search only)[/dim]")
    console.print(f"    text       : {sample['text'][:80]}...")

    # Parent check
    parent_client = SearchClient(
        endpoint=config.AZURE_SEARCH_ENDPOINT,
        index_name=PARENT_INDEX_NAME,
        credential=credential
    )

    parent_count = parent_client.get_document_count()
    console.print(f"\n  Parent docs: {parent_count}")

    # Debug info
    console.print(f"\n[bold cyan]🔎 Debug Info[/bold cyan]")
    console.print(f"  Child chunks indexed : {child_uploaded}")
    console.print(f"  Parent chunks indexed: {parent_uploaded}")
    console.print(f"  Vector dimension     : 1536")
    console.print(f"  HNSW config          : m=4, efC=400, efS=500")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():

    console.rule("Stage 1e — Indexer")

    data_dir = Path(__file__).parent.parent / "data"

    console.print(f"\n📁 Loading documents from: {data_dir}")

    docs = load_all_documents(data_dir)

    child_chunks  = build_child_chunks(docs)
    parent_chunks = build_parent_chunks(docs)

    child_chunks = link_children_to_parents(child_chunks, parent_chunks)

    console.print("\n🔢 Generating embeddings...")
    child_chunks = embed_child_chunks(child_chunks)

    console.print("\n🧹 Cleaning old indexes...")
    delete_index_if_exists(CHILD_INDEX_NAME)
    delete_index_if_exists(PARENT_INDEX_NAME)

    create_child_index()
    create_parent_index()

    child_uploaded  = upload_child_chunks(child_chunks)
    parent_uploaded = upload_parent_chunks(parent_chunks)

    verify_index(child_uploaded, parent_uploaded)

    console.print("\n✅ DONE — Indexing Complete\n")


if __name__ == "__main__":
    main()