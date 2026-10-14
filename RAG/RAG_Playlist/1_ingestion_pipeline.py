import os
import time
import sys
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

# Ensure proper encoding output on Windows terminal
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

# Automatically map GEMINI_API_KEY to GOOGLE_API_KEY if needed
if "GOOGLE_API_KEY" not in os.environ and "GEMINI_API_KEY" in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]




def load_documents(docs_path="docs"):
    """Load all text files from the docs directory"""
    print(f"Loading documents from {docs_path}...")

    # Check if docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add your company files.")

    # Load all .txt files from the docs directory
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8-sig"}
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add your company documents.")

    for i, doc in enumerate(documents[:2]):  # Show first 2 documents
        print(f"\nDocument {i+1}:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:100]}...")
        print(f"  metadata: {doc.metadata}")

    return documents


def is_reference_chunk(text):
    """Filter out Wikipedia citation lists that pollute vector search rankings"""
    return text.count("http") >= 2 or "Retrieved" in text or "Archived" in text


def split_documents(documents, chunk_size=1200, chunk_overlap=150):
    """Split documents using RecursiveCharacterTextSplitter and filter out reference noise"""
    print("Splitting documents into chunks...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    
    raw_chunks = text_splitter.split_documents(documents)
    print(f"Total raw chunks: {len(raw_chunks)}")

    # Drop citation and reference chunks
    chunks = [c for c in raw_chunks if not is_reference_chunk(c.page_content)]
    print(f"Total cleaned chunks after filtering references: {len(chunks)}")
    
    if chunks:
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n--- Sample Chunk {i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:\n{chunk.page_content[:150]}...")
            print("-" * 50)
        
        if len(chunks) > 3:
            print(f"\n... and {len(chunks) - 3} more chunks ready for ingestion")
    
    return chunks



def create_vector_store(chunks, persist_directory="db/chroma_db", batch_size=100):
    """Embed chunks using FastEmbed (HuggingFace ONNX) ultra-fast without rate limits"""
    print(f"Embedding {len(chunks)} chunks using FastEmbed (BAAI/bge-small-en-v1.5)...")
    t0 = time.time()
    
    embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        vectorstore.add_documents(batch)
        print(f"  Embedded {min(i + batch_size, len(chunks))}/{len(chunks)} chunks...")

    elapsed = time.time() - t0
    print(f"⚡ Ingestion completed in {elapsed:.2f} seconds! ({len(chunks)/elapsed:.1f} chunks/sec)")
    return vectorstore




def main():

    """Main ingestion pipeline"""
    print("=== RAG Document Ingestion Pipeline ===\n")
    
    # Define paths
    docs_path = "docs"
    persistent_directory = "db/chroma_db"
    
    # Check if vector store already exists and contains documents
    if os.path.exists(persistent_directory):
        embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        vectorstore = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_model, 
            collection_metadata={"hnsw:space": "cosine"}
        )
        doc_count = vectorstore._collection.count()
        if doc_count > 0:
            print(f"✅ Vector store already exists with {doc_count} documents. No need to re-process documents.")
            return vectorstore
        print("⚠️ Persistent directory exists but contains 0 documents. Populating vector store...\n")
    else:
        print("Persistent directory does not exist. Initializing vector store...\n")



    # Step 1: Load documents
    documents = load_documents()  

    # Step 2: Split into chunks
    chunks = split_documents(documents)

    # Step 3: Create vector store
    vectorstore = create_vector_store(chunks, persistent_directory)


    print("\n✅ Ingestion complete! Your documents are now ready for RAG queries.")
    return vectorstore






if __name__ == "__main__":
    main()









# documents = [
#    Document(
#        page_content="Google LLC is an American multinational corporation and technology company focusing on online advertising, search engine technology, cloud computing, computer software, quantum computing, e-commerce, consumer electronics, and artificial intelligence (AI).",
#        metadata={'source': 'docs/google.txt'}
#    ),
#    Document(
#        page_content="Microsoft Corporation is an American multinational corporation and technology conglomerate headquartered in Redmond, Washington.",
#        metadata={'source': 'docs/microsoft.txt'}
#    ),
#    Document(
#        page_content="Nvidia Corporation is an American technology company headquartered in Santa Clara, California.",
#        metadata={'source': 'docs/nvidia.txt'}
#    ),
#    Document(
#        page_content="Space Exploration Technologies Corp., commonly referred to as SpaceX, is an American space technology company headquartered at the Starbase development site in Starbase, Texas.",
#        metadata={'source': 'docs/spacex.txt'}
#    ),
#    Document(
#        page_content="Tesla, Inc. is an American multinational automotive and clean energy company headquartered in Austin, Texas.",
#        metadata={'source': 'docs/tesla.txt'}
#    )
# ]