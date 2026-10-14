import os
import sys
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from dotenv import load_dotenv

# Ensure proper encoding output on Windows terminal
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

persistent_directory = "db/chroma_db"

# Load ultra-fast HuggingFace ONNX embeddings and vector store
embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}  
)


# Search for relevant documents
query = "How much did Microsoft pay to acquire GitHub?"

print(f"User Query: {query}\n")

# Approach A: Standard similarity search with scores
docs_and_scores = db.similarity_search_with_relevance_scores(query, k=5)

print("--- Context with Relevance Scores ---")
for i, (doc, score) in enumerate(docs_and_scores, 1):
    source = doc.metadata.get("source", "Unknown")
    print(f"Document {i} (Source: {source} | Score: {score:.4f}):")
    print(f"{doc.page_content}\n")
    print("-" * 50)


# Approach B: Threshold or MMR Retriever
# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={"k": 5, "score_threshold": 0.4}
# )
# Or Maximal Marginal Relevance (MMR) to diversify results and eliminate duplicates:
# retriever = db.as_retriever(
#     search_type="mmr",
#     search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.7}
# )




# List of Synthetic Questions:
# 1. "What was NVIDIA's first graphics accelerator called?"
# 2. "Which company did NVIDIA acquire to enter the mobile processor market?"
# 3. "What was Microsoft's first hardware product release?"
# 4. "How much did Microsoft pay to acquire GitHub?"
# 5. "In what year did Tesla begin production of the Roadster?"
# 6. "Who succeeded Ze'ev Drori as CEO in October 2008?"
# 7. "What was the name of the autonomous spaceport drone ship that achieved the first successful sea landing?"
# 8. "What was the original name of Microsoft before it became Microsoft?"



data = db.get(include=["metadatas"])
print("Total chunks:", len(data["ids"]))
print("Sources:", {m.get("source") for m in data["metadatas"]})