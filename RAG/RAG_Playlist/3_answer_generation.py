import os
import sys
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

# Ensure proper encoding output on Windows terminal
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

# Automatically map GEMINI_API_KEY to GOOGLE_API_KEY if needed
if "GOOGLE_API_KEY" not in os.environ and "GEMINI_API_KEY" in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

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

retriever = db.as_retriever(search_kwargs={"k": 5})

# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={
#         "k": 5,
#         "score_threshold": 0.3  # Only return chunks with cosine similarity ≥ 0.3
#     }
# )

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")
# Display results
print("--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    source = doc.metadata.get("source", "Unknown")
    print(f"Document {i} (Source: {source}):\n{doc.page_content}\n")


# Combine the query and the relevant document contents with source tags
context_text = "\n\n".join([f"[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}" for doc in relevant_docs])

combined_input = f"""Based strictly on the following documents, please answer this question: {query}

Documents:
{context_text}

Instructions:
1. Provide a clear, direct answer using ONLY the facts explicitly stated in these documents.
2. Cite the source file name(s) where you found the information.
3. If the answer cannot be found in the documents, respond with: "I don't have enough information to answer that question based on the provided documents."
"""

# Create a ChatGoogleGenerativeAI model
model = ChatGoogleGenerativeAI(
    model="models/gemini-3.5-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
    temperature=0.2
)

# Define the messages for the model
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

# Invoke the model with the combined input
result = model.invoke(messages)

# Display the full result and content only
print("\n--- Generated Response ---")
print("Content only:")
if isinstance(result.content, list):
    clean_text = "".join(part.get("text", "") for part in result.content if isinstance(part, dict))
else:
    clean_text = result.content
print(clean_text)