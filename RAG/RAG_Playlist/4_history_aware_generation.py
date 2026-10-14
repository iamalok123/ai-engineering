import os
import sys
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Ensure proper UTF-8 output on Windows terminal
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables
load_dotenv()

# Automatically map GEMINI_API_KEY to GOOGLE_API_KEY if needed
if "GOOGLE_API_KEY" not in os.environ and "GEMINI_API_KEY" in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

# Helper to cleanly extract text content from Gemini's response
def extract_text(content):
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content)

# Connect to document database with fast HuggingFace ONNX embeddings
persistent_directory = "db/chroma_db"
embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# Set up Gemini Chat Model
model = ChatGoogleGenerativeAI(
    model="models/gemini-3.5-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
    temperature=0.2
)

# Store conversation history
chat_history = []


def ask_question(user_question):
    print(f"\n--- You asked: {user_question} ---")
    
    # Step 1: Make question standalone using conversation history if history exists
    if chat_history:
        messages = [
            SystemMessage(
                content="Given the chat history and the latest user question, formulate a standalone search question that can be used to search a document vector database. Do NOT answer the question, just return the reformulated question."
            ),
        ] + chat_history + [
            HumanMessage(content=f"New question: {user_question}")
        ]
        
        result = model.invoke(messages)
        search_question = extract_text(result.content).strip()
        print(f"Reformulated query for search: {search_question}")
    else:
        search_question = user_question
    
    # Step 2: Retrieve relevant documents
    retriever = db.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(search_question)
    
    print(f"\nFound {len(docs)} relevant documents:")
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        preview = doc.page_content.replace("\n", " ")[:120]
        print(f"  Doc {i} [{source}]: {preview}...")
    
    # Step 3: Format context with source citations
    context_text = "\n\n".join([
        f"[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}"
        for doc in docs
    ])
    
    combined_input = f"""Based strictly on the following documents, please answer this question: {user_question}

Documents:
{context_text}

Instructions:
1. Provide a clear, factual answer using ONLY the explicit facts stated in the documents above.
2. Cite the source file name(s) (e.g. docs\\Microsoft.txt) where you found the information.
3. If the answer cannot be found in the provided documents, state: "I don't have enough information to answer that question based on the provided documents."
"""
    
    # Step 4: Generate response with Gemini
    messages = [
        SystemMessage(content="You are a helpful assistant that answers questions accurately based strictly on provided documents and conversation history."),
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]
    
    result = model.invoke(messages)
    answer = extract_text(result.content).strip()
    
    # Step 5: Update conversation history
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))
    
    print(f"\nAnswer:\n{answer}\n")
    return answer


def start_chat():
    print("=" * 60)
    print("Welcome to History-Aware RAG Assistant (Powered by Gemini)")
    print("Ask me questions about Google, Microsoft, Nvidia, SpaceX, and Tesla!")
    print("Type 'quit' or 'exit' to end the session.")
    print("=" * 60)
    
    while True:
        try:
            question = input("\nYour question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
            
        if not question:
            continue
            
        if question.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
            
        ask_question(question)


if __name__ == "__main__":
    start_chat()