import os
import sys
from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

# Ensure proper encoding output on Windows terminal
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables
load_dotenv()

# Same Tesla text structured to demonstrate semantic grouping
tesla_text = """Tesla's Q3 Results
Tesla reported record revenue of $25.2B in Q3 2024.
The company exceeded analyst expectations by 15%.
Revenue growth was driven by strong vehicle deliveries.

Model Y Performance  
The Model Y became the best-selling vehicle globally, with 350,000 units sold.
Customer satisfaction ratings reached an all-time high of 96%.
Model Y now represents 60% of Tesla's total vehicle sales.

Production Challenges
Supply chain issues caused a 12% increase in production costs.
Tesla is working to diversify its supplier base.
New manufacturing techniques are being implemented to reduce costs."""

# Initialize fast local HuggingFace ONNX embedding model (BAAI/bge-small-en-v1.5)
embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# Semantic Chunker - groups sentences by meaning/similarity rather than fixed character counts
semantic_splitter = SemanticChunker(
    embeddings=embedding_model,
    breakpoint_threshold_type="percentile",  # Options: "percentile", "standard_deviation", "interquartile"
    breakpoint_threshold_amount=70
)

print("Running Semantic Chunker with FastEmbed (BAAI/bge-small-en-v1.5)...")
chunks = semantic_splitter.split_text(tesla_text)

print("\n" + "=" * 55)
print(f"SEMANTIC CHUNKING RESULTS: {len(chunks)} Chunks Created")
print("=" * 55)
for i, chunk in enumerate(chunks, 1):
    print(f"--- Chunk {i} ({len(chunk)} characters) ---")
    print(chunk.strip())
    print()