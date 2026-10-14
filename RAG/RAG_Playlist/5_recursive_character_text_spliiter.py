import sys
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

# Ensure proper encoding output on Windows terminal
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

tesla_text = """Tesla's Q3 Results
Tesla reported record revenue of $25.2B in Q3 2024.

Model Y Performance
The Model Y became the best-selling vehicle globally, with 350,000 units sold.

Production Challenges
Supply chain issues caused a 12% increase in production costs.
This is one very long paragraph that definitely exceeds our 100 character limit and has no double newlines inside it whatsoever making it impossible to split properly."""

print("=" * 65)
print("1. BASIC CHARACTER TEXT SPLITTER (Fixed single separator: '\\n\\n')")
print("=" * 65)

# Notice how single separator fails to split paragraphs longer than chunk_size
char_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=100,
    chunk_overlap=0
)
chunks1 = char_splitter.split_text(tesla_text)
print(f"Total chunks created: {len(chunks1)}")
for i, chunk in enumerate(chunks1, 1):
    print(f"  Chunk {i} ({len(chunk)} chars): {chunk!r}")

print("\n" + "=" * 65)
print("2. RECURSIVE CHARACTER TEXT SPLITTER (Hierarchy: ['\\n\\n', '\\n', '. ', ' ', ''])")
print("=" * 65)

# Recursive splitter falls back to smaller separators when chunks exceed chunk_size
recursive_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=100,
    chunk_overlap=0
)
chunks2 = recursive_splitter.split_text(tesla_text)
print(f"Total chunks created: {len(chunks2)}")
for i, chunk in enumerate(chunks2, 1):
    print(f"  Chunk {i} ({len(chunk)} chars): {chunk!r}")