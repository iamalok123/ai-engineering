import os
import sys
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Ensure proper encoding output on Windows terminal
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables
load_dotenv()

# Automatically map GEMINI_API_KEY to GOOGLE_API_KEY if needed
if "GOOGLE_API_KEY" not in os.environ and "GEMINI_API_KEY" in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

# Initialize the Gemini Chat Model
llm = ChatGoogleGenerativeAI(
    model="models/gemini-3.5-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

# Tesla text to chunk
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

# Create the prompt
prompt = f"""
You are a text chunking expert. Split this text into logical chunks.

Rules:
- Each chunk should be around 200 characters or less
- Split at natural topic boundaries
- Keep related information together
- Put "<<<SPLIT>>>" between chunks

Text:
{tesla_text}

Return the text with <<<SPLIT>>> markers where you want to split:
"""

# Get AI response
print("🤖 Asking Gemini AI to chunk the text agentically...")
response = llm.invoke(prompt)

if isinstance(response.content, list):
    marked_text = "".join(part.get("text", "") for part in response.content if isinstance(part, dict))
else:
    marked_text = response.content

# Split the text at the markers
chunks = marked_text.split("<<<SPLIT>>>")

# Clean up the chunks (remove extra whitespace)
clean_chunks = []
for chunk in chunks:
    cleaned = chunk.strip()
    if cleaned:  # Only keep non-empty chunks
        clean_chunks.append(cleaned)

# Show results
print("\n🎯 AGENTIC CHUNKING RESULTS:")
print("=" * 50)

for i, chunk in enumerate(clean_chunks, 1):
    print(f"Chunk {i}: ({len(chunk)} chars)")
    print(f'"{chunk}"')
    print()