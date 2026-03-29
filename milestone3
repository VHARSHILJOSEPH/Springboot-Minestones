import os
import re
import time
import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
import requests

load_dotenv(override=True)

# ── Credentials ───────────────────────────────────────────────────────────────
PINECONE_API_KEY  = os.getenv("PINECONE_API_KEY")
LLAMA_API_KEY     = os.getenv("LLAMA_API_KEY")
NEO4J_URI         = os.getenv("NEO4J_URI",      "neo4j+s://your-instance.databases.neo4j.io")
NEO4J_USER        = os.getenv("NEO4J_USERNAME", os.getenv("NEO4J_USER", "neo4j"))
NEO4J_PASSWORD    = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE    = os.getenv("NEO4J_DATABASE", "neo4j")

# ── Model / Index settings ────────────────────────────────────────────────────
PINECONE_INDEX    = "email-knowledge-graph"
EMBED_MODEL       = "multilingual-e5-large"
EMBED_DIM         = 1024
OPENROUTER_URL    = "https://ollama.com/v1/chat/completions"
MODEL_NAME        = "gpt-oss:120b-cloud"

# ── Retrieval settings ────────────────────────────────────────────────────────
VECTOR_TOP_K      = 5
GRAPH_FACTS_LIMIT = 5
CHUNK_SIZE        = 500
CHUNK_OVERLAP     = 50

missing = [k for k, v in {
    "PINECONE_API_KEY": PINECONE_API_KEY,
    "LLAMA_API_KEY":    LLAMA_API_KEY,
    "NEO4J_PASSWORD":   NEO4J_PASSWORD,
}.items() if not v]
if missing:
    raise EnvironmentError(f"Missing env vars: {missing}. Add them to .env")


# ── Pinecone helpers ──────────────────────────────────────────────────────────
def get_pinecone_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if PINECONE_INDEX not in [idx.name for idx in pc.list_indexes()]:
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(PINECONE_INDEX).status["ready"]:
            time.sleep(1)
    return pc.Index(PINECONE_INDEX)

def get_embeddings():
    return PineconeEmbeddings(model=EMBED_MODEL)

def get_vectorstore():
    return PineconeVectorStore(index_name=PINECONE_INDEX, embedding=get_embeddings())

def safe_id(text: str) -> str:
    clean = text.encode("ascii", "ignore").decode()
    return re.sub(r"[^a-zA-Z0-9\-_]", "_", clean)


# ── Build vector index (run once) ─────────────────────────────────────────────
def build_vector_index(email_csv: str):
    if not os.path.exists(email_csv):
        raise FileNotFoundError(f"CSV not found: {email_csv}")

    index      = get_pinecone_index()
    embeddings = get_embeddings()
    splitter   = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    df = pd.read_csv(email_csv)
    print(f"Loading {len(df)} emails from {email_csv}...")

    docs, doc_ids = [], []
    for _, row in df.iterrows():
        body = str(row.get("body_cleaned", row.get("body", "")))
        if not body or body.lower() == "nan":
            continue
        msg_id   = str(row["message_id"])
        metadata = {"message_id": msg_id, "subject": str(row.get("subject", "")), "source": "email"}
        for i, chunk in enumerate(splitter.split_text(body)):
            docs.append(Document(page_content=chunk, metadata=metadata))
            doc_ids.append(f"email_{safe_id(msg_id)}_{i}")

    if not docs:
        print("⚠️  No usable email bodies found.")
        return

    print(f"Uploading {len(docs)} chunks...")
    for i in range(0, len(docs), 100):
        b_docs = docs[i : i + 100]
        b_ids  = doc_ids[i : i + 100]
        try:
            existing = set(index.fetch(ids=b_ids).vectors.keys())
        except Exception:
            existing = set()
        new_docs = [d for d, did in zip(b_docs, b_ids) if did not in existing]
        new_ids  = [did for did in b_ids if did not in existing]
        if new_docs:
            PineconeVectorStore(index_name=PINECONE_INDEX, embedding=embeddings).add_documents(new_docs, ids=new_ids)

    print(f"✅ Vector index ready: '{PINECONE_INDEX}'")


# ── Retrieval ─────────────────────────────────────────────────────────────────
GRAPH_STOPWORDS = {
    "what", "which", "where", "when", "who", "whom", "whose",
    "have", "does", "their", "about", "with", "from", "that",
    "this", "were", "been", "being", "most", "more", "many",
    "some", "ever", "mention", "mentioned", "sent", "received",
    "email", "emails", "communicated", "said", "saying", "related",
    "during", "tone", "context", "anywhere", "then", "know",
    "before", "raise", "raised", "internal", "both", "also",
    "there", "those", "these", "they", "them", "than", "only",
    "just", "into", "onto", "over", "under", "after", "between",
}

def extract_keywords(query: str) -> list[str]:
    words = re.sub(r"[^a-zA-Z0-9\s]", " ", query).split()
    return list(dict.fromkeys(w.strip() for w in words if len(w) > 3 and w.lower() not in GRAPH_STOPWORDS))

def retrieve_vector(query: str, top_k: int = VECTOR_TOP_K) -> list[str]:
    return [doc.page_content for doc in get_vectorstore().similarity_search(query, k=top_k)]

def retrieve_graph(query: str) -> list[str]:
    driver   = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    keywords = extract_keywords(query)
    facts    = set()
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            for kw in keywords:
                for row in session.run("""
                    MATCH (n:Entity) WHERE toLower(n.name) CONTAINS toLower($kw)
                    MATCH (n)-[r]->(t:Entity) WHERE type(r) <> 'MENTIONS'
                    RETURN n.name AS src, type(r) AS rel, t.name AS tgt LIMIT $limit
                """, kw=kw, limit=GRAPH_FACTS_LIMIT):
                    facts.add(f"{row['src']} [{row['rel']}] {row['tgt']}")

                for row in session.run("""
                    MATCH (n:Entity) WHERE toLower(n.name) CONTAINS toLower($kw)
                    MATCH (src:Entity)-[r]->(n) WHERE type(r) <> 'MENTIONS'
                    RETURN src.name AS src, type(r) AS rel, n.name AS tgt LIMIT $limit
                """, kw=kw, limit=GRAPH_FACTS_LIMIT):
                    facts.add(f"{row['src']} [{row['rel']}] {row['tgt']}")

                for row in session.run("""
                    MATCH (e:Employee) WHERE toLower(e.name) CONTAINS toLower($kw) OR toLower(e.email) CONTAINS toLower($kw)
                    MATCH (e)-[r:COMMUNICATES_WITH]->(other:Employee)
                    RETURN e.name AS src, type(r) AS rel, other.name AS tgt, r.frequency AS freq
                    ORDER BY r.frequency DESC LIMIT $limit
                """, kw=kw, limit=GRAPH_FACTS_LIMIT):
                    facts.add(f"{row['src']} [{row['rel']} x{row['freq']}] {row['tgt']}")

                for row in session.run("""
                    MATCH (emp:Employee)-[:SENT]->(e:Email)-[:HAS_ENTITY]->(en:Entity)
                    WHERE toLower(en.name) CONTAINS toLower($kw)
                    RETURN DISTINCT emp.name AS sender, en.name AS entity LIMIT $limit
                """, kw=kw, limit=GRAPH_FACTS_LIMIT):
                    facts.add(f"{row['sender']} [MENTIONED] {row['entity']}")

                for row in session.run("""
                    MATCH (emp:Employee)-[:RECEIVED]->(e:Email)-[:HAS_ENTITY]->(en:Entity)
                    WHERE toLower(en.name) CONTAINS toLower($kw)
                    RETURN DISTINCT emp.name AS sender, en.name AS entity LIMIT $limit
                """, kw=kw, limit=GRAPH_FACTS_LIMIT):
                    facts.add(f"{row['sender']} [RECEIVED_EMAIL_ABOUT] {row['entity']}")

                for row in session.run("""
                    MATCH (emp:Employee)-[:SENT]->(e:Email)-[:HAS_ENTITY]->(en:Entity)
                    WHERE toLower(en.name) CONTAINS toLower($kw)
                    RETURN emp.name AS sender, en.name AS entity, count(e) AS email_count
                    ORDER BY email_count DESC LIMIT $limit
                """, kw=kw, limit=GRAPH_FACTS_LIMIT):
                    facts.add(f"{row['sender']} [SENT_MOST_EMAILS_ABOUT] {row['entity']} (count: {row['email_count']})")
    except Exception as e:
        print(f"⚠️  Graph retrieval error: {e}")
    finally:
        driver.close()
    return list(facts)

def retrieve_hybrid(query: str) -> str:
    graph_section  = "\n".join(f"- {f}" for f in retrieve_graph(query))  or "(none found)"
    vector_section = "\n".join(f"- {c}" for c in retrieve_vector(query)) or "(none found)"
    return (
        f"=== GRAPH FACTS (structured, from knowledge graph) ===\n{graph_section}\n\n"
        f"=== EMAIL SNIPPETS (semantic, from Pinecone) ===\n{vector_section}"
    )


# ── Generation ────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an Enterprise Intelligence Assistant specializing in the Enron email dataset.
You think and communicate like a senior analyst — structured, precise, and insightful, in the style of Claude or Gemini.
Your ONLY source of truth is the context provided below. You never go beyond it.

━━━ EVIDENCE SOURCES ━━━
You are given two types of context:
[GRAPH FACTS]    — Structured, verified knowledge-graph triples extracted from Neo4j. Treat as confirmed facts.
[EMAIL SNIPPETS] — Semantic excerpts from real Enron emails retrieved from Pinecone. Treat as direct evidence.

━━━ STRICT GROUNDING RULES (NON-NEGOTIABLE) ━━━
1. Answer using ONLY the information in the provided context. Your training data does not exist for this task.
2. Cite every claim inline with its source:
    — End the sentence with (graph fact)             if it came from GRAPH FACTS.
    — End the sentence with (from email)             if it came from EMAIL SNIPPETS.
    — End the sentence with (graph fact, from email) if supported by both.
3. If context is insufficient, respond with exactly:
    "The provided context does not contain enough information to answer this question."
Then add one sentence specifying what data would be needed to answer it.
4. Never speculate, infer beyond what is stated, or fill gaps with general knowledge.

━━━ RESPONSE STYLE ━━━
Write like a senior analyst briefing an executive — the way Claude or Gemini would respond:

• Opening: 1–2 sentences that directly answer the core question.
• Body: Use **bold** for names, organizations, counts, and key terms.
        Use bullet points for lists of people, entities, or findings.
        Use numbered lists for rankings or sequential facts.
        Group related facts under a short plain-text label like "Key People:" or "Context:".
• If graph facts and email evidence both contribute, present them in clearly separated sections.
• If counts or rankings are available from graph facts, lead with those numbers prominently.
• Closing: End with a 1-sentence "Summary:" that synthesizes the key takeaway.
• Tone: Professional, neutral, and concise. Never begin with filler like "Sure!", "Certainly!", or "Great question!"
"""

HUMAN_TEMPLATE = """Context:
{context}

Question: {question}

Answer (grounded strictly in the context above):"""


def generate_answer(question: str) -> str:
    print(f"\n🔍 Query: {question}")
    context = retrieve_hybrid(question)
    headers = {"Authorization": f"Bearer {LLAMA_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": HUMAN_TEMPLATE.format(context=context, question=question)},
        ],
        "temperature": 0,
    }
    response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# ── Build vector index (uncomment once) ───────────────────────────────────────
# EMAIL_CSV = "../m2/AI-based-Knowledge-Graph-Builder-for-Enterprise-Intelligence-main/sample_email_by_category/sample_email.csv"
# build_vector_index(EMAIL_CSV)


# ── Interactive RAG query loop ────────────────────────────────────────────────
print("=== Enterprise Knowledge Graph — RAG Query Interface ===")
print("Type 'exit' to quit.\n")

while True:
    q = input("Question: ").strip()
    if not q:
        continue
    if q.lower() == "exit":
        print("Goodbye!")
        break
    print(f"\nAnswer:\n{generate_answer(q)}")
    print("-" * 60)
