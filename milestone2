import os
import re
import json
import logging
import requests
import time
from neo4j import GraphDatabase

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

# =====================================================================
# CONFIGURATION
# =====================================================================
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", os.getenv("NEO4J_USER", "neo4j"))
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not NEO4J_PASSWORD or NEO4J_PASSWORD == "your_neo4j_password_here":
    raise EnvironmentError("⚠️ Found placeholder 'your_neo4j_password_here' or missing NEO4J_PASSWORD in .env. Please update it with your local Neo4j password.")

LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")
if not LLAMA_API_KEY:
    raise EnvironmentError("⚠️ LLAMA_API_KEY (Ollama Cloud) not set in .env")

OPENROUTER_URL = "https://ollama.com/v1/chat/completions"
MODEL_NAME = "ministral-3:14b"

BATCH_SIZE = 10
MAX_BODY_CHARS = 3500
MAX_RETRIES = 2

ALLOWED_ENTITY_TYPES = {
    "Person", "Organization", "Location", "FinancialTerm",
    "Project", "EnergyTerm", "Regulation", "Event"
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

# =====================================================================
# TEXT PREPROCESSING 
# =====================================================================
TIMEZONE_WORDS = {"pacific", "central", "eastern", "mountain", "atlantic", "gmt", "utc", "cst", "pst", "est", "mst", "cdt", "pdt", "edt", "mdt"}
STOPWORD_ENTITIES = {"thanks", "thank", "regards", "hi", "hello", "dear", "hey", "please", "update", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "attached", "attachment", "fyi", "note", "meeting", "call", "email", "message", "forward", "original", "subject", "sent", "from", "to"}
UBIQUITOUS_ORGS = {"enron", "enronxgate", "ect", "hou", "corp", "na", "inc", "ltd", "llc", "co"}

HEADER_PATTERNS = [
    re.compile(r"-{3,}.*?Forwarded by.*?\n", re.IGNORECASE),
    re.compile(r"^(To|cc|bcc|From|Subject|Sent|Date|X-[\w-]+)\s*:.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"\S+@\S+\.\S+"),
    re.compile(r'[\w\s,."\']+@\w+(?:\s|$)', re.MULTILINE),
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}(?:\s+\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM))?\b", re.IGNORECASE),
    re.compile(r"\bon\s+\d{1,2}/\d{1,2}/\d{4}.*?\n", re.IGNORECASE),
    re.compile(r"\(?\d{3}\)?[\s\-\.]?\d{3}[\s\-\.]?\d{4}"),
    re.compile(r"\b(?:x|ext\.?)\s*\d{3,6}\b", re.IGNORECASE),
    re.compile(r".*?(passcode|pin\b|dial.in|conference bridge).*?\n", re.IGNORECASE),
    re.compile(r"(privileged|confidential|intended recipient|legal privilege).{0,300}", re.IGNORECASE | re.DOTALL),
    re.compile(r"-{3,}\s*Original Message\s*-{3,}.*", re.IGNORECASE | re.DOTALL),
    re.compile(r"\n{3,}"),
    re.compile(r'^Dear\s+[\w\s\.]+[,.]?\s*$', re.IGNORECASE | re.MULTILINE),
]

TITLE_PREFIXES = re.compile(r'^(commissioner[s]?|governor[s]?|senator[s]?|director[s]?|president|chairman|secretary)\s+', re.IGNORECASE)

def strip_headers(text: str) -> str:
    if not text: return ""
    for pattern in HEADER_PATTERNS:
        repl = "\n" if pattern.pattern != r"\n{3,}" else "\n\n"
        text = pattern.sub(repl, text)
    return text.strip()

def normalize_entity(name: str, etype: str = "") -> str:
    name = name.strip()
    name = re.sub(r"\s{2,}", " ", name)
    name = name.rstrip(".,;:!?'\"-.()")
    if etype == "Person":
        name = TITLE_PREFIXES.sub("", name).strip()
    return name

def is_garbage(name: str, etype: str) -> bool:
    lower = name.lower()
    if len(name) < 3 or lower in STOPWORD_ENTITIES or re.fullmatch(r"[\d\W]+", name): return True
    if etype == "Location" and lower in TIMEZONE_WORDS: return True
    if etype == "Person" and len(name.split()) < 2: return True
    if etype == "Organization" and (not re.search(r"[A-Z]", name) or lower in UBIQUITOUS_ORGS): return True
    return False

# =====================================================================
# SYSTEM PROMPT
# =====================================================================
SYSTEM_MESSAGE = """You are an expert, highly precise Knowledge Graph Extraction Engine.
Your ONLY purpose is to extract named entities and their semantic relationships from corporate email text.

━━━ STRICT ENTITY TYPES (You MUST classify into ONLY these 8 types) ━━━
1. Person        — Full name ONLY. Must have first & last name (e.g., "Ken Lay"). Do not include titles.
2. Organization  — Specific company, agency, or department (e.g., "FERC", "Enron", "Bank of America").
3. Location      — City, state, country, or specific region (e.g., "Houston", "California").
4. FinancialTerm — Distinct financial instrument or metric (e.g., "basis spread", "ISDA master agreement").
5. EnergyTerm    — Specific energy commodity or grid concept (e.g., "natural gas", "megawatt hours").
6. Project       — Named system, initiative, or project (e.g., "EnronOnline", "Project Raptor").
7. Regulation    — Specific law, mandate, or rule (e.g., "FERC Order 636", "Sarbanes-Oxley").
8. Event         — Specific named event or conference (e.g., "Global Energy Summit").

━━━ STRICT RELATIONSHIP RULES ━━━
Extract ALL meaningful, factual relationships between the EXACT entities you extracted.
Form: (subject, predicate, object)
• `subject` and `object` MUST EXACTLY MATCH the "name" of entities in your entities list.
• `predicate` MUST be a concise, snake_case verb or phrase (e.g., "works_at", "manages", "regulates", "discussed_with", "located_in").

━━━ NEGATIVE CONSTRAINTS (Crucial) ━━━
❌ DO NOT extract generic nouns as entities (e.g., "the meeting", "the contract", "market", "price").
❌ DO NOT extract partial names (e.g., just "Jeff" or "Mr. Skilling" -> use "Jeffrey Skilling" if knowable, else ignore).
❌ DO NOT invent relationships that are not explicitly stated or strongly implied by the text.
❌ DO NOT output markdown, conversational text, or reasoning. Output JSON ONLY.

━━━ OUTPUT SCHEMA (JSON ONLY) ━━━
{
  "entities": [{"name": "...", "type": "..."}],
  "relationships": [{"subject": "...", "predicate": "...", "object": "..."}]
}
If no entities or relationships are found, return empty lists: {"entities": [], "relationships": []}
"""

def strip_fences(text: str) -> str:
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)
    return text.strip()

class ExtractionClient:
    def __init__(self):
        self._headers = {"Authorization": f"Bearer {LLAMA_API_KEY}", "Content-Type": "application/json"}
    
    def extract(self, email_body: str) -> dict | None:
        messages = [{"role": "system", "content": SYSTEM_MESSAGE}, {"role": "user", "content": email_body}]
        
        for attempt in range(1, MAX_RETRIES + 2):
            payload = {
                "model": MODEL_NAME,
                "messages": messages,
                "temperature": 0,
            }
            raw_content = None
            try:
                response = requests.post(OPENROUTER_URL, headers=self._headers, json=payload, timeout=120)
                response.raise_for_status()
                raw = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                raw_content = raw  # Save for potential error feedback
                
                raw_clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
                cleaned = strip_fences(raw_clean)
                json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
                if json_match: cleaned = json_match.group(0)
                
                parsed = json.loads(cleaned)
                entities = parsed.get("entities", [])
                relationships = parsed.get("relationships", [])
                if isinstance(entities, list) and isinstance(relationships, list):
                    return {"entities": entities, "relationships": relationships}
            except json.JSONDecodeError as exc:
                log.warning(f"Attempt {attempt} JSON parsing failed: {exc}")
                if raw_content is not None:
                    # Append error context to message history for self-correction next attempt
                    messages.append({"role": "assistant", "content": raw_content})
                    messages.append({
                        "role": "user", 
                        "content": f"The JSON you provided in your last response was malformed. It caused this error: {exc}. Please carefully fix the syntax errors (e.g., missing commas, unescaped quotes, trailing commas) and respond with ONLY the valid JSON object."
                    })
            except Exception as exc:
                log.warning(f"Attempt {attempt} failed: {exc}")
                
            if attempt > MAX_RETRIES: break
        return None

client = ExtractionClient()
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# =====================================================================
# PIPELINE ARCHITECTURE (Neo4j -> LLM -> Neo4j)
# =====================================================================
def fetch_email_batch(skip=0, limit=BATCH_SIZE):
    """1. Fetch emails directly from Neo4j."""
    with driver.session() as session:
        result = session.run("""
            MATCH (m:Email)
            WHERE m.entity_extracted IS NULL OR m.entity_extracted = false
            RETURN m.message_id AS message_id, m.body AS body
            SKIP $skip LIMIT $limit
        """, skip=skip, limit=limit)
        return [{"message_id": r["message_id"], "body": str(r["body"] or "")} for r in result]

def get_remaining_email_count():
    """Get the total count of unprocessed emails remaining in Neo4j."""
    with driver.session() as session:
        result = session.run("""
            MATCH (m:Email)
            WHERE m.entity_extracted IS NULL OR m.entity_extracted = false
            RETURN count(m) AS remaining_count
        """)
        return result.single()["remaining_count"]

def update_neo4j(message_id: str, clean_entities: list, clean_rels: list):
    """3. Write entities + relationships, 4. Update email status"""
    with driver.session() as session:
        # Create Entity nodes and HAS_ENTITY links
        if clean_entities:
            session.run("""
                UNWIND $batch AS row
                MERGE (en:Entity {name: row.name})
                ON CREATE SET en.entity_type = row.type, en.source = "ministral-3:14b", en.created_at = datetime()
                WITH en, row
                MATCH (e:Email {message_id: row.message_id})
                MERGE (e)-[:HAS_ENTITY]->(en)
            """, batch=[{"message_id": message_id, "name": ent["name"], "type": ent["type"]} for ent in clean_entities])
            
        # Create Relationships dynamically
        for rel in clean_rels:
            # Sanitize predicate to only contain alphanumeric characters and underscores
            rel_type = re.sub(r'[^a-zA-Z0-9]', '_', rel["predicate"]).upper()
            rel_type = re.sub(r'_+', '_', rel_type).strip('_')
            if not rel_type:
                rel_type = "RELATED_TO"
                
            session.run(f"""
                MATCH (s:Entity {{name: $subject}})
                MATCH (o:Entity {{name: $object}})
                MERGE (s)-[r:`{rel_type}` {{message_id: $message_id}}]->(o)
                ON CREATE SET r.created_at = datetime()
            """, message_id=message_id, subject=rel["subject"], object=rel["object"])
            
        # Update completion status flag
        session.run("""
            MATCH (m:Email {message_id: $message_id})
            SET m.entity_extracted = true
        """, message_id=message_id)

def run_pure_pipeline():
    print(f"🚀 Starting Neo4j -> LLM -> Neo4j Pipeline (Ollama Cloud)")
    batch_size = 5
    total_processed = 0
    start = time.time()
    
    # Get initial count of emails to process
    total_remaining_at_start = get_remaining_email_count()
    print(f"📊 Total unprocessed emails remaining in database: {total_remaining_at_start}")
    
    while True:
        # Step 1: Fetch
        emails = fetch_email_batch(skip=0, limit=batch_size) # skip=0 because we flag them as extracted
        if not emails:
            print("✅ No more unprocessed emails found in Neo4j.")
            break
            
        current_remaining = get_remaining_email_count()
        print(f"\n--- Batching {len(emails)} emails from Neo4j (Remaining in DB: {current_remaining}) ---")
        
        for email in emails:
            message_id = email["message_id"]
            
            # Preprocessing
            cleaned_body = strip_headers(email["body"])
            if len(cleaned_body) > MAX_BODY_CHARS:
                cleaned_body = cleaned_body[:MAX_BODY_CHARS]
                
            # Step 2: LLM Extraction
            result = client.extract(cleaned_body)
            if not result:
                continue
                
            clean_entities = []
            valid_entity_names = set()
            
            for ent in result.get("entities", []):
                name = normalize_entity(ent["name"], ent["type"])
                if not is_garbage(name, ent["type"]) and ent["type"] in ALLOWED_ENTITY_TYPES:
                    clean_entities.append({"name": name, "type": ent["type"]})
                    valid_entity_names.add(name)
                    
            clean_rels = [
                r for r in result.get("relationships", []) 
                if normalize_entity(r["subject"]) in valid_entity_names 
                and normalize_entity(r["object"]) in valid_entity_names
            ]
            
            # Step 3 & 4: Write Database Updates
            update_neo4j(message_id, clean_entities, clean_rels)
            
            total_processed += 1
            print(f"Processed <{message_id[:15]}...> → {len(clean_entities)} entities, {len(clean_rels)} relationships")
            
        # Rate limit pause for Ollama API
        time.sleep(2)
            
    print("=" * 60)
    print(f"✅ PIPELINE FINISHED: Processed {total_processed} emails in {time.time()-start:.1f}s.")
    print("=" * 60)

if __name__ == "__main__":
    run_pure_pipeline()
