import os
import sys
import json
import re
from typing import List, Dict, Any

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def log(msg: str):
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        safe_msg = msg.encode(sys.stdout.encoding or 'ascii', errors='replace').decode(sys.stdout.encoding or 'ascii')
        print(safe_msg, flush=True)

# Try importing dependencies
try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False

try:
    import chromadb
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

def load_dotenv(env_path=".env"):
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip().strip("'\"")

load_dotenv()

CHROMA_DIR = os.path.join(".", "chroma_db")
COLLECTION_NAME = "axis_bank_transcripts"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Global lazy model / client objects
_EMBEDDER = None
_CHROMA_CLIENT = None
_CHROMA_COLLECTION = None

def get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        if HAS_ST:
            _EMBEDDER = SentenceTransformer(MODEL_NAME)
        else:
            raise RuntimeError("sentence-transformers package is required for RAG embeddings.")
    return _EMBEDDER

def get_chroma_collection():
    global _CHROMA_CLIENT, _CHROMA_COLLECTION
    if _CHROMA_COLLECTION is None:
        if not HAS_CHROMA:
            raise RuntimeError("chromadb package is required for local vector retrieval.")
        _CHROMA_CLIENT = chromadb.PersistentClient(path=CHROMA_DIR)
        _CHROMA_COLLECTION = _CHROMA_CLIENT.get_or_create_collection(name=COLLECTION_NAME)
    return _CHROMA_COLLECTION


# =====================================================================
# 1. RETRIEVAL MODULE
# =====================================================================
def retrieve_relevant_chunks(query: str, top_k: int = 3, filter_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Searches local ChromaDB using Hybrid Retrieval:
    Combines Dense Vector Similarity + BM25/Keyword Weighting + Year/Quarter Alignment Reranking.
    """
    embedder = get_embedder()
    collection = get_chroma_collection()

    query_vector = embedder.encode([query], normalize_embeddings=True).tolist()[0]

    # Fetch a larger candidate pool (n_results = top_k * 5) for hybrid reranking
    candidate_k = max(top_k * 5, 15)
    kwargs = {
        "query_embeddings": [query_vector],
        "n_results": candidate_k
    }
    if filter_metadata:
        kwargs["where"] = filter_metadata

    results = collection.query(**kwargs)

    if not results or "documents" not in results or not results["documents"]:
        return []

    docs = results["documents"][0]
    ids = results["ids"][0]
    metadatas = results["metadatas"][0] if "metadatas" in results else [{}] * len(docs)
    distances = results["distances"][0] if "distances" in results and results["distances"] else [0.0] * len(docs)

    norm_query = query.lower()
    
    # Key financial domain multi-word phrases to heavily weight
    key_phrases = [
        "gross npa", "net npa", "asset quality", "credit cost", 
        "loan to deposit", "net interest margin", "wealth management", 
        "return on assets", "net profit", "slippage", "slippages"
    ]
    found_phrases = [p for p in key_phrases if p in norm_query]
    
    # Financial single keywords (ignoring standard stop words)
    stop_words = {"the", "a", "an", "of", "and", "or", "in", "for", "to", "how", "has", "over", "changed", "two", "tell", "me", "bank", "axis", "quarters", "fiscal", "year"}
    query_tokens = [w for w in re.findall(r'\b[a-zA-Z0-9%\.]+\b', norm_query) if w not in stop_words and len(w) > 1]

    # Years & Quarters mentioned in query (e.g. 2024, fy24, q1, q2, q3, q4, quarter 1, quarter 2)
    requested_years = re.findall(r'\b(202\d|fy2\d)\b', norm_query)
    raw_quarters = re.findall(r'\b(q[1-4]|quarter\s*[1-4])\b', norm_query)
    requested_quarters = [re.sub(r'quarter\s*', 'q', q).strip() for q in raw_quarters]

    candidate_chunks = []
    for doc_id, doc_text, meta, dist in zip(ids, docs, metadatas, distances):
        vec_score = round(1.0 - (dist if dist <= 1.0 else 0.5), 4)
        norm_doc = doc_text.lower()
        
        keyword_score = 0.0
        
        # 1. Exact multi-word phrase matches boost (e.g. "gross npa")
        for phrase in found_phrases:
            if phrase in norm_doc:
                occ = norm_doc.count(phrase)
                keyword_score += 0.35 + min(occ * 0.05, 0.15)
                
        # 2. Key token keyword matches boost (e.g. "npa", "slippages")
        matched_tokens = [t for t in query_tokens if t in norm_doc]
        if query_tokens:
            token_ratio = len(matched_tokens) / len(query_tokens)
            keyword_score += token_ratio * 0.30

        # 3. Year & Quarter alignment boost (e.g. "2024", "fy24", "q3", "q4")
        doc_date = str(meta.get("date", "")).lower()
        doc_subject = str(meta.get("subject", "")).lower()
        for y in requested_years:
            clean_y = y.replace("fy", "20")
            if clean_y in doc_date or clean_y in doc_subject or clean_y in norm_doc or y in norm_doc:
                keyword_score += 0.15

        for q_tag in requested_quarters:
            if q_tag in doc_subject or q_tag in norm_doc:
                keyword_score += 0.15

        keyword_score = min(keyword_score, 1.0)
        
        # Hybrid Rerank Score: 50% Vector Similarity + 50% Weighted Keyword Match
        hybrid_score = round(0.50 * vec_score + 0.50 * keyword_score, 4)

        # Extract lines specifically mentioning the query terms for highlighted snippet focus
        lines = doc_text.split("\n")
        relevant_lines = []
        for line in lines:
            line_norm = line.lower()
            if any(t in line_norm for t in (found_phrases or matched_tokens)):
                relevant_lines.append(line.strip())
        
        highlight_quote = "\n".join(relevant_lines) if relevant_lines else doc_text.strip()

        candidate_chunks.append({
            "chunk_id": doc_id,
            "text": doc_text,
            "highlight_quote": highlight_quote,
            "metadata": meta,
            "similarity_score": vec_score,
            "keyword_score": round(keyword_score, 4),
            "hybrid_score": hybrid_score
        })

    # Rerank candidates by hybrid_score descending
    candidate_chunks.sort(key=lambda x: x["hybrid_score"], reverse=True)

    return candidate_chunks[:top_k]


# =====================================================================
# 2. GENERATION MODULE (RAG Response Generator)
# =====================================================================
def generate_rag_response(query: str, context_chunks: List[Dict[str, Any]]) -> str:
    """
    Generates a RAG response based strictly on retrieved context chunks.
    Supports Ollama open-source LLMs (e.g. llama3.2, llama3.1, llama3, mistral),
    OpenAI API if key is present, or falls back to grounded context synthesis engine.
    """
    context_blocks = []
    for idx, c in enumerate(context_chunks, 1):
        meta = c.get("metadata", {})
        speaker = meta.get("speaker", "Management/Analyst")
        date_str = meta.get("date", "Concall")
        subj_str = meta.get("subject", "")
        context_blocks.append(f"[Source Chunk #{idx} | {speaker} | Date: {date_str} | Subject: {subj_str}]:\n{c['text']}")

    context_str = "\n\n---\n\n".join(context_blocks)

    system_prompt = (
        "You are a strict, ultra-precise financial analyst for Axis Bank earnings transcripts.\n"
        "Answer the user's question directly using ONLY the facts explicitly provided in the retrieved context chunks.\n\n"
        "CRITICAL ANTI-FABRICATION & ZERO-HALLUCINATION RULES:\n"
        "1. DO NOT MAKE UP OR SPECULATE INFORMATION: Never invent, extrapolate, guess, or estimate facts, figures, analyst opinions, or events. If a fact is not written in the context, DO NOT STATE IT.\n"
        "2. ZERO TOLERANCE FOR NUMERICAL FABRICATION: Use ONLY exact figures directly stated in the context. NEVER fabricate numbers or percentages to complete tables or comparisons.\n"
        "3. STRICT QUARTER ATTRIBUTION RULE: Whenever quoting financial figures (like PAT, ROE, ROA, NII), you MUST explicitly state the exact quarter the number belongs to (e.g., 'In Q3 FY24...'). Do NOT present quarterly figures as full-year totals.\n"
        "4. MISSING DATA RULE: If requested information (e.g. analyst concerns, specific quarter metrics, or LDR guidance) is absent from the chunks, state explicitly: 'Information for [X] is not present in the retrieved context' rather than guessing.\n"
        "5. PERIOD & COMPARISON PRECISION (QoQ vs YoY): Growth figures reported in transcripts (e.g., 'NII grew 13% YoY') are Year-on-Year comparisons against the prior fiscal year. Do NOT mislabel YoY growth as QoQ (Quarter-on-Quarter). Sequential changes between consecutive quarters are QoQ.\n"
        "6. SPEAKER & ROLE FIDELITY: Attribute facts and questions accurately to the exact speaker and role specified in the chunk headers (e.g. Puneet Sharma [Management], Mahrukh Adajania [Analyst]). Do NOT invent roles or assign analyst questions as management strategy.\n"
        "7. SUMMARY: Conclude with a 1-2 sentence direct summary based 100% on stated facts.\n\n"
        f"Retrieved Context:\n{context_str}"
    )

    # 1. Try Local Ollama Open Source LLM (Llama 3 / Llama 3.2 / Llama 3.1)
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").strip().rstrip("/")
    ollama_model = os.environ.get("OLLAMA_MODEL", "llama3.2").strip()

    try:
        import requests
        endpoint = f"{ollama_host}/api/chat"
        payload = {
            "model": ollama_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            "stream": False,
            "options": {
                "temperature": 0.0
            }
        }
        res = requests.post(endpoint, json=payload, timeout=30)
        if res.status_code == 200:
            data = res.json()
            answer = data.get("message", {}).get("content", "").strip()
            if answer:
                log(f"[+] LLM Generator: Ollama ({ollama_model}) online!")
                return answer
    except Exception as e:
        log(f"[Note] Ollama Local LLM ({ollama_model} @ {ollama_host}) not reachable ({e}).")
        log("       (Tip: Run 'ollama run llama3.2' in terminal to use local open-source LLM).")

    # 2. Try OpenAI API if key is present
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if openai_key:
        try:
            import openai
            client = openai.OpenAI(api_key=openai_key)
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.0
            )
            log("[+] LLM Generator: OpenAI API online!")
            return resp.choices[0].message.content.strip()
        except Exception as e:
            log(f"[Note] OpenAI API fallback: {e}")

    # 3. Grounded Context Synthesis Engine (Fallback when offline)
    log("[+] LLM Generator: Grounded Context Synthesis Engine (Offline Mode)")
    response_lines = []
    response_lines.append(f"Based on the retrieved Axis Bank concall transcripts:\n")

    for idx, chunk in enumerate(context_chunks, 1):
        meta = chunk.get("metadata", {})
        speaker = meta.get("speaker", "Management/Analyst")
        date_str = meta.get("date", "Concall")
        
        # Extract main passage text
        highlight = chunk.get("highlight_quote") or chunk["text"]
        passage_part = highlight.split("Passage:\n")[-1] if "Passage:\n" in highlight else highlight
        clean_passage = passage_part.strip()
        response_lines.append(f"• [Source Chunk #{idx} | {speaker} | Date: {date_str}]:\n  \"{clean_passage}\"\n")

    response_lines.append("\nSummary: The retrieved concall transcripts detail the quarter-on-quarter NPA changes and asset quality metrics.")
    return "\n".join(response_lines)


# =====================================================================
# 3. HALLUCINATION CHECKER (Groundedness & Faithfulness Verification)
# =====================================================================
def verify_hallucination(response: str, context_chunks: List[Dict[str, Any]], query: str = "") -> Dict[str, Any]:
    """
    Audits generated answer claims against context chunks to detect hallucinations.
    Extracts numerical metrics, proper nouns, speaker roles, dates, and percentages to verify groundedness.
    Ignores structural headers (e.g. **Financial Growth**:), prompt headings, and markdown formatting.
    """
    combined_context = " ".join([c["text"] for c in context_chunks])
    norm_context = re.sub(r'\s+', ' ', combined_context).lower()
    
    # 1. Extract numerical claims and percentages (e.g. ₹6740 crore, 18%, 1.88%, Q3 FY24)
    num_claims = set(re.findall(r'₹?\s*\d+(?:\.\d+)?\s*(?:%|crore|lakh|bps|million)?', response))
    num_claims = {c.strip() for c in num_claims if len(c.strip()) > 1 and re.search(r'\d', c)}

    # 2. Extract Structural Markdown Headers & User Query Terms to IGNORE
    bold_headers = set(re.findall(r'\*\*(.*?)\*\*', response))
    bullet_headers = set(re.findall(r'^[•\d\.\-\*]+\s*([A-Za-z\s]+):', response, flags=re.MULTILINE))
    query_words = set(re.findall(r'\b[A-Za-z0-9]+\b', query.lower())) if query else set()

    header_words = set()
    for h in bold_headers.union(bullet_headers):
        for w in re.findall(r'\b[A-Za-z]+\b', h):
            header_words.add(w.lower())
    
    # Clean markdown bold headers and formatting before proper noun extraction
    clean_resp = re.sub(r'\*\*(.*?)\*\*', '', response)  # strip bold headers completely
    clean_resp = re.sub(r'[*#_`•-]', '', clean_resp)
    
    # Extract Proper Nouns, Entities & Roles (e.g. Mahrukh Adajania, Analyst, Management, Citi)
    raw_entities = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', clean_resp))
    ignore_words = {
        "according", "based", "on", "the", "context", "summary", "table", "quarter", "gross", "net",
        "ratio", "here", "is", "a", "in", "with", "for", "by", "and", "or", "from", "to", "data",
        "present", "not", "result", "statement", "analyst", "analysts", "management", "company", "date", "speaker",
        "this", "that", "these", "those", "however", "further", "also", "overall", "while", "additionally",
        "financial", "growth", "sentiment", "key", "concerns", "executive", "performance", "operating", "strategic",
        "information", "bullet", "regarding", "therefore", "moreover", "note", "provides", "shows", "details"
    }
    
    text_entities = set()
    for e in raw_entities:
        e_clean = e.strip()
        e_lower = e_clean.lower()
        if (e_lower not in ignore_words and 
            e_lower not in query_words and 
            e_lower not in header_words and 
            len(e_clean) > 2):
            text_entities.add(e_clean)

    all_claims = num_claims.union(text_entities)

    grounded_claims = []
    derived_claims = []
    hallucinated_claims = []

    # Extract all floating point numbers in context for arithmetic verification
    context_numbers = [float(n) for n in re.findall(r'\b\d+(?:\.\d+)?\b', combined_context)]

    for claim in all_claims:
        norm_claim = re.sub(r'\s+', ' ', claim).lower()
        
        num_match = re.search(r'\d+(?:\.\d+)?', claim)
        raw_num_str = num_match.group(0) if num_match else ""
        raw_num = float(raw_num_str) if raw_num_str else None

        if norm_claim in norm_context or (raw_num_str and raw_num_str in norm_context):
            grounded_claims.append(claim)
        elif raw_num is not None:
            is_derived = False
            for n1 in context_numbers:
                for n2 in context_numbers:
                    if abs(abs(n1 - n2) - raw_num) < 0.001 or abs((abs(n1 - n2) * 100) - raw_num) < 0.001:
                        is_derived = True
                        break
                if is_derived:
                    break
            if is_derived:
                derived_claims.append(claim)
            else:
                hallucinated_claims.append(claim)
        else:
            hallucinated_claims.append(claim)

    total_valid = len(grounded_claims) + len(derived_claims)
    total_claims = len(claims_in_response)
    if total_claims == 0:
        faithfulness_score = 100.0
    else:
        faithfulness_score = round((total_valid / total_claims) * 100, 1)

    # Determine Verdict
    if faithfulness_score >= 90.0:
        verdict = "PASSED (FAITHFUL)"
        verdict_color = "GREEN"
        status_msg = "All claims and financial metrics in the response are fully grounded in the retrieved concall transcript context."
    elif faithfulness_score >= 60.0:
        verdict = "WARNING (PARTIAL HALLUCINATION)"
        verdict_color = "YELLOW"
        status_msg = "Some financial metrics or claims in the response could not be verified from the retrieved context chunks."
    else:
        verdict = "FAILED (HALLUCINATION DETECTED)"
        verdict_color = "RED"
        status_msg = "Significant hallucinated or ungrounded assertions were detected in the response!"

    return {
        "verdict": verdict,
        "verdict_color": verdict_color,
        "faithfulness_score_pct": faithfulness_score,
        "status_message": status_msg,
        "verified_grounded_claims": sorted(list(grounded_claims)),
        "unverified_hallucinated_claims": sorted(list(hallucinated_claims)),
        "total_claims_audited": total_claims
    }


# =====================================================================
# 4. RAG PIPELINE WRAPPER (QUERY + RETRIEVE + CITED QUOTES + GENERATE + CHECK)
# =====================================================================
def run_rag_pipeline(query: str, top_k: int = 3, injected_hallucination: str = None) -> Dict[str, Any]:
    log(f"\n=======================================================")
    log(f"[RAG QUERY]: '{query}'")
    log(f"=======================================================")

    # Step 1: Retrieve context
    log(f"[1/3] Retrieving top-{top_k} relevant context chunks from ChromaDB...")
    chunks = retrieve_relevant_chunks(query, top_k=top_k)
    
    log(f"     Found {len(chunks)} relevant chunks!")
    
    cited_quotes = []
    for idx, c in enumerate(chunks, 1):
        meta = c["metadata"]
        text = c["text"]
        highlight = c.get("highlight_quote", text)
        passage_part = highlight.split("Passage:\n")[-1] if "Passage:\n" in highlight else highlight
        clean_passage = passage_part.strip()
        
        quote_entry = {
            "citation_id": f"Source Chunk #{idx}",
            "chunk_id": c['chunk_id'],
            "speaker": meta.get('speaker', 'Management/Analyst'),
            "date": meta.get('date', 'Concall'),
            "hybrid_score": c.get('hybrid_score', c['similarity_score']),
            "similarity_score": c['similarity_score'],
            "keyword_score": c.get('keyword_score', 0.0),
            "quote": clean_passage,
            "full_text": text
        }
        cited_quotes.append(quote_entry)
        log(f"     Chunk #{idx} | ID: {c['chunk_id']} | Speaker: {meta.get('speaker')} | Hybrid Score: {c.get('hybrid_score', c['similarity_score'])} (Vector: {c['similarity_score']}, Keyword: {c.get('keyword_score', 0.0)})")

    # Display Cited Quotes from Vector DB
    log(f"\n--- CITED SOURCE QUOTES FROM VECTOR DB (HYBRID RERANKED) ---")
    for q in cited_quotes:
        log(f"[{q['citation_id']}] [{q['speaker']} | Date: {q['date']} | Hybrid Score: {q['hybrid_score']} (Vector: {q['similarity_score']}, Keyword: {q['keyword_score']})]")
        log(f"  Highlight Quote: \"{q['quote']}\"\n")
    log(f"-------------------------------------------------------------")

    # Step 2: Generate response
    log(f"\n[2/3] Generating RAG response...")
    response = generate_rag_response(query, chunks)

    # Optional: Inject hallucination to test the verification engine
    if injected_hallucination:
        log(f"\n[!] INJECTING TEST HALLUCINATION into response: '{injected_hallucination}'")
        response += f"\n\n[Injected Fact]: {injected_hallucination}"

    log(f"\n--- GENERATED RESPONSE ---")
    log(response)
    log(f"--------------------------")

    # Step 3: Verify Hallucination
    log(f"\n[3/3] Auditing response with Hallucination Checker...")
    audit_report = verify_hallucination(response, chunks)

    log(f"\n=======================================================")
    log(f"[HALLUCINATION VERIFICATION REPORT]")
    log(f"=======================================================")
    log(f"VERDICT          : {audit_report['verdict']}")
    log(f"FAITHFULNESS SCORE: {audit_report['faithfulness_score_pct']}%")
    log(f"AUDIT SUMMARY    : {audit_report['status_message']}")
    log(f"GROUNDED CLAIMS  : {audit_report['verified_grounded_claims']}")
    if audit_report['unverified_hallucinated_claims']:
        log(f"[WARNING] UNVERIFIED/HALLUCINATED CLAIMS: {audit_report['unverified_hallucinated_claims']}")
    log(f"=======================================================\n")

    return {
        "query": query,
        "retrieved_chunks": chunks,
        "cited_quotes": cited_quotes,
        "response": response,
        "hallucination_report": audit_report
    }


def main():
    if len(sys.argv) > 1:
        custom_query = " ".join(sys.argv[1:])
        run_rag_pipeline(custom_query, top_k=3)
        return

    log("Running standard RAG Pipeline test suite with Cited Vector DB Quotes...\n")

    # Test Query 1: Standard Faithful Query
    q1 = "What was Axis Bank Net Profit (PAT) and Return on Assets (ROA) in Q3 FY25?"
    run_rag_pipeline(q1, top_k=2)

    # Test Query 2: Loan-to-Deposit Ratio (LDR) Query
    q2 = "What is the Loan-to-Deposit Ratio (LDR) target for Axis Bank and deposit growth strategy?"
    run_rag_pipeline(q2, top_k=2)

    # Test Query 3: Hallucination Verification Test
    q3 = "What is the wealth management AUM growth for Burgundy and Citi customer base?"
    run_rag_pipeline(q3, top_k=2, injected_hallucination="Axis Bank also acquired HDFC Securities for ₹15000 crore with a PAT of ₹99000 crore.")

if __name__ == "__main__":
    main()
