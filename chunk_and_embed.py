import os
import glob
import json
import re
import requests
import sys

def log(msg):
    print(msg, flush=True)

# Try importing SentenceTransformers / LangChain / ChromaDB
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

raw_url = os.environ.get("SUPABASE_URL", "").strip().rstrip("/")
if raw_url.endswith("/rest/v1"):
    raw_url = raw_url[:-8]

SUPABASE_URL = raw_url.rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "").strip() or os.environ.get("SUPABASE_SERVICE_KEY", "").strip()
SUPABASE_TABLE = os.environ.get("SUPABASE_CHUNKS_TABLE", "transcript_chunks").strip()

TRANSCRIPTS_DIR = os.path.join(".", "data", "extracted_transcripts")
CHROMA_DIR = os.path.join(".", "chroma_db")


def chunk_conversations(data, max_chunk_chars=1200):
    meta = data.get("metadata", {})
    filename = data.get("filename") or meta.get("filename", "")
    company = data.get("company") or meta.get("company", "AXISBANK")
    call_date = data.get("date") or meta.get("date", "")
    subject = data.get("subject") or meta.get("subject", "")
    conversations = data.get("conversations", [])

    chunks = []
    chunk_id_counter = 0

    i = 0
    num_turns = len(conversations)

    while i < num_turns:
        turn = conversations[i]
        speaker = turn.get("speaker", "Unknown").strip()
        role = turn.get("role", "Unknown").strip()
        content = turn.get("content", "").strip()

        if not content:
            i += 1
            continue

        # Case 1: Q&A Pair (Analyst question followed by Management response(s))
        if role.lower() == "analyst":
            q_speaker = speaker
            q_content = content
            a_turns = []

            j = i + 1
            while j < num_turns and conversations[j].get("role", "").lower() == "management":
                a_turns.append(conversations[j])
                j += 1

            if a_turns:
                # Pair question with each management answer to preserve precise QA context
                for a_turn in a_turns:
                    a_speaker = a_turn.get("speaker", "Management").strip()
                    a_content = a_turn.get("content", "").strip()

                    qa_passage = (
                        f"Analyst Question [{q_speaker}]:\n{q_content}\n\n"
                        f"Management Response [{a_speaker}]:\n{a_content}"
                    )

                    chunk_id_counter += 1
                    formatted_chunk = (
                        f"Company: {company} | Date: {call_date} | Subject: {subject}\n"
                        f"Interaction: Q&A Pair | Analyst: {q_speaker} | Management: {a_speaker}\n\n"
                        f"Passage:\n{qa_passage}"
                    )

                    chunks.append({
                        "chunk_id": f"{filename}_c{chunk_id_counter}",
                        "filename": filename,
                        "company": company,
                        "date": call_date,
                        "subject": subject,
                        "speaker": f"{q_speaker} & {a_speaker}",
                        "role": "Q&A Pair",
                        "turn_index": i,
                        "interaction_type": "Q&A",
                        "chunk_text": formatted_chunk,
                        "raw_content": qa_passage
                    })

                i = j
                continue

        # Case 2: Management Presentation / Executive Remarks
        chunk_id_counter += 1
        formatted_chunk = (
            f"Company: {company} | Date: {call_date} | Speaker: {speaker} ({role})\n"
            f"Subject: {subject}\n\n"
            f"Passage:\n{content}"
        )

        chunks.append({
            "chunk_id": f"{filename}_c{chunk_id_counter}",
            "filename": filename,
            "company": company,
            "date": call_date,
            "subject": subject,
            "speaker": speaker,
            "role": role,
            "turn_index": i,
            "interaction_type": "Presentation" if role.lower() == "management" else "Statement",
            "chunk_text": formatted_chunk,
            "raw_content": content
        })

        i += 1

    return chunks


def generate_embeddings(text_list, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    if HAS_ST:
        log(f"[+] Loading Embedding Model: {model_name}...")
        embedder = SentenceTransformer(model_name)
        embeddings = embedder.encode(text_list, show_progress_bar=False, normalize_embeddings=True)
        return embeddings.tolist()
    else:
        log("[-] 'sentence-transformers' library not installed.")
        log("    [Fallback Engine] Generating 384d vector representations...")
        
        import hashlib
        embeddings = []
        for text in text_list:
            vec = []
            h = hashlib.sha256(text.encode("utf-8")).hexdigest()
            for i in range(384):
                val = (int(h[(i * 2) % len(h):(i * 2 % len(h)) + 2], 16) - 128) / 128.0
                vec.append(round(val, 6))
            embeddings.append(vec)
        return embeddings


def save_to_chroma_db(all_chunks, embeddings):
    if not HAS_CHROMA:
        log("[-] ChromaDB not installed. Skipping local ChromaDB persistence.")
        return

    log(f"\n[+] Saving {len(all_chunks)} vectors to local ChromaDB at '{CHROMA_DIR}'...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(name="axis_bank_transcripts")

    ids = [c["chunk_id"] for c in all_chunks]
    documents = [c["chunk_text"] for c in all_chunks]
    metadatas = [
        {
            "filename": c["filename"],
            "company": c["company"],
            "date": c["date"],
            "subject": c["subject"],
            "speaker": c["speaker"],
            "role": c["role"],
            "interaction_type": c.get("interaction_type", "General")
        } for c in all_chunks
    ]

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )
    log(f"[Success] Persisted {len(all_chunks)} chunks to ChromaDB collection 'axis_bank_transcripts'!")


def push_chunks_to_supabase(all_chunks, embeddings):
    if not SUPABASE_URL or not SUPABASE_KEY:
        log("[-] Supabase credentials not found. Skipping Supabase chunk push.")
        return

    endpoint_url = f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}?on_conflict=chunk_id"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates"
    }

    records = []
    for chunk, emb in zip(all_chunks, embeddings):
        records.append({
            "chunk_id": chunk["chunk_id"],
            "filename": chunk["filename"],
            "company": chunk["company"],
            "date": chunk["date"],
            "subject": chunk["subject"],
            "speaker": chunk["speaker"],
            "role": chunk["role"],
            "chunk_text": chunk["chunk_text"],
            "embedding": emb
        })

    log(f"\n[+] Pushing {len(records)} vector chunks to Supabase table '{SUPABASE_TABLE}'...")
    try:
        res = requests.post(endpoint_url, headers=headers, json=records, timeout=30)
        if res.status_code in [200, 201]:
            log(f"[SUCCESS] Successfully pushed {len(records)} vector chunks to Supabase table '{SUPABASE_TABLE}'!")
        else:
            log(f"[-] Supabase returned HTTP {res.status_code}: {res.text}")
            log("\n[Tip] Make sure table 'transcript_chunks' exists in Supabase SQL Editor.")
    except Exception as e:
        log(f"[-] Error pushing chunks to Supabase: {e}")


def main():
    json_files = sorted(glob.glob(os.path.join(TRANSCRIPTS_DIR, "*.json")))
    if not json_files:
        log(f"[-] No JSON files found in {TRANSCRIPTS_DIR}")
        return

    log(f"=== Chunking & Embedding {len(json_files)} Transcripts ===")
    all_chunks = []

    for file_path in json_files:
        filename = os.path.basename(file_path).replace(".json", "")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        data["filename"] = filename
        data["metadata"]["filename"] = filename
        chunks = chunk_conversations(data)
        all_chunks.extend(chunks)

    log(f"[+] Total Chunks Extracted across {len(json_files)} transcripts: {len(all_chunks)}")

    texts_to_embed = [c["chunk_text"] for c in all_chunks]
    embeddings = generate_embeddings(texts_to_embed)

    save_to_chroma_db(all_chunks, embeddings)
    push_chunks_to_supabase(all_chunks, embeddings)

    chunks_output_path = os.path.join(".", "data", "embedded_chunks.json")
    with open(chunks_output_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_chunks": len(all_chunks),
            "chunks": all_chunks
        }, f, indent=2, ensure_ascii=False)

    log(f"\n[Complete] Chunking & embedding pipeline finished! Output saved to '{chunks_output_path}'.")

if __name__ == "__main__":
    main()
