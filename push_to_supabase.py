import os
import glob
import json
import re
import requests

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
TABLE_NAME = os.environ.get("SUPABASE_TABLE", "transcripts").strip()
TRANSCRIPTS_DIR = os.path.join(".", "data", "extracted_transcripts")


def push_json_files_to_supabase():
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("\n[!] Supabase Credentials Not Found in .env")
        return

    json_files = sorted(glob.glob(os.path.join(TRANSCRIPTS_DIR, "*.json")))
    if not json_files:
        print(f"[-] No JSON files found in {TRANSCRIPTS_DIR}")
        return

    print(f"=== Sanitizing and Upserting {len(json_files)} Transcripts to Supabase ===")
    print(f"Target URL: {SUPABASE_URL}/rest/v1/{TABLE_NAME}")

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates,return=representation"
    }

    # STEP 1: Process & Sanitize JSON records according to exact rules
    records_to_insert = []

    for file_path in json_files:
        filename = os.path.basename(file_path).replace(".json", "")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        meta = data.get("metadata", {})
        raw_mgmt_names = meta.get("management_names", [])
        conversations = data.get("conversations", [])

        # FIX 1: Aggressively clean management names (strip titles, honorifics, designations after hyphens)
        clean_mgmt_names = []
        for raw_name in raw_mgmt_names:
            name = re.sub(r'(?i)\b(mr\.|ms\.|mrs\.|dr\.)\s*', '', str(raw_name))
            name = name.split('-')[0].split('–')[0].strip()
            if name:
                clean_mgmt_names.append(name)

        # FIX 2: Hard-drop Moderator turns and sanitize speaker names/roles
        cleaned_conversations = []
        for item in conversations:
            speaker_str = str(item.get("speaker", "")).strip()
            content_str = str(item.get("content", "")).strip()

            # Skip Moderator / Operator / Host completely
            if re.search(r'(?i)\b(moderator|operator|host)\b', speaker_str):
                continue

            # Strip honorifics from speaker
            clean_speaker = re.sub(r'(?i)\b(mr\.|ms\.|mrs\.|dr\.)\s*', '', speaker_str).strip()

            # Check if speaker is management
            is_mgmt = any(m.lower() in clean_speaker.lower() for m in clean_mgmt_names)

            cleaned_conversations.append({
                "speaker": clean_speaker,
                "role": "Management" if is_mgmt else "Analyst",
                "content": content_str
            })

        sanitized_data = {
            "metadata": {
                "company": meta.get("company", "AXISBANK"),
                "date": meta.get("date"),
                "subject": meta.get("subject"),
                "management_names": clean_mgmt_names
            },
            "conversations": cleaned_conversations
        }

        record = {
            "filename": filename,
            "company": meta.get("company", "AXISBANK"),
            "date": meta.get("date"),
            "subject": meta.get("subject"),
            "management_names": clean_mgmt_names,
            "conversations": cleaned_conversations,
            "raw_json": sanitized_data
        }
        records_to_insert.append(record)

    # STEP 2: Upsert sanitized records to Supabase with on_conflict resolution
    post_url = f"{SUPABASE_URL}/rest/v1/{TABLE_NAME}?on_conflict=filename"
    print(f"[+] Upserting {len(records_to_insert)} freshly sanitized records to Supabase...")
    
    try:
        response = requests.post(post_url, headers=headers, json=records_to_insert, timeout=30)
        
        if response.status_code in [200, 201]:
            print(f"\n[SUCCESS] Successfully upserted all {len(records_to_insert)} cleaned transcripts into Supabase!")
        else:
            print(f"\n[-] Supabase API returned HTTP {response.status_code}")
            print(f"    Response: {response.text}")
    except Exception as e:
        print(f"[-] Error pushing to Supabase: {e}")

if __name__ == "__main__":
    push_json_files_to_supabase()
