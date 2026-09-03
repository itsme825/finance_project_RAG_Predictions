import os
import re
import json
import time
import requests
from bs4 import BeautifulSoup
from datetime import datetime

# PDF extraction libraries import with fallback mechanism
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    import pypdf
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

if not (HAS_PDFPLUMBER or HAS_PYPDF):
    print("Warning: Neither 'pdfplumber' nor 'pypdf' is installed.")

# Higher management designation keywords for filtering
MANAGEMENT_TITLES = [
    "ceo", "cfo", "md", "managing director", "president",
    "executive director", "chief executive officer", "chief financial officer",
    "chief operating officer", "coo", "chairman", "group executive",
    "head investor relations", "head - investor relations", "treasurer",
    "deputy managing director"
]

HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def fetch_screener_transcript_urls(ticker="AXISBANK", years_interval=3):
    """
    Scrapes Screener.in for concall transcript PDF URLs of a company within the last N years.
    """
    url = f"https://www.screener.in/company/{ticker}/consolidated/"
    print(f"[+] Fetching Screener page for {ticker}: {url}")

    response = requests.get(url, headers=HTTP_HEADERS, timeout=15)
    if response.status_code != 200:
        url = f"https://www.screener.in/company/{ticker}/"
        response = requests.get(url, headers=HTTP_HEADERS, timeout=15)

    if response.status_code != 200:
        raise Exception(f"Failed to fetch Screener page for {ticker}. Status code: {response.status_code}")

    soup = BeautifulSoup(response.text, "html.parser")
    concalls_div = soup.find("div", class_="concalls")

    if not concalls_div:
        print("[-] Could not find concalls section directly. Searching full document for concall links...")
        concalls_div = soup

    current_year = datetime.now().year
    min_year = current_year - years_interval

    transcripts = []
    items = concalls_div.find_all("li")
    for item in items:
        text_content = item.get_text(separator=" ", strip=True)

        year_match = re.search(r'\b(20\d\d)\b', text_content)
        item_year = int(year_match.group(1)) if year_match else None

        link_tag = item.find("a", href=True, string=re.compile(r"Transcript", re.IGNORECASE))
        if not link_tag:
            for a in item.find_all("a", href=True):
                if "transcript" in a["href"].lower() or "call-transcript" in a["href"].lower():
                    link_tag = a
                    break

        if link_tag:
            pdf_url = link_tag["href"]
            if not pdf_url.startswith("http"):
                pdf_url = "https://www.screener.in" + pdf_url

            date_label_el = item.find("div", class_=re.compile(r"ink-600|font-size-15"))
            date_label = date_label_el.get_text(strip=True) if date_label_el else (str(item_year) if item_year else "Concall")

            if item_year is None or item_year >= min_year:
                transcripts.append({
                    "date_label": date_label,
                    "year": item_year,
                    "url": pdf_url
                })

    print(f"[+] Found {len(transcripts)} concall transcripts within the last {years_interval} years.")
    return transcripts


def download_pdf(pdf_url, output_path):
    """
    Downloads PDF from a URL to the local filepath.
    """
    if os.path.exists(output_path):
        print(f"    [Skipped Download] File already exists: {output_path}")
        return True

    print(f"    [Downloading] {pdf_url} -> {output_path}")
    try:
        res = requests.get(pdf_url, headers=HTTP_HEADERS, timeout=30, stream=True)
        if res.status_code == 200:
            with open(output_path, "wb") as f:
                for chunk in res.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        else:
            print(f"    [Error] HTTP {res.status_code} while downloading {pdf_url}")
            return False
    except Exception as e:
        print(f"    [Error] Failed to download {pdf_url}: {e}")
        return False


def extract_text_from_pdf(pdf_path):
    """
    Extracts text page by page from a PDF file using pdfplumber or pypdf.
    """
    text = ""
    if HAS_PDFPLUMBER:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
            return text
        except Exception as e:
            print(f"    [pdfplumber failed: {e}] Falling back to pypdf...")

    if HAS_PYPDF:
        try:
            reader = pypdf.PdfReader(pdf_path)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            return text
        except Exception as e:
            print(f"    [pypdf failed: {e}]")

    return text


def clean_speaker_name(speaker_raw):
    """
    Cleans raw speaker string (removes titles, suffixes, designations).
    """
    cleaned = re.sub(r'^(Mr\.|Ms\.|Mrs\.|Dr\.|Prof\.)\s+', '', speaker_raw, flags=re.IGNORECASE)
    cleaned = re.split(r'\s*[\–\-\:]\s*', cleaned)[0]
    cleaned = re.sub(r'\s*\([^)]*\)', '', cleaned)
    return cleaned.strip()


def parse_transcript_data(raw_text, company_name="AXISBANK"):
    """
    Parses metadata, management names, and dialogue turns from raw transcript text.
    """
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

    # 1. Date extraction
    date_match = re.search(
        r'(\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+20\d\d\b|\b\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December),?\s+20\d\d\b|\b20\d\d-\d{2}-\d{2}\b)',
        raw_text, re.IGNORECASE
    )
    call_date = date_match.group(0) if date_match else "Unknown Date"

    try:
        cleaned_date_str = re.sub(r'(st|nd|rd|th),?', '', call_date)
        for fmt in ("%B %d, %Y", "%d %B %Y", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(cleaned_date_str, fmt)
                call_date = dt.strftime("%Y-%m-%d")
                break
            except ValueError:
                pass
    except Exception:
        pass

    # 2. Subject extraction
    subject_match = re.search(r'(Q[1-4]\s*FY\d\d\b.*?(?:Call|Conference|Earnings|Results)?|Quarter\s+[1-4]\s+FY\d\d.*?(?:Call|Conference)?)', raw_text, re.IGNORECASE)
    subject = subject_match.group(0).strip() if subject_match else "Earnings Call Transcript"

    # 3. Extract Management Team (Higher Management Level)
    management_names = []
    management_names_set = set()

    header_block = raw_text[:4000]
    mgmt_section_match = re.search(
        r'(MANAGEMENT|CORPORATE PARTICIPANTS|SPEAKERS|EXECUTIVES)\s*:?\s*(.*?)(MODERATOR|ANALYST|PRESENTATION|QUESTIONS AND ANSWERS|\n\s*\n\s*\n)',
        header_block, re.DOTALL | re.IGNORECASE
    )

    mgmt_lines = mgmt_section_match.group(2).splitlines() if mgmt_section_match else header_block.splitlines()[:100]

    for line in mgmt_lines:
        line_clean = line.strip()
        if not line_clean:
            continue
        
        line_lower = line_clean.lower()
        if any(title in line_lower for title in MANAGEMENT_TITLES):
            parts = re.split(r'[\–\-\:]', line_clean, maxsplit=1)
            if parts:
                name_candidate = clean_speaker_name(parts[0])
                if name_candidate and len(name_candidate.split()) <= 4 and name_candidate.lower() not in ["management", "speakers", "corporate participants"]:
                    if name_candidate not in management_names_set:
                        management_names.append(name_candidate)
                        management_names_set.add(name_candidate)

    # 4. Conversation Dialogue Turns Extraction
    conversations = []
    current_speaker = None
    current_role = None
    current_content = []

    speaker_regex = re.compile(r'^([A-Z][a-zA-Za-z\.\s\-\'\(\)]+?)\s*:\s*(.*)', re.UNICODE)

    for line in lines:
        match = speaker_regex.match(line)
        if match:
            raw_speaker = match.group(1).strip()
            text_part = match.group(2).strip()

            if raw_speaker.lower() in ["date", "time", "page", "note", "subject", "ticker", "company"]:
                if current_speaker:
                    current_content.append(line)
                continue

            cleaned_name = clean_speaker_name(raw_speaker)
            raw_speaker_lower = raw_speaker.lower()

            if "moderator" in raw_speaker_lower or "operator" in raw_speaker_lower:
                role = "Moderator"
            elif any(m_name.lower() in cleaned_name.lower() or cleaned_name.lower() in m_name.lower() for m_name in management_names) or any(t in raw_speaker_lower for t in MANAGEMENT_TITLES):
                role = "Management"
            else:
                role = "Analyst"

            if current_speaker and current_content:
                full_text = " ".join(current_content).strip()
                if full_text:
                    conversations.append({
                        "speaker": current_speaker,
                        "role": current_role,
                        "content": full_text
                    })

            current_speaker = cleaned_name if role != "Moderator" else "Moderator"
            current_role = role
            current_content = [text_part] if text_part else []
        else:
            if current_speaker:
                current_content.append(line)

    if current_speaker and current_content:
        full_text = " ".join(current_content).strip()
        if full_text:
            conversations.append({
                "speaker": current_speaker,
                "role": current_role,
                "content": full_text
            })

    return {
        "metadata": {
            "company": company_name,
            "date": call_date,
            "subject": subject,
            "management_names": management_names
        },
        "conversations": conversations
    }


def process_transcripts(ticker="AXISBANK", years_interval=3, output_dir="axis_bank_transcripts"):
    pdf_dir = os.path.join(output_dir, "pdfs")
    txt_dir = os.path.join(output_dir, "txt_transcripts")
    json_dir = os.path.join(output_dir, "json_transcripts")

    for d in [pdf_dir, txt_dir, json_dir]:
        os.makedirs(d, exist_ok=True)

    print(f"=== Starting Axis Bank Transcript Extractor ({years_interval} Years Interval) ===")
    transcripts = fetch_screener_transcript_urls(ticker=ticker, years_interval=years_interval)

    for idx, item in enumerate(transcripts, start=1):
        url = item["url"]
        date_label = item["date_label"].replace(" ", "_").replace("/", "_")
        filename_base = f"{ticker}_{date_label}_{idx}"

        pdf_path = os.path.join(pdf_dir, f"{filename_base}.pdf")
        txt_path = os.path.join(txt_dir, f"{filename_base}.txt")
        json_path = os.path.join(json_dir, f"{filename_base}.json")

        print(f"\n[{idx}/{len(transcripts)}] Processing: {date_label} ({url})")

        # 1. Download PDF
        if not download_pdf(url, pdf_path):
            continue

        # 2. Extract Raw Text to TXT
        print(f"    [Extracting Text] -> {txt_path}")
        raw_text = extract_text_from_pdf(pdf_path)

        if not raw_text.strip():
            print(f"    [Warning] No text extracted from {pdf_path}. Skipping.")
            continue

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(raw_text)

        # 3. Parse Metadata & Dialogue into JSON
        print(f"    [Parsing JSON] -> {json_path}")
        parsed_data = parse_transcript_data(raw_text, company_name=ticker)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(parsed_data, f, indent=2, ensure_ascii=False)

        print(f"    [Success] {len(parsed_data['metadata']['management_names'])} management members identified, {len(parsed_data['conversations'])} dialogue turns parsed.")
        time.sleep(1)

    print(f"\n=== Processing finished! Results saved to directory: {output_dir} ===")


if __name__ == "__main__":
    process_transcripts(ticker="AXISBANK", years_interval=3, output_dir="axis_bank_transcripts")
