import os
import json
import re

# Setup Directory Structure
base_data_dir = os.path.join(".", "data")
pdf_dir = os.path.join(base_data_dir, "extracted_pdf")
transcripts_dir = os.path.join(base_data_dir, "extracted_transcripts")

os.makedirs(pdf_dir, exist_ok=True)
os.makedirs(transcripts_dir, exist_ok=True)

years = [2024, 2025, 2026, 2027]
quarters_config = [
    {
        "t_code": "T1",
        "month_day": "01-23",
        "q_name": "Q3",
        "date_str": "23rd January"
    },
    {
        "t_code": "T2",
        "month_day": "04-24",
        "q_name": "Q4",
        "date_str": "24th April"
    },
    {
        "t_code": "T3",
        "month_day": "07-24",
        "q_name": "Q1",
        "date_str": "24th July"
    }
]

# Raw Management Names with titles & designations
raw_management_names = [
    "MR. AMITABH CHAUDHRY – MANAGING DIRECTOR & CEO, AXIS BANK LIMITED",
    "MR. RAJIV ANAND – DEPUTY MANAGING DIRECTOR, AXIS BANK LIMITED",
    "MR. SUBRAT MOHANTY - GROUP EXECUTIVE & HEAD - BANKING OPERATIONS AND TRANSFORMATION",
    "MR. PUNEET SHARMA – CHIEF FINANCIAL OFFICER, AXIS BANK LIMITED",
    "MR. MUNISH SHARDA - GROUP EXECUTIVE & HEAD - BHARAT BANKING",
    "MR. SUMIT BALI - GROUP EXECUTIVE & HEAD - RETAIL LENDING"
]

# FIX 1: Aggressively clean the management names
clean_mgmt_names = []
for raw_name in raw_management_names:
    name = re.sub(r'(?i)\b(mr\.|ms\.|mrs\.|dr\.)\s*', '', raw_name)
    name = name.split('-')[0].split('–')[0].strip()
    clean_mgmt_names.append(name)

print("=== Generating Cleaned Concall Transcripts & Metadata ===")
print("Cleaned Management Names:", clean_mgmt_names)

count = 0
for year in years:
    for q_idx, q in enumerate(quarters_config):
        count += 1
        fy_number = year % 100 if q["q_name"] in ["Q3", "Q4"] else (year + 1) % 100
        call_date = f"{year}-{q['month_day']}"
        subject = f"Axis Bank {q['q_name']} FY{fy_number} Financial Results & Strategy Call"
        filename_base = f"AXISBANK_{year}_{q['t_code']}"

        # Dynamic financial metrics per quarter
        nii_crores = 12500 + count * 310
        pat_crores = 6000 + count * 185
        opprofit_crores = 9800 + count * 260
        nim_pct = round(3.88 + (count * 0.03) % 0.30, 2)
        roe_pct = round(18.8 + (count * 0.35) % 3.0, 2)
        roa_pct = round(1.80 + (count * 0.02) % 0.22, 2)
        gnpa_pct = round(1.52 - (count * 0.02) % 0.25, 2)
        nnpa_pct = round(0.42 - (count * 0.01) % 0.12, 2)
        pcr_pct = round(76.5 + (count * 0.4) % 4.5, 1)
        credit_cost_bps = 48 + (count * 3) % 20
        cet1_pct = round(14.05 + (count * 0.12) % 1.2, 2)
        crar_pct = round(16.40 + (count * 0.15) % 1.5, 2)
        casa_ratio_pct = round(41.2 + (count * 0.2) % 2.5, 1)
        ldr_ratio_pct = round(89.2 - (count * 0.4) % 4.0, 1)
        cost_of_funds_pct = round(5.25 + (count * 0.04) % 0.45, 2)
        yield_on_advances_pct = round(9.65 + (count * 0.03) % 0.40, 2)
        cost_to_income_pct = round(48.2 - (count * 0.2) % 2.0, 1)
        burgundy_aum_lakh_cr = round(4.10 + count * 0.18, 2)
        branch_count = 4920 + count * 40
        unsecured_pct = round(10.1 + (count * 0.1) % 1.2, 1)

        raw_conversations_list = [
            {
                "speaker": "Moderator",
                "content": (
                    f"Ladies and gentlemen, good day and welcome to Axis Bank's {subject} held on {call_date}.\n"
                    f"Participation in this call is by invitation only. All lines are in listen-only mode."
                )
            },
            {
                "speaker": "Amitabh Chaudhry",
                "content": (
                    f"Executive Strategic Summary & Operating Performance:\n"
                    f"Good evening and thank you for joining. In {q['q_name']} FY{fy_number}, Axis Bank delivered a Net Profit (PAT) of ₹{pat_crores} crore, "
                    f"reflecting a YoY growth of {14 + (count % 6)}%. Our Return on Assets (ROA) expanded to {roa_pct}% "
                    f"and consolidated Return on Equity (ROE) stood at {roe_pct}%.\n\n"
                    f"Strategic Execution under GPS Pillars:\n"
                    f"1. Resilient Franchise: Total balance sheet size crossed ₹{14 + count * 0.5:.1f} lakh crore. Branch network expanded to {branch_count} branches.\n"
                    f"2. Distinctiveness & Digital Adoption: Monthly active users on our flagship retail app 'open' reached {14.5 + count * 0.6:.1f} million. "
                    f"Our wholesale digital platform 'NEO' now handles over {65 + (count % 15)}% of corporate cash management volumes.\n"
                    f"3. Customer Experience: In the independent Cantor benchmarking study, Axis Bank secured the #2 rank in Net Promoter Score (NPS) across peer private banks."
                )
            },
            {
                "speaker": "Puneet Sharma",
                "content": (
                    f"Financial Analytics & Asset Quality Metrics:\n"
                    f"1. Revenue & Margins: Net Interest Income (NII) for the quarter grew {12 + (count % 5)}% YoY to ₹{nii_crores} crore. "
                    f"Net Interest Margin (NIM) was {nim_pct}%. Core Operating Profit rose to ₹{opprofit_crores} crore. Cost-to-Income ratio improved to {cost_to_income_pct}%.\n\n"
                    f"2. Balance Sheet Yields & Costs: Yield on advances stood at {yield_on_advances_pct}%, while cost of funds was {cost_of_funds_pct}%, resulting in a healthy interest spread of {yield_on_advances_pct - cost_of_funds_pct:.2f}%.\n\n"
                    f"3. Asset Quality & Capital Ratios: Gross NPA ratio declined by {4 + (count % 3)} bps QoQ to {gnpa_pct}%, and Net NPA ratio stood at {nnpa_pct}%. "
                    f"Provision Coverage Ratio (PCR) on gross NPAs reached {pcr_pct}%. Annualized credit cost was {credit_cost_bps} bps.\n\n"
                    f"4. Capital & Liquidity: Basel III Capital Adequacy Ratio (CRAR) stood at {crar_pct}%, with CET-1 ratio at {cet1_pct}%. "
                    f"Loan-to-Deposit Ratio (LDR) was controlled at {ldr_ratio_pct}%."
                )
            },
            {
                "speaker": "Mahrukh Adajania (HSBC)",
                "content": (
                    f"Analytical Query - Deposit Repricing, LDR & Citi Integration Synergy:\n"
                    f"Thank you. I have two specific analytical questions:\n"
                    f"Q1: Given system liquidity conditions, what is your guidance on terminal cost of deposits, and how fast will LDR reach your targeted 85% mark?\n"
                    f"Q2: For Rajiv—could you quantify the wealth management AUM growth, cross-sell ratio, and annual churn rate within the acquired Citi customer base?"
                )
            },
            {
                "speaker": "Rajiv Anand",
                "content": (
                    f"Response on Citi Integration Metrics & Wealth Management AUM:\n"
                    f"Thanks Mahrukh. On Citi integration:\n"
                    f"- Wealth Management AUM: Overall AUM under 'Burgundy' and Citi wealth accounts increased to ₹{burgundy_aum_lakh_cr} lakh crore, representing a YoY growth of {22 + (count % 6)}%.\n"
                    f"- Attrition Rate: Customer retention post-integration remains best-in-class, with annual attrition under 1.8% across credit cards and high-net-worth accounts.\n"
                    f"- Cross-Sell Multiplier: Cross-sell ratio of liability products and home loans to Citi cardholders increased by {32 + (count % 8)}% YoY, delivering higher fee intensity per customer."
                )
            },
            {
                "speaker": "Puneet Sharma",
                "content": (
                    f"Response on LDR Trajectory & Deposit Cost Guidance:\n"
                    f"Regarding LDR and cost of funds:\n"
                    f"- CASA & Deposit Breakdown: CASA ratio stood at {casa_ratio_pct}%. Retail term deposits grew {17 + (count % 4)}% YoY, providing granular liquidity support.\n"
                    f"- LDR Target: LDR reduced to {ldr_ratio_pct}% this quarter. We are growing advances at {13 + (count % 3)}% while targeting deposit growth at {15 + (count % 3)}%, ensuring LDR smoothly converges to 85% over the next 2-3 quarters.\n"
                    f"- Cost of Funds Trajectory: Incremental cost of deposits has plateaued, with sequential increase contained to just {3 + (count % 3)} bps."
                )
            },
            {
                "speaker": "Suresh Ganapathy (Macquarie)",
                "content": (
                    f"Analytical Query - Unsecured Credit Stress, Delinquencies & RBI Risk Weights:\n"
                    f"My question is on unsecured retail assets (Personal Loans & Credit Cards). "
                    f"What is the current share of unsecured loans in overall advances, what are the 30+ and 90+ DPD delinquency trends, and how did the RBI risk weight recalibration impact CET-1?"
                )
            },
            {
                "speaker": "Sumit Bali",
                "content": (
                    f"Response on Unsecured Retail Portfolio & Risk Analytics:\n"
                    f"Hi Suresh, Sumit here. Here are the precise retail lending metrics:\n"
                    f"- Portfolio Mix: Unsecured retail advances (personal loans + credit cards) represent exactly {unsecured_pct}% of total bank advances.\n"
                    f"- Customer Profile: Over 83% of personal loan borrowers are salaried individuals working with ET 500 / MNC corporates, or existing high-vintage deposit customers.\n"
                    f"- Delinquency Trends: Gross 30+ DPD stands at {1.42 + (count * 0.02):.2f}% and 90+ DPD delinquency is flat at {0.52 + (count * 0.01):.2f}%, demonstrating high credit stability.\n"
                    f"- Capital Impact: The RBI risk weight increase had a one-time impact of ~71 bps on CET-1, which we have fully absorbed through organic internal capital generation of {42 + (count % 6)} bps per fiscal year."
                )
            },
            {
                "speaker": "Amit Sachdeva (Equirus)",
                "content": (
                    f"Analytical Query - Bharat Banking Growth, Branch Productivity & Opex Guidance:\n"
                    f"Could management break down Bharat Banking (RUSU) loan growth, agri-loan disbursement trends, and overall cost-to-income guidance?"
                )
            },
            {
                "speaker": "Munish Sharda",
                "content": (
                    f"Response on Bharat Banking Growth & RUSU Credit:\n"
                    f"Thanks Amit. Munish here. Key metrics for Bharat Banking:\n"
                    f"- RUSU Credit Growth: Rural and Semi-Urban advances expanded {19 + (count % 4)}% YoY, driven by micro-banking and agri-enterprise financing.\n"
                    f"- Dedicated Agri Infrastructure: We operate dedicated agri-loan hubs in {2150 + count * 25} branches, yielding a deposit growth of {16 + (count % 4)}% YoY in RUSU pincodes.\n"
                    f"- Cost-to-Income Trajectory: Operating expenses grew {10 + (count % 3)}% YoY, outpaced by core revenue growth. Cost-to-income ratio for the quarter improved to {cost_to_income_pct}%, and we guide for a medium-term target of ~46-47%."
                )
            }
        ]

        # FIX 2: Hard-drop the Moderator and classify Role (Management vs Analyst) using clean_mgmt_names
        cleaned_conversations = []
        for item in raw_conversations_list:
            speaker_str = item["speaker"].strip()
            content_str = item["content"].strip()

            # Skip Moderator / Operator / Host entirely
            if re.search(r'(?i)\b(moderator|operator|host)\b', speaker_str):
                continue

            # Check against cleaned management names
            is_management = any(mgmt.lower() in speaker_str.lower() for mgmt in clean_mgmt_names)

            cleaned_conversations.append({
                "speaker": speaker_str,
                "role": "Management" if is_management else "Analyst",
                "content": content_str
            })

        # 1. Write formatted .txt file
        txt_path = os.path.join(transcripts_dir, f"{filename_base}.txt")
        txt_lines = [
            f"================================================================================",
            f"COMPANY: AXIS BANK LIMITED",
            f"TICKER: AXISBANK",
            f"DATE: {call_date}",
            f"SUBJECT: {subject}",
            f"MANAGEMENT TEAM: {', '.join(clean_mgmt_names)}",
            f"================================================================================",
            f"TRANSCRIPT CONTENT:\n"
        ]
        for conv in cleaned_conversations:
            txt_lines.append(f"{conv['speaker']} [{conv['role']}]:")
            txt_lines.append(f"{conv['content']}\n")

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(txt_lines))

        # 2. Write formatted .json file with clean management_names and cleaned_conversations
        json_path = os.path.join(transcripts_dir, f"{filename_base}.json")
        json_data = {
            "metadata": {
                "company": "AXISBANK",
                "date": call_date,
                "subject": subject,
                "management_names": clean_mgmt_names
            },
            "conversations": cleaned_conversations
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        # 3. Create PDF file in extracted_pdf
        pdf_path = os.path.join(pdf_dir, f"{filename_base}.pdf")
        pdf_header = (
            f"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            f"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n"
            f"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<</Font<</F1 4 0 R>>>>/Contents 5 0 R>>endobj\n"
            f"4 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
            f"5 0 obj<</Length 300>>stream\nBT /F1 12 Tf 50 700 TD ({subject}) Tj 0 -20 TD (Date: {call_date}) Tj 0 -20 TD (Company: AXISBANK) Tj ET\nendstream\nendobj\n"
            f"xref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000056 00000 n \n0000000111 00000 n \n0000000212 00000 n \n0000000283 00000 n \n"
            f"trailer<</Size 6/Root 1 0 R>>\nstartxref\n630\n%%EOF"
        )
        with open(pdf_path, "w", encoding="utf-8") as f:
            f.write(pdf_header)

        print(f"[{count}/12] Generated Cleaned JSON & TXT for {filename_base}")

print(f"\n[Success] Completed generating cleaned transcripts!")
