import re
import hashlib
from datetime import datetime, date
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Patient Insight Analyzer", layout="wide")

# ----------------------------
# Helpers
# ----------------------------

MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

def mask_pii(text: str) -> str:
    """Mask obvious emails and phone-like numbers to reduce accidental PII retention."""
    if not text:
        return text
    # emails
    text = re.sub(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", "[EMAIL]", text, flags=re.I)
    # phone-ish: long digit runs incl +61 / spaces
    text = re.sub(r"\b(?:\+?\d[\d\s-]{7,}\d)\b", "[PHONE]", text)
    return text

def stable_hash(s: str) -> str:
    s_norm = re.sub(r"\s+", " ", (s or "").strip().lower())
    return hashlib.sha256(s_norm.encode("utf-8")).hexdigest()[:16]

def split_cases(raw: str) -> list[str]:
    """
    Split bulk paste into case blocks.
    Your data repeats 'Certificate Details' — that’s a strong delimiter.
    """
    raw = raw.strip()
    if not raw:
        return []
    # Keep delimiter in blocks
    parts = re.split(r"(?=Certificate Details)", raw)
    # Filter tiny junk chunks
    blocks = [p.strip() for p in parts if len(p.strip()) > 80]
    return blocks

def parse_suburb_state(block: str):
    # Common pattern: "Wolli Creek NSW, Australia"
    m = re.search(r"\n\s*([A-Za-z][A-Za-z\s'.-]+)\s+(NSW|VIC|QLD|SA|WA|TAS|ACT|NT)\s*,\s*Australia", block)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None, None

def parse_leave_category(block: str):
    m = re.search(r"Leave Category\s*\n\s*\n\s*(Work|University|School|Other)\b", block, flags=re.I)
    return m.group(1).title() if m else None

def parse_certificate_detail(block: str):
    # After "Certificate Details" the next non-empty line is usually the reason bucket (e.g., Period Pain)
    m = re.search(r"Certificate Details\s*\n\s*([^\n]+)", block)
    if m:
        val = m.group(1).strip()
        # Sometimes the next line is empty; guard against headings
        if val.lower() not in {"leave category", "certificate period", "symptoms"}:
            return val
    return None

def parse_symptoms(block: str):
    m = re.search(r"Symptoms\s*\n\s*\n\s*(.+?)\n\s*\n\s*Certificate Period", block, flags=re.S)
    if m:
        return re.sub(r"\s+", " ", m.group(1).strip())
    return None

def parse_cert_date(block: str):
    """
    Try multiple patterns:
    - "15/12/25, 11:36 pm"
    - "15/12/2025"
    - Certificate Period lines "15 Dec → 15 Dec" (assume current year if none)
    """
    # dd/mm/yy or dd/mm/yyyy
    m = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b", block)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if y < 100:
            y += 2000
        try:
            return date(y, mo, d)
        except ValueError:
            pass

    # "15 Dec" style
    m2 = re.search(r"Certificate Period\s*\n\s*\n\s*(\d{1,2})\s+([A-Za-z]+)", block)
    if m2:
        d = int(m2.group(1))
        mon = MONTHS.get(m2.group(2).strip().lower()[:3])
        if mon:
            y = datetime.now().year
            try:
                return date(y, mon, d)
            except ValueError:
                pass

    return None

def parse_gender(block: str):
    """
    User said: female and male only.
    We’ll look for explicit 'male'/'female' mentions if present.
    If not present, leave blank (don’t guess).
    """
    if re.search(r"\bfemale\b", block, flags=re.I):
        return "Female"
    if re.search(r"\bmale\b", block, flags=re.I):
        return "Male"
    return None

def compute_age_from_dob(dob: date, asof: date | None = None) -> int | None:
    asof = asof or date.today()
    if dob > asof:
        return None
    years = asof.year - dob.year - ((asof.month, asof.day) < (dob.month, dob.day))
    if 0 <= years <= 110:
        return years
    return None

def parse_age(block: str):
    """
    Fix for your '322' issue:
    - Only accept 1–2 digit ages when using 'yrs' pattern.
    - If DOB exists (e.g., '18 November 2003'), compute age.
    - Do NOT treat any 3+ digit number as age.
    """
    # "22yrs" pattern (limit to 1-2 digits)
    m = re.search(r"\b(\d{1,2})\s*(?:yrs|years|yo|y/o)\b", block, flags=re.I)
    if m:
        age = int(m.group(1))
        if 0 < age <= 110:
            return age

    # DOB pattern: "18 November 2003"
    m2 = re.search(r"\b(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})\b", block)
    if m2:
        d = int(m2.group(1))
        mon_name = m2.group(2).strip().lower()
        y = int(m2.group(3))
        mon = MONTHS.get(mon_name[:3])
        if mon:
            try:
                dob = date(y, mon, d)
                return compute_age_from_dob(dob)
            except ValueError:
                pass

    return None

def age_cluster_3yr(age: int | None):
    if age is None:
        return None
    base = (age // 3) * 3
    return f"{base}-{base+2}"

# ----------------------------
# Psychology / intent tagging
# ----------------------------

INTENT_RULES = [
    ("Same-day urgency", [
        r"\btoday\b", r"\btonight\b", r"\bnow\b", r"\bASAP\b", r"\bcan't attend\b", r"\bunable to attend\b"
    ]),
    ("Short leave (1 day)", [
        r"\b1 day\b", r"\bone day\b", r"\bjust today\b"
    ]),
    ("Legitimacy reassurance", [
        r"\blegit\b", r"\bvalid\b", r"\baccepted\b", r"\bonline\b", r"\btelehealth\b"
    ]),
    ("Work compliance pressure", [
        r"\bwork\b", r"\bmanager\b", r"\bemployer\b", r"\bHR\b", r"\bsupervisor\b"
    ]),
    ("Caregiver responsibility", [
        r"\bkid\b", r"\bkids\b", r"\bdaughter\b", r"\bson\b", r"\blooking after\b"
    ]),
    ("Pain-driven", [
        r"\bpain\b", r"\bdysmenorrhea\b", r"\bheadache\b", r"\bback pain\b", r"\bcramps?\b"
    ]),
    ("Infectious concern", [
        r"\bflu\b", r"\bcold\b", r"\bfever\b", r"\bvomi(t|ting)\b", r"\bdiarrh\w+\b", r"\bgastro\b"
    ]),
    ("Mental wellbeing & functioning", [
        r"\banxiety\b", r"\bstress\b", r"\blow mood\b", r"\bdepress\w+\b", r"\bconcentration\b"
    ]),
]

def tag_intents(text: str) -> list[str]:
    t = (text or "")
    tags = []
    for name, pats in INTENT_RULES:
        for p in pats:
            if re.search(p, t, flags=re.I):
                tags.append(name)
                break
    return tags

def suggest_ad_baskets(df: pd.DataFrame) -> pd.DataFrame:
    """
    High-level themed clusters for ad strategy.
    Predictability = how tightly the theme maps to common, explicit intent phrases in your dataset.
    (This is a heuristic score from the text you pasted — not Google’s actual search data.)
    """
    baskets = [
        ("Same-Day Certificate (Work)", ["Same-day urgency", "Work compliance pressure"], 0.80),
        ("1-Day Sick Leave Proof", ["Short leave (1 day)", "Work compliance pressure"], 0.75),
        ("Cold/Flu Quick Note", ["Infectious concern", "Same-day urgency"], 0.70),
        ("Gastro / Up All Night", ["Infectious concern", "Short leave (1 day)"], 0.70),
        ("Period Pain Privacy", ["Pain-driven", "Short leave (1 day)"], 0.65),
        ("Back Pain / Injury", ["Pain-driven", "Work compliance pressure"], 0.60),
        ("Caregiver (Child Sick)", ["Caregiver responsibility", "Work compliance pressure"], 0.60),
        ("Mental Health Study/Work Day Off", ["Mental wellbeing & functioning"], 0.55),
        ("Legit/Accepted Online Cert", ["Legitimacy reassurance"], 0.50),
    ]
    rows = []
    for name, required, score in baskets:
        # Count cases that contain all required tags
        hit = df["intent_tags"].apply(lambda tags: all(r in (tags or []) for r in required)).sum()
        rows.append({
            "Basket": name,
            "Required intent tags": ", ".join(required),
            "Matches in your data": int(hit),
            "Predictability (heuristic)": score
        })
    return pd.DataFrame(rows).sort_values(["Matches in your data", "Predictability (heuristic)"], ascending=False)

def negative_keywords_seed():
    """
    Starter negatives to reduce junk spend.
    You MUST review these for your market + compliance.
    """
    return [
        # Employment/legal
        "fake", "forged", "template", "blank", "download", "pdf template", "free", "edit", "photoshop",
        # Medical diagnosis / emergencies
        "chest pain", "stroke", "slurred speech", "difficulty breathing", "emergency",
        # Employer-side
        "verify certificate", "call doctor", "contact provider",
        # Competitor / irrelevant
        "Centrelink medical certificate form", "workers comp claim form",
        # Price hunters / misfit intent
        "cheapest", "free medical certificate", "bulk discount",
        # Students if you only want work (remove if you DO want students)
        "uni special consideration", "exam deferral",
    ]

# ----------------------------
# UI
# ----------------------------

st.title("Bulk Paste → De-dupe → Insights (Streamlit)")

if "cases_df" not in st.session_state:
    st.session_state.cases_df = pd.DataFrame()

with st.sidebar:
    st.header("Settings")
    st.caption("You can paste repeatedly; duplicates are automatically ignored.")
    show_raw = st.toggle("Show parsed rows", value=True)
    show_blocks = st.toggle("Show detected blocks", value=False)

st.subheader("1) Paste transcripts / case text")
raw = st.text_area(
    "Paste here (you can paste many cases at once).",
    height=240,
    placeholder="Paste the de-identified cases here…"
)

colA, colB, colC = st.columns([1, 1, 2])
with colA:
    do_add = st.button("Add to dataset", type="primary")
with colB:
    do_clear = st.button("Clear dataset")
with colC:
    st.write("")

if do_clear:
    st.session_state.cases_df = pd.DataFrame()
    st.success("Dataset cleared.")

if do_add:
    clean = mask_pii(raw)
    blocks = split_cases(clean)

    if show_blocks and blocks:
        st.info(f"Detected {len(blocks)} case blocks")
        st.code(blocks[0][:2000])

    rows = []
    for b in blocks:
        rows.append({
            "case_id": stable_hash(b),
            "suburb": parse_suburb_state(b)[0],
            "state": parse_suburb_state(b)[1],
            "leave_category": parse_leave_category(b),
            "certificate_detail": parse_certificate_detail(b),
            "symptoms": parse_symptoms(b),
            "cert_date": parse_cert_date(b),
            "gender": parse_gender(b),
            "age": parse_age(b),
            "raw_block": b
        })

    new_df = pd.DataFrame(rows).drop_duplicates(subset=["case_id"])
    if st.session_state.cases_df.empty:
        st.session_state.cases_df = new_df
    else:
        merged = pd.concat([st.session_state.cases_df, new_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["case_id"]).reset_index(drop=True)
        st.session_state.cases_df = merged

    st.success(f"Added {len(new_df)} unique cases. Total now: {len(st.session_state.cases_df)}")

df = st.session_state.cases_df.copy()

st.divider()
st.subheader("2) Parsed dataset + quality checks")

if df.empty:
    st.warning("No cases yet. Paste text above and click **Add to dataset**.")
    st.stop()

# Derived fields
df["age_cluster_3yr"] = df["age"].apply(age_cluster_3yr)
df["text_for_intent"] = (
    df["certificate_detail"].fillna("") + " | " +
    df["leave_category"].fillna("") + " | " +
    df["symptoms"].fillna("") + " | " +
    df["raw_block"].fillna("")
)
df["intent_tags"] = df["text_for_intent"].apply(tag_intents)

# Data quality checks (helps you see why insights might be empty)
qc1, qc2, qc3, qc4 = st.columns(4)
qc1.metric("Cases", len(df))
qc2.metric("Age present", int(df["age"].notna().sum()))
qc3.metric("Suburb present", int(df["suburb"].notna().sum()))
qc4.metric("Symptoms present", int(df["symptoms"].notna().sum()))

if show_raw:
    st.dataframe(
        df.drop(columns=["raw_block", "text_for_intent"]),
        use_container_width=True,
        hide_index=True
    )

st.divider()
st.subheader("3) Insights that should always render")

left, right = st.columns(2)

with left:
    st.markdown("**Top certificate reasons**")
    st.dataframe(
        df["certificate_detail"].fillna("Unknown").value_counts().head(12).reset_index()
          .rename(columns={"index": "certificate_detail", "certificate_detail": "count"}),
        use_container_width=True,
        hide_index=True
    )

with right:
    st.markdown("**Top intent tags (psychology/decision drivers)**")
    tag_counts = (
        df["intent_tags"]
        .explode()
        .dropna()
        .value_counts()
        .reset_index()
        .rename(columns={"index": "intent_tag", "intent_tags": "count"})
    )
    st.dataframe(tag_counts, use_container_width=True, hide_index=True)

st.divider()
st.subheader("4) Ad strategy baskets (with predictability %)")

baskets_df = suggest_ad_baskets(df)
st.dataframe(baskets_df, use_container_width=True, hide_index=True)

st.divider()
st.subheader("5) Negative keyword seed list (starter)")

neg = negative_keywords_seed()
st.code("\n".join(neg))

st.divider()
st.subheader("6) Export")

export_df = df.drop(columns=["raw_block", "text_for_intent"])
csv = export_df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", data=csv, file_name="patient_insights.csv", mime="text/csv")
