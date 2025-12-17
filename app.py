import re
import hashlib
import datetime as dt
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import streamlit as st
from difflib import SequenceMatcher


# -----------------------------
# Helpers: normalization & hashing
# -----------------------------
def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()

def similarity(a: str, b: str) -> float:
    """0..1 similarity; use only on normalized strings"""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


# -----------------------------
# Age banding
# -----------------------------
def age_to_band_3yr(age: Optional[int]) -> Optional[str]:
    if age is None:
        return None
    if age < 0:
        return None
    start = (age // 3) * 3
    end = start + 2
    return f"{start}-{end}"


# -----------------------------
# Core record
# -----------------------------
@dataclass
class ConsultRecord:
    consult_id: Optional[str] = None
    created_at: Optional[str] = None

    suburb: Optional[str] = None
    state: Optional[str] = None

    age: Optional[int] = None
    age_band_3yr: Optional[str] = None
    gender: Optional[str] = None  # "Female" | "Male" only

    leave_category: Optional[str] = None
    reason: Optional[str] = None
    days: Optional[int] = None
    certificate_start: Optional[str] = None
    certificate_end: Optional[str] = None

    symptoms_text: Optional[str] = None
    transcript_text: Optional[str] = None
    transcript_present: bool = False

    raw_block_hash: Optional[str] = None
    fingerprint: Optional[str] = None


# -----------------------------
# Parsing: messy copy/paste blocks
# -----------------------------
STATE_RE = r"(NSW|VIC|QLD|SA|WA|TAS|ACT|NT)"
SUBURB_STATE_RE = re.compile(rf"([A-Za-z][A-Za-z\s'\-]+)\s+({STATE_RE}),\s*Australia", re.IGNORECASE)
DURATION_RE = re.compile(r"Duration\s*\n\s*(\d+)\s*day", re.IGNORECASE)
LEAVE_CAT_RE = re.compile(r"Leave Category\s*\n\s*([A-Za-z]+)", re.IGNORECASE)
REASON_RE = re.compile(r"Certificate Details\s*\n\s*([A-Za-z ]+)", re.IGNORECASE)
SYMPTOMS_RE = re.compile(r"Symptoms\s*\n\s*(.+?)(?:\n\s*Certificate Period|\Z)", re.IGNORECASE | re.DOTALL)
CERT_PERIOD_RE = re.compile(r"Certificate Period\s*\n\s*([0-9]{1,2}\s+\w+)\s*→\s*([0-9]{1,2}\s+\w+)", re.IGNORECASE)
AGE_INLINE_RE = re.compile(r"(\d{1,3})\s*yrs|\b(\d{1,3})\s*years?\s*old\b", re.IGNORECASE)
DOB_RE = re.compile(r"\b(\d{1,2})\s+(\w+)\s+(\d{4})\b", re.IGNORECASE)  # e.g. 18 November 2003
GENDER_RE = re.compile(r"\b(Female|Male)\b", re.IGNORECASE)

TRANSCRIPT_MARKERS = [
    "Doccy Agent",
    "I'm the pre consult",
    "preconsult",
    "pre consult",
    "transfer you to a clinician",
]


def split_into_blocks(raw: str) -> List[str]:
    """
    Split large paste into likely consult blocks.
    Heuristic: split on repeated "Medical Certificate" headings.
    """
    raw = raw.strip()
    if not raw:
        return []

    # Insert a separator before "Medical Certificate" if it appears multiple times
    parts = re.split(r"\n\s*Medical Certificate\s*\n", raw, flags=re.IGNORECASE)
    if len(parts) == 1:
        return [raw]

    blocks = []
    # The first chunk may contain address header; subsequent chunks are true blocks
    # Re-add the marker for consistency
    for i, p in enumerate(parts):
        p = p.strip()
        if not p:
            continue
        if i == 0 and len(parts) > 1:
            # keep it attached to next block if very small
            if len(p) < 200:
                continue
        blocks.append(("Medical Certificate\n" + p) if i > 0 else p)

    return blocks if blocks else [raw]


def extract_transcript(block: str) -> Optional[str]:
    # If it contains any transcript markers, try to capture from the first "Doccy Agent" onward
    idx = None
    for m in ["Doccy Agent", "Michael Addis", "Doctor", "00:00", "00:01"]:
        pos = block.find(m)
        if pos != -1:
            idx = pos if idx is None else min(idx, pos)
    if idx is None:
        # fallback if contains markers
        if any(k.lower() in block.lower() for k in TRANSCRIPT_MARKERS):
            return block
        return None

    snippet = block[idx:].strip()
    # Avoid accidentally storing giant admin logs only
    return snippet if len(snippet) > 50 else None


def parse_block(block: str) -> ConsultRecord:
    rec = ConsultRecord()
    rec.raw_block_hash = sha256(normalize_text(block))

    # suburb/state
    m = SUBURB_STATE_RE.search(block)
    if m:
        rec.suburb = m.group(1).strip()
        rec.state = m.group(2).upper()

    # reason
    m = REASON_RE.search(block)
    if m:
        rec.reason = m.group(1).strip()

    # leave category
    m = LEAVE_CAT_RE.search(block)
    if m:
        rec.leave_category = m.group(1).strip()

    # duration/days
    m = DURATION_RE.search(block)
    if m:
        try:
            rec.days = int(m.group(1))
        except:
            rec.days = None

    # certificate period
    m = CERT_PERIOD_RE.search(block)
    if m:
        rec.certificate_start = m.group(1).strip()
        rec.certificate_end = m.group(2).strip()

    # symptoms
    m = SYMPTOMS_RE.search(block)
    if m:
        rec.symptoms_text = m.group(1).strip()
        # clean repeated whitespace
        rec.symptoms_text = re.sub(r"\s+", " ", rec.symptoms_text).strip()

    # gender (strict)
    m = GENDER_RE.search(block)
    if m:
        g = m.group(1).strip().lower()
        rec.gender = "Female" if g == "female" else "Male"

    # age (if present)
    m = AGE_INLINE_RE.search(block)
    age = None
    if m:
        age_str = m.group(1) or m.group(2)
        try:
            age = int(age_str)
        except:
            age = None

    # fallback: compute age from DOB if you pasted DOB
    if age is None:
        dob = DOB_RE.search(block)
        if dob:
            day = int(dob.group(1))
            month_name = dob.group(2).lower()
            year = int(dob.group(3))
            try:
                month_map = {
                    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
                    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
                }
                month = month_map.get(month_name)
                if month:
                    dob_dt = dt.date(year, month, day)
                    today = dt.date.today()
                    age = today.year - dob_dt.year - ((today.month, today.day) < (dob_dt.month, dob_dt.day))
            except:
                age = None

    rec.age = age
    rec.age_band_3yr = age_to_band_3yr(age)

    # transcript
    transcript = extract_transcript(block)
    if transcript:
        rec.transcript_text = transcript
        rec.transcript_present = True

    # fingerprint: used for merging duplicates across pastes
    # Use stable, low-PHI-ish fields. (You can adjust)
    fp_source = "|".join([
        normalize_text(rec.suburb or ""),
        normalize_text(rec.state or ""),
        normalize_text(rec.reason or ""),
        normalize_text(rec.leave_category or ""),
        normalize_text(rec.certificate_start or ""),
        normalize_text(rec.certificate_end or ""),
        str(rec.days or ""),
        normalize_text((rec.symptoms_text or "")[:120]),   # limit for stability + privacy
    ])
    rec.fingerprint = sha256(fp_source)

    return rec


# -----------------------------
# Deduplication & merge
# -----------------------------
def merge_records(a: ConsultRecord, b: ConsultRecord) -> ConsultRecord:
    """
    Merge b into a, keeping the "best" / most complete fields.
    - Prefer non-empty values
    - For transcript/symptoms, keep longer
    """
    def pick(x, y):
        return x if x not in [None, ""] else y

    a.consult_id = pick(a.consult_id, b.consult_id)
    a.created_at = pick(a.created_at, b.created_at)
    a.suburb = pick(a.suburb, b.suburb)
    a.state = pick(a.state, b.state)
    a.gender = pick(a.gender, b.gender)
    a.leave_category = pick(a.leave_category, b.leave_category)
    a.reason = pick(a.reason, b.reason)
    a.days = a.days if a.days is not None else b.days
    a.certificate_start = pick(a.certificate_start, b.certificate_start)
    a.certificate_end = pick(a.certificate_end, b.certificate_end)

    # age / age_band
    a.age = a.age if a.age is not None else b.age
    a.age_band_3yr = a.age_band_3yr if a.age_band_3yr is not None else b.age_band_3yr

    # symptoms: keep longer
    if (b.symptoms_text or "") and len(b.symptoms_text) > len(a.symptoms_text or ""):
        a.symptoms_text = b.symptoms_text

    # transcript: keep longer / more complete
    if (b.transcript_text or "") and len(b.transcript_text) > len(a.transcript_text or ""):
        a.transcript_text = b.transcript_text

    a.transcript_present = bool(a.transcript_text)
    return a


def dedupe_and_merge(records: List[ConsultRecord], near_dup_threshold: float = 0.97) -> Tuple[List[ConsultRecord], Dict[str, Any]]:
    """
    Two-layer dedupe:
    1) Exact: same raw_block_hash or same fingerprint
    2) Near-dup: compare normalized (symptoms+transcript) similarity
    """
    stats = {"input": len(records), "exact_merged": 0, "near_merged": 0}

    # Exact merge by fingerprint (preferred) then raw_block_hash
    by_fp: Dict[str, ConsultRecord] = {}
    for r in records:
        key = r.fingerprint or r.raw_block_hash
        if key in by_fp:
            by_fp[key] = merge_records(by_fp[key], r)
            stats["exact_merged"] += 1
        else:
            by_fp[key] = r

    deduped = list(by_fp.values())

    # Near-dup merge (expensive but ok for early v1)
    used = [False] * len(deduped)
    merged: List[ConsultRecord] = []

    for i in range(len(deduped)):
        if used[i]:
            continue
        base = deduped[i]
        base_text = normalize_text((base.symptoms_text or "") + " " + (base.transcript_text or ""))
        for j in range(i + 1, len(deduped)):
            if used[j]:
                continue
            cand = deduped[j]
            cand_text = normalize_text((cand.symptoms_text or "") + " " + (cand.transcript_text or ""))

            # Only compare if they share some anchors to avoid accidental merges
            same_reason = normalize_text(base.reason or "") == normalize_text(cand.reason or "")
            same_days = (base.days == cand.days) or (base.days is None or cand.days is None)
            same_suburb = normalize_text(base.suburb or "") == normalize_text(cand.suburb or "")

            if not (same_reason and same_days and (same_suburb or not base.suburb or not cand.suburb)):
                continue

            sim = similarity(base_text, cand_text)
            if sim >= near_dup_threshold:
                base = merge_records(base, cand)
                used[j] = True
                stats["near_merged"] += 1

        merged.append(base)

    stats["output"] = len(merged)
    return merged, stats


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Med Cert Market Analyzer (v1)", layout="wide")

st.title("Med Cert Market Analyzer (v1) — Bulk paste → parse → dedupe → analyse")

st.info(
    "Tip: Paste in batches. This app stores every paste in session state until you clear it. "
    "Duplicates and near-duplicates are merged automatically."
)

if "raw_pastes" not in st.session_state:
    st.session_state.raw_pastes = []  # list[str]
if "records" not in st.session_state:
    st.session_state.records = []  # list[ConsultRecord]

colA, colB = st.columns([2, 1])

with colA:
    raw = st.text_area("Paste consult blocks here (you can paste multiple consults at once):", height=240)

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        if st.button("Add paste to batch"):
            if raw.strip():
                st.session_state.raw_pastes.append(raw.strip())
                st.success(f"Added paste #{len(st.session_state.raw_pastes)} to batch.")
            else:
                st.warning("Nothing pasted.")

    with c2:
        if st.button("Parse batch (no dedupe)"):
            all_blocks = []
            for p in st.session_state.raw_pastes:
                all_blocks.extend(split_into_blocks(p))
            new_records = [parse_block(b) for b in all_blocks]
            st.session_state.records = new_records
            st.success(f"Parsed {len(new_records)} records from {len(all_blocks)} blocks.")

    with c3:
        if st.button("Parse + Dedupe + Merge"):
            all_blocks = []
            for p in st.session_state.raw_pastes:
                all_blocks.extend(split_into_blocks(p))
            parsed = [parse_block(b) for b in all_blocks]
            merged, stats = dedupe_and_merge(parsed, near_dup_threshold=0.97)
            st.session_state.records = merged
            st.success(f"Done. Input: {stats['input']} → Output: {stats['output']} "
                       f"(exact merged: {stats['exact_merged']}, near merged: {stats['near_merged']}).")

with colB:
    st.subheader("Batch controls")
    st.write(f"Raw pastes stored: **{len(st.session_state.raw_pastes)}**")
    st.write(f"Current records: **{len(st.session_state.records)}**")

    if st.button("Clear batch + records"):
        st.session_state.raw_pastes = []
        st.session_state.records = []
        st.success("Cleared.")

    st.subheader("Dedup sensitivity")
    near_thr = st.slider("Near-duplicate threshold", 0.90, 0.995, 0.97, 0.001)
    st.caption("Higher = merges only very similar blocks. If you paste lots of repeats, keep ~0.97–0.985.")

# Display records
st.divider()
st.subheader("Parsed records")

if st.session_state.records:
    df = pd.DataFrame([asdict(r) for r in st.session_state.records])

    # Enforce gender rule for reporting
    df["gender"] = df["gender"].apply(lambda x: x if x in ["Female", "Male"] else None)

    # Ensure 3-year clustering always present if age exists
    df["age_band_3yr"] = df.apply(lambda row: row["age_band_3yr"] or age_to_band_3yr(row["age"]), axis=1)

    # Quick filters
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        reason_filter = st.multiselect("Reason", sorted([x for x in df["reason"].dropna().unique()]))
    with f2:
        state_filter = st.multiselect("State", sorted([x for x in df["state"].dropna().unique()]))
    with f3:
        gender_filter = st.multiselect("Gender", ["Female", "Male"])
    with f4:
        transcript_filter = st.selectbox("Transcript present", ["All", "Yes", "No"], index=0)

    fdf = df.copy()
    if reason_filter:
        fdf = fdf[fdf["reason"].isin(reason_filter)]
    if state_filter:
        fdf = fdf[fdf["state"].isin(state_filter)]
    if gender_filter:
        fdf = fdf[fdf["gender"].isin(gender_filter)]
    if transcript_filter == "Yes":
        fdf = fdf[fdf["transcript_present"] == True]
    elif transcript_filter == "No":
        fdf = fdf[fdf["transcript_present"] == False]

    st.dataframe(
        fdf[
            [
                "suburb", "state", "age", "age_band_3yr", "gender",
                "leave_category", "reason", "days",
                "certificate_start", "certificate_end",
                "transcript_present",
                "symptoms_text",
            ]
        ],
        use_container_width=True,
        height=420
    )

    # Export
    csv = fdf.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered CSV", data=csv, file_name="consults_filtered.csv", mime="text/csv")

else:
    st.warning("No records yet. Paste blocks → Add to batch → Parse + Dedupe + Merge.")
