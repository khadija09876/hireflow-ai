import os
import io
import re
import json
import hashlib
from typing import List, Dict, Any

import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
from docx import Document
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss


# ============================================================
# HireFlow AI
# Intelligent Resume Screening & Recruitment Assistant
# ============================================================

st.set_page_config(
    page_title="HireFlow AI | Recruitment Assistant",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Professional Maroon & Grey theme
# -----------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #1b1a1c;
        --surface: #232224;
        --surface-2: #2a2829;
        --border: #3d3a3c;
        --text: #f5f3f4;
        --muted: #b8b3b6;
        --accent: #7c1f2b;
        --accent-2: #a83246;
        --success: #38d39f;
        --warning: #f5b942;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(124,31,43,.18), transparent 28%),
            radial-gradient(circle at 90% 15%, rgba(168,50,70,.10), transparent 25%),
            var(--bg);
        color: var(--text);
    }

    [data-testid="stSidebar"] {
        background: #201f21;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] * {
        color: var(--text);
    }

    .hero {
        padding: 26px 30px;
        border: 1px solid var(--border);
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(124,31,43,.22), rgba(35,34,36,.92) 52%);
        box-shadow: 0 18px 60px rgba(0,0,0,.25);
        margin-bottom: 22px;
    }

    .hero h1 {
        font-size: 38px;
        margin: 0 0 7px 0;
        font-weight: 800;
        letter-spacing: -1px;
        color: var(--text);
    }

    .hero p {
        color: var(--muted);
        font-size: 15px;
        margin: 0;
        line-height: 1.6;
    }

    .badge {
        display: inline-block;
        padding: 6px 11px;
        border-radius: 999px;
        background: rgba(124,31,43,.22);
        border: 1px solid rgba(168,50,70,.45);
        color: #ff9aa8;
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 13px;
    }

    .section-title {
        font-size: 20px;
        font-weight: 750;
        margin: 18px 0 10px 0;
        color: var(--text);
    }

    .section-subtitle {
        color: var(--muted);
        font-size: 13px;
        margin-bottom: 14px;
    }

    .metric-card {
        background: linear-gradient(145deg, var(--surface-2), var(--surface));
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 18px;
        min-height: 108px;
    }

    .metric-label {
        color: var(--muted);
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: .6px;
    }

    .metric-value {
        font-size: 28px;
        font-weight: 800;
        margin-top: 7px;
        color: var(--text);
    }

    .candidate-card {
        background: linear-gradient(145deg, #262425, #201e1f);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 18px;
        margin: 8px 0;
    }

    .candidate-name {
        font-size: 18px;
        font-weight: 750;
        color: var(--text);
    }

    .small-muted {
        color: var(--muted);
        font-size: 12px;
    }

    .pill {
        display: inline-block;
        padding: 4px 9px;
        margin: 3px 4px 3px 0;
        border-radius: 999px;
        background: #332e30;
        color: #f0e9eb;
        font-size: 11px;
        border: 1px solid #4a4345;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 700;
        border: 1px solid #4a4547;
        min-height: 42px;
        color: var(--text) !important;
        background: var(--surface-2);
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--accent-2), var(--accent));
        border: none;
        color: #ffffff !important;
    }

    div[data-testid="stFileUploader"] {
        background: rgba(35,34,36,.72);
        border: 1px dashed #55494c;
        border-radius: 14px;
        padding: 8px;
    }

    div[data-testid="stFileUploader"] * {
        color: var(--text) !important;
    }

    .info-box {
        padding: 13px 15px;
        border-radius: 12px;
        background: #262425;
        border: 1px solid var(--border);
        color: #ece6e8;
        font-size: 13px;
        line-height: 1.55;
    }

    .footer {
        text-align: center;
        color: #9a9296;
        font-size: 11px;
        padding: 28px 0 10px;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* ---------- Keep only the top toolbar hidden; sidebar toggle stays visible ---------- */
    header [data-testid="stToolbar"] {visibility: hidden;}

    [data-testid="collapsedControl"] {
        visibility: visible !important;
        display: flex !important;
        align-items: center;
        justify-content: center;
        background: var(--surface-2);
        border: 1px solid var(--border);
        border-radius: 8px;
        color: var(--text) !important;
        box-shadow: 0 10px 30px rgba(0,0,0,.35);
    }

    [data-testid="collapsedControl"] svg {
        fill: var(--text) !important;
    }

    /* ---------- Text area / text input / select box: readable on dark theme ---------- */
    .stTextArea textarea,
    .stTextInput input {
        background-color: var(--surface) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        caret-color: var(--accent-2);
    }

    .stTextArea textarea::placeholder,
    .stTextInput input::placeholder {
        color: var(--muted) !important;
        opacity: 1 !important;
    }

    .stTextArea textarea:focus,
    .stTextInput input:focus {
        border-color: var(--accent-2) !important;
        box-shadow: 0 0 0 2px rgba(168,50,70,.25) !important;
    }

    .stSelectbox div[data-baseweb="select"] > div {
        background-color: var(--surface) !important;
        color: var(--text) !important;
        border-color: var(--border) !important;
    }

    .stSelectbox div[data-baseweb="select"] * {
        color: var(--text) !important;
    }

    /* ---------- Radio buttons (sidebar nav) ---------- */
    [data-testid="stSidebar"] div[role="radiogroup"] label {
        padding: 6px 4px;
        border-radius: 8px;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: rgba(255,255,255,.05);
    }

    /* ---------- Dataframes / tables text ---------- */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid var(--border);
    }

    /* ---------- Progress bar ---------- */
    div[data-testid="stProgress"] > div > div {
        background: linear-gradient(135deg, var(--accent-2), var(--accent)) !important;
    }

    /* ---------- Body text: only apply to plain markdown/labels, NEVER to
       success/warning/info/error alert boxes (they need their own contrast) ---------- */
    .stMarkdown p, .stMarkdown li, .stMarkdown span,
    [data-testid="stWidgetLabel"] label,
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
        color: var(--text);
    }

    /* ---------- Sidebar radio (Workspace) options: make label + selected dot clearly visible ---------- */
    [data-testid="stRadio"] label {
        color: var(--text) !important;
    }

    [data-testid="stRadio"] label p {
        color: var(--text) !important;
        font-size: 14px;
    }

    /* ---------- Dropdown / selectbox popover (candidate picker) ----------
       This menu renders in a portal outside the main app container, so it
       needs its own background + text color or it becomes invisible. */
    div[data-baseweb="popover"] {
        background-color: var(--surface) !important;
    }

    ul[data-testid="stSelectboxVirtualDropdown"] {
        background-color: var(--surface) !important;
        border: 1px solid var(--border) !important;
    }

    ul[data-testid="stSelectboxVirtualDropdown"] li {
        background-color: var(--surface) !important;
        color: var(--text) !important;
    }

    ul[data-testid="stSelectboxVirtualDropdown"] li:hover {
        background-color: var(--surface-2) !important;
    }

    /* ---------- Alert boxes (success / warning / info / error): keep Streamlit's
       own light backgrounds with dark, readable text — do not override with
       the dark-theme text color. ---------- */
    div[data-testid="stAlert"] {
        color: #1a1a1a !important;
    }

    div[data-testid="stAlert"] p, div[data-testid="stAlert"] span {
        color: #1a1a1a !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Session state
# -----------------------------
DEFAULT_STATE = {
    "job_text": "",
    "job_name": "",
    "candidates": [],
    "results": [],
    "analysis_ready": False,
    "analysis_id": None,
    "jd_index": None,
    "embedding_model": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# -----------------------------
# Helpers
# -----------------------------
def get_secret(name: str) -> str:
    """Read a secret from Streamlit secrets first, then environment."""
    try:
        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:
        pass
    return os.getenv(name, "")


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf(file_bytes: bytes) -> str:
    text_parts = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            text_parts.append(page.get_text("text"))
    return clean_text("\n".join(text_parts))


def extract_docx(file_bytes: bytes) -> str:
    doc = Document(io.BytesIO(file_bytes))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]

    for table in doc.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text.strip() for cell in row.cells))

    return clean_text("\n".join(parts))


def extract_txt(file_bytes: bytes) -> str:
    for encoding in ("utf-8", "utf-16", "cp1252", "latin-1"):
        try:
            return clean_text(file_bytes.decode(encoding))
        except UnicodeDecodeError:
            continue
    return clean_text(file_bytes.decode("utf-8", errors="ignore"))


def extract_text(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    data = uploaded_file.getvalue()

    if name.endswith(".pdf"):
        return extract_pdf(data)
    if name.endswith(".docx"):
        return extract_docx(data)
    if name.endswith(".txt"):
        return extract_txt(data)

    return ""


def split_chunks(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(end - overlap, start + 1)

    return chunks


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def build_faiss_index(chunks: List[str]):
    if not chunks:
        return None

    model = load_embedding_model()
    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def retrieve_context(query: str, chunks: List[str], index, top_k: int = 5) -> str:
    if not chunks or index is None:
        return ""

    model = load_embedding_model()
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    k = min(top_k, len(chunks))
    _, indices = index.search(query_embedding, k)

    selected = []
    for idx in indices[0]:
        if 0 <= idx < len(chunks):
            selected.append(chunks[idx])

    return "\n\n---\n\n".join(selected)


def get_groq_client():
    key = get_secret("GROQ_API_KEY")
    if not key:
        return None
    return Groq(api_key=key)


def groq_text(prompt: str, temperature: float = 0.2) -> str:
    client = get_groq_client()
    if client is None:
        raise RuntimeError(
            "GROQ_API_KEY is not configured. Add it in Streamlit Cloud → Settings → Secrets."
        )

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are HireFlow AI, an HR recruitment assistant. "
                    "Provide factual, evidence-based resume analysis. "
                    "Never make a final hiring decision. Do not infer protected "
                    "characteristics or personality/mental-health traits. "
                    "If information is not present in the supplied documents, "
                    "say 'Not found in provided material'."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content.strip()


def parse_json_response(text: str) -> Dict[str, Any]:
    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.I).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

    return {}


def safe_list(value):
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [str(value)]


def candidate_name_from_result(result: Dict[str, Any], fallback: str) -> str:
    profile = result.get("candidate_profile", {})
    name = profile.get("name", "")
    return name.strip() if isinstance(name, str) and name.strip() else fallback


def analyze_candidate(job_text: str, resume_text: str, candidate_filename: str) -> Dict[str, Any]:
    # Retrieve JD context relevant to this resume.
    jd_chunks = split_chunks(job_text)
    jd_index = build_faiss_index(jd_chunks)
    jd_context = retrieve_context(
        "required skills qualifications experience education responsibilities technologies",
        jd_chunks,
        jd_index,
        top_k=6,
    )

    # Retrieve resume context around core candidate information.
    resume_chunks = split_chunks(resume_text)
    resume_index = build_faiss_index(resume_chunks)
    resume_context = retrieve_context(
        "candidate name contact education skills work experience projects certifications technologies achievements",
        resume_chunks,
        resume_index,
        top_k=8,
    )

    prompt = f"""
Analyze the candidate resume against the supplied job description.

IMPORTANT:
- Use only information explicitly present in the supplied material.
- Do not infer age, gender, religion, race, nationality, disability, marital status,
  health, personality, or any other protected/sensitive characteristic.
- Do not make a final hiring/rejection decision.
- Treat missing evidence as "Not found in provided material".
- The output is HR decision-support only.
- Keep claims traceable to the provided documents.

JOB DESCRIPTION CONTEXT:
{jd_context}

CANDIDATE RESUME CONTEXT:
{resume_context}

Return ONLY valid JSON using this schema:
{{
  "candidate_profile": {{
    "name": "",
    "email": "",
    "phone": "",
    "location": ""
  }},
  "skills": [],
  "education": [],
  "experience": [],
  "certifications": [],
  "projects": [],
  "matched_requirements": [],
  "missing_or_unverified_requirements": [],
  "relevant_experience_summary": "",
  "evidence_notes": [],
  "interview_questions": []
}}

Interview questions must be job-relevant and based on the candidate's documented background.
Generate 4 to 6 useful questions.
"""

    raw = groq_text(prompt, temperature=0.15)
    data = parse_json_response(raw)

    if not data:
        # Graceful plain-text fallback if the model returns malformed JSON.
        return {
            "candidate_profile": {"name": "", "email": "", "phone": "", "location": ""},
            "skills": [],
            "education": [],
            "experience": [],
            "certifications": [],
            "projects": [],
            "matched_requirements": [],
            "missing_or_unverified_requirements": [],
            "relevant_experience_summary": raw,
            "evidence_notes": ["The AI returned a non-JSON analysis."],
            "interview_questions": [],
            "_filename": candidate_filename,
        }

    data["_filename"] = candidate_filename
    return data


def make_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {
            "candidate_count": 0,
            "skills_count": 0,
            "education_count": 0,
            "matched_count": 0,
            "missing_count": 0,
        }

    skills = sum(len(safe_list(r.get("skills"))) for r in results)
    education = sum(len(safe_list(r.get("education"))) for r in results)
    matched = sum(len(safe_list(r.get("matched_requirements"))) for r in results)
    missing = sum(len(safe_list(r.get("missing_or_unverified_requirements"))) for r in results)

    return {
        "candidate_count": len(results),
        "skills_count": skills,
        "education_count": education,
        "matched_count": matched,
        "missing_count": missing,
    }


def pills(items: List[str], empty_text: str = "Not found in provided material"):
    items = [str(x) for x in safe_list(items) if str(x).strip()]
    if not items:
        st.markdown(f"<span class='small-muted'>{empty_text}</span>", unsafe_allow_html=True)
        return

    html = "".join(f"<span class='pill'>{x}</span>" for x in items[:25])
    st.markdown(html, unsafe_allow_html=True)


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown(
        """
        <div style="padding:8px 2px 18px;">
            <div style="font-size:25px;font-weight:800;">💼 HireFlow AI</div>
            <div style="color:#b8b3b6;font-size:12px;margin-top:5px;">
                Recruitment Intelligence
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Workspace",
        ["Screening Dashboard", "Candidate Analysis", "How It Works"],
        index=0,
    )

    st.markdown("---")
    st.markdown(
        """
        <div class="info-box">
        <b>AI-assisted screening</b><br>
        HireFlow extracts documented candidate information and compares it
        with job requirements. Final recruitment decisions remain with HR.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("↻ Reset Workspace", use_container_width=True):
        for key, value in DEFAULT_STATE.items():
            st.session_state[key] = value
        st.rerun()


# ============================================================
# Hero
# ============================================================
st.markdown(
    """
    <div class="hero">
        <div class="badge">AI-POWERED RECRUITMENT ASSISTANT</div>
        <h1>HireFlow AI</h1>
        <p>
            Intelligent resume screening, requirement matching and
            interview preparation — designed to reduce repetitive HR work
            while keeping people in control of hiring decisions.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Screening Dashboard
# ============================================================
if page == "Screening Dashboard":
    st.markdown("<div class='section-title'>1. Define the vacancy</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-subtitle'>Upload a job description or paste it directly.</div>",
        unsafe_allow_html=True,
    )

    jd_file = st.file_uploader(
        "Job Description",
        type=["pdf", "docx", "txt"],
        key="jd_uploader",
        help="Supported formats: PDF, DOCX and TXT.",
    )

    jd_text_input = st.text_area(
        "Or paste Job Description",
        value=st.session_state.job_text,
        height=170,
        placeholder="Paste the role requirements, responsibilities, qualifications and required skills here...",
    )

    if jd_file:
        extracted_jd = extract_text(jd_file)
        if extracted_jd:
            st.session_state.job_text = extracted_jd
            st.session_state.job_name = jd_file.name
            st.success(f"Job description loaded: {jd_file.name}")
        else:
            st.error("No readable text was found in this job-description file.")
    elif jd_text_input.strip():
        st.session_state.job_text = clean_text(jd_text_input)
        st.session_state.job_name = "Pasted Job Description"

    st.markdown("<div class='section-title'>2. Add candidate resumes</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-subtitle'>Upload multiple candidate files at once. PDF, DOCX and TXT are supported.</div>",
        unsafe_allow_html=True,
    )

    resumes = st.file_uploader(
        "Candidate Resumes",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key="resume_uploader",
        help="You can upload multiple resumes in one batch.",
    )

    if resumes:
        candidates = []
        unreadable = []

        for file in resumes:
            text = extract_text(file)
            if text:
                candidates.append(
                    {
                        "filename": file.name,
                        "text": text,
                        "characters": len(text),
                    }
                )
            else:
                unreadable.append(file.name)

        st.session_state.candidates = candidates

        if unreadable:
            st.warning(
                "Could not extract readable text from: "
                + ", ".join(unreadable)
                + ". For scanned PDFs, add an OCR-enabled version or OCR layer."
            )

    # Metrics
    candidate_count = len(st.session_state.candidates)
    jd_loaded = bool(st.session_state.job_text.strip())

    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        ("JOB DESCRIPTION", "Ready" if jd_loaded else "Missing"),
        ("CANDIDATES", candidate_count),
        ("SUPPORTED FILES", "PDF · DOCX · TXT"),
        ("AI ENGINE", "Groq"),
    ]

    for col, (label, value) in zip([c1, c2, c3, c4], metrics):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.session_state.candidates:
        st.markdown("<div class='section-title'>Candidate files</div>", unsafe_allow_html=True)
        file_df = pd.DataFrame(
            [
                {
                    "Candidate File": c["filename"],
                    "Extracted Characters": c["characters"],
                    "Status": "Ready for analysis",
                }
                for c in st.session_state.candidates
            ]
        )
        st.dataframe(file_df, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    ready = jd_loaded and candidate_count > 0
    if st.button(
        "Analyze Candidates →",
        type="primary",
        use_container_width=True,
        disabled=not ready,
    ):
        try:
            with st.status("Running AI-assisted screening...", expanded=True) as status:
                st.write("Preparing job-description context...")
                results = []

                progress = st.progress(0)
                total = len(st.session_state.candidates)

                for i, candidate in enumerate(st.session_state.candidates):
                    st.write(f"Analyzing {candidate['filename']}...")
                    result = analyze_candidate(
                        st.session_state.job_text,
                        candidate["text"],
                        candidate["filename"],
                    )
                    results.append(result)
                    progress.progress((i + 1) / total)

                st.session_state.results = results
                st.session_state.analysis_ready = True
                st.session_state.analysis_id = hashlib.md5(
                    ("".join(r.get("_filename", "") for r in results) + st.session_state.job_text).encode()
                ).hexdigest()

                status.update(label="Analysis completed", state="complete")

            st.success("Candidate analysis is ready.")
            st.rerun()

        except Exception as e:
            st.error(f"Analysis could not be completed: {e}")

    if st.session_state.analysis_ready and st.session_state.results:
        st.markdown("---")
        st.markdown("<div class='section-title'>Screening overview</div>", unsafe_allow_html=True)

        summary = make_summary(st.session_state.results)
        m1, m2, m3, m4, m5 = st.columns(5)

        overview = [
            (m1, "CANDIDATES", summary["candidate_count"]),
            (m2, "SKILL ITEMS", summary["skills_count"]),
            (m3, "EDUCATION ITEMS", summary["education_count"]),
            (m4, "MATCHED REQUIREMENTS", summary["matched_count"]),
            (m5, "MISSING / UNVERIFIED", summary["missing_count"]),
        ]

        for col, label, value in overview:
            with col:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value">{value}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        table_rows = []
        for result in st.session_state.results:
            name = candidate_name_from_result(
                result,
                result.get("_filename", "Candidate"),
            )
            table_rows.append(
                {
                    "Candidate": name,
                    "Skills": len(safe_list(result.get("skills"))),
                    "Experience": len(safe_list(result.get("experience"))),
                    "Matched Requirements": len(safe_list(result.get("matched_requirements"))),
                    "Missing / Unverified": len(
                        safe_list(result.get("missing_or_unverified_requirements"))
                    ),
                }
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame(table_rows),
            use_container_width=True,
            hide_index=True,
        )

        export_rows = []
        for result in st.session_state.results:
            export_rows.append(
                {
                    "Candidate": candidate_name_from_result(result, result.get("_filename", "Candidate")),
                    "File": result.get("_filename", ""),
                    "Skills": "; ".join(map(str, safe_list(result.get("skills")))),
                    "Education": "; ".join(map(str, safe_list(result.get("education")))),
                    "Experience": "; ".join(map(str, safe_list(result.get("experience")))),
                    "Matched Requirements": "; ".join(
                        map(str, safe_list(result.get("matched_requirements")))
                    ),
                    "Missing / Unverified": "; ".join(
                        map(str, safe_list(result.get("missing_or_unverified_requirements")))
                    ),
                    "Relevant Experience Summary": result.get("relevant_experience_summary", ""),
                }
            )

        csv_bytes = pd.DataFrame(export_rows).to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Screening Report (CSV)",
            data=csv_bytes,
            file_name="hireflow_screening_report.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ============================================================
# Candidate Analysis
# ============================================================
elif page == "Candidate Analysis":
    if not st.session_state.results:
        st.info("Run candidate analysis from the Screening Dashboard first.")
    else:
        st.markdown("<div class='section-title'>Candidate Analysis</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-subtitle'>Review documented evidence, requirement matching and interview preparation.</div>",
            unsafe_allow_html=True,
        )

        names = [
            candidate_name_from_result(r, r.get("_filename", "Candidate"))
            for r in st.session_state.results
        ]

        selected_name = st.selectbox("Select candidate", names)
        result = st.session_state.results[names.index(selected_name)]

        profile = result.get("candidate_profile", {})

        st.markdown(
            f"""
            <div class="candidate-card">
                <div class="candidate-name">{selected_name}</div>
                <div class="small-muted">{result.get("_filename", "")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        p1, p2, p3, p4 = st.columns(4)
        profile_items = [
            ("EMAIL", profile.get("email", "Not found")),
            ("PHONE", profile.get("phone", "Not found")),
            ("LOCATION", profile.get("location", "Not found")),
            ("EDUCATION", len(safe_list(result.get("education")))),
        ]

        for col, (label, value) in zip([p1, p2, p3, p4], profile_items):
            with col:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div style="font-size:15px;font-weight:700;margin-top:8px;">{value}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        left, right = st.columns(2)

        with left:
            st.markdown("#### Skills")
            pills(result.get("skills"))

            st.markdown("#### Education")
            for item in safe_list(result.get("education")):
                st.write(f"• {item}")

            st.markdown("#### Experience")
            for item in safe_list(result.get("experience")):
                st.write(f"• {item}")

            st.markdown("#### Certifications")
            for item in safe_list(result.get("certifications")):
                st.write(f"• {item}")

        with right:
            st.markdown("#### Matched requirements")
            for item in safe_list(result.get("matched_requirements")):
                st.success(str(item))

            st.markdown("#### Missing / unverified")
            for item in safe_list(result.get("missing_or_unverified_requirements")):
                st.warning(str(item))

            st.markdown("#### Evidence notes")
            for item in safe_list(result.get("evidence_notes")):
                st.write(f"• {item}")

        st.markdown("#### Relevant experience summary")
        st.markdown(
            f"<div class='info-box'>{result.get('relevant_experience_summary', 'Not found in provided material')}</div>",
            unsafe_allow_html=True,
        )

        st.markdown("#### Suggested interview questions")
        questions = safe_list(result.get("interview_questions"))
        if questions:
            for i, q in enumerate(questions, 1):
                st.markdown(f"**{i}.** {q}")
        else:
            st.info("No interview questions were generated.")


# ============================================================
# How It Works
# ============================================================
else:
    st.markdown("<div class='section-title'>How HireFlow AI works</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-subtitle'>The technical workflow runs behind the interface so HR can focus on the results.</div>",
        unsafe_allow_html=True,
    )

    steps = [
        ("01", "Upload", "Add a job description and multiple candidate resumes."),
        ("02", "Extract", "Python extracts text from PDF, DOCX and TXT documents."),
        ("03", "Retrieve", "The system chunks documents, creates embeddings and retrieves relevant context."),
        ("04", "Analyze", "Groq AI compares documented candidate information with job requirements."),
        ("05", "Review", "HR reviews skills, education, experience, matching evidence and missing/unverified requirements."),
        ("06", "Prepare", "The system generates job-relevant interview questions from the supplied context."),
    ]

    for num, title, description in steps:
        st.markdown(
            f"""
            <div class="candidate-card">
                <div style="display:flex;gap:16px;align-items:flex-start;">
                    <div style="font-size:13px;font-weight:800;color:#ff9aa8;">{num}</div>
                    <div>
                        <div class="candidate-name">{title}</div>
                        <div class="small-muted" style="margin-top:5px;">{description}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Technology")
    tech_df = pd.DataFrame(
        [
            ["Python", "Application logic and document processing"],
            ["Streamlit", "Web interface and HR dashboard"],
            ["Groq", "Fast LLM-based candidate analysis"],
            ["RAG", "Context retrieval from job descriptions and resumes"],
            ["FAISS", "Vector similarity search"],
            ["Sentence Transformers", "Document embeddings"],
            ["PyMuPDF", "PDF text extraction"],
            ["python-docx", "DOCX text extraction"],
        ],
        columns=["Technology", "Role"],
    )
    st.dataframe(tech_df, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="info-box">
        <b>Responsible use:</b> HireFlow AI is designed as a recruitment
        support tool. It should not infer protected characteristics or make
        the final hiring decision. HR professionals should verify important
        information against the original application materials.
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    "<div class='footer'>HireFlow AI · Intelligent Recruitment Assistant · AI-assisted, human-led hiring</div>",
    unsafe_allow_html=True,
)
