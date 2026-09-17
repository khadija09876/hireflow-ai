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
# HireFlow AI - Professional Recruitment Assistant
# ============================================================

st.set_page_config(
    page_title="HireFlow AI | Professional Recruitment Suite",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Professional Dark Theme & Styling
# -----------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #080b12;
        --surface: #10151f;
        --surface-2: #151b27;
        --border: #263044;
        --text: #f5f7fb;
        --muted: #9aa6b8;
        --accent: #e63946;
        --accent-2: #ff6b6b;
        --success: #38d39f;
        --warning: #f5b942;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(230,57,70,.12), transparent 28%),
            radial-gradient(circle at 90% 15%, rgba(255,107,107,.07), transparent 25%),
            var(--bg);
        color: var(--text);
    }

    [data-testid="stSidebar"] {
        background: #0c1018;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] * {
        color: var(--text);
    }

    .hero {
        padding: 26px 30px;
        border: 1px solid var(--border);
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(230,57,70,.15), rgba(16,21,31,.92) 52%);
        box-shadow: 0 18px 60px rgba(0,0,0,.25);
        margin-bottom: 22px;
    }

    .hero h1 {
        font-size: 38px;
        margin: 0 0 7px 0;
        font-weight: 800;
        letter-spacing: -1px;
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
        background: rgba(230,57,70,.14);
        border: 1px solid rgba(230,57,70,.35);
        color: #ff8a91;
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 13px;
    }

    .section-title {
        font-size: 20px;
        font-weight: 750;
        margin: 18px 0 10px 0;
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
    }

    .candidate-card {
        background: linear-gradient(145deg, #111722, #0e131c);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 18px;
        margin: 8px 0;
    }

    .candidate-name {
        font-size: 18px;
        font-weight: 750;
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
        background: #1b2433;
        color: #dce3ef;
        font-size: 11px;
        border: 1px solid #2a3548;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 700;
        border: 1px solid #303b50;
        min-height: 42px;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #e63946, #b91f2c);
        border: none;
        color: white;
    }

    div[data-testid="stFileUploader"] {
        background: rgba(16,21,31,.72);
        border: 1px dashed #39465e;
        border-radius: 14px;
        padding: 8px;
    }

    .info-box {
        padding: 13px 15px;
        border-radius: 12px;
        background: #111823;
        border: 1px solid var(--border);
        color: #cbd4e2;
        font-size: 13px;
        line-height: 1.55;
    }

    .footer {
        text-align: center;
        color: #657187;
        font-size: 11px;
        padding: 28px 0 10px;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Session State Setup
# -----------------------------
DEFAULT_STATE = {
    "job_text": "",
    "job_name": "",
    "candidates": [],
    "results": [],
    "analysis_ready": False,
    "analysis_id": None,
    "hr_feedback": {},  # Stores ratings and private notes per candidate file
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# -----------------------------
# Helper Functions
# -----------------------------
def get_secret(name: str) -> str:
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
    selected = [chunks[idx] for idx in indices[0] if 0 <= idx < len(chunks)]
    return "\n\n---\n\n".join(selected)


def get_groq_client():
    key = get_secret("GROQ_API_KEY")
    if not key:
        return None
    return Groq(api_key=key)


def groq_text(prompt: str, temperature: float = 0.2) -> str:
    client = get_groq_client()
    if client is None:
        raise RuntimeError("GROQ_API_KEY is not configured in Secrets or environment variables.")
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are HireFlow AI, an advanced HR recruitment assistant. "
                    "Provide factual, evidence-based resume analysis. "
                    "Never make final hiring decisions. Do not infer protected characteristics."
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
    jd_chunks = split_chunks(job_text)
    jd_index = build_faiss_index(jd_chunks)
    jd_context = retrieve_context(
        "required skills qualifications experience education responsibilities technologies",
        jd_chunks,
        jd_index,
        top_k=6,
    )

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
JOB DESCRIPTION CONTEXT:
{jd_context}

CANDIDATE RESUME CONTEXT:
{resume_context}
"""
    raw = groq_text(prompt, temperature=0.15)
    data = parse_json_response(raw)

    if not data:
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
            "evidence_notes": ["Parsed fallback due to formatting."],
            "interview_questions": [],
            "_filename": candidate_filename,
        }

    data["_filename"] = candidate_filename
    return data


def make_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {"candidate_count": 0, "skills_count": 0, "education_count": 0, "matched_count": 0, "missing_count": 0}
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
# Sidebar Configuration
# ============================================================
with st.sidebar:
    st.markdown(
        """
        <div style="padding:8px 2px 18px;">
            <div style="font-size:25px;font-weight:800;">💼 HireFlow AI</div>
            <div style="color:#9aa6b8;font-size:12px;margin-top:5px;">
                Professional Recruitment Suite
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Workspace",
        ["Screening Dashboard", "Candidate Evaluation & Notes", "How It Works"],
        index=0,
    )

    st.markdown("---")
    st.markdown(
        """
        <div class="info-box">
        <b>HR Professional Mode</b><br>
        Add custom notes, score candidate suitability, and export structured evaluation sheets seamlessly.
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
# Hero Section
# ============================================================
st.markdown(
    """
    <div class="hero">
        <div class="badge">ENTERPRISE RECRUITMENT ASSISTANT</div>
        <h1>HireFlow AI Suite</h1>
        <p>
            AI-powered semantic resume screening equipped with private HR notes, 
            structured candidate scoring, and robust decision-support workflows.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Page 1: Screening Dashboard
# ============================================================
if page == "Screening Dashboard":
    st.markdown("<div class='section-title'>1. Define Vacancy & Job Description</div>", unsafe_allow_html=True)
    
    jd_file = st.file_uploader("Job Description Document", type=["pdf", "docx", "txt"], key="jd_uploader")
    jd_text_input = st.text_area(
        "Or Paste Job Description text",
        value=st.session_state.job_text,
        height=140,
        placeholder="Paste role scope, requirements, tech stacks...",
    )

    if jd_file:
        extracted_jd = extract_text(jd_file)
        if extracted_jd:
            st.session_state.job_text = extracted_jd
            st.session_state.job_name = jd_file.name
            st.success(f"Loaded JD: {jd_file.name}")
    elif jd_text_input.strip():
        st.session_state.job_text = clean_text(jd_text_input)
        st.session_state.job_name = "Pasted Job Description"

    st.markdown("<div class='section-title'>2. Upload Candidate Resumes (Batch)</div>", unsafe_allow_html=True)
    resumes = st.file_uploader(
        "Resumes Batch Upload",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key="resume_uploader",
    )

    if resumes:
        candidates = []
        unreadable = []
        for file in resumes:
            text = extract_text(file)
            if text:
                candidates.append({"filename": file.name, "text": text, "characters": len(text)})
            else:
                unreadable.append(file.name)
        st.session_state.candidates = candidates
        if unreadable:
            st.warning("Could not read: " + ", ".join(unreadable))

    candidate_count = len(st.session_state.candidates)
    jd_loaded = bool(st.session_state.job_text.strip())

    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        ("JOB STATUS", "Ready" if jd_loaded else "Pending"),
        ("CANDIDATES", candidate_count),
        ("PARSING ENGINE", "PyMuPDF / Docx"),
        ("BACKEND LLM", "Groq AI"),
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
    ready = jd_loaded and candidate_count > 0
    if st.button("Run AI Screening Analysis →", type="primary", use_container_width=True, disabled=not ready):
        try:
            with st.status("Running cross-document semantic analysis...", expanded=True) as status:
                results = []
                progress = st.progress(0)
                total = len(st.session_state.candidates)
                for i, candidate in enumerate(st.session_state.candidates):
                    st.write(f"Evaluating candidate: {candidate['filename']}...")
                    res = analyze_candidate(st.session_state.job_text, candidate["text"], candidate["filename"])
                    results.append(res)
                    progress.progress((i + 1) / total)

                st.session_state.results = results
                st.session_state.analysis_ready = True
                status.update(label="Screening complete successfully!", state="complete")
            st.rerun()
        except Exception as e:
            st.error(f"Error during execution: {e}")

    if st.session_state.analysis_ready and st.session_state.results:
        st.markdown("---")
        st.markdown("<div class='section-title'>Screening Overview & Executive Table</div>", unsafe_allow_html=True)

        table_rows = []
        for result in st.session_state.results:
            fname = result.get("_filename", "")
            name = candidate_name_from_result(result, fname)
            fb = st.session_state.hr_feedback.get(fname, {"rating": "Unrated", "notes": ""})
            
            table_rows.append({
                "Candidate Name": name,
                "File": fname,
                "HR Rating": fb.get("rating", "Unrated"),
                "Matched Criteria": len(safe_list(result.get("matched_requirements"))),
                "Missing / Flags": len(safe_list(result.get("missing_or_unverified_requirements"))),
                "HR Notes Added": "Yes" if fb.get("notes") else "No"
            })

        df_results = pd.DataFrame(table_rows)
        st.dataframe(df_results, use_container_width=True, hide_index=True)

        # Export with HR Feedback merged
        export_rows = []
        for result in st.session_state.results:
            fname = result.get("_filename", "")
            fb = st.session_state.hr_feedback.get(fname, {"rating": "Unrated", "notes": ""})
            export_rows.append({
                "Candidate": candidate_name_from_result(result, fname),
                "File": fname,
                "HR Rating": fb.get("rating"),
                "HR Private Notes": fb.get("notes"),
                "Skills": "; ".join(map(str, safe_list(result.get("skills")))),
                "Matched Requirements": "; ".join(map(str, safe_list(result.get("matched_requirements")))),
                "Missing Requirements": "; ".join(map(str, safe_list(result.get("missing_or_unverified_requirements")))),
            })

        csv_data = pd.DataFrame(export_rows).to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Professional Evaluation Report (CSV)",
            data=csv_data,
            file_name="hireflow_hr_executive_report.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ============================================================
# Page 2: Candidate Evaluation & Private HR Notes
# ============================================================
elif page == "Candidate Evaluation & Notes":
    if not st.session_state.results:
        st.info("Please perform a batch screening run from the Screening Dashboard first.")
    else:
        st.markdown("<div class='section-title'>Deep Candidate Evaluation & HR Log</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-subtitle'>Inspect technical profiles, assign interview scores, and append confidential internal notes.</div>", unsafe_allow_html=True)

        names = [candidate_name_from_result(r, r.get("_filename", "Candidate")) for r in st.session_state.results]
        selected_name = st.selectbox("Select Candidate to Evaluate", names)
        
        result = next((r for r in st.session_state.results if candidate_name_from_result(r, r.get("_filename", "Candidate")) == selected_name), None)
        
        if result:
            fname = result.get("_filename", "")
            profile = result.get("candidate_profile", {})

            # Initialize state slot for candidate feedback
            if fname not in st.session_state.hr_feedback:
                st.session_state.hr_feedback[fname] = {"rating": "Select Rating", "notes": ""}

            st.markdown(
                f"""
                <div class="candidate-card">
                    <div class="candidate-name">{selected_name}</div>
                    <div class="small-muted">Source File: {fname} | Email: {profile.get('email', 'N/A')} | Phone: {profile.get('phone', 'N/A')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Interactive HR Feedback Section
            with st.form(key=f"hr_form_{fname}"):
                st.markdown("#### 📝 Confidential HR Evaluation & Feedback")
                col_r1, col_r2 = st.columns(2)
                
                rating_options = ["Select Rating", "⭐ 1 - Poor Match", "⭐⭐ 2 - Borderline", "⭐⭐⭐ 3 - Moderate", "⭐⭐⭐⭐ 4 - Strong Match", "⭐⭐⭐⭐⭐ 5 - Top Hire"]
                current_rating = st.session_state.hr_feedback[fname].get("rating", "Select Rating")
                try:
                    default_idx = rating_options.index(current_rating)
                except ValueError:
                    default_idx = 0

                with col_r1:
                    selected_rating = st.selectbox("Candidate Suitability Rating", rating_options, index=default_idx)
                
                current_notes = st.session_state.hr_feedback[fname].get("notes", "")
                with col_r2:
                    hr_notes_input = st.text_area("Private Interviewer Notes / Remarks", value=current_notes, height=68)

                save_feedback = st.form_submit_button("Save HR Feedback")
                if save_feedback:
                    st.session_state.hr_feedback[fname] = {
                        "rating": selected_rating,
                        "notes": hr_notes_input
                    }
                    st.success("HR Feedback updated successfully!")

            st.markdown("---")
            left, right = st.columns(2)

            with left:
                st.markdown("#### Documented Skills")
                pills(result.get("skills"))

                st.markdown("#### Education Background")
                for item in safe_list(result.get("education")):
                    st.write(f"• {item}")

                st.markdown("#### Work Experience Timeline")
                for item in safe_list(result.get("experience")):
                    st.write(f"• {item}")

            with right:
                st.markdown("#### Matched Requirements")
                for item in safe_list(result.get("matched_requirements")):
                    st.success(str(item))

                st.markdown("#### Missing / Unverified Items")
                for item in safe_list(result.get("missing_or_unverified_requirements")):
                    st.warning(str(item))

                st.markdown("#### Suggested Technical Interview Questions")
                for i, q in enumerate(safe_list(result.get("interview_questions")), 1):
                    st.markdown(f"**Q{i}:** {q}")


# ============================================================
# Page 3: How It Works
# ============================================================
else:
    st.markdown("<div class='section-title'>Architecture & Security Overview</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="info-box" style="line-height: 1.8;">
        <b>HireFlow AI Enterprise Framework:</b><br>
        - <b>RAG + FAISS Architecture:</b> Splits incoming long texts into indexed mathematical vector blocks to fetch precise contextual alignment.<br>
        - <b>Groq LLM Acceleration:</b> Processes structural constraints safely without exposing underlying personal sensitive identifiers.<br>
        - <b>HR Control Panel:</b> Keeps final hiring authority exclusively in human hands while eliminating repetitive sorting overhead.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div class='footer'>HireFlow AI &bull; Professional Recruitment Suite &bull; All Rights Reserved</div>", unsafe_allow_html=True)
