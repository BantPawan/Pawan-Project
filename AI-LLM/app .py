# app.py
import streamlit as st
import requests
import PyPDF2
import hashlib
import textwrap
import os
from io import BytesIO

# --- CONFIG ---
OLLAMA_URL = "http://localhost:11434/api/generate"  # Internal to Render
MODEL_NAME = "paper-analyzer"

# --- SESSION STATE ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "last_file_hash" not in st.session_state:
    st.session_state.last_file_hash = None
if "processed" not in st.session_state:
    st.session_state.processed = False

# --- PROMPTS ---
ANSWER_PROMPT = """[INST] <<SYS>>
You are a professional academic assistant analyzing research papers. Structure your answer with these exact section headers:

1. KEY CONCEPT: Identify the main concept (1-2 sentences)
2. MATHEMATICAL FORMULATION: Provide equations/formulas with explanations
3. MATHEMATICAL INTUITION: Explain the meaning and significance
4. PRACTICAL IMPLICATIONS: Describe 3-5 applications/benefits
5. SUMMARY: Brief 2-3 sentence recap

Format equations using $$ for LaTeX and wrap code in ```.
<</SYS>>

CONTEXT: {context}

QUESTION: {question}
[/INST]"""

SUMMARY_PROMPT = """[INST] <<SYS>>
Summarize the following research paper content in 100 words using bullet points.
<</SYS>>

CONTEXT: {context}
[/INST]"""

QUIZ_PROMPT = """[INST] <<SYS>>
Generate 3 true/false questions based on this content with answers explained.
<</SYS>>

CONTEXT: {context}
[/INST]"""

# --- HELPER FUNCTIONS ---
def get_file_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()

def extract_text_from_pdf(uploaded_file) -> str:
    pdf = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text.strip()

def split_text(text: str, chunk_size: int = 1000, overlap: int = 200):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def query_ollama(prompt: str) -> str:
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.5}
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json().get("response", "No response from model.")
    except Exception as e:
        return f"Error: {str(e)}"

def format_response(raw: str) -> str:
    sections = {
        "KEY CONCEPT": "Not found",
        "MATHEMATICAL FORMULATION": "No equations",
        "MATHEMATICAL INTUITION": "No intuition",
        "PRACTICAL IMPLICATIONS": "No implications",
        "SUMMARY": "No summary"
    }
    current = None
    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        for sec in sections:
            if sec in line.upper():
                current = sec
                line = line.replace(sec, "").strip()
                break
        if current and line:
            sections[current] += "\n" + line

    return f"""
## Research Paper Analysis

### Key Concept
{sections['KEY CONCEPT']}

### Mathematical Formulation
{sections['MATHEMATICAL FORMULATION']}

### Mathematical Intuition
{sections['MATHEMATICAL INTUITION']}

### Practical Implications
{sections['PRACTICAL IMPLICATIONS']}

### Summary
{sections['SUMMARY']}
"""

# --- MAIN APP ---
def main():
    st.set_page_config(page_title="Research Paper Analyzer", layout="wide", page_icon="magnifying glass")

    st.markdown("""
    <style>
    .main-header {font-size: 2.5rem; color: #1E90FF; text-align: center; margin-bottom: 1rem;}
    .section {padding: 1rem; border-radius: 10px; margin: 1rem 0; border-left: 5px solid #1E90FF;}
    .stButton>button {background: #4CAF50; color: white; border-radius: 8px; padding: 0.5rem 1rem;}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='main-header'>Research Paper Q&A Assistant</h1>", unsafe_allow_html=True)
    st.markdown("**Upload a PDF or paste a URL → Ask questions → Get structured AI analysis**")

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("Input Source")
        uploaded_files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)
        url = st.text_input("Or enter paper URL (arXiv, etc.):", placeholder="https://arxiv.org/pdf/...")
        
        if st.button("Process Input", type="primary"):
            if uploaded_files or url:
                with st.spinner("Processing..."):
                    if uploaded_files:
                        all_text = ""
                        for f in uploaded_files:
                            current_hash = get_file_hash(f.getvalue())
                            if st.session_state.last_file_hash == current_hash:
                                continue
                            text = extract_text_from_pdf(BytesIO(f.getvalue()))
                            all_text += text + "\n\n"
                            st.session_state.last_file_hash = current_hash
                        if all_text.strip():
                            st.session_state.chunks = split_text(all_text)
                            st.session_state.processed = True
                            st.success(f"Extracted & split into {len(st.session_state.chunks)} chunks")
                    elif url:
                        try:
                            import urllib.request
                            with urllib.request.urlopen(url) as response:
                                if "pdf" in response.headers.get("Content-Type", ""):
                                    text = extract_text_from_pdf(BytesIO(response.read()))
                                else:
                                    text = response.read().decode("utf-8")
                            st.session_state.chunks = split_text(text)
                            st.session_state.processed = True
                            st.success("URL processed!")
                        except:
                            st.error("Failed to load URL")
            else:
                st.warning("Upload a file or enter a URL")

    # --- MAIN CONTENT ---
    if st.session_state.get("processed", False):
        st.success("Ready! Ask questions, generate summary, or quiz.")

        tab1, tab2, tab3 = st.tabs(["Q&A", "Summary", "Quiz"])

        with tab1:
            question = st.text_area("Ask about the paper:", height=100, placeholder="Explain the attention mechanism...")
            if st.button("Analyze", type="primary"):
                context = "\n\n".join(st.session_state.chunks[:3])  # Top 3 chunks
                prompt = ANSWER_PROMPT.format(context=context, question=question)
                with st.spinner("Thinking..."):
                    raw = query_ollama(prompt)
                    st.markdown(format_response(raw), unsafe_allow_html=True)

        with tab2:
            if st.button("Generate Summary"):
                context = "\n\n".join(st.session_state.chunks[:5])
                prompt = SUMMARY_PROMPT.format(context=context)
                with st.spinner("Summarizing..."):
                    summary = query_ollama(prompt)
                    st.markdown(f"<div class='section'>{textwrap.fill(summary, 80)}</div>", unsafe_allow_html=True)

        with tab3:
            if st.button("Generate Quiz"):
                context = "\n\n".join(st.session_state.chunks[:4])
                prompt = QUIZ_PROMPT.format(context=context)
                with st.spinner("Creating quiz..."):
                    quiz = query_ollama(prompt)
                    st.markdown(f"<div class='section'>{textwrap.fill(quiz, 80)}</div>", unsafe_allow_html=True)
    else:
        st.info("Upload a research paper PDF or URL to begin.")
        st.image("https://images.unsplash.com/photo-1456513080510-7bf3a84b82f8?auto=format&fit=crop&w=1200", use_column_width=True)

if __name__ == "__main__":
    main()
