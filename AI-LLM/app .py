# app.py
import streamlit as st
import requests
import PyPDF2
import hashlib
import textwrap
import os
from io import BytesIO

# --- CONFIG ---
# Get Ollama URL from environment variable (set in Render)
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434/api/generate')
MODEL_NAME = "paper-analyzer"

# --- SESSION STATE ---
if "chunks" not in st.session_state:
    st.session_state.chunks = []
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
def extract_text_from_pdf(uploaded_file) -> str:
    """Extract text from PDF file with error handling"""
    try:
        pdf = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def split_text(text: str, chunk_size: int = 800, overlap: int = 100):
    """Split text into manageable chunks"""
    if not text:
        return []
    words = text.split()
    if not words:
        return []
    
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
        if i >= len(words):
            break
    return chunks

def query_ollama(prompt: str) -> str:
    """Query Ollama API with robust error handling"""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.5}
            },
            timeout=180  # Increased timeout for Render
        )
        
        if response.status_code == 200:
            return response.json().get("response", "No response from model.")
        else:
            return f"API Error: {response.status_code} - {response.text}"
            
    except requests.exceptions.Timeout:
        return "Error: Request timeout - the AI service is taking too long to respond."
    except requests.exceptions.ConnectionError:
        return f"Error: Cannot connect to AI service at {OLLAMA_URL}. Please check if the backend is running."
    except Exception as e:
        return f"Error: {str(e)}"

def format_response(raw: str) -> str:
    """Format the AI response into structured sections"""
    # Initialize sections
    sections = {
        "KEY CONCEPT": "",
        "MATHEMATICAL FORMULATION": "", 
        "MATHEMATICAL INTUITION": "",
        "PRACTICAL IMPLICATIONS": "",
        "SUMMARY": ""
    }
    
    current_section = None
    lines = raw.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if this line starts a new section
        for section in sections:
            if line.upper().startswith(section):
                current_section = section
                # Remove section header from content
                line = line[len(section):].lstrip(' :')
                break
        else:
            # If no section header found, check for numbered sections
            if line.startswith('1.') and 'KEY CONCEPT' in line.upper():
                current_section = "KEY CONCEPT"
                line = line[2:].lstrip()
            elif line.startswith('2.') and 'MATHEMATICAL' in line.upper():
                current_section = "MATHEMATICAL FORMULATION" 
                line = line[2:].lstrip()
            elif line.startswith('3.') and 'INTUITION' in line.upper():
                current_section = "MATHEMATICAL INTUITION"
                line = line[2:].lstrip()
            elif line.startswith('4.') and 'PRACTICAL' in line.upper():
                current_section = "PRACTICAL IMPLICATIONS"
                line = line[2:].lstrip()
            elif line.startswith('5.') and 'SUMMARY' in line.upper():
                current_section = "SUMMARY"
                line = line[2:].lstrip()
                
        # Add content to current section
        if current_section and line:
            if sections[current_section]:
                sections[current_section] += " " + line
            else:
                sections[current_section] = line

    # Format the output
    formatted = "## 📊 Research Paper Analysis\n\n"
    for section, content in sections.items():
        if content and content.strip():
            formatted += f"### {section}\n{content.strip()}\n\n"
        else:
            formatted += f"### {section}\n*Information not specified in response*\n\n"
            
    return formatted

# --- MAIN APP ---
def main():
    st.set_page_config(
        page_title="Research Paper Analyzer", 
        layout="wide", 
        page_icon="🔍"
    )

    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem; 
        color: #1E90FF; 
        text-align: center; 
        margin-bottom: 1rem;
    }
    .info-box {
        padding: 1rem;
        background-color: #f0f8ff;
        border-radius: 10px;
        border-left: 5px solid #1E90FF;
        margin: 1rem 0;
    }
    .stButton>button {
        background: #4CAF50; 
        color: white; 
        border-radius: 8px; 
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background: #45a049;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='main-header'>🔍 Research Paper Q&A Assistant</h1>", unsafe_allow_html=True)
    st.markdown("**Upload research papers → Ask questions → Get structured AI analysis**")

    # Info box about the demo
    st.markdown("""
    <div class="info-box">
    <strong>💡 Demo Note:</strong> This application uses a lightweight AI model for fast responses. 
    For complex research papers, responses focus on key concepts and practical insights.
    </div>
    """, unsafe_allow_html=True)

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("📁 Upload Research Paper")
        uploaded_file = st.file_uploader("Choose PDF file", type="pdf")
        
        if st.button("🚀 Process Paper", type="primary", use_container_width=True):
            if uploaded_file:
                with st.spinner("Processing PDF..."):
                    text = extract_text_from_pdf(uploaded_file)
                    
                    if text and len(text) > 100:  # Ensure meaningful content
                        st.session_state.chunks = split_text(text)
                        st.session_state.processed = True
                        st.success(f"✅ Processed! Split into {len(st.session_state.chunks)} sections")
                    else:
                        st.error("❌ Could not extract sufficient text from PDF. Please try another file.")
            else:
                st.warning("⚠️ Please upload a PDF file first")

    # --- MAIN CONTENT ---
    if st.session_state.get("processed", False) and st.session_state.chunks:
        st.success("🎉 Ready! Use the tabs below to analyze the paper.")
        
        # Show basic info
        total_words = sum(len(chunk.split()) for chunk in st.session_state.chunks)
        st.caption(f"📄 Document stats: {len(st.session_state.chunks)} sections, ~{total_words} words")

        tab1, tab2, tab3 = st.tabs(["💬 Q&A Analysis", "📝 Summary", "❓ Quiz"])

        with tab1:
            st.subheader("Ask Questions About the Paper")
            question = st.text_area(
                "Enter your question:", 
                height=100, 
                placeholder="e.g., Explain the main methodology...\nWhat are the key findings?\nDescribe the experimental setup..."
            )
            
            if st.button("🤖 Analyze", type="primary", use_container_width=True):
                if question.strip():
                    # Use first 3 chunks for context (adjust based on content)
                    context = "\n\n".join(st.session_state.chunks[:3])
                    prompt = ANSWER_PROMPT.format(context=context, question=question)
                    
                    with st.spinner("🔍 Analyzing paper content..."):
                        raw_response = query_ollama(prompt)
                        formatted_response = format_response(raw_response)
                        st.markdown(formatted_response, unsafe_allow_html=True)
                else:
                    st.warning("Please enter a question")

        with tab2:
            st.subheader("Generate Summary")
            if st.button("📋 Generate Summary", use_container_width=True):
                context = "\n\n".join(st.session_state.chunks[:5])  # More context for summary
                prompt = SUMMARY_PROMPT.format(context=context)
                
                with st.spinner("📝 Generating summary..."):
                    summary = query_ollama(prompt)
                    st.markdown(f"### 📄 Paper Summary\n{summary}")

        with tab3:
            st.subheader("Generate Quiz")
            if st.button("🎯 Generate Quiz", use_container_width=True):
                context = "\n\n".join(st.session_state.chunks[:4])
                prompt = QUIZ_PROMPT.format(context=context)
                
                with st.spinner("🎲 Creating quiz questions..."):
                    quiz = query_ollama(prompt)
                    st.markdown(f"### ❓ Comprehension Quiz\n{quiz}")

    else:
        # Welcome screen
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 🎯 How to Use:
            
            1. **Upload** a research paper PDF in the sidebar
            2. **Click** "Process Paper" to extract content  
            3. **Choose** your analysis method:
               - **Q&A**: Ask specific questions about the paper
               - **Summary**: Get a concise overview
               - **Quiz**: Test understanding with true/false questions
            
            ### 📚 Supported Content:
            - Academic research papers
            - Conference proceedings  
            - Technical reports
            - Scientific articles
            """)
        
        with col2:
            st.image(
                "https://images.unsplash.com/photo-1456513080510-7bf3a84b82f8?auto=format&fit=crop&w=400", 
                use_column_width=True,
                caption="Research Paper Analysis"
            )

if __name__ == "__main__":
    main()
