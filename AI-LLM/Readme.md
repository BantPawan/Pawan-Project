Research Paper Q&A Assistant
Overview
The Research Paper Q&A Assistant is a web-based application built with Streamlit that enables users to analyze research papers by uploading PDFs or providing URLs. It leverages advanced natural language processing (NLP) and large language models (LLMs) to answer questions, generate summaries, and create quizzes based on the content of research papers. The application uses Retrieval-Augmented Generation (RAG) to provide accurate, context-based responses, making it a valuable tool for students, researchers, and professionals.
This project demonstrates proficiency in Python, NLP, machine learning, and web development, integrating libraries like LangChain, FAISS, Hugging Face Transformers, and PyPDF2 to process and analyze complex academic documents.
Features

Document Processing: Upload PDF research papers or provide URLs (e.g., arXiv links) to extract and process text.
Question Answering: Ask questions about the paper and receive structured responses with sections like Key Concept, Mathematical Formulation, Mathematical Intuition, Practical Implications, and Summary.
Summary Generation: Generate concise, bullet-point summaries of the paper's content.
Quiz Generation: Create true/false quizzes with explanations to test understanding of the paper.
Efficient Retrieval: Uses FAISS for fast similarity search to retrieve relevant document chunks.
Optimized AI Model: Employs Llama-2-7b-chat-hf with 4-bit quantization for memory-efficient text generation.
User-Friendly Interface: Built with Streamlit, featuring a clean layout, custom CSS styling, and clipboard functionality for copying responses.

Technologies Used

Python: Core programming language.
Streamlit: Web framework for building the interactive user interface.
LangChain: Handles document processing, text splitting, and prompt formatting.
FAISS: Vector database for efficient similarity search.
Hugging Face Transformers: Provides the Llama-2-7b-chat-hf model and all-MiniLM-L6-v2 embeddings.
PyPDF2: Extracts text from PDF files.
Pyperclip: Enables copying responses to the clipboard.
python-dotenv: Manages environment variables for secure API key storage.
Torch: Supports machine learning operations for the AI model.



Usage

Launch the App: Run streamlit run app.py to open the web interface.
Upload Documents: Use the sidebar to upload PDF research papers or enter a URL (e.g., https://arxiv.org/pdf/...).
Process Documents: Click "Process PDFs" or "Process URL" to extract and index the content.
Ask Questions: Enter a question about the paper (e.g., "What is the main methodology?") and click "Get Answer" to receive a structured response.
Generate Summaries or Quizzes:
Click "Generate Summary" for a concise overview of the paper.
Click "Generate Quiz" for true/false questions with explanations.


Copy Responses: Use the "Copy to Clipboard" button to copy answers for further use.

Project Structure
research-paper-qa-assistant/
├── app.py              # Main Streamlit application script
├── requirements.txt    # List of dependencies
├── .env                # Environment variables (not tracked in Git)
└── README.md           # Project documentation

Example

Input: Upload a PDF from arXiv or enter a URL like https://arxiv.org/pdf/2307.09288.pdf.
Question: "Explain the main methodology of the paper."
Output: A structured response with sections like:
Key Concept: [Brief description of the paper’s main idea]
Mathematical Formulation: [Equations with explanations]
Mathematical Intuition: [Significance of the math]
Practical Implications: [3-5 applications]
Summary: [2-3 sentence recap]


Summary: A 100-word bullet-point summary of the paper.
Quiz: Three true/false questions with explanations.

