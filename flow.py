import os
import streamlit as st
import pdfplumber
import torch
import tempfile
import pandas as pd
import re
import nltk
import io
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from langchain_ollama.llms import OllamaLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import docx
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Image, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from PIL import Image as PILImage
import base64
import json
from pandas_profiling import ProfileReport

# Set page config as the very first Streamlit command
st.set_page_config(page_title="Student Study Assistant", page_icon="📚", layout="wide")

# Try to download NLTK resources if needed
try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    st.warning(f"Failed to download NLTK resources: {e}. Some functionality might be limited.")

# Define constants
DEFAULT_MODEL = "all-MiniLM-L6-v2"  # Lighter model for better performance
DEFAULT_LLM = "llama3:8b"  # Default model

# Configure page appearance
def set_page_style():
    st.markdown("""
    <style>
    .stApp {
        background-color: #000000; /* Black background */
        color: #ff69b4; /* Pink text */
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #ff69b4; /* Pink headers */
    }
    .stButton > button {
        background-color: #ff69b4; /* Pink button */
        color: white;
        border-radius: 8px;
    }
    .stButton > button:hover {
        background-color: #ff1493; /* Darker pink on hover */
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 16px;
        font-weight: 500;
        color: #ff69b4; /* Pink tabs */
    }
    .stExpander {
        border-radius: 8px;
        border: 1px solid #ff69b4; /* Pink border */
    }
    .card {
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        background-color: #1a1a1a; /* Dark gray background */
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        color: #ff69b4; /* Pink text */
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'documents' not in st.session_state:
        st.session_state.documents = []
    if 'document_texts' not in st.session_state:
        st.session_state.document_texts = {}
    if 'document_chunks' not in st.session_state:
        st.session_state.document_chunks = {}
    if 'summarized_documents' not in st.session_state:
        st.session_state.summarized_documents = {}
    if 'created_flashcards' not in st.session_state:
        st.session_state.created_flashcards = []
    if 'study_timer' not in st.session_state:
        st.session_state.study_timer = {"active": False, "start_time": None, "total_time": 0}
    if 'uploaded_dataset' not in st.session_state:
        st.session_state.uploaded_dataset = None

# Load models
@st.cache_resource
def load_embedding_model():
    try:
        return SentenceTransformer(DEFAULT_MODEL, device='cpu')
    except Exception as e:
        st.error(f"Failed to load embedding model: {e}")
        return None

@st.cache_resource
def load_llm():
    try:
        return OllamaLLM(
            model=DEFAULT_LLM, 
            system_prompt="You are Study Assistant, an AI helping students with their academic questions based on their uploaded study materials."
        )
    except Exception as e:
        st.error(f"Failed to load LLM model '{DEFAULT_LLM}'. Please check Ollama or your model list.")
        # Return a dummy LLM for graceful degradation
        class DummyLLM:
            def run(self, **kwargs):
                return "I'm currently unable to process this request. Please check your model configuration."
        return DummyLLM()

# Document extraction utilities
def extract_text_from_file(file, file_type):
    """Extract text from various file types"""
    try:
        if file_type == 'pdf':
            with pdfplumber.open(file) as pdf:
                return "\n".join([page.extract_text() or "" for page in pdf.pages])
        elif file_type == 'docx':
            doc = docx.Document(file)
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])
        elif file_type == 'txt':
            return file.read().decode('utf-8')
        else:
            st.warning(f"Unsupported file type: {file_type}")
            return ""
    except Exception as e:
        st.error(f"Error processing {file_type}: {str(e)}")
        return ""

# Text preprocessing
def preprocess_text(text, chunk_size=1000, overlap=200):
    """Split text into chunks for processing"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Split text into sentences
    try:
        sentences = nltk.sent_tokenize(text)
    except:
        # Fallback to simple splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += " " + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    
    if current_chunk:  # Add the last chunk
        chunks.append(current_chunk.strip())
    
    return chunks

# LLM interaction functions
def generate_summary(text, llm):
    """Generate a summary of the text using LLM"""
    if len(text) < 100:
        return text
    
    prompt = PromptTemplate(
        input_variables=["text"],
        template="""
        Provide a concise summary of the following text in 3-5 sentences:
        
        {text}
        
        Summary:"""
    )
    
    summary_chain = LLMChain(llm=llm, prompt=prompt)
    
    # Process in chunks if text is too long
    if len(text) > 2000:
        chunks = preprocess_text(text)
        summaries = []
        for chunk in chunks[:3]:  # Limit to first 3 chunks for speed
            if len(chunk) > 100:
                result = summary_chain.run(text=chunk[:1500])
                summaries.append(result)
        return " ".join(summaries)
    else:
        return summary_chain.run(text=text[:1500])

def generate_flashcards(text, llm, max_cards=5):
    """Generate flashcards from text using LLM"""
    prompt = PromptTemplate(
        input_variables=["text", "max_cards"],
        template="""
        Create {max_cards} flashcards from the following text. 
        Format each flashcard exactly as:
        Question: [question]
        Answer: [answer]
        
        Make the questions conceptual and thought-provoking rather than simple facts.
        
        Text: {text}
        """
    )
    
    flashcard_chain = LLMChain(llm=llm, prompt=prompt)
    result = flashcard_chain.run(text=text[:4000], max_cards=max_cards)  # Limit text length
    
    # Parse the response into flashcard pairs
    flashcards = []
    pairs = re.findall(r"Question: (.*?)\nAnswer: (.*?)(?=\n\nQuestion:|$)", result, re.DOTALL)
    for question, answer in pairs:
        flashcards.append({"question": question.strip(), "answer": answer.strip()})
    
    return flashcards

def extract_key_terms(text, llm, max_terms=10):
    """Extract key terms/vocabulary from text"""
    prompt = PromptTemplate(
        input_variables=["text", "max_terms"],
        template="""
        Extract {max_terms} key technical terms or vocabulary words from the following text.
        For each term, provide a brief definition.
        Format as "Term: [term] | Definition: [definition]"
        
        Text: {text}
        """
    )
    
    # Limit text length to avoid token limits
    text_to_use = text[:min(len(text), 4000)]
    
    terms_chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        response = terms_chain.run(text=text_to_use, max_terms=max_terms)
        
        # Parse response into key terms
        key_terms = []
        for line in response.split('\n'):
            if '|' in line and 'Term:' in line and 'Definition:' in line:
                parts = line.split('|')
                term = parts[0].replace('Term:', '').strip()
                definition = parts[1].replace('Definition:', '').strip()
                if term and definition:
                    key_terms.append({"term": term, "definition": definition})
        
        return key_terms
    except Exception as e:
        print(f"Error extracting key terms: {str(e)}")
        return []

def answer_question(query, llm, context, history=None):
    """Answer a question based on document context"""
    if history is None:
        history = []
    
    # Format conversation history
    formatted_history = ""
    for i, (q, a) in enumerate(history[-3:]):  # Only use last 3 exchanges
        formatted_history += f"Question {i+1}: {q}\nAnswer {i+1}: {a}\n\n"
    
    # Define the prompt template
    prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template="""
        You are a helpful study assistant for students.
        
        Previous conversation:
        {history}
        
        Based on the following context and previous conversation, answer the student's question.
        If the answer cannot be found in the context, say so and provide what information you can.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:
        """
    )
    
    # Create the LLMChain with the provided LLM and prompt
    qa_chain = LLMChain(llm=llm, prompt=prompt)
    
    # Run the chain with the input data
    return qa_chain.run(
        history=formatted_history, 
        context=context[:3000],  # Limit context length
        question=query
    )

# Study timer functions
def start_timer():
    """Start study timer"""
    st.session_state.study_timer["active"] = True
    st.session_state.study_timer["start_time"] = pd.Timestamp.now()

def stop_timer():
    """Stop study timer and record elapsed time"""
    if st.session_state.study_timer["active"]:
        elapsed = pd.Timestamp.now() - st.session_state.study_timer["start_time"]
        st.session_state.study_timer["total_time"] += elapsed.total_seconds()
        st.session_state.study_timer["active"] = False
        return elapsed.total_seconds()
    return 0

# Search functionality
def search_documents(query, texts, embedding_model):
    """Simple search implementation using cosine similarity"""
    if not texts:
        return []
    
    query_embedding = embedding_model.encode(query)
    results = []
    
    for doc_name, chunks in texts.items():
        for i, chunk in enumerate(chunks):
            chunk_embedding = embedding_model.encode(chunk)
            similarity = torch.nn.functional.cosine_similarity(
                torch.tensor(query_embedding).unsqueeze(0),
                torch.tensor(chunk_embedding).unsqueeze(0)
            ).item()
            
            results.append({
                "doc_name": doc_name,
                "chunk_id": i,
                "text": chunk,
                "score": similarity
            })
    
    # Sort by similarity score
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:5]  # Return top 5 results

# Data analysis
def data_analysis_explore(df):
    """Analyze a pandas dataframe and generate visualizations"""
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("#### 📊 Basic Statistics")
    st.dataframe(df.describe())
    st.markdown('</div>', unsafe_allow_html=True)
    
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if numeric_cols:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("#### 🔥 Correlation Heatmap")
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("#### 📈 Distribution Plots")
        selected_col = st.selectbox("Select column for histogram:", numeric_cols)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df[selected_col], kde=True, ax=ax)
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
    
    if categorical_cols and numeric_cols:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("#### 📊 Categorical Analysis")
        cat_col = st.selectbox("Select categorical column:", categorical_cols)
        num_col = st.selectbox("Select numeric column:", numeric_cols)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x=cat_col, y=num_col, data=df, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

# Report generation
def generate_study_report(study_data):
    """Generate comprehensive study report"""
    try:
        # Create a temporary PDF file with a unique name
        pdf_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        filename = pdf_file.name
        pdf_file.close()  # Close the file so ReportLab can open it
        
        # Create the PDF document
        doc = SimpleDocTemplate(filename, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []
        
        # Title
        elements.append(Paragraph("Study Session Report", styles['Title']))
        elements.append(Spacer(1, 12))
        
        # Study session statistics
        elements.append(Paragraph("Study Session Statistics", styles['Heading2']))
        
        # Add default values and type checking
        total_time = study_data.get('total_time', 0)
        if not isinstance(total_time, (int, float)):
            total_time = 0
            
        doc_count = len(study_data.get('documents', []))
        question_count = len(study_data.get('questions', []))
        flashcard_count = len(study_data.get('flashcards', []))
        
        stats_data = [
            ["Total Study Time", f"{total_time // 60:.0f} minutes"],
            ["Documents Reviewed", str(doc_count)],
            ["Questions Asked", str(question_count)],
            ["Flashcards Created", str(flashcard_count)]
        ]
        
        stats_table = Table(stats_data, colWidths=[200, 200])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(stats_table)
        elements.append(Spacer(1, 12))
        
        # Document summaries
        if study_data.get('documents'):
            elements.append(Paragraph("Document Summaries", styles['Heading2']))
            for document in study_data['documents']:
                doc_name = document.get('name', 'Untitled Document')
                doc_summary = document.get('summary', 'No summary available')
                
                elements.append(Paragraph(doc_name, styles['Heading3']))
                elements.append(Paragraph(doc_summary, styles['Normal']))
                elements.append(Spacer(1, 12))
        
        # Key terms
        if study_data.get('key_terms'):
            elements.append(Paragraph("Key Terms and Definitions", styles['Heading2']))
            terms_data = [["Term", "Definition"]]
            
            key_terms = study_data['key_terms']
            for term_item in key_terms:
                if isinstance(term_item, dict):
                    # Format: {"term": "...", "definition": "..."}
                    term = term_item.get('term', '')
                    definition = term_item.get('definition', '')
                    terms_data.append([term, definition])
                elif isinstance(term_item, (list, tuple)) and len(term_item) >= 2:
                    # Format: ["term", "definition"]
                    terms_data.append([term_item[0], term_item[1]])
                elif isinstance(term_item, str):
                    # Fallback for string format
                    terms_data.append([term_item, ""])
            
            terms_table = Table(terms_data, colWidths=[150, 300])
            terms_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(terms_table)
            elements.append(Spacer(1, 12))
        
        # Flashcards
        if study_data.get('flashcards'):
            elements.append(Paragraph("Study Flashcards", styles['Heading2']))
            cards_data = [["Question", "Answer"]]
            
            for card in study_data['flashcards']:
                question = card.get('question', 'No question')
                answer = card.get('answer', 'No answer')
                cards_data.append([question, answer])
            
            cards_table = Table(cards_data, colWidths=[225, 225])
            cards_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(cards_table)
            elements.append(Spacer(1, 12))
        
        # Word cloud if available
        if 'wordcloud' in study_data:
            elements.append(Paragraph("Content Word Cloud", styles['Heading2']))
            wordcloud_data = study_data['wordcloud']
            if isinstance(wordcloud_data, io.BytesIO):
                img = Image(wordcloud_data, width=400, height=200)
                elements.append(img)
            elif isinstance(wordcloud_data, str):
                try:
                    img = Image(wordcloud_data, width=400, height=200)
                    elements.append(img)
                except Exception as e:
                    print(f"Error loading wordcloud image: {str(e)}")
        
        # Build PDF
        doc.build(elements)
        return filename
    except Exception as e:
        print(f"Error generating PDF: {str(e)}")
        raise Exception(f"Error generating full report: {str(e)}")

def generate_word_cloud(text):
    """Generate word cloud from text"""
    try:
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            colormap='viridis',
            max_words=100
        ).generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        
        return buf
    except Exception as e:
        st.error(f"Error generating word cloud: {str(e)}")
        return None
    
def generate_html_eda_report(df):
    """
    Generate an HTML EDA report for the given dataframe
    
    Parameters:
    df (pandas.DataFrame): DataFrame to analyze
    
    Returns:
    str: Path to the generated HTML report
    """
    import pandas as pd
    from pandas_profiling import ProfileReport
    import tempfile
    import os
    
    # Create a temp file for the report
    temp_dir = tempfile.mkdtemp()
    report_path = os.path.join(temp_dir, "eda_report.html")
    
    # Generate the report
    profile = ProfileReport(df, title="Exploratory Data Analysis Report", explorative=True)
    profile.to_file(report_path)
    
    return report_path

# Main app
def main():
    set_page_style()
    init_session_state()
    
    st.title("📚 Student Study Assistant")
    
    # Sidebar with study timer
    with st.sidebar:
        st.header("⏱️ Study Timer")
        
        timer_active = st.session_state.study_timer.get("active", False)
        total_mins = st.session_state.study_timer.get("total_time", 0) // 60
        
        st.metric("Total Study Time", f"{int(total_mins)} minutes")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if not timer_active:
                if st.button("▶️ Start Timer"):
                    start_timer()
                    st.rerun()  # Fixed: Replaced st.experimental_rerun() with st.rerun()
        with col2:
            if timer_active:
                if st.button("⏹️ Stop Timer"):
                    stop_timer()
                    st.rerun()  # Fixed: Replaced st.experimental_rerun() with st.rerun()
        
        st.divider()
        
        st.header("🔄 Reset Data")
        if st.button("Clear All Data"):
            # Clear session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            init_session_state()
            st.success("All data cleared!")
            st.rerun()  # Fixed: Replaced st.experimental_rerun() with st.rerun())
    
    # Main area tabs
    tabs = st.tabs(["📄 Documents", "❓ Q&A", "🎴 Flashcards", "📊 Analysis", "📝 Study Report", "📊 Dataset Analysis"])
    
    # Documents Tab
    with tabs[0]:
        st.header("Upload and Manage Documents")
        
        uploaded_file = st.file_uploader("Upload a study document", type=["pdf", "docx", "txt"])
        
        if uploaded_file:
            file_type = uploaded_file.name.split(".")[-1].lower()
            
            if st.button("Process Document"):
                with st.spinner("Processing document..."):
                    # Extract text
                    document_text = extract_text_from_file(uploaded_file, file_type)
                    
                    if document_text:
                        # Add to session state
                        doc_id = f"{uploaded_file.name}"
                        
                        st.session_state.document_texts[doc_id] = document_text
                        
                        # Create chunks for searching
                        chunks = preprocess_text(document_text)
                        st.session_state.document_chunks[doc_id] = chunks
                        
                        # Generate summary
                        llm = load_llm()
                        summary = generate_summary(document_text, llm)
                        
                        # Add to documents list
                        st.session_state.documents.append({
                            "id": doc_id,
                            "name": uploaded_file.name,
                            "type": file_type,
                            "summary": summary
                        })
                        
                        st.success(f"Document '{uploaded_file.name}' processed successfully!")
                        
        # Display documents
        if st.session_state.documents:
            st.subheader("Your Documents")
            
            for i, doc in enumerate(st.session_state.documents):
                with st.expander(f"{doc['name']}"):
                    st.markdown(f"**Summary:** {doc.get('summary', 'No summary available')}")
                    
                    # View full text button
                    if st.button(f"View Full Text", key=f"view_{i}"):
                        doc_text = st.session_state.document_texts.get(doc['id'], "Text not available")
                        st.text_area("Document Content", doc_text, height=300)
        else:
            st.info("No documents uploaded yet. Upload a document to get started.")
    
    # Q&A Tab
    with tabs[1]:
        st.header("Ask Questions About Your Documents")
        
        if not st.session_state.documents:
            st.warning("Please upload documents in the Documents tab first.")
        else:
            # Select document
            doc_names = [doc["name"] for doc in st.session_state.documents]
            selected_doc = st.selectbox("Select a document to query", doc_names)
            
            # Get the document ID
            doc_id = None
            for doc in st.session_state.documents:
                if doc["name"] == selected_doc:
                    doc_id = doc["id"]
                    break
            
            # Question input
            question = st.text_input("Enter your question:")
            
            if st.button("Ask Question") and question and doc_id:
                with st.spinner("Finding answer..."):
                    # Get document text
                    doc_text = st.session_state.document_texts.get(doc_id, "")
                    
                    # Get answer
                    llm = load_llm()
                    answer = answer_question(
                        query=question,
                        llm=llm,
                        context=doc_text,
                        history=st.session_state.conversation_history
                    )
                    
                    # Add to conversation history
                    st.session_state.conversation_history.append((question, answer))
            
            # Display conversation history
            if st.session_state.conversation_history:
                st.subheader("Conversation History")
                
                for i, (q, a) in enumerate(st.session_state.conversation_history):
                    st.markdown(f"**Question {i+1}:** {q}")
                    st.markdown(f"**Answer {i+1}:** {a}")
                    st.divider()
    
    # Flashcards Tab
    with tabs[2]:
        st.header("Create and Study Flashcards")
        
        if not st.session_state.documents:
            st.warning("Please upload documents in the Documents tab first.")
        else:
            doc_names = [doc["name"] for doc in st.session_state.documents]
            selected_doc = st.selectbox("Select a document for flashcards", doc_names, key="fc_doc_select")
            
            # Get the document ID
            doc_id = None
            for doc in st.session_state.documents:
                if doc["name"] == selected_doc:
                    doc_id = doc["id"]
                    break
            
            num_cards = st.slider("Number of flashcards to generate", 3, 10, 5)
            
            if st.button("Generate Flashcards") and doc_id:
                with st.spinner("Creating flashcards..."):
                    # Get document text
                    doc_text = st.session_state.document_texts.get(doc_id, "")
                    
                    # Generate flashcards
                    llm = load_llm()
                    flashcards = generate_flashcards(doc_text, llm, max_cards=num_cards)
                    
                    # Store flashcards
                    st.session_state.created_flashcards = flashcards
            
            # Display flashcards
            if st.session_state.created_flashcards:
                st.subheader("Your Flashcards")
                
                for i, card in enumerate(st.session_state.created_flashcards):
                    with st.expander(f"Card {i+1}: {card['question'][:50]}..."):
                        st.markdown(f"**Question:** {card['question']}")
                        st.markdown(f"**Answer:** {card['answer']}")
    
    # Analysis Tab
    with tabs[3]:
        st.header("Analyze Documents")
        
        if not st.session_state.documents:
            st.warning("Please upload documents in the Documents tab first.")
        else:
            # Select document
            doc_names = [doc["name"] for doc in st.session_state.documents]
            selected_doc = st.selectbox("Select a document to analyze", doc_names, key="analysis_doc_select")
            
            # Get the document ID
            doc_id = None
            selected_doc_idx = 0
            for i, doc in enumerate(st.session_state.documents):
                if doc["name"] == selected_doc:
                    doc_id = doc["id"]
                    selected_doc_idx = i
                    break
            
            if doc_id:
                # Text analysis
                if st.button("Analyze Document"):
                    with st.spinner("Analyzing document..."):
                        doc_text = st.session_state.document_texts.get(doc_id, "")
                        
                        # Word cloud
                        wordcloud_img = generate_word_cloud(doc_text)
                        if wordcloud_img:
                            st.image(wordcloud_img)
                        
                        # Extract key terms
                        llm = load_llm()
                        key_terms = extract_key_terms(doc_text, llm)
                        
                        if key_terms:
                            st.subheader("Key Terms and Definitions")
                            
                            for term in key_terms:
                                with st.expander(term["term"]):
                                    st.write(term["definition"])
    
    # Study Report Tab
    with tabs[4]:
        st.header("Generate Study Report")
        
        if not st.session_state.documents:
            st.warning("Please upload documents in the Documents tab first.")
        else:
            if st.button("Generate Study Report"):
                with st.spinner("Creating study report..."):
                    try:
                        # Prepare data for report
                        report_data = {
                            "total_time": st.session_state.study_timer.get("total_time", 0),
                            "documents": st.session_state.documents,
                            "questions": st.session_state.conversation_history,
                            "flashcards": st.session_state.created_flashcards,
                        }

                        # Combine all document texts into one string
                        all_text = " ".join([
                            st.session_state.document_texts.get(doc["id"], "") 
                            for doc in st.session_state.documents
                        ])

                        # Generate word cloud from combined text
                        wordcloud_img = generate_word_cloud(all_text)
                        if wordcloud_img:
                            report_data["wordcloud"] = wordcloud_img

                        # Extract key terms from the combined text
                        llm = load_llm()
                        key_terms = extract_key_terms(all_text[:4000], llm, max_terms=15)
                        if key_terms:
                            report_data["key_terms"] = key_terms

                        # Generate PDF report
                        report_file = generate_study_report(report_data)

                        # Display success and provide download
                        st.success("Study report generated!")

                        # Read the PDF file
                        with open(report_file, "rb") as f:
                            pdf_bytes = f.read()

                        # Download button
                        st.download_button(
                            label="Download Study Report (PDF)",
                            data=pdf_bytes,
                            file_name="Study_Report.pdf",
                            mime="application/pdf"
                        )

                        # Clean up temp file
                        try:
                            os.unlink(report_file)
                        except:
                            pass

                    except Exception as e:
                        st.error(f"Error generating report: {str(e)}")

    # Dataset Analysis Tab
    with tabs[5]:
        st.header("📊 Dataset Analysis")
        
        # Upload dataset
        uploaded_dataset = st.file_uploader("Upload a dataset (CSV or Excel)", type=["csv", "xlsx"])
        
        if uploaded_dataset:
            try:
                # Load dataset
                if uploaded_dataset.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_dataset)
                else:
                    df = pd.read_excel(uploaded_dataset)
                
                st.session_state.uploaded_dataset = df
                st.success("Dataset uploaded successfully!")
                
                # Display dataset preview
                st.subheader("Dataset Preview")
                st.dataframe(df.head())
                
                # Perform EDA
                st.subheader("Exploratory Data Analysis (EDA)")
                
                # Basic statistics
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.write("#### 📊 Basic Statistics")
                st.dataframe(df.describe())
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Correlation heatmap
                numeric_cols = df.select_dtypes(include='number').columns.tolist()
                if numeric_cols:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.write("#### 🔥 Correlation Heatmap")
                    corr = df[numeric_cols].corr()
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
                    st.pyplot(fig)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Generate HTML EDA report
                if st.button("Generate EDA Report"):
                    with st.spinner("Generating EDA report..."):
                        report_path = generate_html_eda_report(df)
                        
                        # Provide download link
                        with open(report_path, "rb") as file:
                            st.download_button(
                                label="Download EDA Report (HTML)",
                                data=file,
                                file_name="eda_report.html",
                                mime="text/html"
                            )
                        
                        st.success("EDA report generated!")
            except Exception as e:
                st.error(f"Error processing dataset: {str(e)}")

# Run the app
if __name__ == "__main__":
    main()