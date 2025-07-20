import json
import streamlit as st
from datetime import datetime, timedelta
from langchain.document_loaders import PyPDFLoader, TextLoader
# Import the DOCX loader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import torch
import qrcode
import os
import io
import zipfile
import base64
import hashlib
import time

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from openai import OpenAI
import numpy as np

# --- QR SHARING & STORAGE SETUP (Unified) ---

STORAGE_DIR = os.path.join(os.getcwd(), ".streamlit", "files")
METADATA_FILE = os.path.join(STORAGE_DIR, "files.json")

def init_storage():
    os.makedirs(STORAGE_DIR, exist_ok=True)

def load_files_metadata():
    init_storage()
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, 'r') as f:
                data = json.load(f)
            for fid in data:
                if 'expiry' in data[fid] and isinstance(data[fid]['expiry'], str):
                    data[fid]['expiry'] = datetime.fromisoformat(data[fid]['expiry'])
                elif 'expiry' not in data[fid]:
                    data[fid]['expiry'] = datetime.now() + timedelta(hours=24) 
            return data
        except (json.JSONDecodeError, IOError):
            return {}
    return {}

def save_files_metadata(data):
    serializable = {}
    for fid, info in data.items():
        serializable_info = info.copy()
        if isinstance(serializable_info.get('expiry'), datetime):
            serializable_info['expiry'] = serializable_info['expiry'].isoformat()
        else:
            serializable_info['expiry'] = (datetime.now() + timedelta(hours=24)).isoformat()
        serializable[fid] = serializable_info
    with open(METADATA_FILE, 'w') as f:
        json.dump(serializable, f, indent=2)

def cleanup_expired_files():
    files = load_files_metadata()
    current_time = datetime.now()
    expired_ids = [fid for fid, info in files.items() if current_time >= info.get('expiry', datetime.max)]

    for fid in expired_ids:
        try:
            file_path = os.path.join(STORAGE_DIR, f"{fid}.dat")
            if os.path.exists(file_path):
                os.remove(file_path)
            del files[fid]
        except (OSError, KeyError):
            pass

    if expired_ids:
        save_files_metadata(files)
    return files

def generate_file_id(length=10):
    return hashlib.md5(f"{time.time()}{os.urandom(8)}".encode()).hexdigest()[:length]

def create_qr_code(data):
    qr = qrcode.QRCode(version=1, box_size=10, border=3, error_correction=qrcode.constants.ERROR_CORRECT_L)
    qr.add_data(data)
    qr.make(fit=True)
    return qr.make_image(fill_color="black", back_color="white")

def save_single_file_for_sharing(uploaded_file, expiry_hours=24):
    files_data = load_files_metadata()
    file_id = generate_file_id()
    
    file_data_bytes = uploaded_file.read()
    
    with open(os.path.join(STORAGE_DIR, f"{file_id}.dat"), 'wb') as f:
        f.write(file_data_bytes)
    
    file_info = {
        'name': uploaded_file.name,
        'size': len(file_data_bytes),
        'type': uploaded_file.type,
        'expiry': datetime.now() + timedelta(hours=expiry_hours),
        'share_type': 'single_file'
    }
    files_data[file_id] = file_info
    save_files_metadata(files_data)
    return file_id, file_info

def save_batch_files_for_sharing(uploaded_files, expiry_hours=24):
    files_data = load_files_metadata()
    batch_id = generate_file_id(length=8)
    
    file_infos = []
    for i, uploaded_file in enumerate(uploaded_files):
        file_id = f"{batch_id}_{i}"
        file_data_bytes = uploaded_file.read()

        with open(os.path.join(STORAGE_DIR, f"{file_id}.dat"), 'wb') as f:
            f.write(file_data_bytes)
        
        file_info = {
            'name': uploaded_file.name,
            'size': len(file_data_bytes),
            'type': uploaded_file.type,
            'expiry': datetime.now() + timedelta(hours=expiry_hours),
            'batch_id': batch_id,
            'share_type': 'batch_file'
        }
        files_data[file_id] = file_info
        file_infos.append(file_info)
    
    save_files_metadata(files_data)
    return file_infos, batch_id

def save_document_package_for_sharing(package_data, filename="document_package.zip", expiry_hours=24):
    files_data = cleanup_expired_files()
    file_id = generate_file_id()

    with open(os.path.join(STORAGE_DIR, f"{file_id}.dat"), 'wb') as f:
        f.write(package_data)

    file_info = {
        'name': filename,
        'size': len(package_data),
        'type': 'application/zip',
        'expiry': datetime.now() + timedelta(hours=expiry_hours),
        'share_type': 'document_package'
    }
    files_data[file_id] = file_info
    save_files_metadata(files_data)
    return file_id

def get_shared_data(item_id):
    files = cleanup_expired_files()
    if item_id not in files:
        return None
    try:
        with open(os.path.join(STORAGE_DIR, f"{item_id}.dat"), 'rb') as f:
            return files[item_id], f.read()
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Error reading file {item_id}: {e}")
        return None

def format_file_size(byte_size):
    if byte_size is None: return "0 B"
    for unit in ['B', 'KB', 'MB', 'GB']:
        if byte_size < 1024.0:
            return f"{byte_size:.1f} {unit}"
        byte_size /= 1024.0
    return f"{byte_size:.1f} TB"

# --- Download Page Handler (Unified) ---

query_params = st.query_params

if 'file' in query_params or 'batch' in query_params:
    st.set_page_config(page_title="Download Files", layout="centered")
    st.title("📥 Download Your Files")
    st.markdown("---")

    if 'batch' in query_params:
        batch_id = query_params['batch']
        files = cleanup_expired_files()
        batch_files = {fid: info for fid, info in files.items() if info.get('batch_id') == batch_id}
        
        if batch_files:
            st.success(f"Found {len(batch_files)} files in batch: {batch_id}")
            st.write(f"Expires: {list(batch_files.values())[0]['expiry'].strftime('%Y-%m-%d %H:%M')}")
            
            for file_id, file_info in batch_files.items():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"📄 **{file_info['name']}** ({format_file_size(file_info['size'])})")
                with col2:
                    retrieved_data = get_shared_data(file_id)
                    if retrieved_data:
                        st.download_button("📥 Download", retrieved_data[1], file_info['name'], file_info['type'], key=f"dl_batch_{file_id}")
                    else:
                        st.error("File content not available or expired.")
        else:
            st.error("Batch not found or expired.")

    elif 'file' in query_params:
        file_id = query_params['file']
        file_data_tuple = get_shared_data(file_id)
        
        if file_data_tuple:
            file_info, data = file_data_tuple
            st.success(f"File found: {file_info['name']}")
            st.write(f"Size: {format_file_size(file_info['size'])} | Expires: {file_info['expiry'].strftime('%Y-%m-%d %H:%M')}")
            st.download_button(
                "📥 Download File",
                data,
                file_info['name'],
                file_info['type'],
                use_container_width=True,
                type="primary"
            )
        else:
            st.error("File not found or has expired.")

    st.markdown("---")
    if st.button("🏠 Go to Main App", use_container_width=True):
        st.query_params.clear()
        st.rerun()

# --- MAIN APPLICATION LOGIC ---
else:
    class SimpleRAG:
        def __init__(self):
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.client = OpenAI(
                base_url="https://router.huggingface.co/novita",
                api_key="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"
            )
            self.documents = []
            self.embeddings = None
        
        def add_docs(self, texts):
            self.documents = texts
            self.embeddings = self.model.encode(texts)
        
        def retrieve(self, query, top_k=3):
            if self.embeddings is None or len(self.embeddings) == 0:
                return []
            
            query_emb = self.model.encode([query])
            similarities = cos_sim(query_emb, self.embeddings)[0]
            top_indices = torch.argsort(similarities, descending=True)[:top_k]
            return [self.documents[i] for i in top_indices]
        
        def query(self, question):
            context = "\n".join(self.retrieve(question))
            prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer based on the context provided:"
            
            try:
                response = self.client.chat.completions.create(
                    model="meta-llama/llama-3.2-1b-instruct",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.7
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"Error generating response: {str(e)}"

    class CustomLLM:
        def __init__(self):
            self.client = OpenAI(
                base_url="https://router.huggingface.co/novita",
                api_key="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"
            )
        
        def __call__(self, prompt):
            try:
                response = self.client.chat.completions.create(
                    model="meta-llama/llama-3.2-1b-instruct",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.7
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"Error: {str(e)}"

    llm = CustomLLM()

    st.set_page_config(
        page_title="Smart Document Assistant", 
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
        .main-header { text-align: center; padding: 2rem 0; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 2rem; }
        .upload-section { background: #f8f9fa; padding: 1.5rem; border-radius: 10px; border: 2px dashed #ddd; }
        .summary-card { background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 1rem 0; border-left: 4px solid #667eea; }
        .qr-section { text-align: center; background: #f0f8ff; padding: 2rem; border-radius: 10px; margin: 1rem 0; }
        .file-status { background: #d4edda; border: 1px solid #c3e6cb; border-radius: 5px; padding: 0.5rem; margin: 0.25rem 0; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-header"><h1>Smart Document Assistant</h1><p>Upload, Summarize, Ask Questions & Download</p></div>', unsafe_allow_html=True)

    def create_download_package(documents, summaries):
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for doc_info in documents:
                # Ensure the 'text' is encoded to bytes for zipping
                zip_file.writestr(f"documents/{doc_info['name']}", doc_info['text'].encode('utf-8'))
            
            if summaries:
                all_summaries = "\n\n" + "="*50 + "\n\n".join(summaries)
                zip_file.writestr("summaries/all_summaries.txt", all_summaries.encode('utf-8'))
                
                for i, summary in enumerate(summaries):
                    doc_name = documents[i]['name'] if i < len(documents) else f"document_{i+1}"
                    summary_filename = f"summaries/{os.path.splitext(doc_name)[0]}_summary.txt"
                    zip_file.writestr(summary_filename, summary.encode('utf-8'))
            
            metadata = {
                "created": datetime.now().isoformat(), "total_documents": len(documents),
                "document_names": [doc['name'] for doc in documents], "ai_model": "meta-llama/llama-3.2-1b-instruct",
                "embedding_model": "all-MiniLM-L6-v2"
            }
            zip_file.writestr("metadata.json", json.dumps(metadata, indent=2).encode('utf-8'))
        
        zip_buffer.seek(0)
        return zip_buffer.getvalue()

    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.subheader("Upload Your Documents")
    
    uploaded_files_for_processing = st.file_uploader(
        "Choose files (PDF, TXT, DOCX):", 
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        key="main_uploader"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_files_for_processing:
        if "processed_docs" not in st.session_state or "all_texts" not in st.session_state or st.button("Re-process Documents", key="reprocess_button"):
            st.session_state.processed_docs = []
            st.session_state.all_texts = []
            st.session_state.summaries = []
            st.session_state.pop('sharable_link', None) 
            st.session_state.pop('sharable_qr', None)
            st.session_state.pop('rag_system', None)
            
            with st.spinner("Processing documents..."):
                progress_bar = st.progress(0)
                for i, uploaded_file in enumerate(uploaded_files_for_processing):
                    text_content = ""
                    try:
                        if uploaded_file.name.lower().endswith(".pdf"):
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                                tmp_file.write(uploaded_file.read())
                                tmp_path = tmp_file.name
                            loader = PyPDFLoader(tmp_path)
                            docs = loader.load()
                            text_content = "\n".join([page.page_content for page in docs])
                            os.unlink(tmp_path)
                        elif uploaded_file.name.lower().endswith(".txt"):
                            text_content = uploaded_file.read().decode('utf-8')
                        elif uploaded_file.name.lower().endswith(".docx"):
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
                                tmp_file.write(uploaded_file.read())
                                tmp_path = tmp_file.name
                            # Use UnstructuredWordDocumentLoader for DOCX
                            loader = UnstructuredWordDocumentLoader(tmp_path)
                            docs = loader.load()
                            text_content = "\n".join([page.page_content for page in docs])
                            os.unlink(tmp_path)
                        
                        if text_content: # Only add if it's a PDF, TXT, or DOCX that yielded content
                            st.session_state.processed_docs.append({
                                'name': uploaded_file.name, 
                                'text': text_content, 
                                'size': len(text_content),
                                'original_file': uploaded_file # Store original for direct sharing
                            })
                            st.session_state.all_texts.append(text_content)
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                    progress_bar.progress((i + 1) / len(uploaded_files_for_processing))
                
                ai_processed_count = sum(1 for doc in st.session_state.processed_docs if doc['text'])
                # st.success(f"Processed {ai_processed_count} documents for AI features. {len(st.session_state.processed_docs) - ai_processed_count} files stored for sharing only (if any failed text extraction).")
        
        st.subheader("Processed Documents")
        if st.session_state.processed_docs:
            cols = st.columns(min(3, len(st.session_state.processed_docs)))
            for i, doc_info in enumerate(st.session_state.processed_docs):
                with cols[i % 3]:
                    display_size = doc_info['size'] if doc_info['text'] else doc_info['original_file'].getbuffer().nbytes
                    file_type_note = ""
                    if not doc_info['text']: # This would now only happen if DOCX extraction failed or it's a non-text file
                        file_type_note = "(Text extraction failed/not applicable)"
                    st.markdown(f"""<div class="file-status"><strong>{doc_info['name']}</strong> {file_type_note}<br><small>Size: {format_file_size(display_size)}</small></div>""", unsafe_allow_html=True)
        else:
            st.info("No supported documents uploaded for AI processing.")


        tab1, tab2, tab3 = st.tabs(["📊 Summaries", "❓ Q&A", "📲 QR Generator"])
        
        with tab1:
            st.subheader("Document Summaries")
            if st.session_state.all_texts:
                summary_type = st.radio("Summary type:", ["Quick Summary", "Detailed Summary", "Key Points"], horizontal=True, key="summary_radio")
                if st.button("Generate Summaries", type="primary", key="generate_summary_button"):
                    st.session_state.summaries = []
                    progress_bar = st.progress(0)
                    for i, doc_info in enumerate(st.session_state.processed_docs):
                        if doc_info['text']: # Only summarize if text content is available
                            with st.status(f"Summarizing: {doc_info['name']}", expanded=True):
                                try:
                                    # Ensure prompt fits within LLM context window (e.g., 4000 characters for initial check)
                                    text_to_summarize = doc_info['text'][:4000] # Truncate for prompt
                                    if summary_type == "Quick Summary":
                                        prompt = f"Provide a concise summary in 2-3 sentences of the following text:\n\n{text_to_summarize}"
                                    elif summary_type == "Detailed Summary":
                                        prompt = f"Provide a comprehensive summary with main points and details of the following text:\n\n{text_to_summarize}"
                                    else:
                                        prompt = f"Extract and list the key points and important information from the following text as bullet points:\n\n{text_to_summarize}"
                                    
                                    result = llm(prompt)
                                    summary = f"**{doc_info['name']}:**\n{result}\n"
                                    st.session_state.summaries.append(summary)
                                    st.write(f"Completed: {doc_info['name']}")
                                except Exception as e:
                                    st.error(f"Error with {doc_info['name']}: {str(e)}")
                            progress_bar.progress((i + 1) / len(st.session_state.processed_docs))
                    st.success("All summaries generated!")
                
                if "summaries" in st.session_state and st.session_state.summaries:
                    st.subheader("Generated Summaries")
                    for summary in st.session_state.summaries:
                        st.markdown(f'<div class="summary-card">{summary}</div>', unsafe_allow_html=True)
                    all_summaries_text = "\n".join(st.session_state.summaries)
                    st.download_button("Download All Summaries", all_summaries_text, file_name=f"summaries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", mime="text/plain", use_container_width=True)
            else:
                st.info("Upload PDF, TXT, or DOCX documents to generate summaries.")

        with tab2:
            st.subheader("Ask Questions About Your Documents")
            if st.session_state.all_texts:
                if "rag_system" not in st.session_state:
                    with st.spinner("Setting up RAG system..."):
                        try:
                            st.session_state.rag_system = SimpleRAG()
                            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                            all_chunks = [chunk for text in st.session_state.all_texts for chunk in splitter.split_text(text)]
                            st.session_state.rag_system.add_docs(all_chunks)
                            # st.success(f"RAG system is ready! Indexed {len(all_chunks)} chunks.")
                        except Exception as e:
                            st.error(f"Error setting up RAG: {str(e)}")
                
                if "rag_system" in st.session_state:
                    question = st.text_area("Your Question:", height=100, placeholder="Type your question here...", key="rag_question")
                    if st.button("Get Answer", type="primary", key="rag_button"):
                        if question:
                            with st.spinner("Thinking..."):
                                answer = st.session_state.rag_system.query(question)
                                st.markdown("### Answer:")
                                st.write(answer)
                                with st.expander("View Retrieved Context"):
                                    for i, doc in enumerate(st.session_state.rag_system.retrieve(question), 1):
                                        st.text(f"Context {i}:\n{doc[:300]}...")
                                        st.markdown("---")
                        else:
                            st.warning("Please enter a question.")
            else:
                st.info("Upload PDF, TXT, or DOCX documents first for Q&A.")

        with tab3:
            st.subheader("Download Processed Package with QR Code")
            st.markdown("Download originals + summaries. Link valid for 24 hours.")

            if st.session_state.processed_docs and any(doc['text'] for doc in st.session_state.processed_docs): # Ensure there are docs with content
                summaries_to_include = st.session_state.get('summaries', [])
                package_filename = f"documents_package_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
                download_data = create_download_package(
                    [doc for doc in st.session_state.processed_docs if doc['text']], # Only include AI processed docs
                    summaries_to_include
                )

                col1_pkg, col2_pkg = st.columns(2)
                with col1_pkg:
                    st.markdown("#### For This Device")
                    st.download_button("Download Package Directly", download_data, file_name=package_filename, mime="application/zip", use_container_width=True, type="primary", key="direct_pkg_download")
                    st.write(f"Package Size: {format_file_size(len(download_data))}")

                with col2_pkg:
                    st.markdown("#### For Another Device use QR Code")
                    # if st.button("Generate Sharable QR Code (Package)", use_container_width=True, key="generate_pkg_qr"):
                    with st.spinner("Generating secure link..."):
                        file_id = save_document_package_for_sharing(download_data, package_filename)
                        app_url = "https://koqndcvr7kfjrnroznzfws.streamlit.app" # Replace with your deployed app URL
                        st.session_state.sharable_link_pkg = f"{app_url}/?file={file_id}"
                        st.session_state.sharable_qr_pkg = create_qr_code(st.session_state.sharable_link_pkg)

                if 'sharable_link_pkg' in st.session_state and 'sharable_qr_pkg' in st.session_state:
                    st.markdown("---")
                    st.success("Link and QR Code Generated!")
                    
                    st.markdown('<div class="qr-section">', unsafe_allow_html=True)
                    qr_img = st.session_state.sharable_qr_pkg
                    img_buffer = io.BytesIO()
                    qr_img.save(img_buffer, format='PNG')
                    img_b64 = base64.b64encode(img_buffer.getvalue()).decode()
                    
                    st.markdown(f'<img src="data:image/png;base64,{img_b64}" style="max-width: 250px; border: 3px solid #667eea; border-radius: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">', unsafe_allow_html=True)
                    st.code(st.session_state.sharable_link_pkg, language=None)
                    st.caption("Scan QR or use link to download.")
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("Upload PDF, TXT, or DOCX documents for package download.")

    st.markdown('---\n<div style="text-align: center; color: #666;"><p>Smart Document Assistant | Built with Streamlit & Custom AI Models</p></div>', unsafe_allow_html=True)