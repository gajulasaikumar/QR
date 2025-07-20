import streamlit as st
import qrcode
import io
import hashlib
import time
import os
import json
from datetime import datetime, timedelta

st.set_page_config(page_title="FliQR - Multi File Share", page_icon="📱", layout="wide")

STORAGE_DIR = os.path.join(os.getcwd(), ".streamlit", "files")
METADATA_FILE = os.path.join(STORAGE_DIR, "files.json")

def init_storage():
    os.makedirs(STORAGE_DIR, exist_ok=True)

def load_files():
    init_storage()
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, 'r') as f:
                data = json.load(f)
                for fid in data:
                    data[fid]['expiry'] = datetime.fromisoformat(data[fid]['expiry'])
                return data
        except:
            return {}
    return {}

def save_files(data):
    serializable = {}
    for fid, info in data.items():
        serializable[fid] = {**info, 'expiry': info['expiry'].isoformat()}
    with open(METADATA_FILE, 'w') as f:
        json.dump(serializable, f)

def cleanup_expired():
    files = load_files()
    current = datetime.now()
    expired = [fid for fid, info in files.items() if current >= info['expiry']]
    
    for fid in expired:
        try:
            os.remove(os.path.join(STORAGE_DIR, f"{fid}.dat"))
            del files[fid]
        except:
            pass
    
    if expired:
        save_files(files)
    return files

def generate_id():
    return hashlib.md5(f"{time.time()}{os.urandom(8)}".encode()).hexdigest()[:8]

def create_qr(data):
    qr = qrcode.QRCode(version=1, box_size=8, border=2)
    qr.add_data(data)
    qr.make(fit=True)
    return qr.make_image(fill_color="black", back_color="white")

def save_files_batch(uploaded_files):
    files_data = load_files()
    batch_id = generate_id()
    file_infos = []
    
    for uploaded_file in uploaded_files:
        if uploaded_file.type not in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/plain']:
            continue
            
        file_id = f"{batch_id}_{len(file_infos)}"
        file_data = uploaded_file.read()
        
        with open(os.path.join(STORAGE_DIR, f"{file_id}.dat"), 'wb') as f:
            f.write(file_data)
        
        file_info = {
            'name': uploaded_file.name,
            'size': len(file_data),
            'type': uploaded_file.type,
            'expiry': datetime.now() + timedelta(hours=24),
            'batch': batch_id
        }
        
        files_data[file_id] = file_info
        file_infos.append((file_id, file_info))
    
    save_files(files_data)
    return file_infos, batch_id

def get_file(file_id):
    files = cleanup_expired()
    if file_id not in files:
        return None
    
    try:
        with open(os.path.join(STORAGE_DIR, f"{file_id}.dat"), 'rb') as f:
            return files[file_id], f.read()
    except:
        return None

def format_size(bytes):
    for unit in ['B', 'KB', 'MB']:
        if bytes < 1024:
            return f"{bytes:.1f} {unit}"
        bytes /= 1024
    return f"{bytes:.1f} GB"

query_params = st.query_params

if 'batch' in query_params or 'file' in query_params:
    st.title("📥 Download Files")
    
    if 'batch' in query_params:
        batch_id = query_params['batch']
        files = cleanup_expired()
        batch_files = {fid: info for fid, info in files.items() if info.get('batch') == batch_id}
        
        if batch_files:
            st.success(f"✅ Found {len(batch_files)} files in batch: {batch_id}")
            
            for file_id, file_info in batch_files.items():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"📄 **{file_info['name']}** ({format_size(file_info['size'])})")
                with col2:
                    file_data = get_file(file_id)
                    if file_data:
                        st.download_button("📥", file_data[1], file_info['name'], file_info['type'], key=file_id)
        else:
            st.error("❌ Batch not found or expired")
    
    elif 'file' in query_params:
        file_id = query_params['file']
        file_data = get_file(file_id)
        
        if file_data:
            file_info, data = file_data
            st.success(f"✅ File found: {file_info['name']}")
            st.download_button("📥 Download", data, file_info['name'], file_info['type'], type="primary")
        else:
            st.error("❌ File not found or expired")
    
    if st.button("🏠 Upload New Files"):
        st.query_params.clear()
        st.rerun()

else:
    st.title("📱 FliQR - Multi File Share")
    st.caption("Upload DOCX and TXT files • 24 hour availability")
    
    tab1, tab2 = st.tabs(["📤 Upload", "📋 Files"])
    
    with tab1:
        uploaded_files = st.file_uploader(
            "Select DOCX or TXT files:",
            type=['docx', 'txt'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            valid_files = [f for f in uploaded_files if f.type in [
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                'text/plain'
            ]]
            
            if valid_files:
                with st.spinner("Uploading files..."):
                    file_infos, batch_id = save_files_batch(valid_files)
                
                st.success(f"✅ Uploaded {len(file_infos)} files!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📄 Files")
                    for file_id, file_info in file_infos:
                        st.write(f"• {file_info['name']} ({format_size(file_info['size'])})")
                    
                    batch_url = f"https://koqndcvr7kfjrnroznzfws.streamlit.app/?batch={batch_id}"
                    st.code(batch_url)
                
                with col2:
                    st.subheader("📱 QR Code")
                    qr_img = create_qr(batch_url)
                    
                    img_buffer = io.BytesIO()
                    qr_img.save(img_buffer, format='PNG')
                    
                    st.image(img_buffer, width=250)
                    st.download_button("💾 Save QR", img_buffer.getvalue(), f"fliqr_{batch_id}.png", "image/png")
            else:
                st.error("❌ Please upload only DOCX or TXT files")
    
    with tab2:
        st.subheader("📋 Active Files")
        
        files = cleanup_expired()
        
        if not files:
            st.info("📭 No files uploaded")
        else:
            batches = {}
            singles = {}
            
            for file_id, file_info in files.items():
                batch_id = file_info.get('batch')
                if batch_id:
                    if batch_id not in batches:
                        batches[batch_id] = []
                    batches[batch_id].append((file_id, file_info))
                else:
                    singles[file_id] = file_info
            
            for batch_id, batch_files in batches.items():
                with st.expander(f"📦 Batch {batch_id} ({len(batch_files)} files)"):
                    for file_id, file_info in batch_files:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"📄 {file_info['name']} ({format_size(file_info['size'])})")
                        with col2:
                            file_data = get_file(file_id)
                            if file_data:
                                st.download_button("📥", file_data[1], file_info['name'], key=f"dl_{file_id}")
                    
                    batch_url = f"https://koqndcvr7kfjrnroznzfws.streamlit.app/?batch={batch_id}"
                    st.code(batch_url)
            
            for file_id, file_info in singles.items():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"📄 {file_info['name']} ({format_size(file_info['size'])})")
                with col2:
                    file_data = get_file(file_id)
                    if file_data:
                        st.download_button("📥", file_data[1], file_info['name'], key=f"single_{file_id}")

if __name__ == "__main__":
    pass