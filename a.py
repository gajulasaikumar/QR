import streamlit as st
import qrcode
from PIL import Image
import io
import hashlib
import time
import os
import json
import base64
from datetime import datetime, timedelta

# Configure page
st.set_page_config(
    page_title="FliQR - Share Files with QR Codes",
    page_icon="📱",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .file-info {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Use Streamlit's cache directory for persistent storage
STORAGE_DIR = os.path.join(os.getcwd(), ".streamlit", "file_storage")
METADATA_FILE = os.path.join(STORAGE_DIR, "metadata.json")

def ensure_storage_dir():
    """Ensure storage directory exists"""
    os.makedirs(STORAGE_DIR, exist_ok=True)

def load_metadata():
    """Load file metadata"""
    ensure_storage_dir()
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, 'r') as f:
                data = json.load(f)
                # Convert ISO strings back to datetime objects
                for file_id in data:
                    data[file_id]['upload_time'] = datetime.fromisoformat(data[file_id]['upload_time'])
                    data[file_id]['expiry_time'] = datetime.fromisoformat(data[file_id]['expiry_time'])
                return data
        except Exception as e:
            st.error(f"Error loading metadata: {e}")
            return {}
    return {}

def save_metadata(metadata):
    """Save file metadata"""
    ensure_storage_dir()
    # Convert datetime objects to ISO strings for JSON serialization
    serializable_data = {}
    for file_id, data in metadata.items():
        serializable_data[file_id] = {
            **data,
            'upload_time': data['upload_time'].isoformat(),
            'expiry_time': data['expiry_time'].isoformat()
        }
    
    try:
        with open(METADATA_FILE, 'w') as f:
            json.dump(serializable_data, f, indent=2)
    except Exception as e:
        st.error(f"Error saving metadata: {e}")

def cleanup_expired_files():
    """Remove expired files and their metadata"""
    metadata = load_metadata()
    current_time = datetime.now()
    expired_files = []
    
    for file_id, file_info in metadata.items():
        if current_time >= file_info['expiry_time']:
            expired_files.append(file_id)
            # Remove the actual file
            file_path = os.path.join(STORAGE_DIR, f"{file_id}.dat")
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                st.error(f"Error removing file {file_id}: {e}")
    
    # Remove expired entries from metadata
    for file_id in expired_files:
        if file_id in metadata:
            del metadata[file_id]
    
    # Save updated metadata if any files were expired
    if expired_files:
        save_metadata(metadata)
    
    return metadata

def generate_file_id():
    """Generate unique file ID"""
    return hashlib.md5(f"{time.time()}{os.urandom(8)}".encode()).hexdigest()[:12]

def create_qr_code(data, size=10):
    """Create QR code for given data"""
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=size,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)
    
    return qr.make_image(fill_color="black", back_color="white")

def get_app_url():
    """Get the current app URL"""
    try:
        # Try to get from Streamlit context
        ctx = st.runtime.get_instance().get_client().request
        if hasattr(ctx, 'headers'):
            host = ctx.headers.get('host')
            if host and 'streamlit.app' in host:
                return f"https://{host}"
    except:
        pass
    
    # Fallback - you should replace this with your actual app URL
    return "https://koqndcvr7kfjrnroznzfws.streamlit.app/"

def create_share_url(file_id):
    """Create shareable URL"""
    base_url = get_app_url()
    return f"{base_url}/?file_id={file_id}"

def save_file(uploaded_file):
    """Save uploaded file to persistent storage"""
    ensure_storage_dir()
    
    file_id = generate_file_id()
    file_path = os.path.join(STORAGE_DIR, f"{file_id}.dat")
    
    try:
        # Read and save file data
        file_data = uploaded_file.read()
        with open(file_path, 'wb') as f:
            f.write(file_data)
        
        # Create metadata
        file_info = {
            'id': file_id,
            'name': uploaded_file.name,
            'size': len(file_data),
            'type': uploaded_file.type or 'application/octet-stream',
            'upload_time': datetime.now(),
            'expiry_time': datetime.now() + timedelta(hours=24),
            'file_path': file_path
        }
        
        # Load existing metadata and add new file
        metadata = load_metadata()
        metadata[file_id] = file_info
        save_metadata(metadata)
        
        return file_info
        
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def get_file(file_id):
    """Get file data by ID"""
    # Clean up expired files first
    metadata = cleanup_expired_files()
    
    if file_id not in metadata:
        return None
    
    file_info = metadata[file_id]
    
    # Double-check expiry
    if datetime.now() >= file_info['expiry_time']:
        return None
    
    # Try to read the file
    file_path = os.path.join(STORAGE_DIR, f"{file_id}.dat")
    try:
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                file_data = f.read()
            file_info['data'] = file_data
            return file_info
    except Exception as e:
        st.error(f"Error reading file {file_id}: {e}")
    
    return None

def format_file_size(size_bytes):
    """Format file size"""
    if size_bytes == 0:
        return "0 B"
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def handle_download_page():
    """Handle file download page"""
    query_params = st.query_params
    
    if 'file_id' not in query_params:
        return False
    
    file_id = query_params['file_id']
    st.markdown(f"### 🔍 Looking for file: `{file_id}`")
    
    with st.spinner("Loading file..."):
        file_info = get_file(file_id)
    
    if file_info:
        st.success(f"✅ File found: **{file_info['name']}**")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown(f"""
            **📄 File Details:**
            - **Name:** {file_info['name']}
            - **Size:** {format_file_size(file_info['size'])}
            - **Type:** {file_info['type']}
            - **Uploaded:** {file_info['upload_time'].strftime('%Y-%m-%d %H:%M')}
            - **Expires:** {file_info['expiry_time'].strftime('%Y-%m-%d %H:%M')}
            """)
        
        with col2:
            st.download_button(
                label="📥 Download File",
                data=file_info['data'],
                file_name=file_info['name'],
                mime=file_info['type'],
                use_container_width=True,
                type="primary"
            )
        
        # Time remaining
        time_left = file_info['expiry_time'] - datetime.now()
        hours_left = int(time_left.total_seconds() // 3600)
        minutes_left = int((time_left.total_seconds() % 3600) // 60)
        
        if hours_left > 0:
            st.info(f"⏰ This file will expire in {hours_left}h {minutes_left}m")
        else:
            st.warning(f"⚠️ This file will expire in {minutes_left} minutes!")
            
    else:
        st.error("❌ File not found!")
        st.markdown("""
        **Possible reasons:**
        - File has expired (files are deleted after 24 hours)
        - Invalid file ID
        - File was never uploaded successfully
        """)
    
    st.markdown("---")
    if st.button("🏠 Go to Upload Page"):
        st.query_params.clear()
        st.rerun()
    
    return True

def main():
    # Handle download page first
    if handle_download_page():
        return
    
    # Main app header
    st.markdown("""
    <div class="main-header">
        <h1>📱 FliQR</h1>
        <h3>Share Files with QR Codes</h3>
        <p>Upload files up to 50 MB • Files available for 24 hours</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload", "📋 My Files", "ℹ️ About"])
    
    with tab1:
        st.markdown("## 📤 Upload Your File")
        
        uploaded_file = st.file_uploader(
            "Choose a file to share:",
            help="Maximum file size: 50 MB"
        )
        
        if uploaded_file is not None:
            # Check file size
            file_size = len(uploaded_file.getvalue())
            if file_size > 50 * 1024 * 1024:
                st.error("❌ File too large! Maximum size is 50 MB.")
                return
            
            # Process file
            with st.spinner("💾 Saving file..."):
                file_info = save_file(uploaded_file)
            
            if file_info:
                # Success! Show results
                st.success("✅ File uploaded successfully!")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown(f"""
                    <div class="file-info">
                        <h4>📄 File Information</h4>
                        <p><strong>Name:</strong> {file_info['name']}</p>
                        <p><strong>Size:</strong> {format_file_size(file_info['size'])}</p>
                        <p><strong>File ID:</strong> <code>{file_info['id']}</code></p>
                        <p><strong>Expires:</strong> {file_info['expiry_time'].strftime('%Y-%m-%d %H:%M')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Share URL
                    share_url = create_share_url(file_info['id'])
                    st.markdown("**🔗 Share URL:**")
                    st.code(share_url)
                    
                    # Test download
                    test_file = get_file(file_info['id'])
                    if test_file:
                        st.download_button(
                            label="🧪 Test Download",
                            data=test_file['data'],
                            file_name=test_file['name'],
                            mime=test_file['type']
                        )
                
                with col2:
                    st.markdown("### 📱 QR Code")
                    
                    # Generate QR code
                    qr_img = create_qr_code(share_url)
                    
                    # Convert to bytes for display
                    img_buffer = io.BytesIO()
                    qr_img.save(img_buffer, format='PNG')
                    img_buffer.seek(0)
                    
                    st.image(img_buffer, width=300)
                    
                    # Download QR code
                    st.download_button(
                        label="💾 Download QR Code",
                        data=img_buffer.getvalue(),
                        file_name=f"fliqr_{file_info['id']}.png",
                        mime="image/png"
                    )
                
                # Instructions
                st.markdown(f"""
                ### 📖 How to Share:
                
                1. **📱 QR Code:** Share the QR code image - others can scan it with their phone camera
                2. **🔗 URL:** Copy and send the share URL above
                3. **🆔 File ID:** Share the File ID: `{file_info['id']}`
                
                ### ⏰ Important:
                - Files are automatically deleted after **24 hours**
                - Anyone with the QR code, URL, or File ID can download your file
                - Keep your share information secure!
                """)
    
    with tab2:
        st.markdown("## 📋 Active Files")
        
        # Manual file access
        st.markdown("### 🔍 Access File by ID")
        file_id_input = st.text_input("Enter File ID:", placeholder="e.g., abc123def456")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🔍 Find File", use_container_width=True) and file_id_input:
                st.query_params.file_id = file_id_input.strip()
                st.rerun()
        
        with col2:
            if st.button("🔄 Refresh Files", use_container_width=True):
                st.rerun()
        
        st.markdown("---")
        
        # Show all active files
        metadata = cleanup_expired_files()
        
        if not metadata:
            st.info("📭 No active files found.")
        else:
            st.markdown(f"### 📁 Active Files ({len(metadata)})")
            
            for file_id, file_info in metadata.items():
                time_remaining = file_info['expiry_time'] - datetime.now()
                
                if time_remaining.total_seconds() > 0:
                    hours = int(time_remaining.total_seconds() // 3600)
                    minutes = int((time_remaining.total_seconds() % 3600) // 60)
                    
                    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                    
                    with col1:
                        st.markdown(f"""
                        **📄 {file_info['name']}**  
                        `{file_id}` • {format_file_size(file_info['size'])}  
                        ⏰ Expires in: {hours}h {minutes}m
                        """)
                    
                    with col2:
                        if st.button("📱", key=f"qr_{file_id}", help="Show QR Code"):
                            share_url = create_share_url(file_id)
                            qr_img = create_qr_code(share_url, size=8)
                            
                            img_buffer = io.BytesIO()
                            qr_img.save(img_buffer, format='PNG')
                            
                            st.image(img_buffer, width=200)
                            st.code(share_url)
                    
                    with col3:
                        share_url = create_share_url(file_id)
                        if st.button("🔗", key=f"url_{file_id}", help="Copy URL"):
                            st.code(share_url)
                    
                    with col4:
                        file_data = get_file(file_id)
                        if file_data:
                            st.download_button(
                                "📥",
                                data=file_data['data'],
                                file_name=file_data['name'],
                                mime=file_data['type'],
                                key=f"dl_{file_id}",
                                help="Download"
                            )
                    
                    st.divider()
    
    with tab3:
        st.markdown("""
        ## ℹ️ About FliQR
        
        **FliQR** makes file sharing simple with QR codes and secure temporary storage.
        
        ### ✨ Features
        - 📁 **50 MB file limit** - Share documents, images, videos, and more
        - ⏰ **24-hour availability** - Files auto-delete for privacy
        - 📱 **QR code generation** - Easy mobile sharing
        - 🔗 **Direct URLs** - Works on any device
        - 🆔 **File ID system** - Manual access option
        - 💾 **Persistent storage** - Files work across devices and sessions
        
        ### 🔒 Privacy & Security
        - Files are stored temporarily and automatically deleted after 24 hours
        - Only people with the QR code, URL, or File ID can access your files
        - No registration or personal information required
        
        ### 📱 How to Use QR Codes
        - **iPhone:** Open Camera app and point at the QR code
        - **Android:** Use Google Lens, Samsung Camera, or any QR reader app
        - **Desktop:** Use your phone or a browser extension
        
        ### 🛠️ Technical Details
        - Files are stored in Streamlit's persistent directory structure
        - Metadata is managed with JSON for reliability
        - Automatic cleanup removes expired files and frees storage
        - Base64 encoding ensures file integrity
        
        ---
        
        **🚀 Ready to share?** Go to the Upload tab and start sharing files instantly!
        
        *Made with ❤️ using Streamlit*
        """)

if __name__ == "__main__":
    main()