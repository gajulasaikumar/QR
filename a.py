import streamlit as st
import qrcode
from PIL import Image
import io
import base64
import os
import hashlib
import time
from datetime import datetime, timedelta
import json
import urllib.parse

# Configure page
st.set_page_config(
    page_title="FliQR - Share Files with QR Codes",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling
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
    .upload-area {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f8f9ff;
        margin: 1rem 0;
    }
    .qr-container {
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .file-info {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    .download-link {
        background-color: #667eea;
        color: white;
        padding: 0.5rem 1rem;
        text-decoration: none;
        border-radius: 5px;
        display: inline-block;
        margin: 0.5rem;
    }
    .share-url {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 5px;
        padding: 10px;
        font-family: monospace;
        word-break: break-all;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = {}

if 'file_metadata' not in st.session_state:
    st.session_state.file_metadata = {}

def generate_file_id():
    """Generate unique file ID"""
    return hashlib.md5(str(time.time()).encode()).hexdigest()[:8]

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
    
    img = qr.make_image(fill_color="black", back_color="white")
    return img

def get_current_url():
    """Get the current Streamlit app URL"""
    # Update this with your deployed URL
    return "https://koqndcvr7kfjrnroznzfws.streamlit.app"

def create_share_url(file_id):
    """Create a shareable URL with the file ID as a query parameter"""
    base_url = get_current_url()
    return f"{base_url}/?file_id={file_id}"

def save_uploaded_file(uploaded_file):
    """Save uploaded file and return file info"""
    file_id = generate_file_id()
    
    # Store file data in session state (in production, you'd save to disk/cloud)
    file_data = uploaded_file.read()
    
    # Reset file pointer for later operations
    uploaded_file.seek(0)
    
    # Store file information
    file_info = {
        'id': file_id,
        'name': uploaded_file.name,
        'size': len(file_data),
        'type': uploaded_file.type,
        'upload_time': datetime.now(),
        'expiry_time': datetime.now() + timedelta(hours=24),
        'data': file_data
    }
    
    st.session_state.uploaded_files[file_id] = file_info
    st.session_state.file_metadata[file_id] = {
        'name': file_info['name'],
        'size': file_info['size'],
        'type': file_info['type'],
        'upload_time': file_info['upload_time'].isoformat(),
        'expiry_time': file_info['expiry_time'].isoformat()
    }
    
    return file_info

def format_file_size(size_bytes):
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.2f} {size_names[i]}"

def handle_file_download():
    """Handle file download from QR code scan"""
    query_params = st.query_params
    
    if 'file_id' in query_params:
        file_id = query_params['file_id']
        
        if file_id in st.session_state.uploaded_files:
            file_info = st.session_state.uploaded_files[file_id]
            
            # Check if file hasn't expired
            if datetime.now() < file_info['expiry_time']:
                st.markdown("### 📥 File Download")
                st.success(f"File found: **{file_info['name']}**")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown(f"""
                    **File Information:**
                    - **Name:** {file_info['name']}
                    - **Size:** {format_file_size(file_info['size'])}
                    - **Type:** {file_info['type'] or 'Unknown'}
                    - **Uploaded:** {file_info['upload_time'].strftime('%Y-%m-%d %H:%M')}
                    """)
                
                with col2:
                    st.download_button(
                        label="📥 Download File",
                        data=file_info['data'],
                        file_name=file_info['name'],
                        mime=file_info['type'],
                        use_container_width=True
                    )
                
                st.info("💡 This file will be automatically deleted after 24 hours for privacy.")
                return True
            else:
                st.error("⏰ This file has expired and is no longer available.")
                return True
        else:
            st.error("❌ File not found. It may have expired or the link is invalid.")
            return True
    
    return False

def main():
    # Check if this is a file download request first
    if handle_file_download():
        st.markdown("---")
        if st.button("🏠 Go to Home"):
            st.query_params.clear()
            st.rerun()
        return
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📱 FliQR</h1>
        <h3>Share Files with QR Codes</h3>
        <p>Upload files up to 50 MB. Files available for 24 hours.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload File", "📋 Manage Files", "ℹ️ About"])
    
    with tab1:
        st.markdown("## Upload Your File")
        
        # File upload area
        st.markdown('<div class="upload-area">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drag and drop your file here or click to browse",
            type=None,
            help="Maximum file size: 50 MB"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            # Check file size (50 MB limit)
            file_size = len(uploaded_file.getvalue())
            if file_size > 50 * 1024 * 1024:  # 50 MB
                st.error("File size exceeds 50 MB limit. Please choose a smaller file.")
                return
            
            # Save file and generate QR code
            with st.spinner("Processing file..."):
                file_info = save_uploaded_file(uploaded_file)
                
            # Create share URL
            share_url = create_share_url(file_info['id'])
            
            # Display file info
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown(f"""
                <div class="file-info">
                    <h4>📄 File Information</h4>
                    <p><strong>Name:</strong> {file_info['name']}</p>
                    <p><strong>Size:</strong> {format_file_size(file_info['size'])}</p>
                    <p><strong>Type:</strong> {file_info['type'] or 'Unknown'}</p>
                    <p><strong>File ID:</strong> {file_info['id']}</p>
                    <p><strong>Expires:</strong> {file_info['expiry_time'].strftime('%Y-%m-%d %H:%M')}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Share URL
                st.markdown("**Share URL:**")
                st.code(share_url)
                
                # Direct download for testing
                st.download_button(
                    label="📥 Direct Download (for testing)",
                    data=file_info['data'],
                    file_name=file_info['name'],
                    mime=file_info['type']
                )
            
            with col2:
                st.markdown("### 📱 QR Code")
                
                # Generate and display QR code with the share URL
                qr_img = create_qr_code(share_url)
                
                # Convert PIL image to bytes for display
                img_buffer = io.BytesIO()
                qr_img.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                
                st.image(img_buffer, width=300)
                
                # QR code download
                st.download_button(
                    label="💾 Download QR Code",
                    data=img_buffer.getvalue(),
                    file_name=f"qr_code_{file_info['id']}.png",
                    mime="image/png"
                )
                
            st.success("✅ File uploaded successfully! Share the QR code to let others download your file.")
            
            # Instructions
            st.markdown(f"""
            ### 📖 How to share:
            1. **Save the QR code** or take a screenshot
            2. **Send the QR code** to anyone you want to share the file with
            3. **They scan the QR code** with their phone camera or QR code reader
            4. **They'll be directed to this app** where they can download the file
            
            ### 🔗 Alternative sharing:
            - **Copy and send the share URL** above
            - **Share the File ID**: `{file_info['id']}` (others can enter this manually)
            """)
    
    with tab2:
        st.markdown("## 📋 Your Uploaded Files")
        
        # Manual file ID entry
        st.markdown("### 🔍 Access File by ID")
        manual_file_id = st.text_input("Enter File ID to download:", placeholder="e.g., abc12345")
        if st.button("🔍 Find File") and manual_file_id:
            st.query_params.file_id = manual_file_id
            st.rerun()
        
        st.markdown("---")
        
        if not st.session_state.uploaded_files:
            st.info("No files uploaded yet. Go to the Upload tab to share your first file!")
        else:
            # Display all uploaded files
            for file_id, file_info in list(st.session_state.uploaded_files.items()):
                expiry_time = file_info['expiry_time']
                time_remaining = expiry_time - datetime.now()
                
                if time_remaining.total_seconds() > 0:
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.markdown(f"""
                        **{file_info['name']}**  
                        Size: {format_file_size(file_info['size'])} | ID: `{file_id}`  
                        Expires in: {int(time_remaining.total_seconds() // 3600)}h {int((time_remaining.total_seconds() % 3600) // 60)}m
                        """)
                    
                    with col2:
                        share_url = create_share_url(file_id)
                        # Generate QR for this file
                        if st.button(f"📱 Show QR", key=f"qr_{file_id}"):
                            qr_img = create_qr_code(share_url, size=6)
                            
                            img_buffer = io.BytesIO()
                            qr_img.save(img_buffer, format='PNG')
                            img_buffer.seek(0)
                            
                            st.image(img_buffer, width=150)
                            st.code(share_url)
                    
                    with col3:
                        st.download_button(
                            label="📥 Download",
                            data=file_info['data'],
                            file_name=file_info['name'],
                            mime=file_info['type'],
                            key=f"dl_{file_id}"
                        )
                    
                    st.divider()
                else:
                    # Remove expired files
                    del st.session_state.uploaded_files[file_id]
                    if file_id in st.session_state.file_metadata:
                        del st.session_state.file_metadata[file_id]
    
    with tab3:
        st.markdown("""
        ## ℹ️ About FliQR
        
        **FliQR** is a simple and secure way to share files using QR codes. Perfect for:
        
        - 📱 **Quick transfers** between devices
        - 👥 **Sharing with friends** and family
        - 💼 **Professional file sharing** without complex links
        - 🔒 **Temporary sharing** with automatic expiry
        
        ### 🚀 Features
        
        - **50 MB file limit** for free users
        - **24-hour availability** - files auto-delete for privacy
        - **Any file type** supported
        - **Instant QR generation** for easy sharing
        - **No registration** required
        - **Mobile-friendly** QR codes
        
        ### 🔒 Privacy & Security
        
        - Files are temporarily stored and automatically deleted after 24 hours
        - No permanent storage of your data
        - QR codes contain secure, unique download links
        - Files are only accessible to those with the QR code or File ID
        
        ### 📱 How to Use QR Codes
        
        1. **iPhone:** Open Camera app and point at QR code
        2. **Android:** Use Google Lens or Camera app
        3. **Any device:** Use any QR code reader app
        
        ---
        
        Made with ❤️ using Streamlit
        """)

if __name__ == "__main__":
    main()