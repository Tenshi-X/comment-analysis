import streamlit as st
import pandas as pd
import os
import hashlib
from config import Config
from utils.file_utils import allowed_file
import tempfile

# Set page config
st.set_page_config(page_title="TikTok Comments Analyzer", page_icon="📊", layout="wide")

# Initialize session state
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

# Initialize session state for BERTopic
if 'bertopic_cache' not in st.session_state:
    st.session_state.bertopic_cache = {}  # Dictionary to cache models per file hash

def get_file_hash(file_path):
    """Calculate SHA256 hash of file content to identify unique files."""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

def main():
    st.title("📊 TikTok Comments Analyzer")

    # Check if file is uploaded
    if st.session_state.uploaded_file and os.path.exists(st.session_state.uploaded_file):
        # Sidebar navigation (without Upload option)
        page = st.sidebar.selectbox(
            "Pilih Menu",
            ["Home", "Komentar Asli", "Komentar Preprocessing", "Analisis"]
        )

        if page == "Home":
            home_page()
        elif page == "Komentar Asli":
            comments_raw_page()
        elif page == "Komentar Preprocessing":
            comments_preprocessed_page()
        elif page == "Analisis":
            analysis_page()
    else:
        # No file uploaded, show only upload page without sidebar
        upload_page()

def upload_page():
    st.header("Upload CSV File")
    uploaded_file = st.file_uploader("Pilih file CSV komentar TikTok", type=['csv'])

    if uploaded_file is not None:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            st.session_state.uploaded_file = tmp_file.name

        st.success("File berhasil diupload!")
        st.info("Sedang memuat halaman Home...")
        st.rerun()

def home_page():
    st.header("Dashboard")

    if st.session_state.uploaded_file and os.path.exists(st.session_state.uploaded_file):
        st.success("File sudah diupload dan siap untuk analisis.")

        # Option to upload new file
        st.subheader("Upload File Baru (Opsional)")
        new_file = st.file_uploader("Pilih file CSV baru", type=['csv'], key="new_upload")
        if new_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                tmp_file.write(new_file.getvalue())
                st.session_state.uploaded_file = tmp_file.name

            st.success("File berhasil diganti!")
            st.info("Model analisis akan dimuat ulang untuk file baru.")
            # Clear caches
            st.session_state.bertopic_cache = {}
    else:
        st.warning("Silakan upload file CSV terlebih dahulu melalui menu Upload.")

def comments_raw_page():
    st.header("💬 Komentar Asli")

    if not st.session_state.uploaded_file or not os.path.exists(st.session_state.uploaded_file):
        st.warning("Silakan upload file CSV terlebih dahulu.")
        return

    try:
        df = pd.read_csv(st.session_state.uploaded_file, encoding='utf-8-sig')
    except Exception as e:
        st.error(f'Gagal membaca file CSV: {e}')
        return

    # Normalisasi nama kolom
    columns_map = {c.strip().lower(): c for c in df.columns}
    if 'text' not in columns_map or 'createtimeiso' not in columns_map:
        st.error(f'File CSV harus memiliki kolom "text" dan "createTimeISO". Kolom ditemukan: {df.columns.tolist()}')
        return

    text_col = columns_map['text']
    date_col = columns_map['createtimeiso']

    comments = [
        {'text': str(row[text_col]), 'date': str(row[date_col])}
        for _, row in df[[text_col, date_col]].dropna(subset=[text_col]).iterrows()
    ]

    # Statistics Section
    st.subheader("📊 Statistik Komentar")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Komentar", len(comments))

    with col2:
        avg_length = sum(len(c['text']) for c in comments) / len(comments) if comments else 0
        st.metric("Rata-rata Panjang", f"{avg_length:.1f} karakter")

    with col3:
        max_length = max(len(c['text']) for c in comments) if comments else 0
        st.metric("Komentar Terpanjang", f"{max_length} karakter")

    with col4:
        min_length = min(len(c['text']) for c in comments) if comments else 0
        st.metric("Komentar Terpendek", f"{min_length} karakter")

    # Search and Filter Section
    st.subheader("🔍 Cari Kata Kunci")
    col_search, col_limit = st.columns([2, 1])

    with col_search:
        search_term = st.text_input("Cari komentar:", placeholder="Ketik kata kunci...")

    with col_limit:
        display_limit = st.selectbox(
            "Jumlah komentar:",
            [10, 25, 50, 100, len(comments)],
            index=2,
            format_func=lambda x: f"{x} komentar" if x != len(comments) else "Semua komentar"
        )

    # Filter comments based on search
    if search_term:
        filtered_comments = [
            comment for comment in comments
            if search_term.lower() in comment['text'].lower()
        ]
        st.info(f"Ditemukan {len(filtered_comments)} komentar yang mengandung '{search_term}'")
    else:
        filtered_comments = comments

    # Display comments
    st.subheader("📝 Daftar Komentar")

    if not filtered_comments:
        st.warning("Tidak ada komentar yang sesuai dengan kriteria pencarian.")
        return

    # Show only selected limit
    comments_to_show = filtered_comments[:display_limit]

    for i, comment in enumerate(comments_to_show, 1):
        # Create a container for each comment
        with st.container():
            # Date header
            st.markdown(f"**📅 {comment['date'][:10]}** - *{comment['date']}*")

            # Comment text in a styled box
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin: 5px 0;">
                {comment['text']}
            </div>
            """, unsafe_allow_html=True)

            # Comment length info
            st.caption(f"📏 Panjang: {len(comment['text'])} karakter")

            # Separator
            st.markdown("---")

    # Pagination info
    if len(filtered_comments) > display_limit:
        st.info(f"Menampilkan {display_limit} dari {len(filtered_comments)} komentar yang sesuai.")
    else:
        st.success(f"Menampilkan semua {len(filtered_comments)} komentar yang sesuai.")

def comments_preprocessed_page():
    # Lazy import to speed up initial load
    from services.preprocessing import get_preprocessing_steps
    
    st.header("🔧 Komentar Setelah Preprocessing")

    if not st.session_state.uploaded_file or not os.path.exists(st.session_state.uploaded_file):
        st.warning("Silakan upload file CSV terlebih dahulu.")
        return

    try:
        df = pd.read_csv(st.session_state.uploaded_file, encoding='utf-8-sig')

        # Get preprocessing steps
        preprocessing_results = get_preprocessing_steps(df)

        original_comments = df['text'].dropna().tolist()
        preprocessed_comments = preprocessing_results['hasil_preprocessing']

        # Statistics Comparison Section
        st.subheader("📊 Perbandingan Statistik")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Komentar Asli", len(original_comments))
            st.metric("Total Setelah Preprocessing", len(preprocessed_comments), delta=len(preprocessed_comments)-len(original_comments))

        with col2:
            avg_orig = sum(len(c) for c in original_comments) / len(original_comments) if original_comments else 0
            avg_prep = sum(len(c) for c in preprocessed_comments) / len(preprocessed_comments) if preprocessed_comments else 0
            st.metric("Rata-rata Panjang Asli", f"{avg_orig:.1f} karakter")
            st.metric("Rata-rata Panjang Preprocessing", f"{avg_prep:.1f} karakter", delta=f"{avg_prep-avg_orig:.1f}")

        with col3:
            max_orig = max(len(c) for c in original_comments) if original_comments else 0
            max_prep = max(len(c) for c in preprocessed_comments) if preprocessed_comments else 0
            st.metric("Komentar Terpanjang Asli", f"{max_orig} karakter")
            st.metric("Komentar Terpanjang Preprocessing", f"{max_prep} karakter", delta=max_prep-max_orig)

        with col4:
            min_orig = min(len(c) for c in original_comments) if original_comments else 0
            min_prep = min(len(c) for c in preprocessed_comments) if preprocessed_comments else 0
            st.metric("Komentar Terpendek Asli", f"{min_orig} karakter")
            st.metric("Komentar Terpendek Preprocessing", f"{min_prep} karakter", delta=min_prep-min_orig)

        # Comparison Table Section
        st.subheader("📋 Perbandingan Komentar")

        # Create comparison table
        comparison_data = []
        for original, processed in zip(original_comments, preprocessed_comments):
            # Clean the processed comment to remove unwanted text and symbols
            cleaned_processed = str(processed).strip('{}').replace('"text":', '').strip('"').strip()
            comparison_data.append({
                "Komentar Asli": original,
                "Komentar Setelah Preprocessing": cleaned_processed
            })

        # Display as dataframe with custom styling
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(
            df_comparison,
            column_config={
                "Komentar Asli": st.column_config.TextColumn("Komentar Asli", width="large"),
                "Komentar Setelah Preprocessing": st.column_config.TextColumn("Komentar Setelah Preprocessing", width="large")
            },
            hide_index=True,
            use_container_width=True
        )

        st.markdown("""
        <style>
        .dataframe { font-size: 18px !important; }
        .dataframe th { font-size: 20px !important; font-weight: bold !important; }
        .dataframe td { font-size: 18px !important; }
        </style>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f'Error dalam preprocessing: {str(e)}')

def analysis_page():
    st.title("🧩 Analisis Topik")
    
    if not st.session_state.uploaded_file or not os.path.exists(st.session_state.uploaded_file):
        st.warning("Silakan upload file CSV terlebih dahulu.")
        return

    tab1, tab2 = st.tabs(["BERTopic + Sentence Transformer", "BERTopic + USE"])

    with tab1:
        bertopic_page(embedding_type="indobert")

    with tab2:
        bertopic_page(embedding_type="use")

def bertopic_page(embedding_type="indobert"):
    # Determine model name and import appropriate builder
    if embedding_type == "indobert":
        from services.bertopic_indobert_service import build_bertopic_indobert as build_model
        model_name = "Sentence Transformer (IndoBERT)"
    else:
        from services.bertopic_use_service import build_bertopic_use as build_model
        model_name = "Universal Sentence Encoder (USE)"
    
    st.header(f"Metode BERTopic (+ {model_name})")

    try:
        # Calculate file hash to identify unique files (plus embedding type)
        # get_file_hash is defined at module level in app.py
        file_hash = get_file_hash(st.session_state.uploaded_file)
        cache_key = f"{file_hash}_{embedding_type}"

        # Check if model is already cached for this file
        if cache_key in st.session_state.bertopic_cache:
            data = st.session_state.bertopic_cache[cache_key]
            st.success(f"Model BERTopic ({model_name}) dimuat dari cache!")
        else:
            # Build specific model
            with st.spinner(f"Membangun model BERTopic dengan {model_name}... Ini mungkin memakan waktu beberapa menit."):
                data = build_model(st.session_state.uploaded_file)
            st.session_state.bertopic_cache[cache_key] = data
            st.success(f"Model BERTopic ({model_name}) berhasil dibangun!")

        # Display results
        if 'error' in data:
            st.error(data['error'])
            return

        # 1. Coherence Table (As requested by user: Topik | Daftar Kata | Coherence Score cv)
        if 'topics_details' in data:
             st.subheader("📋 Detail Topik & Coherence Score")
             
             # Format for display: Topik, Daftar Kata (underscored), Coherence
             table_data = []
             for topic in data['topics_details']:
                 # Join words with underscores as shown in user image
                 words_str = "_".join(topic['keywords'])
                 table_data.append({
                     "Topik": topic['topic_id'],
                     "Daftar Kata": words_str,
                     "Coherence Score cv": f"{topic['coherence']:.3f}"
                 })
             
             df_table = pd.DataFrame(table_data)
             st.dataframe(
                 df_table,
                 column_config={
                    "Topik": st.column_config.NumberColumn("Topik", width="small"),
                    "Daftar Kata": st.column_config.TextColumn("Daftar Kata", width="large"),
                    "Coherence Score cv": st.column_config.TextColumn("Coherence Score cv", width="medium"), 
                 },
                 hide_index=True,
                 use_container_width=True
             )

        # 2. Visualizations
        if 'visualizations' in data:
            st.subheader("📈 Visualisasi")
            
            # Barchart
            if data['visualizations'].get('barchart'):
                st.markdown("**Barchart - Kata-Kata Utama per Topik**")
                st.components.v1.html(data['visualizations']['barchart'], height=400, scrolling=True)
            
            # 2D Plot
            if data['visualizations'].get('topics'):
                 st.markdown("**Visualisasi Topik**")
                 st.components.v1.html(data['visualizations']['topics'], height=800)

    except Exception as e:
        import traceback
        st.error(f'Error dalam analisis BERTopic: {str(e)}')
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
