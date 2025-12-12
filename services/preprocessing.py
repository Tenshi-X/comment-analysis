import re
import string
import nltk
import pandas as pd
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import os
import pickle

# Download resource NLTK yang diperlukan
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

factory = StemmerFactory()
stemmer = factory.create_stemmer()
stopwords_id = set(stopwords.words('indonesian'))

# Cache untuk stemming dengan persistensi
STEM_CACHE_FILE = 'models/stem_cache.pkl'
stem_cache = {}
cache_lock = threading.Lock()

def load_stem_cache():
    """Memuat cache stemming dari file jika ada"""
    global stem_cache
    try:
        if os.path.exists(STEM_CACHE_FILE):
            with open(STEM_CACHE_FILE, 'rb') as f:
                stem_cache = pickle.load(f)
            print(f"Cache stemming dimuat: {len(stem_cache)} kata")
    except Exception as e:
        print(f"Gagal memuat cache stemming: {e}")
        stem_cache = {}

def save_stem_cache():
    """Menyimpan cache stemming ke file"""
    try:
        os.makedirs(os.path.dirname(STEM_CACHE_FILE), exist_ok=True)
        with open(STEM_CACHE_FILE, 'wb') as f:
            pickle.dump(stem_cache, f)
        print(f"Cache stemming disimpan: {len(stem_cache)} kata")
    except Exception as e:
        print(f"Gagal menyimpan cache stemming: {e}")

# Muat cache saat modul diimpor
load_stem_cache()

# Tentukan jumlah worker berdasarkan CPU cores
MAX_WORKERS = min(multiprocessing.cpu_count(), 4)  # Max 4 workers untuk menghindari overload

def get_preprocessing_steps(df):
    """Fungsi untuk melakukan semua tahap preprocessing"""
    
    # 1. Pembersihan Data Awal
    results = {
        'data_awal': {
            'jumlah_data': len(df),
            'jumlah_duplikat': df.duplicated().sum(),
            'data_kosong': df.isnull().sum().to_dict()
        }
    }
    
    # 2. Hapus Duplikat
    df_clean = df.drop_duplicates()
    results['setelah_hapus_duplikat'] = {
        'jumlah_data': len(df_clean)
    }
    
    # 3. Hapus Data Kosong
    df_clean = df_clean.dropna(subset=['text'])
    results['setelah_hapus_kosong'] = {
        'jumlah_data': len(df_clean)
    }
    
    # 4. Preprocessing Text
    def clean_text(text):
        if not isinstance(text, str):
            return ""
        # Lowercase
        text = text.lower()
        # Hapus tanda baca
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Hapus karakter khusus
        text = re.sub(r'[^a-z0-9\s]', '', text)
        # Tokenisasi sederhana
        tokens = text.split()
        # Hapus stopwords
        tokens = [t for t in tokens if t not in stopwords_id]
        # Stemming dengan cache
        stemmed_tokens = []
        for t in tokens:
            with cache_lock:
                if t in stem_cache:
                    stemmed_tokens.append(stem_cache[t])
                else:
                    stemmed = stemmer.stem(t)
                    stem_cache[t] = stemmed
                    stemmed_tokens.append(stemmed)
        return ' '.join(stemmed_tokens)
    
    # Terapkan preprocessing ke semua text dengan parallel processing
    print(f"Memulai preprocessing {len(df_clean)} komentar dengan {MAX_WORKERS} workers...")
    start_time = time.time()

    # Gunakan parallel processing untuk text preprocessing
    texts = df_clean['text'].tolist()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_text = {executor.submit(clean_text, text): text for text in texts}

        # Collect results in order - maintain original order
        cleaned_texts = [None] * len(texts)
        for future in as_completed(future_to_text):
            try:
                original_text = future_to_text[future]
                index = texts.index(original_text)
                cleaned_texts[index] = future.result()
            except Exception as exc:
                print(f'Generated an exception: {exc}')
                original_text = future_to_text[future]
                index = texts.index(original_text)
                cleaned_texts[index] = ""  # Fallback untuk error

    df_clean['text_clean'] = cleaned_texts

    end_time = time.time()
    print(f"Preprocessing selesai dalam {end_time - start_time:.2f} detik")

    # Simpan cache stemming setelah preprocessing
    save_stem_cache()

    # Simpan hasil
    results['hasil_preprocessing'] = df_clean[['text', 'text_clean']].to_dict('records')

    return results
