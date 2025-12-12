import os
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from services.preprocessing import get_preprocessing_steps

def load_word2vec_model(model_path='models/word2vec.model'):
    """Load Word2Vec model"""
    if os.path.exists(model_path):
        return Word2Vec.load(model_path)
    return None

def get_word2vec_info(model):
    """Get Word2Vec model information"""
    if model is None:
        return None

    return {
        'vocab_size': len(model.wv),
        'vector_size': model.vector_size,
        'window_size': model.window,
        'min_count': model.min_count,
        'workers': model.workers
    }

def get_sample_embeddings(model, n_samples=10):
    """Get sample word embeddings"""
    if model is None:
        return []

    words = list(model.wv.index_to_key)[:n_samples]
    embeddings = []

    for word in words:
        vector = model.wv[word][:5]  # Show first 5 dimensions
        embeddings.append({
            'word': word,
            'vector': vector.tolist()
        })

    return embeddings

def get_similar_words(model, word, topn=5):
    """Get similar words for a given word"""
    if model is None or word not in model.wv:
        return []

    similar_words = model.wv.most_similar(word, topn=topn)
    return [{'word': w, 'similarity': float(s)} for w, s in similar_words]

def create_2d_visualization(model, words=None, n_words=50):
    """Create 2D visualization using PCA"""
    if model is None:
        return None

    if words is None:
        words = list(model.wv.index_to_key)[:n_words]

    # Filter words that exist in vocabulary
    words = [w for w in words if w in model.wv]

    if len(words) < 2:
        return None

    # Get word vectors
    vectors = np.array([model.wv[word] for word in words])

    # Apply PCA
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)

    # Create DataFrame for plotting
    df = pd.DataFrame({
        'word': words,
        'x': vectors_2d[:, 0],
        'y': vectors_2d[:, 1]
    })

    # Create interactive plot
    fig = px.scatter(
        df,
        x='x',
        y='y',
        text='word',
        title='Word2Vec Embeddings (PCA 2D)',
        labels={'x': 'PC1', 'y': 'PC2'}
    )

    fig.update_traces(
        textposition='top center',
        marker=dict(size=8, opacity=0.7)
    )

    fig.update_layout(
        height=600,
        showlegend=False
    )

    return fig.to_html(full_html=False)

def get_word2vec_analysis(filepath=None):
    """Main function to get Word2Vec analysis data"""
    # If no filepath provided, try to load existing model
    if filepath is None:
        model = load_word2vec_model()
        if model is None:
            return {
                'error': 'Model Word2Vec tidak ditemukan. Pastikan file models/word2vec.model ada.'
            }
    else:
        # Train new model from uploaded data
        try:
            df = pd.read_csv(filepath, encoding='utf-8-sig')
            preprocessing_results = get_preprocessing_steps(df)
            text_data = [item['text_clean'].split() for item in preprocessing_results['hasil_preprocessing'] if item['text_clean']]

            # Train Word2Vec model
            model = Word2Vec(sentences=text_data, vector_size=100, window=5, min_count=1, workers=4)
            model.save('models/word2vec.model')
        except Exception as e:
            return {
                'error': f'Gagal melatih model Word2Vec: {str(e)}'
            }

    info = get_word2vec_info(model)
    embeddings = get_sample_embeddings(model)
    visualization = create_2d_visualization(model)

    # Get similar words for some common words
    sample_words = ['wisata', 'liburan', 'jalan', 'kuliner', 'indonesia']
    similar_words = {}
    for word in sample_words:
        if word in model.wv:
            similar_words[word] = get_similar_words(model, word)

    return {
        'model_info': info,
        'sample_embeddings': embeddings,
        'visualization': visualization,
        'similar_words': similar_words
    }
