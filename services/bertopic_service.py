import os
import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import plotly.express as px
import joblib
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer, models
import hashlib
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try importing tensorflow dependencies for USE
try:
    import tensorflow as tf
    import tensorflow_hub as hub
    import tensorflow_text  # Required for some TF Hub models
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow/TF Hub/TF Text not installed. Universal Sentence Encoder (Original) may fail.")

# --- Custom wrapper for USE as per reference file ---
class UniversalSentenceEncoderModel(SentenceTransformer):
    def __init__(self, tfhub_module_callable, model_name="UniversalSentenceEncoder"):
        # Initialize base SentenceTransformer without model_name_or_path first
        # We manually handle the encoder
        super().__init__('sentence-transformers/all-MiniLM-L6-v2') # Dummy initialization to get base structure
        # But we override the encode logic
        self.tfhub_module = tfhub_module_callable
        self.model_name = model_name
        
        # Create a Keras model that uses the tfhub_module_callable wrapped in Lambda.
        if TF_AVAILABLE:
            try:
                text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
                embeddings_output = tf.keras.layers.Lambda(lambda x: self.tfhub_module(x), name=self.model_name)(text_input)
                self.keras_encoder = tf.keras.Model(inputs=text_input, outputs=embeddings_output)
            except Exception as e:
                logger.error(f"Failed to create Keras model for USE: {e}")
        
        # Override module definition to avoid saving errors or confusion
        self._modules = {} 

    def encode(self, sentences, batch_size=32, show_progress_bar=False, convert_to_numpy=True):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is not available. Cannot use Universal Sentence Encoder (TF Hub).")

        if isinstance(sentences, str):
            sentences = [sentences]
            
        # Use the internal Keras model to get embeddings
        # predict returns numpy array by default in explicit calls usually, but let's be safe
        embeddings = self.keras_encoder.predict(sentences, batch_size=batch_size, verbose=0)
        
        if convert_to_numpy:
            return embeddings
        return tf.convert_to_tensor(embeddings)

def load_bertopic_model(filepath=None, model_path=None):
    """Load topic model for specific CSV file"""
    if filepath and not model_path:
        # Generate unique model path based on file hash
        file_hash = hashlib.md5(filepath.encode()).hexdigest()[:8]
        model_path = f'models/bertopic_model_{file_hash}.pkl'

    if model_path and os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except:
            pass
    return None

def build_bertopic_model(filepath=None, embedding_type="indobert"):
    """Build topic model using BERTopic from uploaded CSV data"""
    if filepath is None:
        return {
            'error': 'Filepath tidak diberikan. Pastikan file CSV sudah diupload.'
        }

    try:
        # Load and preprocess data
        df = pd.read_csv(filepath, encoding='utf-8-sig')
        from services.preprocessing import get_preprocessing_steps
        preprocessing_results = get_preprocessing_steps(df)
        
        # Original text and clean text
        text_data = [item['text_clean'] for item in preprocessing_results['hasil_preprocessing'] if item['text_clean']]

        if not text_data:
            return {
                'error': 'Tidak ada data teks yang valid setelah preprocessing.'
            }

        # Setup Indonesian stopwords safe load
        try:
            indonesian_stopwords = stopwords.words('indonesian')
        except:
            indonesian_stopwords = []

        # Vectorizer
        vectorizer_model = CountVectorizer(
            stop_words=indonesian_stopwords,
            ngram_range=(1, 2),
            min_df=5 
        )
        
        # Tokenized data for Coherence Calculation (Use analyzer to match topic n-grams)
        analyzer = vectorizer_model.build_analyzer()
        tokenized_data = [analyzer(text) for text in text_data]

        # Select Embedding Model
        if embedding_type == "indobert":
            # --- Method 1: IndoBERT from Reference (Explicit Construction) ---
            print("Loading IndoBERT model (Explicit Construction)...")
            try:
                # 1. Load the pre-trained IndoBERT model from Hugging Face's transformers
                word_embedding_model = models.Transformer('indolem/indobert-base-uncased')
                # 2. Add a pooling layer
                pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
                # 3. Combine
                embedding_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
            except Exception as e:
                print(f"Failed to load indolem/indobert-base-uncased: {e}. Fallback to firqaaa/indo-sentence-bert-base.")
                embedding_model = SentenceTransformer("firqaaa/indo-sentence-bert-base")
                
        else:
            # --- Method 2: USE from Reference (TF Hub Wrapper) ---
            print("Loading Universal Sentence Encoder (TF Hub)...")
            if TF_AVAILABLE:
                try:
                    use_model_url = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
                    tfhub_module_callable = hub.load(use_model_url)
                    embedding_model = UniversalSentenceEncoderModel(tfhub_module_callable)
                except Exception as e:
                    print(f"Failed to load USE from TF Hub: {e}. Fallback to distiluse.")
                    embedding_model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')
            else:
                 print("TensorFlow not available. Fallback to distiluse.")
                 embedding_model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')

        # Create BERTopic model
        # Reference: min_topic_size=30, but reduced to 10 to allow more topics for smaller datasets
        topic_model = BERTopic(
            embedding_model=embedding_model,
            vectorizer_model=vectorizer_model,
            representation_model=KeyBERTInspired(),
            ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
            calculate_probabilities=True,
            verbose=True,
            min_topic_size=10 
        )

        # Fit the model
        topics, probs = topic_model.fit_transform(text_data)

        # Get topic information
        topic_info = topic_model.get_topic_info()
        valid_topics = topic_info[topic_info['Topic'] != -1]

        # --- Coherence Calculation (C_V) per Topic ---
        
        # 1. Get Topics list (list of list of words)
        topics_list = []
        topic_ids = valid_topics['Topic'].tolist()
        
        for tid in topic_ids:
            # Get only words (no scores)
            words_data = topic_model.get_topic(tid)
            if words_data:
                words = [word for word, _ in words_data]
                topics_list.append(words)
            else:
                topics_list.append([])

        # 2. Gensim Dictionary
        dictionary = Dictionary(tokenized_data)
        
        # Filter topic words
        filtered_topics_list = []
        for topic in topics_list:
            filtered_topic = [word for word in topic if word in dictionary.token2id]
            filtered_topics_list.append(filtered_topic)
        
        # 3. Calculate Coherence
        coherence_per_topic = []
        
        # Prepare valid topics for Gensim (Gensim fails if a topic is empty list)
        valid_topics_indices = [i for i, t in enumerate(filtered_topics_list) if len(t) > 0]
        valid_topics_payload = [filtered_topics_list[i] for i in valid_topics_indices]

        if valid_topics_payload:
            try:
                cm = CoherenceModel(topics=valid_topics_payload, texts=tokenized_data, dictionary=dictionary, coherence='c_v')
                coherence_scores_raw = cm.get_coherence_per_topic()
                
                # Map back to original indices
                score_map = {original_idx: score for original_idx, score in zip(valid_topics_indices, coherence_scores_raw)}
                
                # Fill full list
                for i in range(len(topic_ids)):
                    coherence_per_topic.append(score_map.get(i, 0.0))
                    
            except Exception as e:
                logger.error(f"Coherence calculation failed: {e}")
                coherence_per_topic = [0.0] * len(topic_ids)
        else:
             coherence_per_topic = [0.0] * len(topic_ids)

        # Prepare detailed results for Table
        topics_details = []
        for idx, tid in enumerate(topic_ids):
            keywords = topics_list[idx] if idx < len(topics_list) else []
            score = coherence_per_topic[idx] if idx < len(coherence_per_topic) else 0.0
            
            topics_details.append({
                'topic_id': tid,
                'name': topic_model.get_topic_info(tid)['Name'].values[0],
                'keywords': keywords,
                'coherence': score
            })

        topics_details = sorted(topics_details, key=lambda x: x['coherence'], reverse=True)

        # Generate Visualizations
        visualizations = generate_bertopic_visualizations(topic_model)

        return {
            'topics_details': topics_details,
            'total_topics': len(topics_details),
            'model_type': f'BERTopic ({ "Sentence Transformer" if embedding_type == "indobert" else embedding_type})',
            'visualizations': visualizations
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            'error': f'Gagal membangun model BERTopic: {str(e)}'
        }

def get_bertopic_analysis(filepath=None):
    pass

def generate_bertopic_visualizations(topic_model):
    """Generate visualizations for BERTopic model"""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import plotly.express as px

    visualizations = {}

    try:
        # 1. Custom Barchart (Colorful)
        topic_info = topic_model.get_topic_info()
        topic_info = topic_info[topic_info['Topic'] != -1]
        
        # Take top 8 by count for barchart
        top_topics = topic_info.sort_values('Count', ascending=False).head(8)
        
        colors = px.colors.qualitative.Plotly + px.colors.qualitative.Bold
        
        rows = (len(top_topics) + 1) // 2
        cols = 2
        
        fig_barchart = make_subplots(
            rows=rows, 
            cols=cols, 
            subplot_titles=[f"Topic {row['Topic']}" for _, row in top_topics.iterrows()],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )

        for idx, (_, row) in enumerate(top_topics.iterrows()):
            topic_id = row['Topic']
            words_data = topic_model.get_topic(topic_id)
            if not words_data:
                continue
                
            words = [w[0] for w in words_data][:5][::-1]
            scores = [w[1] for w in words_data][:5][::-1]
            
            row_idx = (idx // 2) + 1
            col_idx = (idx % 2) + 1
            color = colors[idx % len(colors)]
            
            fig_barchart.add_trace(
                go.Bar(
                    y=words, 
                    x=scores, 
                    orientation='h', 
                    name=f"Topic {topic_id}",
                    marker_color=color,
                    showlegend=False
                ),
                row=row_idx, 
                col=col_idx
            )
            
        fig_barchart.update_layout(
            height=350 * rows, 
            title_text="Top Words per Topic",
            showlegend=False
        )
        visualizations['barchart'] = fig_barchart.to_html(full_html=False)

        # 2. Topics visualization (2D scatter plot)
        try:
             fig_topics = topic_model.visualize_topics()
             visualizations['topics'] = fig_topics.to_html(full_html=False)
        except:
             visualizations['topics'] = "<p>Visualisasi 2D tidak tersedia (mungkin kurang topik)</p>"

    except Exception as viz_error:
        visualizations['barchart'] = f"<p>Visualisasi gagal dimuat: {str(viz_error)}</p>"
        visualizations['topics'] = "<p>Visualisasi topik 2D gagal dimuat</p>"

    return visualizations
