import os
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, models
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
import logging
import joblib

from services.bertopic_utils import generate_bertopic_visualizations

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_bertopic_indobert(filepath=None):
    """Build topic model using BERTopic + IndoBERT (Sentence Transformer)"""
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

        # Determine min_df based on dataset size
        # Prevent ValueError: max_df corresponds to < documents than min_df
        # Use a more conservative minimal document frequency
        n_docs = len(text_data)
        print(f"DEBUG: Valid text data count: {n_docs}")
        
        if n_docs < 200:
            min_df = 1
        elif n_docs < 1000:
            min_df = 2
        else:
            min_df = 5
        
        print(f"DEBUG: Using min_df={min_df}")

        # Vectorizer
        vectorizer_model = CountVectorizer(
            stop_words=indonesian_stopwords,
            ngram_range=(1, 2),
            min_df=min_df 
        )
        
        # Tokenized data for Coherence Calculation (Use analyzer to match topic n-grams)
        analyzer = vectorizer_model.build_analyzer()
        tokenized_data = [analyzer(text) for text in text_data]

        # --- Method 1: IndoBERT Lite (Sentene Transformer) ---
        print("Loading IndoBERT Lite model (Lighter version)...")
        try:
            # Menggunakan IndoBERT Lite (ALBERT based) yang lebih ringan ("min L2"/Lite) tapi tetap IndoBERT
            word_embedding_model = models.Transformer('indolem/indobert-lite-base-uncased')
            # Add pooling layer
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
            # Combine to create SentenceTransformer
            embedding_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        except Exception as e:
            print(f"Failed to load indolem/indobert-lite-base-uncased: {e}. Fallback to firqaaa/indo-sentence-bert-base.")
            embedding_model = SentenceTransformer("firqaaa/indo-sentence-bert-base")

        # Dynamic topic size: aggressive for small data
        if n_docs < 500:
            calculated_min_topic = 3
        else:
            calculated_min_topic = 10

        # Create BERTopic model
        topic_model = BERTopic(
            embedding_model=embedding_model,
            vectorizer_model=vectorizer_model,
            representation_model=KeyBERTInspired(),
            ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
            calculate_probabilities=True,
            verbose=True,
            min_topic_size=calculated_min_topic
        )

        # Fit the model
        topics, probs = topic_model.fit_transform(text_data)
        
        # Debugging topic counts
        unique_topics_count = len(set(topics)) - (1 if -1 in topics else 0)
        print(f"DEBUG: Found {unique_topics_count} topic(s).")

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
            'model_type': 'BERTopic (Sentence Transformer)',
            'visualizations': visualizations
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            'error': f'Gagal membangun model BERTopic (IndoBERT): {str(e)}'
        }
