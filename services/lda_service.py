import os
import pandas as pd
import gensim
from gensim import corpora
from gensim.models import CoherenceModel
import joblib
import hashlib
import plotly.express as px
import plotly.graph_objects as go
from services.preprocessing import get_preprocessing_steps

def load_lda_model(filepath=None, model_path=None):
    """Load LDA model for specific CSV file"""
    if filepath and not model_path:
        # Generate unique model path based on file hash
        file_hash = hashlib.md5(filepath.encode()).hexdigest()[:8]
        model_path = f'models/lda_model_{file_hash}.pkl'

    if model_path and os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except:
            pass
    return None

def build_lda_model(filepath=None, num_topics=5):
    """Build LDA model from uploaded CSV data"""
    if filepath is None:
        return {
            'error': 'Filepath tidak diberikan. Pastikan file CSV sudah diupload.'
        }

    try:
        # Load and preprocess data
        df = pd.read_csv(filepath, encoding='utf-8-sig')
        preprocessing_results = get_preprocessing_steps(df)
        
        # Get tokenized data (list of lists of strings)
        # Assuming preprocessing returns a list of dicts with 'text_clean'
        # We need to tokenize 'text_clean' for Gensim
        text_data = [item['text_clean'].split() for item in preprocessing_results['hasil_preprocessing'] if item['text_clean']]

        if not text_data:
            return {
                'error': 'Tidak ada data teks yang valid setelah preprocessing.'
            }

        # Create Dictionary
        id2word = corpora.Dictionary(text_data)
        # Filter extremes to remove very rare and very common words
        id2word.filter_extremes(no_below=2, no_above=0.95)

        # Create Corpus
        corpus = [id2word.doc2bow(text) for text in text_data]

        # Train LDA Model
        lda_model = gensim.models.LdaModel(
            corpus=corpus,
            id2word=id2word,
            num_topics=num_topics,
            random_state=42,
            update_every=1,
            chunksize=100,
            passes=10,
            alpha='auto',
            per_word_topics=True
        )

        # Calculate Coherence Score
        coherence_model_lda = CoherenceModel(
            model=lda_model,
            texts=text_data,
            dictionary=id2word,
            coherence='c_v'
        )
        coherence_score = coherence_model_lda.get_coherence()

        # Prepare Topics Summary
        topics_summary = []
        for topic_id in range(num_topics):
            # Get top words
            top_words = lda_model.show_topic(topic_id, topn=10)
            keywords = [word for word, prob in top_words]
            
            # Estimate topic prevalence (count in corpus)
            # This is a bit rough for LDA as documents successfuly mix topics
            # We can sum the contributions of this topic across all documents
            count = 0
            for doc in corpus:
                topic_dist = dict(lda_model.get_document_topics(doc))
                if topic_id in topic_dist:
                    count += topic_dist[topic_id]
            
            topics_summary.append({
                'topic_id': topic_id,
                'name': f'Topik {topic_id + 1}',
                'keywords': keywords,
                'count': int(count), # Weighted count
                'top_words_probs': top_words # [(word, prob), ...]
            })

        # Save model
        file_hash = hashlib.md5(filepath.encode()).hexdigest()[:8]
        model_path = f'models/lda_model_{file_hash}.pkl'
        model_data = {
            'lda_model': lda_model,
            'corpus': corpus,
            'id2word': id2word,
            'topics_summary': topics_summary,
            'coherence_score': coherence_score,
            'num_topics': num_topics
        }
        joblib.dump(model_data, model_path)

        # Generate Visualizations
        visualizations = generate_lda_visualizations(topics_summary)

        return {
            'topics_summary': topics_summary,
            'coherence_score': coherence_score,
            'total_topics': num_topics,
            'model_type': 'LDA (Latent Dirichlet Allocation)',
            'visualizations': visualizations
        }

    except Exception as e:
        return {
            'error': f'Gagal membangun model LDA: {str(e)}'
        }

def get_lda_analysis(filepath=None):
    """Get LDA analysis data from saved model"""
    model_data = load_lda_model(filepath=filepath)
    if model_data is None:
        return {
            'error': 'Model LDA tidak ditemukan. Pastikan model sudah dibangun.'
        }

    topics_summary = model_data['topics_summary']
    visualizations = generate_lda_visualizations(topics_summary)

    return {
        'topics_summary': topics_summary,
        'coherence_score': model_data.get('coherence_score', 0.0),
        'total_topics': model_data.get('num_topics', 0),
        'model_type': 'LDA (Latent Dirichlet Allocation)',
        'visualizations': visualizations
    }

def generate_lda_visualizations(topics_summary):
    # Visualize Barchart: Top words per topic
    barchart_htmls = {}
    
    # Combined barchart (just top words for all topics nicely?)
    # Or separate charts? BERTopic does separate bars in one big chart usually or subplot.
    # Let's do what user asked: Barchart.
    # We can create a dropdown-like behavior or just show top 5 topics' top words.
    
    # Let's make a horizontal bar chart for each topic or a Facet plot
    # Simpler: Bar chart of Top 10 words for the Topic with highest count?
    # Or just Stacked?
    # Let's try to mimic BERTopic's barchart: Horizontal bars for each topic.
    
    # Since we can't easily return multiple widgets, let's create one big figure with subplots
    # OR just one figure for the most prominent topic?
    # Actually, let's just make one combined chart for the top topic for now, 
    # or better, return a list of figures if the UI can handle it.
    # But current pattern returns HTML strings.
    
    # Let's create a single chart that allows selecting topics or just shows top words for Topic 1
    # Actually, standard BERTopic "visualize_barchart" shows a grid of bar charts.
    # We can replicate "Top 5 Topics" bar charts using Plotly Subplots.
    
    try:
        from plotly.subplots import make_subplots
        
        # Take top 5 topics by count
        top_topics = sorted(topics_summary, key=lambda x: x['count'], reverse=True)[:8] # Max 8
        rows = (len(top_topics) + 1) // 2
        cols = 2
        
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=[t['name'] for t in top_topics])
        
        for idx, topic in enumerate(top_topics):
            words = [w[0] for w in topic['top_words_probs']][:5][::-1] # Top 5 words, reversed for horiz bar
            probs = [w[1] for w in topic['top_words_probs']][:5][::-1]
            
            row = (idx // 2) + 1
            col = (idx % 2) + 1
            
            fig.add_trace(
                go.Bar(y=words, x=probs, orientation='h', name=topic['name']),
                row=row, col=col
            )
            
        fig.update_layout(height=300 * rows, title_text="Top Words per Topic", showlegend=False)
        barchart_html = fig.to_html(full_html=False)
    except Exception as e:
        barchart_html = f"<p>Gagal membuat visualisasi: {str(e)}</p>"

    return {
        'barchart': barchart_html
    }
