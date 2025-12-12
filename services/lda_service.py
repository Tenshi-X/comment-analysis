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
    visualizations = {}
    
    try:
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        import plotly.express as px
        import plotly.colors
        
        # 1. Barchart: Top words per topic (Colorful)
        
        # Take top 8 topics by count
        top_topics = sorted(topics_summary, key=lambda x: x['count'], reverse=True)[:8]
        rows = (len(top_topics) + 1) // 2
        cols = 2
        
        fig = make_subplots(
            rows=rows, 
            cols=cols, 
            subplot_titles=[t['name'] for t in top_topics],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # Colors
        colors = plotly.colors.qualitative.Plotly + plotly.colors.qualitative.Bold
        
        for idx, topic in enumerate(top_topics):
            words = [w[0] for w in topic['top_words_probs']][:5][::-1] 
            probs = [w[1] for w in topic['top_words_probs']][:5][::-1]
            
            row = (idx // 2) + 1
            col = (idx % 2) + 1
            
            # Use specific color
            color = colors[idx % len(colors)]
            
            fig.add_trace(
                go.Bar(
                    y=words, 
                    x=probs, 
                    orientation='h', 
                    name=topic['name'],
                    marker_color=color,
                    showlegend=False
                ),
                row=row, col=col
            )
            
        fig.update_layout(height=350 * rows, title_text="Top Words per Topic", showlegend=False)
        visualizations['barchart'] = fig.to_html(full_html=False)
        
        # 2. Intertopic Distance Map (Simulated like BERTopic using MDS)
        try:
            from sklearn.manifold import MDS
            from sklearn.metrics.pairwise import cosine_distances
            import numpy as np

            # Create vectors based on the union of all top words.
            all_top_words = set()
            for t in topics_summary:
                for w, _ in t['top_words_probs']:
                    all_top_words.add(w)
            
            vocab = list(all_top_words)
            vocab_idx = {w: i for i, w in enumerate(vocab)}
            
            # Create vectors
            vectors = []
            sizes = []
            labels = []
            
            for t in topics_summary:
                vec = np.zeros(len(vocab))
                for w, p in t['top_words_probs']:
                    if w in vocab_idx:
                        vec[vocab_idx[w]] = p
                vectors.append(vec)
                sizes.append(t['count'])
                labels.append(t['name'])
            
            vectors = np.array(vectors)
            
            if len(vectors) > 2:
                # Compute distance matrix
                dist_matrix = cosine_distances(vectors)
                
                # MDS
                mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
                coords = mds.fit_transform(dist_matrix)
                
                # Plot
                df_map = pd.DataFrame({
                    'x': coords[:, 0],
                    'y': coords[:, 1],
                    'Topic': labels,
                    'Size': sizes,
                    'Keywords': [", ".join(t['keywords'][:5]) for t in topics_summary]
                })
                
                fig_map = px.scatter(
                    df_map, 
                    x='x', 
                    y='y', 
                    size='Size', 
                    color='Topic',
                    hover_name='Topic',
                    hover_data={'Keywords': True, 'x': False, 'y': False},
                    title='Intertopic Distance Map (LDA)',
                    size_max=60
                )
                
                fig_map.update_layout(
                    showlegend=False, 
                    height=800,
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                )
                visualizations['intertopic_map'] = fig_map.to_html(full_html=False)
            else:
                visualizations['intertopic_map'] = "<p>Not enough topics for Intertopic Map</p>"
                
        except Exception as e:
             visualizations['intertopic_map'] = f"<p>Gagal membuat map: {str(e)}</p>"

    except Exception as e:
        visualizations['barchart'] = f"<p>Gagal membuat visualisasi: {str(e)}</p>"

    return visualizations
