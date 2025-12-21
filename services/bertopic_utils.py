import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def generate_bertopic_visualizations(topic_model):
    """Generate visualizations for BERTopic model"""
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
        
        if len(top_topics) > 0:
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
                height=350 * rows if rows > 0 else 400, 
                title_text="Top Words per Topic",
                showlegend=False
            )
            visualizations['barchart'] = fig_barchart.to_html(full_html=False)
        else:
            visualizations['barchart'] = "<p>Tidak cukup topik untuk menampilkan barchart.</p>"

        # 2. Topics visualization (2D scatter plot)
        try:
             # Check if we have enough topics for distance map (needs at least 2 valid topics)
             distinct_topics = set(topic_model.topics_)
             if -1 in distinct_topics:
                 distinct_topics.remove(-1)
             
             if len(distinct_topics) < 2:
                 msg = f"Hanya ditemukan {len(distinct_topics)} topik. Visualisasi 2D membutuhkan minimal 2 topik untuk menampilkan jarak."
                 visualizations['topics'] = f"<div style='padding:20px; text-align:center; color: #888;'>{msg}</div>"
             else:
                 fig_topics = topic_model.visualize_topics()
                 visualizations['topics'] = fig_topics.to_html(full_html=False)
        except Exception as e:
             visualizations['topics'] = f"<p>Visualisasi 2D tidak tersedia: {str(e)}</p>"

    except Exception as viz_error:
        visualizations['barchart'] = f"<p>Visualisasi gagal dimuat: {str(viz_error)}</p>"
        visualizations['topics'] = "<p>Visualisasi topik 2D gagal dimuat</p>"

    return visualizations
