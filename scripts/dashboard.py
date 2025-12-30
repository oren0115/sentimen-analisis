# -*- coding: utf-8 -*-
"""
Dashboard Interaktif untuk Analisis Sentimen
Analisis Sentimen Opini Publik Terhadap Pilpres

Jalankan dengan:
    streamlit run scripts/dashboard.py
    atau
    python -m streamlit run scripts/dashboard.py
"""

import sys
import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    WordCloud = None
from collections import Counter

# Get project root directory (parent of scripts folder)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add project root to path so we can import src module
sys.path.insert(0, PROJECT_ROOT)

from src.config import DATA_PATH, DATA_TEST_SIZE, SENTIMENT_MAP, SENTIMENT_LABELS
from src.data_processing import (
    load_data, split_data, preprocess_data, encode_sentiments,
    process_token_lengths, prepare_data_for_training
)
from src.text_cleaning import clean_texts
from src.models import train_naive_bayes

# Page configuration
st.set_page_config(
    page_title="Analisis Sentimen Pilpres",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Cache data loading
@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess data with caching."""
    try:
        # Load data from single CSV file
        df = load_data(DATA_PATH)
        
        # Split into train and test sets
        df_train, df_test = split_data(df, test_size=DATA_TEST_SIZE)
        
        # Preprocess
        df_processed, df_test_processed = preprocess_data(df_train.copy(), df_test.copy())
        df_processed, df_test_processed = encode_sentiments(df_processed, df_test_processed)
        
        return df_train, df_test, df_processed, df_test_processed
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None

# Load data
df, df_test, df_processed, df_test_processed = load_and_preprocess_data()

if df is None:
    st.error("Tidak dapat memuat data. Pastikan file CSV ada di folder data/")
    st.stop()

# Sidebar
st.sidebar.title("📊 Navigasi Dashboard")
page = st.sidebar.radio(
    "Pilih Halaman",
    ["🏠 Overview", "📈 Analisis Data", "💬 Analisis Sentimen", "🏆 Evaluasi Model", "📊 Visualisasi Detail"]
)

# Main title
st.markdown('<h1 class="main-header">📊 Dashboard Analisis Sentimen Opini Publik Terhadap Pilpres</h1>', unsafe_allow_html=True)

# ============================================================================
# PAGE 1: OVERVIEW
# ============================================================================
if page == "🏠 Overview":
    st.header("📋 Overview Data")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Data Training", f"{len(df):,}")
    
    with col2:
        st.metric("Total Data Test", f"{len(df_test):,}")
    
    with col3:
        st.metric("Data Setelah Preprocessing", f"{len(df_processed):,}")
    
    with col4:
        st.metric("Data Test Setelah Preprocessing", f"{len(df_test_processed):,}")
    
    st.divider()
    
    # Data preview
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Preview Data Training")
        st.dataframe(df.head(10), width='stretch')
    
    with col2:
        st.subheader("📄 Preview Data Test")
        st.dataframe(df_test.head(10), width='stretch')
    
    st.divider()
    
    # Statistics
    st.subheader("📊 Statistik Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Data Training:**")
        st.write(f"- Jumlah baris: {len(df):,}")
        st.write(f"- Jumlah kolom: {len(df.columns)}")
        st.write(f"- Kolom: {', '.join(df.columns.tolist())}")
        if 'Date' in df.columns:
            st.write(f"- Rentang tanggal: {df['Date'].min()} hingga {df['Date'].max()}")
    
    with col2:
        st.write("**Data Test:**")
        st.write(f"- Jumlah baris: {len(df_test):,}")
        st.write(f"- Jumlah kolom: {len(df_test.columns)}")
        st.write(f"- Kolom: {', '.join(df_test.columns.tolist())}")

# ============================================================================
# PAGE 2: ANALISIS DATA
# ============================================================================
elif page == "📈 Analisis Data":
    st.header("📈 Analisis Data")
    
    # Sentiment distribution
    st.subheader("📊 Distribusi Sentimen")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Training data sentiment
        if 'Sentiment' in df_processed.columns:
            sentiment_counts = df_processed['Sentiment'].value_counts().sort_index()
            sentiment_labels = [SENTIMENT_LABELS[int(i)] for i in sentiment_counts.index]
            
            fig = px.pie(
                values=sentiment_counts.values,
                names=sentiment_labels,
                title="Distribusi Sentimen (Training Data)",
                color_discrete_map={
                    'Negative': '#FF6B6B',
                    'Neutral': '#FFD93D',
                    'Positive': '#6BCF7F'
                }
            )
            st.plotly_chart(fig, width='stretch')
    
    with col2:
        # Test data sentiment
        if 'Sentiment' in df_test_processed.columns:
            sentiment_counts_test = df_test_processed['Sentiment'].value_counts().sort_index()
            sentiment_labels_test = [SENTIMENT_LABELS[int(i)] for i in sentiment_counts_test.index]
            
            fig = px.pie(
                values=sentiment_counts_test.values,
                names=sentiment_labels_test,
                title="Distribusi Sentimen (Test Data)",
                color_discrete_map={
                    'Negative': '#FF6B6B',
                    'Neutral': '#FFD93D',
                    'Positive': '#6BCF7F'
                }
            )
            st.plotly_chart(fig, width='stretch')
    
    st.divider()
    
    # Tweets by date
    if 'Date' in df.columns:
        st.subheader("📅 Jumlah Tweet per Tanggal")
        
        df['Date'] = pd.to_datetime(df['Date'])
        tweets_per_date = df['Date'].dt.strftime('%Y-%m-%d').value_counts().sort_index().reset_index()
        tweets_per_date.columns = ['Date', 'Count']
        
        fig = px.bar(
            tweets_per_date,
            x='Date',
            y='Count',
            title="Jumlah Tweet per Tanggal",
            labels={'Date': 'Tanggal', 'Count': 'Jumlah Tweet'},
            color='Count',
            color_continuous_scale='Blues'
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, width='stretch')
    
    st.divider()
    
    # Text length distribution
    st.subheader("📏 Distribusi Panjang Teks")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'text_len' in df_processed.columns:
            fig = px.histogram(
                df_processed,
                x='text_len',
                nbins=30,
                title="Distribusi Panjang Teks (Training)",
                labels={'text_len': 'Panjang Teks (kata)', 'count': 'Jumlah'},
                color_discrete_sequence=['#1f77b4']
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width='stretch')
    
    with col2:
        if 'text_len' in df_test_processed.columns:
            fig = px.histogram(
                df_test_processed,
                x='text_len',
                nbins=30,
                title="Distribusi Panjang Teks (Test)",
                labels={'text_len': 'Panjang Teks (kata)', 'count': 'Jumlah'},
                color_discrete_sequence=['#ff7f0e']
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width='stretch')
    
    # Statistics
    if 'text_len' in df_processed.columns:
        st.subheader("📊 Statistik Panjang Teks")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Rata-rata (Training)", f"{df_processed['text_len'].mean():.1f} kata")
        with col2:
            st.metric("Median (Training)", f"{df_processed['text_len'].median():.1f} kata")
        with col3:
            st.metric("Rata-rata (Test)", f"{df_test_processed['text_len'].mean():.1f} kata")
        with col4:
            st.metric("Median (Test)", f"{df_test_processed['text_len'].median():.1f} kata")

# ============================================================================
# PAGE 3: ANALISIS SENTIMEN
# ============================================================================
elif page == "💬 Analisis Sentimen":
    st.header("💬 Analisis Sentimen Detail")
    
    # Sentiment comparison
    st.subheader("📊 Perbandingan Sentimen")
    
    if 'Sentiment' in df_processed.columns and 'Sentiment' in df_test_processed.columns:
        train_sentiment = df_processed['Sentiment'].value_counts().sort_index()
        test_sentiment = df_test_processed['Sentiment'].value_counts().sort_index()
        
        comparison_df = pd.DataFrame({
            'Sentiment': [SENTIMENT_LABELS[int(i)] for i in train_sentiment.index],
            'Training': train_sentiment.values,
            'Test': test_sentiment.values
        })
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Training',
            x=comparison_df['Sentiment'],
            y=comparison_df['Training'],
            marker_color='#1f77b4'
        ))
        fig.add_trace(go.Bar(
            name='Test',
            x=comparison_df['Sentiment'],
            y=comparison_df['Test'],
            marker_color='#ff7f0e'
        ))
        
        fig.update_layout(
            title="Perbandingan Distribusi Sentimen",
            xaxis_title="Sentimen",
            yaxis_title="Jumlah",
            barmode='group'
        )
        
        st.plotly_chart(fig, width='stretch')
    
    st.divider()
    
    # Sample texts by sentiment
    st.subheader("📝 Contoh Teks per Sentimen")
    
    sentiment_filter = st.selectbox(
        "Pilih Sentimen",
        ["All", "Negative", "Neutral", "Positive"]
    )
    
    if sentiment_filter != "All":
        sentiment_value = SENTIMENT_LABELS.index(sentiment_filter)
        filtered_df = df_processed[df_processed['Sentiment'] == sentiment_value]
    else:
        filtered_df = df_processed
    
    num_samples = st.slider("Jumlah Contoh", 1, 20, 5)
    
    if len(filtered_df) > 0:
        samples = filtered_df.sample(min(num_samples, len(filtered_df)))
        
        for idx, row in samples.iterrows():
            sentiment_label = SENTIMENT_LABELS[int(row['Sentiment'])]
            color = {
                'Negative': '🔴',
                'Neutral': '🟡',
                'Positive': '🟢'
            }[sentiment_label]
            
            with st.expander(f"{color} {sentiment_label} - Teks #{idx}"):
                st.write("**Teks Asli:**")
                st.write(row['Text'] if 'Text' in row else "N/A")
                if 'text_clean' in row:
                    st.write("**Teks Setelah Cleaning:**")
                    st.write(row['text_clean'])
    
    st.divider()
    
    # Word Cloud (if text_clean exists)
    if 'text_clean' in df_processed.columns:
        st.subheader("☁️ Word Cloud")
        
        sentiment_for_wc = st.selectbox(
            "Pilih Sentimen untuk Word Cloud",
            ["All", "Negative", "Neutral", "Positive"],
            key="wc_sentiment"
        )
        
        if sentiment_for_wc != "All":
            sentiment_value_wc = SENTIMENT_LABELS.index(sentiment_for_wc)
            texts_for_wc = df_processed[df_processed['Sentiment'] == sentiment_value_wc]['text_clean'].str.cat(sep=' ')
        else:
            texts_for_wc = df_processed['text_clean'].str.cat(sep=' ')
        
        if texts_for_wc and len(texts_for_wc) > 0:
            if WORDCLOUD_AVAILABLE:
                try:
                    wordcloud = WordCloud(
                        width=800,
                        height=400,
                        background_color='white',
                        max_words=100
                    ).generate(texts_for_wc)
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"Tidak dapat membuat word cloud: {str(e)}")
            else:
                st.info("⚠️ WordCloud tidak tersedia. Install dengan: `pip install wordcloud`")

# ============================================================================
# PAGE 4: EVALUASI MODEL
# ============================================================================
elif page == "🏆 Evaluasi Model":
    st.header("🏆 Evaluasi Model Naive Bayes")
    
    # Cache model training
    @st.cache_data
    def train_and_evaluate_model():
        """Train model and return metrics."""
        try:
            # Prepare data for training
            df_train_prep = df_processed.copy()
            df_test_prep = df_test_processed.copy()
            
            # Process token lengths
            from src.config import MAX_TOKEN_LENGTH
            df_train_prep, df_test_prep = process_token_lengths(
                df_train_prep, df_test_prep, max_length=MAX_TOKEN_LENGTH
            )
            
            # Prepare data
            (X_train, X_valid, X_test, y_train, y_valid, y_test,
             y_train_le, y_valid_le, y_test_le) = prepare_data_for_training(
                df_train_prep, df_test_prep
            )
            
            # Train model
            nb_model, vectorizer, tf_transformer, metrics = train_naive_bayes(
                X_train, y_train_le, X_test, y_test_le
            )
            
            return metrics, nb_model, vectorizer, tf_transformer
        except Exception as e:
            st.error(f"Error training model: {str(e)}")
            return None, None, None, None
    
    # Train model
    with st.spinner("Training model... Ini mungkin memakan waktu beberapa saat."):
        metrics, model, vectorizer, tf_transformer = train_and_evaluate_model()
    
    if metrics is None:
        st.error("Gagal melatih model. Silakan cek error di atas.")
        st.stop()
    
    # Overall Accuracy
    st.subheader("🎯 Akurasi Keseluruhan")
    
    accuracy = metrics['accuracy']
    accuracy_color = '#10b981' if accuracy >= 0.8 else '#f59e0b' if accuracy >= 0.6 else '#ef4444'
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
            <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, {accuracy_color}15 0%, {accuracy_color}05 100%); border-radius: 1rem; border: 2px solid {accuracy_color}; margin: 1rem 0;">
                <div style="font-size: 0.875rem; color: #64748b; font-weight: 600; margin-bottom: 0.5rem;">AKURASI MODEL</div>
                <div style="font-size: 4rem; font-weight: 700; color: {accuracy_color}; line-height: 1;">
                    {accuracy*100:.2f}%
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Metrics per class
    st.subheader("📊 Metrik per Kelas")
    
    # Create metrics dataframe
    metrics_data = []
    for label in SENTIMENT_LABELS:
        metrics_data.append({
            'Kelas': label,
            'Precision': metrics['precision']['per_class'][label] * 100,
            'Recall': metrics['recall']['per_class'][label] * 100,
            'F1-Score': metrics['f1_score']['per_class'][label] * 100,
            'Support': int(metrics['classification_report'][label]['support'])
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Display metrics table
    st.dataframe(metrics_df, width='stretch', hide_index=True)
    
    # Bar charts for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig = px.bar(
            metrics_df,
            x='Kelas',
            y='Precision',
            title="Precision per Kelas",
            labels={'Precision': 'Precision (%)', 'Kelas': 'Sentimen'},
            color='Kelas',
            color_discrete_map={
                'Negative': '#FF6B6B',
                'Neutral': '#FFD93D',
                'Positive': '#6BCF7F'
            }
        )
        fig.update_layout(showlegend=False, yaxis_range=[0, 100])
        st.plotly_chart(fig, width='stretch', use_container_width=True)
    
    with col2:
        fig = px.bar(
            metrics_df,
            x='Kelas',
            y='Recall',
            title="Recall per Kelas",
            labels={'Recall': 'Recall (%)', 'Kelas': 'Sentimen'},
            color='Kelas',
            color_discrete_map={
                'Negative': '#FF6B6B',
                'Neutral': '#FFD93D',
                'Positive': '#6BCF7F'
            }
        )
        fig.update_layout(showlegend=False, yaxis_range=[0, 100])
        st.plotly_chart(fig, width='stretch', use_container_width=True)
    
    with col3:
        fig = px.bar(
            metrics_df,
            x='Kelas',
            y='F1-Score',
            title="F1-Score per Kelas",
            labels={'F1-Score': 'F1-Score (%)', 'Kelas': 'Sentimen'},
            color='Kelas',
            color_discrete_map={
                'Negative': '#FF6B6B',
                'Neutral': '#FFD93D',
                'Positive': '#6BCF7F'
            }
        )
        fig.update_layout(showlegend=False, yaxis_range=[0, 100])
        st.plotly_chart(fig, width='stretch', use_container_width=True)
    
    st.divider()
    
    # Confusion Matrix
    st.subheader("📋 Confusion Matrix")
    
    cm = np.array(metrics['confusion_matrix'])
    
    # Create confusion matrix heatmap
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=SENTIMENT_LABELS,
        y=SENTIMENT_LABELS,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='Blues',
        title="Confusion Matrix"
    )
    fig.update_layout(width=600, height=500)
    st.plotly_chart(fig, width='stretch')
    
    st.divider()
    
    # Detailed Classification Report
    st.subheader("📄 Classification Report Detail")
    
    # Create detailed report table
    report_data = []
    for label in SENTIMENT_LABELS:
        report = metrics['classification_report'][label]
        report_data.append({
            'Kelas': label,
            'Precision': f"{report['precision']*100:.2f}%",
            'Recall': f"{report['recall']*100:.2f}%",
            'F1-Score': f"{report['f1-score']*100:.2f}%",
            'Support': int(report['support'])
        })
    
    # Add averages
    macro_avg = metrics['classification_report']['macro avg']
    weighted_avg = metrics['classification_report']['weighted avg']
    
    report_data.append({
        'Kelas': 'Macro Avg',
        'Precision': f"{macro_avg['precision']*100:.2f}%",
        'Recall': f"{macro_avg['recall']*100:.2f}%",
        'F1-Score': f"{macro_avg['f1-score']*100:.2f}%",
        'Support': int(macro_avg['support'])
    })
    
    report_data.append({
        'Kelas': 'Weighted Avg',
        'Precision': f"{weighted_avg['precision']*100:.2f}%",
        'Recall': f"{weighted_avg['recall']*100:.2f}%",
        'F1-Score': f"{weighted_avg['f1-score']*100:.2f}%",
        'Support': int(weighted_avg['support'])
    })
    
    report_df = pd.DataFrame(report_data)
    st.dataframe(report_df, width='stretch', hide_index=True)
    
    st.divider()
    
    # Summary insights
    st.subheader("💡 Insight & Analisis")
    
    best_class = max(metrics['f1_score']['per_class'].items(), key=lambda x: x[1])
    worst_class = min(metrics['f1_score']['per_class'].items(), key=lambda x: x[1])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
            <div style="padding: 1.5rem; background: linear-gradient(135deg, #10b98115 0%, #10b98105 100%); border-radius: 0.75rem; border-left: 4px solid #10b981;">
                <h4 style="color: #10b981; font-weight: 600; margin-bottom: 1rem;">
                    🏆 Best Performing Class
                </h4>
                <div style="font-size: 1.5rem; font-weight: 700; color: #1e293b; margin-bottom: 0.5rem;">
                    {best_class[0]}
                </div>
                <div style="color: #64748b;">
                    F1-Score: <strong style="color: #10b981;">{best_class[1]*100:.2f}%</strong>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div style="padding: 1.5rem; background: linear-gradient(135deg, #f59e0b15 0%, #f59e0b05 100%); border-radius: 0.75rem; border-left: 4px solid #f59e0b;">
                <h4 style="color: #f59e0b; font-weight: 600; margin-bottom: 1rem;">
                    ⚠️ Worst Performing Class
                </h4>
                <div style="font-size: 1.5rem; font-weight: 700; color: #1e293b; margin-bottom: 0.5rem;">
                    {worst_class[0]}
                </div>
                <div style="color: #64748b;">
                    F1-Score: <strong style="color: #f59e0b;">{worst_class[1]*100:.2f}%</strong>
                </div>
            </div>
        """, unsafe_allow_html=True)

# ============================================================================
# PAGE 5: VISUALISASI DETAIL
# ============================================================================
elif page == "📊 Visualisasi Detail":
    st.header("📊 Visualisasi Detail")
    
    # Interactive filters
    st.subheader("🔍 Filter Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        min_length = st.slider(
            "Panjang Teks Minimum",
            min_value=1,
            max_value=50,
            value=1,
            key="min_len"
        )
    
    with col2:
        max_length = st.slider(
            "Panjang Teks Maksimum",
            min_value=1,
            max_value=200,
            value=200,
            key="max_len"
        )
    
    # Filter data
    filtered_df = df_processed[
        (df_processed['text_len'] >= min_length) & 
        (df_processed['text_len'] <= max_length)
    ]
    
    st.metric("Jumlah Data Setelah Filter", len(filtered_df))
    
    st.divider()
    
    # Detailed charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Box Plot Panjang Teks")
        if 'text_len' in filtered_df.columns and 'Sentiment' in filtered_df.columns:
            fig = px.box(
                filtered_df,
                x='Sentiment',
                y='text_len',
                color='Sentiment',
                title="Box Plot Panjang Teks per Sentimen",
                labels={
                    'text_len': 'Panjang Teks (kata)',
                    'Sentiment': 'Sentimen'
                },
                color_discrete_map={
                    '0': '#FF6B6B',
                    '1': '#FFD93D',
                    '2': '#6BCF7F'
                }
            )
            # Update x-axis labels
            fig.update_xaxes(
                tickmode='array',
                tickvals=[0, 1, 2],
                ticktext=SENTIMENT_LABELS
            )
            st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.subheader("📈 Violin Plot")
        if 'text_len' in filtered_df.columns and 'Sentiment' in filtered_df.columns:
            fig = px.violin(
                filtered_df,
                x='Sentiment',
                y='text_len',
                color='Sentiment',
                title="Violin Plot Panjang Teks per Sentimen",
                labels={
                    'text_len': 'Panjang Teks (kata)',
                    'Sentiment': 'Sentimen'
                },
                color_discrete_map={
                    '0': '#FF6B6B',
                    '1': '#FFD93D',
                    '2': '#6BCF7F'
                }
            )
            fig.update_xaxes(
                tickmode='array',
                tickvals=[0, 1, 2],
                ticktext=SENTIMENT_LABELS
            )
            st.plotly_chart(fig, width='stretch')
    
    st.divider()
    
    # Data table
    st.subheader("📋 Data Detail")
    st.dataframe(
        filtered_df[['Text', 'text_clean', 'text_len', 'Sentiment']].head(100),
        width='stretch'
    )

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>Dashboard Analisis Sentimen Opini Publik Terhadap Pilpres</p>
        <p>Dibuat dengan Streamlit</p>
    </div>
""", unsafe_allow_html=True)

