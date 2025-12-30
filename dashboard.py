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

# Get project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# Add project root to path so we can import src module
sys.path.insert(0, PROJECT_ROOT)

from src.config import DATA_PATH, DATA_TEST_SIZE, SENTIMENT_MAP, SENTIMENT_LABELS
from src.data_processing import load_data, split_data, preprocess_data, encode_sentiments
from src.text_cleaning import clean_texts

# Page configuration
st.set_page_config(
    page_title="Analisis Sentimen Pilpres",
    page_icon="📊",  # Favicon - Streamlit doesn't support Font Awesome icons here
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper function untuk icons
def icon(icon_name, size="1rem", color="#1f77b4", style="solid"):
    """Generate Font Awesome icon HTML"""
    icon_class = f"fa{style[0]} fa-{icon_name}" if style else f"fa-{icon_name}"
    return f'<i class="{icon_class}" style="font-size: {size}; color: {color}; margin-right: 0.5rem;"></i>'

# Custom CSS untuk modern UI/UX - Disembunyikan dari tampilan
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" integrity="sha512-iecdLmaskl7CVkqkXNQ/ZH/XLlvWZOJyj7Yy7tcenmpD1ypASozpmT/E0iPtmFIB46ZmdtAc9eNBvH0H/ZpiBw==" crossorigin="anonymous" referrerpolicy="no-referrer" />
<style>
    /* Global Styles */
    :root {
        --primary-color: #2563eb;
        --secondary-color: #1e40af;
        --accent-color: #3b82f6;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --danger-color: #ef4444;
        --bg-color: #f8fafc;
        --card-bg: #ffffff;
        --text-primary: #1e293b;
        --text-secondary: #64748b;
        --border-color: #e2e8f0;
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    /* Main Header */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        padding: 1.5rem 0;
        margin-bottom: 2rem;
        letter-spacing: -0.02em;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background-color: var(--card-bg);
        padding-top: 2rem;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
    }
    
    /* Navigation Items */
    .stRadio label {
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        transition: all 0.3s ease;
        margin: 0.25rem 0;
        font-weight: 500;
    }
    
    .stRadio label:hover {
        background-color: #f1f5f9;
        transform: translateX(4px);
    }
    
    /* Metric Cards */
    .metric-card {
        background: var(--card-bg);
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin: 0.5rem 0;
        box-shadow: var(--shadow-md);
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        box-shadow: var(--shadow-lg);
        transform: translateY(-2px);
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--border-color);
        display: flex;
        align-items: center;
    }
    
    /* Subheader Styling */
    h3 {
        color: var(--text-primary);
        font-weight: 600;
        margin-top: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Card Containers */
    .card-container {
        background: var(--card-bg);
        padding: 1.5rem;
        border-radius: 0.75rem;
        box-shadow: var(--shadow-md);
        border: 1px solid var(--border-color);
        margin: 1rem 0;
    }
    
    /* Sentiment Badges */
    .sentiment-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    
    .sentiment-negative {
        background-color: #fee2e2;
        color: #991b1b;
    }
    
    .sentiment-neutral {
        background-color: #fef3c7;
        color: #92400e;
    }
    
    .sentiment-positive {
        background-color: #d1fae5;
        color: #065f46;
    }
    
    /* Icon Styling */
    .icon-wrapper {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Divider Styling */
    hr {
        border: none;
        border-top: 2px solid var(--border-color);
        margin: 2rem 0;
    }
    
    /* Dataframe Styling */
    .dataframe {
        border-radius: 0.5rem;
        overflow: hidden;
        box-shadow: var(--shadow-sm);
    }
    
    /* Button Enhancements */
    .stButton > button {
        border-radius: 0.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
        border: 1px solid var(--border-color);
    }
    
    .stButton > button:hover {
        box-shadow: var(--shadow-md);
        transform: translateY(-1px);
    }
    
    /* Selectbox Styling */
    .stSelectbox label {
        font-weight: 600;
        color: var(--text-primary);
    }
    
    /* Slider Styling */
    .stSlider label {
        font-weight: 600;
        color: var(--text-primary);
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        background-color: #f8fafc;
        border-radius: 0.5rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: var(--text-secondary);
        padding: 2rem 0;
        margin-top: 3rem;
        border-top: 1px solid var(--border-color);
        font-size: 0.875rem;
    }
    
    /* Loading Spinner */
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    .spinner {
        display: inline-block;
        animation: spin 1s linear infinite;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
    }
    
    /* Remove Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
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
st.sidebar.markdown("""
    <div style='padding: 1rem 0; margin-bottom: 1rem;'>
        <h2 style='color: #1e293b; font-weight: 700; font-size: 1.5rem; display: flex; align-items: center; gap: 0.5rem;'>
            <i class='fas fa-chart-line' style='color: #2563eb;'></i>
            Navigasi Dashboard
        </h2>
    </div>
""", unsafe_allow_html=True)

# Navigation
page = st.sidebar.radio(
    "Pilih Halaman",
    ["Overview", "Analisis Data", "Analisis Sentimen", "Visualisasi Detail"]
)

# Main title
st.markdown("""
    <h1 class="main-header">
        <i class="fas fa-chart-bar" style="font-size: 2.5rem; color: #2563eb; vertical-align: middle; margin-right: 1rem;"></i>
        Dashboard Analisis Sentimen Opini Publik Terhadap Pilpres
    </h1>
""", unsafe_allow_html=True)

# ============================================================================
# PAGE 1: OVERVIEW
# ============================================================================
if page == "Overview":
    st.markdown("""
        <div class="section-header">
            <i class="fas fa-clipboard-list" style="color: #2563eb;"></i>
            Overview Data
        </div>
    """, unsafe_allow_html=True)
    
    # Metrics dengan icon
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                    <i class="fas fa-database" style="color: #2563eb; font-size: 1.5rem; margin-right: 0.75rem;"></i>
                    <span style="font-weight: 600; color: #64748b;">Total Data Training</span>
                </div>
                <div style="font-size: 2rem; font-weight: 700; color: #1e293b;">{len(df):,}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                    <i class="fas fa-vial" style="color: #10b981; font-size: 1.5rem; margin-right: 0.75rem;"></i>
                    <span style="font-weight: 600; color: #64748b;">Total Data Test</span>
                </div>
                <div style="font-size: 2rem; font-weight: 700; color: #1e293b;">{len(df_test):,}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                    <i class="fas fa-spinner" style="color: #f59e0b; font-size: 1.5rem; margin-right: 0.75rem;"></i>
                    <span style="font-weight: 600; color: #64748b;">Data Setelah Preprocessing</span>
                </div>
                <div style="font-size: 2rem; font-weight: 700; color: #1e293b;">{len(df_processed):,}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
            <div class="metric-card">
                <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                    <i class="fas fa-check-circle" style="color: #8b5cf6; font-size: 1.5rem; margin-right: 0.75rem;"></i>
                    <span style="font-weight: 600; color: #64748b;">Test Setelah Preprocessing</span>
                </div>
                <div style="font-size: 2rem; font-weight: 700; color: #1e293b;">{len(df_test_processed):,}</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Data preview
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <h3>
                <i class="fas fa-table" style="color: #2563eb;"></i>
                Preview Data Training
            </h3>
        """, unsafe_allow_html=True)
        st.dataframe(df.head(10), width='stretch')
    
    with col2:
        st.markdown("""
            <h3>
                <i class="fas fa-table" style="color: #10b981;"></i>
                Preview Data Test
            </h3>
        """, unsafe_allow_html=True)
        st.dataframe(df_test.head(10), width='stretch')
    
    st.divider()
    
    # Statistics
    st.markdown("""
        <div class="section-header">
            <i class="fas fa-chart-pie" style="color: #2563eb;"></i>
            Statistik Data
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class="card-container">
                <h4 style="color: #2563eb; font-weight: 600; margin-bottom: 1rem;">
                    <i class="fas fa-database"></i> Data Training
                </h4>
        """, unsafe_allow_html=True)
        st.markdown(f"""
                <ul style="list-style: none; padding: 0; line-height: 2;">
                    <li><i class="fas fa-list-ol" style="color: #64748b; margin-right: 0.5rem;"></i> Jumlah baris: <strong>{len(df):,}</strong></li>
                    <li><i class="fas fa-columns" style="color: #64748b; margin-right: 0.5rem;"></i> Jumlah kolom: <strong>{len(df.columns)}</strong></li>
                    <li><i class="fas fa-tags" style="color: #64748b; margin-right: 0.5rem;"></i> Kolom: <strong>{', '.join(df.columns.tolist())}</strong></li>
        """, unsafe_allow_html=True)
        if 'Date' in df.columns:
            st.markdown(f"""
                    <li><i class="fas fa-calendar-alt" style="color: #64748b; margin-right: 0.5rem;"></i> Rentang tanggal: <strong>{df['Date'].min()} hingga {df['Date'].max()}</strong></li>
            """, unsafe_allow_html=True)
        st.markdown("</ul></div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="card-container">
                <h4 style="color: #10b981; font-weight: 600; margin-bottom: 1rem;">
                    <i class="fas fa-vial"></i> Data Test
                </h4>
        """, unsafe_allow_html=True)
        st.markdown(f"""
                <ul style="list-style: none; padding: 0; line-height: 2;">
                    <li><i class="fas fa-list-ol" style="color: #64748b; margin-right: 0.5rem;"></i> Jumlah baris: <strong>{len(df_test):,}</strong></li>
                    <li><i class="fas fa-columns" style="color: #64748b; margin-right: 0.5rem;"></i> Jumlah kolom: <strong>{len(df_test.columns)}</strong></li>
                    <li><i class="fas fa-tags" style="color: #64748b; margin-right: 0.5rem;"></i> Kolom: <strong>{', '.join(df_test.columns.tolist())}</strong></li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

# ============================================================================
# PAGE 2: ANALISIS DATA
# ============================================================================
elif page == "Analisis Data":
    st.markdown("""
        <div class="section-header">
            <i class="fas fa-chart-line" style="color: #2563eb;"></i>
            Analisis Data
        </div>
    """, unsafe_allow_html=True)
    
    # Sentiment distribution
    st.markdown("""
        <h3>
            <i class="fas fa-pie-chart" style="color: #2563eb;"></i>
            Distribusi Sentimen
        </h3>
    """, unsafe_allow_html=True)
    
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
        st.markdown("""
            <h3>
                <i class="fas fa-calendar-day" style="color: #2563eb;"></i>
                Jumlah Tweet per Tanggal
            </h3>
        """, unsafe_allow_html=True)
        
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
    st.markdown("""
        <h3>
            <i class="fas fa-ruler-horizontal" style="color: #2563eb;"></i>
            Distribusi Panjang Teks
        </h3>
    """, unsafe_allow_html=True)
    
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
        st.markdown("""
            <h3>
                <i class="fas fa-chart-bar" style="color: #2563eb;"></i>
                Statistik Panjang Teks
            </h3>
        """, unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                        <i class="fas fa-calculator" style="color: #2563eb; font-size: 1.2rem; margin-right: 0.5rem;"></i>
                        <span style="font-weight: 600; color: #64748b; font-size: 0.875rem;">Rata-rata (Training)</span>
                    </div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: #1e293b;">{df_processed['text_len'].mean():.1f} <small style="font-size: 0.875rem; color: #64748b;">kata</small></div>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
                <div class="metric-card">
                    <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                        <i class="fas fa-equals" style="color: #10b981; font-size: 1.2rem; margin-right: 0.5rem;"></i>
                        <span style="font-weight: 600; color: #64748b; font-size: 0.875rem;">Median (Training)</span>
                    </div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: #1e293b;">{df_processed['text_len'].median():.1f} <small style="font-size: 0.875rem; color: #64748b;">kata</small></div>
                </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
                <div class="metric-card">
                    <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                        <i class="fas fa-calculator" style="color: #f59e0b; font-size: 1.2rem; margin-right: 0.5rem;"></i>
                        <span style="font-weight: 600; color: #64748b; font-size: 0.875rem;">Rata-rata (Test)</span>
                    </div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: #1e293b;">{df_test_processed['text_len'].mean():.1f} <small style="font-size: 0.875rem; color: #64748b;">kata</small></div>
                </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
                <div class="metric-card">
                    <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                        <i class="fas fa-equals" style="color: #8b5cf6; font-size: 1.2rem; margin-right: 0.5rem;"></i>
                        <span style="font-weight: 600; color: #64748b; font-size: 0.875rem;">Median (Test)</span>
                    </div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: #1e293b;">{df_test_processed['text_len'].median():.1f} <small style="font-size: 0.875rem; color: #64748b;">kata</small></div>
                </div>
            """, unsafe_allow_html=True)

# ============================================================================
# PAGE 3: ANALISIS SENTIMEN
# ============================================================================
elif page == "Analisis Sentimen":
    st.markdown("""
        <div class="section-header">
            <i class="fas fa-comments" style="color: #2563eb;"></i>
            Analisis Sentimen Detail
        </div>
    """, unsafe_allow_html=True)
    
    # Sentiment comparison
    st.markdown("""
        <h3>
            <i class="fas fa-balance-scale" style="color: #2563eb;"></i>
            Perbandingan Sentimen
        </h3>
    """, unsafe_allow_html=True)
    
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
    st.markdown("""
        <h3>
            <i class="fas fa-file-text" style="color: #2563eb;"></i>
            Contoh Teks per Sentimen
        </h3>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    with col1:
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
            icon_class = {
                'Negative': 'fa-frown',
                'Neutral': 'fa-meh',
                'Positive': 'fa-smile'
            }[sentiment_label]
            icon_color = {
                'Negative': '#ef4444',
                'Neutral': '#f59e0b',
                'Positive': '#10b981'
            }[sentiment_label]
            badge_class = {
                'Negative': 'sentiment-negative',
                'Neutral': 'sentiment-neutral',
                'Positive': 'sentiment-positive'
            }[sentiment_label]
            
            # Create expander with sentiment badge (icons are shown inside)
            with st.expander(f"{sentiment_label} - Teks #{idx}", expanded=False):
                st.markdown(f"""
                    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem; padding: 0.5rem; background: {'#fee2e2' if sentiment_label == 'Negative' else '#fef3c7' if sentiment_label == 'Neutral' else '#d1fae5'}; border-radius: 0.5rem;">
                        <i class="fas {icon_class}" style="color: {icon_color}; font-size: 1.25rem;"></i>
                        <span class="{badge_class}" style="font-weight: 600;">{sentiment_label}</span>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                    <div class="card-container">
                        <div style="margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 1px solid #e2e8f0;">
                            <strong><i class="fas fa-align-left" style="color: #2563eb; margin-right: 0.5rem;"></i>Teks Asli:</strong>
                        </div>
                        <div style="padding: 0.75rem; background: #f8fafc; border-radius: 0.5rem; margin-bottom: 1rem;">
                            {row['Text'] if 'Text' in row else "N/A"}
                        </div>
                """, unsafe_allow_html=True)
                if 'text_clean' in row:
                    st.markdown(f"""
                        <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #e2e8f0;">
                            <strong><i class="fas fa-broom" style="color: #10b981; margin-right: 0.5rem;"></i>Teks Setelah Cleaning:</strong>
                        </div>
                        <div style="padding: 0.75rem; background: #f0fdf4; border-radius: 0.5rem; margin-top: 0.5rem;">
                            {row['text_clean']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("</div>", unsafe_allow_html=True)
    
    st.divider()
    
    # Word Cloud (if text_clean exists)
    if 'text_clean' in df_processed.columns:
        st.markdown("""
            <h3>
                <i class="fas fa-cloud" style="color: #2563eb;"></i>
                Word Cloud
            </h3>
        """, unsafe_allow_html=True)
        
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
# PAGE 4: VISUALISASI DETAIL
# ============================================================================
elif page == "Visualisasi Detail":
    st.markdown("""
        <div class="section-header">
            <i class="fas fa-chart-bar" style="color: #2563eb;"></i>
            Visualisasi Detail
        </div>
    """, unsafe_allow_html=True)
    
    # Interactive filters
    st.markdown("""
        <h3>
            <i class="fas fa-filter" style="color: #2563eb;"></i>
            Filter Data
        </h3>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div style="margin-bottom: 1rem;">
                <label style="display: flex; align-items: center; font-weight: 600; color: #1e293b; margin-bottom: 0.5rem;">
                    <i class="fas fa-arrow-down" style="color: #2563eb; margin-right: 0.5rem;"></i>
                    Panjang Teks Minimum
                </label>
            </div>
        """, unsafe_allow_html=True)
        min_length = st.slider(
            "",
            min_value=1,
            max_value=50,
            value=1,
            key="min_len",
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("""
            <div style="margin-bottom: 1rem;">
                <label style="display: flex; align-items: center; font-weight: 600; color: #1e293b; margin-bottom: 0.5rem;">
                    <i class="fas fa-arrow-up" style="color: #2563eb; margin-right: 0.5rem;"></i>
                    Panjang Teks Maksimum
                </label>
            </div>
        """, unsafe_allow_html=True)
        max_length = st.slider(
            "",
            min_value=1,
            max_value=200,
            value=200,
            key="max_len",
            label_visibility="collapsed"
        )
    
    # Filter data
    filtered_df = df_processed[
        (df_processed['text_len'] >= min_length) & 
        (df_processed['text_len'] <= max_length)
    ]
    
    st.markdown(f"""
        <div class="metric-card" style="margin-top: 1rem;">
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <i class="fas fa-database" style="color: #2563eb; font-size: 1.5rem; margin-right: 0.75rem;"></i>
                <span style="font-weight: 600; color: #64748b;">Jumlah Data Setelah Filter</span>
            </div>
            <div style="font-size: 2rem; font-weight: 700; color: #1e293b;">{len(filtered_df):,}</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Detailed charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <h3>
                <i class="fas fa-chart-box" style="color: #2563eb;"></i>
                Box Plot Panjang Teks
            </h3>
        """, unsafe_allow_html=True)
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
        st.markdown("""
            <h3>
                <i class="fas fa-guitar" style="color: #2563eb;"></i>
                Violin Plot
            </h3>
        """, unsafe_allow_html=True)
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
    st.markdown("""
        <h3>
            <i class="fas fa-table" style="color: #2563eb;"></i>
            Data Detail
        </h3>
    """, unsafe_allow_html=True)
    st.dataframe(
        filtered_df[['Text', 'text_clean', 'text_len', 'Sentiment']].head(100),
        width='stretch'
    )

# Footer
st.divider()
st.markdown("""
    <div class="footer">
        <div style="margin-bottom: 0.5rem;">
            <i class="fas fa-chart-bar" style="color: #2563eb; margin-right: 0.5rem;"></i>
            <strong>Dashboard Analisis Sentimen Opini Publik Terhadap Pilpres</strong>
        </div>
        <div style="color: #64748b; font-size: 0.875rem;">
            <i class="fab fa-python" style="margin-right: 0.5rem;"></i>
            Dibuat dengan Streamlit
        </div>
    </div>
""", unsafe_allow_html=True)

