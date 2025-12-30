# -*- coding: utf-8 -*-
"""
Configuration file for sentiment analysis project
Contains all constants and configuration settings
"""

import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
SEED = 42
MAX_TOKEN_LENGTH = 80
MIN_TEXT_LENGTH = 4
TEST_SIZE = 0.1  # For train/validation split

# ============================================================================
# SENTIMENT MAPPING
# ============================================================================
SENTIMENT_MAP = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
SENTIMENT_LABELS = ['Negative', 'Neutral', 'Positive']

# ============================================================================
# FILE PATHS CONFIGURATION
# ============================================================================
# Set USE_COLAB to False if not using Google Colab
USE_COLAB = True

import os

# Get the project root directory (parent of src folder)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# Test size for splitting data into train and test sets
DATA_TEST_SIZE = 0.2  # 20% for testing, 80% for training

if USE_COLAB:
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        DATA_PATH = '/content/drive/My Drive/tugas/Sentiment1.csv'
    except ImportError:
        print("Warning: google.colab not available, using local paths")
        DATA_PATH = os.path.join(DATA_DIR, 'Sentiment1.csv')
else:
    DATA_PATH = os.path.join(DATA_DIR, 'Sentiment1.csv')

# Keep TRAIN_PATH and TEST_PATH for backward compatibility (will be set after splitting)
TRAIN_PATH = None
TEST_PATH = None

# ============================================================================
# VISUALIZATION STYLE CONFIGURATION
# ============================================================================
def setup_plot_style():
    """Configure matplotlib and seaborn styles."""
    sns.set_style("whitegrid")
    plt.rc("figure", autolayout=True)
    plt.rc("axes", labelweight="bold", labelsize="large", titleweight="bold", titlepad=10)

# Initialize plot style
setup_plot_style()

