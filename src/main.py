# -*- coding: utf-8 -*-
"""
Main execution file for sentiment analysis
Analisis Sentimen Opini Publik Terhadap Pilpres

CATATAN: Untuk menjalankan program, gunakan main.py di root directory:
    python main.py
    
Atau jika ingin menjalankan file ini langsung, pastikan dijalankan dari root:
    python -m src.main
"""

import sys
import os
import pandas as pd

# Handle both relative and absolute imports
try:
    # Try relative imports first (when run as module)
    from .config import DATA_PATH, DATA_TEST_SIZE
    from .data_processing import (
        load_data,
        split_data,
        preprocess_data,
        process_token_lengths,
        encode_sentiments,
        prepare_data_for_training
    )
    from .visualization import plot_tweets_by_date, plot_text_length_distribution
    from .models import train_naive_bayes
except ImportError:
    # Fall back to absolute imports (when run directly)
    # Add parent directory to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    from src.config import DATA_PATH, DATA_TEST_SIZE
    from src.data_processing import (
        load_data,
        split_data,
        preprocess_data,
        process_token_lengths,
        encode_sentiments,
        prepare_data_for_training
    )
    from src.visualization import plot_tweets_by_date, plot_text_length_distribution
    from src.models import train_naive_bayes


def main():
    """Main execution function."""
    # Load data
    print("Loading data...")
    df = load_data(DATA_PATH)
    print(f"Total data shape: {df.shape}")
    
    # Display basic info
    print("\nFirst few rows of data:")
    print(df.head())
    print("\nDataFrame info:")
    df.info()
    
    # Display sentiment distribution before splitting
    print("\nSentiment distribution (before splitting):")
    print(df['Sentiment'].value_counts())

    # Split data into training and testing sets
    print("\n" + "="*50)
    print("Splitting data into training and testing sets...")
    print("="*50)
    df_train, df_test = split_data(df, test_size=DATA_TEST_SIZE)

    # Plot tweets by date (if Date column exists)
    if 'Date' in df_train.columns:
        df_train['Date'] = pd.to_datetime(df_train['Date'])
        plot_tweets_by_date(df_train)

    # Preprocess data
    df_train, df_test = preprocess_data(df_train, df_test)

    # Visualize text length distribution
    plot_text_length_distribution(df_train, 'Training tweets with less than 10 words')
    plot_text_length_distribution(df_test, 'Test tweets with less than 10 words')

    # Process token lengths
    df_train, df_test = process_token_lengths(df_train, df_test)

    # Encode sentiments
    df_train, df_test = encode_sentiments(df_train, df_test)
    print("\nTraining set sentiment distribution (after encoding):")
    print(df_train['Sentiment'].value_counts())

    # Prepare data for training
    (X_train, X_valid, X_test, y_train, y_valid, y_test,
     y_train_le, y_valid_le, y_test_le) = prepare_data_for_training(df_train, df_test)

    # Train Naive Bayes
    nb_model, vectorizer, tf_transformer, metrics = train_naive_bayes(
        X_train, y_train_le, X_test, y_test_le
    )

    # Display summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\n✅ Model Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"\n📊 Best Performing Class:")
    best_class = max(metrics['f1_score']['per_class'].items(), key=lambda x: x[1])
    print(f"   {best_class[0]}: F1-Score = {best_class[1]*100:.2f}%")
    print(f"\n📊 Worst Performing Class:")
    worst_class = min(metrics['f1_score']['per_class'].items(), key=lambda x: x[1])
    print(f"   {worst_class[0]}: F1-Score = {worst_class[1]*100:.2f}%")
    
    print("\n" + "="*60)
    print("Processing complete!")
    print("="*60)


if __name__ == "__main__":
    main()

