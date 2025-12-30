# -*- coding: utf-8 -*-
"""
Main entry point for sentiment analysis
Analisis Sentimen Opini Publik Terhadap Pilpres

Run this file from the project root directory:
    python scripts/main.py
    atau
    cd scripts && python main.py
"""

import sys
import os

# Get project root directory (parent of scripts folder)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

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
import pandas as pd


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

