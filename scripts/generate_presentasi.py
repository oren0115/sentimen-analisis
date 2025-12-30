# -*- coding: utf-8 -*-
"""
Script untuk mengumpulkan hasil analisis dan mengisi template presentasi
Jalankan setelah menjalankan main.py untuk mendapatkan hasil analisis
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Get project root directory (parent of scripts folder)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from src.config import DATA_PATH, DATA_TEST_SIZE, SENTIMENT_LABELS
from src.data_processing import (
    load_data,
    split_data,
    preprocess_data,
    process_token_lengths,
    encode_sentiments,
    prepare_data_for_training
)
from src.models import train_naive_bayes
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


def collect_results():
    """Collect all analysis results."""
    print("=" * 60)
    print("MENGUMPULKAN HASIL ANALISIS DATA")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data...")
    df = load_data(DATA_PATH)
    print(f"   ✓ Total data: {len(df):,} rows")
    
    # Split data into train and test sets
    print("\n2. Splitting data into train and test sets...")
    df_train, df_test = split_data(df, test_size=DATA_TEST_SIZE)
    print(f"   ✓ Training data: {len(df_train):,} rows")
    print(f"   ✓ Test data: {len(df_test):,} rows")
    
    # Preprocess
    print("\n3. Preprocessing data...")
    df_processed, df_test_processed = preprocess_data(df_train.copy(), df_test.copy())
    print(f"   ✓ Training setelah preprocessing: {len(df_processed):,} rows")
    print(f"   ✓ Test setelah preprocessing: {len(df_test_processed):,} rows")
    
    # Process token lengths
    print("\n4. Processing token lengths...")
    df_processed, df_test_processed = process_token_lengths(df_processed, df_test_processed)
    print(f"   ✓ Training setelah filtering: {len(df_processed):,} rows")
    print(f"   ✓ Test setelah filtering: {len(df_test_processed):,} rows")
    
    # Encode sentiments
    df_processed, df_test_processed = encode_sentiments(df_processed, df_test_processed)
    
    # Get sentiment distribution before oversampling
    sentiment_dist_before = df_processed['Sentiment'].value_counts().sort_index()
    
    # Prepare data for training
    print("\n4. Preparing data for training...")
    (X_train, X_valid, X_test, y_train, y_valid, y_test,
     y_train_le, y_valid_le, y_test_le) = prepare_data_for_training(df_processed, df_test_processed)
    
    # Get sentiment distribution after oversampling
    sentiment_dist_after = pd.Series(y_train_le).value_counts().sort_index()
    
    # Train model
    print("\n5. Training model...")
    nb_model, vectorizer, tf_transformer, metrics = train_naive_bayes(
        X_train, y_train_le, X_test, y_test_le
    )
    
    # Use metrics from training function
    accuracy = metrics['accuracy']
    report = metrics['classification_report']
    cm = confusion_matrix(y_test_le, nb_model.predict(
        tf_transformer.transform(vectorizer.transform(X_test))
    ))
    
    # Calculate text length statistics
    text_len_stats = {
        'mean': df_processed['text_len'].mean(),
        'median': df_processed['text_len'].median(),
        'min': df_processed['text_len'].min(),
        'max': df_processed['text_len'].max()
    }
    
    text_len_stats_test = {
        'mean': df_test_processed['text_len'].mean(),
        'median': df_test_processed['text_len'].median(),
        'min': df_test_processed['text_len'].min(),
        'max': df_test_processed['text_len'].max()
    }
    
    # Compile results
    results = {
        'dataset': {
            'train_initial': len(df_train),
            'test_initial': len(df_test),
            'train_after_preprocessing': len(df_processed),
            'test_after_preprocessing': len(df_test_processed),
            'train_after_oversampling': len(X_train),
            'validation': len(X_valid),
            'test': len(X_test)
        },
        'sentiment_dist_before': {
            label: int(sentiment_dist_before.get(i, 0)) 
            for i, label in enumerate(SENTIMENT_LABELS)
        },
        'sentiment_dist_after': {
            label: int(sentiment_dist_after.get(i, 0)) 
            for i, label in enumerate(SENTIMENT_LABELS)
        },
        'text_length_stats': text_len_stats,
        'text_length_stats_test': text_len_stats_test,
        'model_performance': {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }
    }
    
    return results


def format_results_for_presentation(results):
    """Format results for presentation."""
    output = []
    output.append("=" * 60)
    output.append("HASIL ANALISIS DATA UNTUK PRESENTASI")
    output.append("=" * 60)
    output.append("")
    
    # Dataset statistics
    output.append("## STATISTIK DATASET")
    output.append("")
    output.append("| Metrik | Training Data | Test Data |")
    output.append("|--------|--------------|-----------|")
    output.append(f"| **Jumlah Data Awal** | {results['dataset']['train_initial']:,} | {results['dataset']['test_initial']:,} |")
    output.append(f"| **Setelah Preprocessing** | {results['dataset']['train_after_preprocessing']:,} | {results['dataset']['test_after_preprocessing']:,} |")
    output.append(f"| **Setelah Oversampling** | {results['dataset']['train_after_oversampling']:,} | - |")
    output.append(f"| **Validation Data** | {results['dataset']['validation']:,} | - |")
    output.append(f"| **Test Data** | - | {results['dataset']['test']:,} |")
    output.append("")
    
    # Sentiment distribution before
    output.append("## DISTRIBUSI SENTIMEN (SEBELUM OVERSAMPLING)")
    total_before = sum(results['sentiment_dist_before'].values())
    for label in SENTIMENT_LABELS:
        count = results['sentiment_dist_before'][label]
        percentage = (count / total_before * 100) if total_before > 0 else 0
        output.append(f"- **{label}**: {percentage:.1f}% ({count:,} data)")
    output.append("")
    
    # Sentiment distribution after
    output.append("## DISTRIBUSI SENTIMEN (SETELAH OVERSAMPLING)")
    total_after = sum(results['sentiment_dist_after'].values())
    for label in SENTIMENT_LABELS:
        count = results['sentiment_dist_after'][label]
        percentage = (count / total_after * 100) if total_after > 0 else 0
        output.append(f"- **{label}**: {percentage:.1f}% ({count:,} data)")
    output.append("")
    
    # Text length statistics
    output.append("## STATISTIK PANJANG TEKS")
    output.append("")
    output.append("### Training Data:")
    output.append(f"- Rata-rata: {results['text_length_stats']['mean']:.1f} kata")
    output.append(f"- Median: {results['text_length_stats']['median']:.1f} kata")
    output.append(f"- Min: {results['text_length_stats']['min']} kata")
    output.append(f"- Max: {results['text_length_stats']['max']} kata")
    output.append("")
    output.append("### Test Data:")
    output.append(f"- Rata-rata: {results['text_length_stats_test']['mean']:.1f} kata")
    output.append(f"- Median: {results['text_length_stats_test']['median']:.1f} kata")
    output.append(f"- Min: {results['text_length_stats_test']['min']} kata")
    output.append(f"- Max: {results['text_length_stats_test']['max']} kata")
    output.append("")
    
    # Model performance
    output.append("## PERFORMANSI MODEL")
    output.append("")
    output.append(f"### Akurasi Keseluruhan: {results['model_performance']['accuracy']*100:.2f}%")
    output.append("")
    output.append("| Metrik | Negative | Neutral | Positive | Average |")
    output.append("|--------|----------|---------|----------|---------|")
    
    report = results['model_performance']['classification_report']
    metrics = ['precision', 'recall', 'f1-score']
    
    for metric in metrics:
        row = [metric.capitalize()]
        for label in SENTIMENT_LABELS:
            value = report[label][metric] * 100
            row.append(f"{value:.2f}%")
        avg_value = report['macro avg'][metric] * 100
        row.append(f"{avg_value:.2f}%")
        output.append("| " + " | ".join(row) + " |")
    
    # Support row
    row = ["Support"]
    for label in SENTIMENT_LABELS:
        row.append(str(int(report[label]['support'])))
    row.append(str(int(report['macro avg']['support'])))
    output.append("| " + " | ".join(row) + " |")
    output.append("")
    
    # Confusion matrix
    output.append("## CONFUSION MATRIX")
    output.append("")
    cm = results['model_performance']['confusion_matrix']
    output.append("```")
    output.append("                Predicted")
    output.append("            Neg    Neu    Pos")
    output.append(f"Actual Neg  {cm[0][0]:4d}    {cm[0][1]:4d}    {cm[0][2]:4d}")
    output.append(f"       Neu  {cm[1][0]:4d}    {cm[1][1]:4d}    {cm[1][2]:4d}")
    output.append(f"       Pos  {cm[2][0]:4d}    {cm[2][1]:4d}    {cm[2][2]:4d}")
    output.append("```")
    output.append("")
    
    return "\n".join(output)


def main():
    """Main function."""
    try:
        # Collect results
        results = collect_results()
        
        # Format for presentation
        formatted_output = format_results_for_presentation(results)
        
        # Save to file in outputs directory
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        outputs_dir = os.path.join(PROJECT_ROOT, 'outputs')
        os.makedirs(outputs_dir, exist_ok=True)
        
        output_file = Path(outputs_dir) / "HASIL_ANALISIS.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(formatted_output)
        
        print("\n" + "=" * 60)
        print("HASIL ANALISIS TELAH DISIMPAN")
        print("=" * 60)
        print(f"\nFile: {output_file.absolute()}")
        print("\nGunakan hasil ini untuk mengisi template presentasi!")
        print("\nHasil:")
        print(formatted_output)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

