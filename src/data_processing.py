# -*- coding: utf-8 -*-
"""
Data processing module
Contains functions for loading, preprocessing, and preparing data
"""

import numpy as np
import pandas as pd
from typing import Tuple

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from transformers import BertTokenizerFast

from .config import SEED, MAX_TOKEN_LENGTH, MIN_TEXT_LENGTH, TEST_SIZE, SENTIMENT_MAP
from .text_cleaning import clean_texts


def load_data(data_path: str) -> pd.DataFrame:
    """Load data from CSV file."""
    df = pd.read_csv(data_path, encoding='ISO-8859-1')
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = SEED) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and testing sets with stratification.
    
    Args:
        df: DataFrame containing the data
        test_size: Proportion of data to use for testing (default: 0.2)
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (train_df, test_df)
    """
    # Check if Sentiment column exists
    if 'Sentiment' not in df.columns:
        raise ValueError("DataFrame must contain 'Sentiment' column for stratified splitting")
    
    # Split data with stratification to maintain class distribution
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df['Sentiment'],
        random_state=random_state
    )
    
    # Reset indices
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    print(f"Data split completed:")
    print(f"  Training set: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Testing set: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    print(f"\nTraining set sentiment distribution:")
    print(train_df['Sentiment'].value_counts())
    print(f"\nTesting set sentiment distribution:")
    print(test_df['Sentiment'].value_counts())
    
    return train_df, test_df


def preprocess_data(df: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Preprocess data: clean, remove duplicates, filter by length."""
    # Select relevant columns
    df = df[['Text', 'Sentiment']].copy()
    df_test = df_test[['Text', 'Sentiment']].copy()

    # Remove duplicates
    df.drop_duplicates(subset='Text', inplace=True)

    # Clean texts
    print("Cleaning training texts...")
    df['text_clean'] = clean_texts(df['Text'].tolist())
    
    print("Cleaning test texts...")
    df_test['text_clean'] = clean_texts(df_test['Text'].tolist())

    # Calculate text lengths
    df['text_len'] = df['text_clean'].apply(lambda x: len(x.split()))
    df_test['text_len'] = df_test['text_clean'].apply(lambda x: len(x.split()))

    # Filter by minimum text length
    print(f"Before filtering: Train={df.shape[0]}, Test={df_test.shape[0]}")
    df = df[df['text_len'] > MIN_TEXT_LENGTH].copy()
    df_test = df_test[df_test['text_len'] > MIN_TEXT_LENGTH].copy()
    print(f"After filtering: Train={df.shape[0]}, Test={df_test.shape[0]}")

    return df, df_test


def process_token_lengths(df: pd.DataFrame, df_test: pd.DataFrame, 
                         max_length: int = MAX_TOKEN_LENGTH) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Process and filter data based on token lengths."""
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # Process training data
    print("Processing training token lengths...")
    token_lens = []
    for txt in df['text_clean'].values:
        tokens = tokenizer.encode(txt, max_length=512, truncation=True)
        token_lens.append(len(tokens))

    df['token_lens'] = token_lens
    max_len = np.max(token_lens)
    print(f"MAX TOKENIZED SENTENCE LENGTH (train): {max_len}")

    # Find and remove outliers
    df = df.sort_values(by='token_lens', ascending=False)
    outliers = df[df['token_lens'] > max_length]
    if len(outliers) > 0:
        print(f"Removing {len(outliers)} outliers from training data")
        df = df[df['token_lens'] <= max_length].copy()
    
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Process test data
    print("Processing test token lengths...")
    token_lens_test = []
    for txt in df_test['text_clean'].values:
        tokens = tokenizer.encode(txt, max_length=512, truncation=True)
        token_lens_test.append(len(tokens))

    df_test['token_lens'] = token_lens_test
    max_len_test = np.max(token_lens_test)
    print(f"MAX TOKENIZED SENTENCE LENGTH (test): {max_len_test}")

    # Find and remove outliers
    df_test = df_test.sort_values(by='token_lens', ascending=False)
    outliers_test = df_test[df_test['token_lens'] > max_length]
    if len(outliers_test) > 0:
        print(f"Removing {len(outliers_test)} outliers from test data")
        df_test = df_test[df_test['token_lens'] <= max_length].copy()
    
    df_test = df_test.sample(frac=1, random_state=SEED).reset_index(drop=True)

    return df, df_test


def encode_sentiments(df: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Encode sentiment labels to numeric values."""
    df['Sentiment'] = df['Sentiment'].map(SENTIMENT_MAP)
    df_test['Sentiment'] = df_test['Sentiment'].map(SENTIMENT_MAP)
    return df, df_test


def prepare_data_for_training(df: pd.DataFrame, df_test: pd.DataFrame) -> Tuple:
    """Prepare data for training with oversampling and train/validation split."""
    # Oversample training data
    ros = RandomOverSampler(random_state=SEED)
    train_x, train_y = ros.fit_resample(
        np.array(df['text_clean']).reshape(-1, 1),
        np.array(df['Sentiment']).reshape(-1, 1)
    )
    train_os = pd.DataFrame(
        list(zip([x[0] for x in train_x], train_y)),
        columns=['text_clean', 'Sentiment']
    )

    # Split into train and validation
    X = train_os['text_clean'].values
    y = train_os['Sentiment'].values
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )

    # Prepare test data
    X_test = df_test['text_clean'].values
    y_test = df_test['Sentiment'].values

    # Keep label encoded versions
    y_train_le = y_train.copy()
    y_valid_le = y_valid.copy()
    y_test_le = y_test.copy()

    # One-hot encode
    ohe = preprocessing.OneHotEncoder()
    y_train = ohe.fit_transform(np.array(y_train).reshape(-1, 1)).toarray()
    y_valid = ohe.fit_transform(np.array(y_valid).reshape(-1, 1)).toarray()
    y_test = ohe.fit_transform(np.array(y_test).reshape(-1, 1)).toarray()

    print(f"TRAINING DATA: {X_train.shape[0]}")
    print(f"VALIDATION DATA: {X_valid.shape[0]}")
    print(f"TESTING DATA: {X_test.shape[0]}")

    return (X_train, X_valid, X_test, y_train, y_valid, y_test,
            y_train_le, y_valid_le, y_test_le)

