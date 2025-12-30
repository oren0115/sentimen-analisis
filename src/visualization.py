# -*- coding: utf-8 -*-
"""
Visualization module
Contains functions for plotting and data visualization
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, title: str = "Confusion Matrix"):
    """Plot confusion matrix."""
    fig, ax = plt.subplots(figsize=(5, 5))
    labels = ['Negative', 'Neutral', 'Positive']
    ax = sns.heatmap(
        confusion_matrix(y_true, y_pred),
        annot=True,
        cmap="Blues",
        fmt='g',
        cbar=False,
        annot_kws={"size": 25}
    )
    plt.title(title, fontsize=20)
    ax.xaxis.set_ticklabels(labels, fontsize=17)
    ax.yaxis.set_ticklabels(labels, fontsize=17)
    ax.set_ylabel('Test', fontsize=20)
    ax.set_xlabel('Predicted', fontsize=20)
    plt.show()


def plot_tweets_by_date(df: pd.DataFrame):
    """Plot tweets count by date."""
    df['Date'] = pd.to_datetime(df['Date'])
    tweets_per_date = df['Date'].dt.strftime('%Y-%m-%d').value_counts().sort_index().reset_index()
    tweets_per_date.columns = ['Date', 'counts']

    plt.figure(figsize=(20, 5))
    ax = sns.barplot(
        x='Date',
        y='counts',
        data=tweets_per_date,
        edgecolor='black',
        ci=False,
        palette='Blues_r'
    )
    plt.title('Tweets count by date')
    plt.yticks([])
    ax.bar_label(ax.containers[0])
    plt.ylabel('Count')
    plt.xlabel('Date')
    plt.xticks(rotation=90)
    plt.show()


def plot_text_length_distribution(df: pd.DataFrame, title: str = "Text Length Distribution"):
    """Plot distribution of text lengths less than 10 words."""
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(
        x='text_len',
        data=df[df['text_len'] < 10],
        palette='mako'
    )
    plt.title(title)
    plt.yticks([])
    ax.bar_label(ax.containers[0])
    plt.ylabel('count')
    plt.xlabel('')
    plt.show()

