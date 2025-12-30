# -*- coding: utf-8 -*-
"""
Models module
Contains functions for training and evaluating machine learning models
"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

from .config import SENTIMENT_LABELS
from .visualization import plot_confusion_matrix


def train_naive_bayes(X_train, y_train_le, X_test, y_test_le):
    """Train and evaluate Naive Bayes classifier."""
    print("\nTraining Naive Bayes Classifier...")
    
    # Vectorize
    clf = CountVectorizer()
    X_train_cv = clf.fit_transform(X_train)
    X_test_cv = clf.transform(X_test)

    # TF-IDF transformation
    tf_transformer = TfidfTransformer(use_idf=True).fit(X_train_cv)
    X_train_tf = tf_transformer.transform(X_train_cv)
    X_test_tf = tf_transformer.transform(X_test_cv)

    # Train model
    nb_clf = MultinomialNB()
    nb_clf.fit(X_train_tf, y_train_le)

    # Predict
    nb_pred = nb_clf.predict(X_test_tf)

    # Evaluate
    print('\nClassification Report for Naive Bayes:\n')
    print(classification_report(
        y_test_le,
        nb_pred,
        target_names=SENTIMENT_LABELS
    ))

    # Plot confusion matrix
    plot_confusion_matrix(y_test_le, nb_pred, "Naive Bayes Confusion Matrix")

    return nb_clf, clf, tf_transformer

