# -*- coding: utf-8 -*-
"""
Models module
Contains functions for training and evaluating machine learning models
"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from .config import SENTIMENT_LABELS
from .visualization import plot_confusion_matrix


def train_naive_bayes(X_train, y_train_le, X_test, y_test_le):
    """
    Train and evaluate Naive Bayes classifier.
    
    Returns:
        tuple: (model, vectorizer, tf_transformer, metrics_dict)
        metrics_dict contains: accuracy, precision, recall, f1, classification_report, confusion_matrix
    """
    print("\n" + "="*60)
    print("TRAINING NAIVE BAYES CLASSIFIER")
    print("="*60)
    
    # Vectorize
    print("\n1. Vectorizing text data...")
    clf = CountVectorizer()
    X_train_cv = clf.fit_transform(X_train)
    X_test_cv = clf.transform(X_test)
    print(f"   ✓ Training features: {X_train_cv.shape}")
    print(f"   ✓ Test features: {X_test_cv.shape}")

    # TF-IDF transformation
    print("\n2. Applying TF-IDF transformation...")
    tf_transformer = TfidfTransformer(use_idf=True).fit(X_train_cv)
    X_train_tf = tf_transformer.transform(X_train_cv)
    X_test_tf = tf_transformer.transform(X_test_cv)
    print("   ✓ TF-IDF transformation completed")

    # Train model
    print("\n3. Training Naive Bayes model...")
    nb_clf = MultinomialNB()
    nb_clf.fit(X_train_tf, y_train_le)
    print("   ✓ Model training completed")

    # Predict
    print("\n4. Making predictions on test set...")
    nb_pred = nb_clf.predict(X_test_tf)
    print(f"   ✓ Predicted {len(nb_pred)} samples")

    # Calculate metrics
    print("\n5. Calculating evaluation metrics...")
    accuracy = accuracy_score(y_test_le, nb_pred)
    precision = precision_score(y_test_le, nb_pred, average=None, zero_division=0)
    recall = recall_score(y_test_le, nb_pred, average=None, zero_division=0)
    f1 = f1_score(y_test_le, nb_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_test_le, nb_pred)
    
    # Macro averages
    precision_macro = precision_score(y_test_le, nb_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_test_le, nb_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test_le, nb_pred, average='macro', zero_division=0)
    
    # Weighted averages
    precision_weighted = precision_score(y_test_le, nb_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_test_le, nb_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_test_le, nb_pred, average='weighted', zero_division=0)

    # Classification report
    report = classification_report(
        y_test_le,
        nb_pred,
        target_names=SENTIMENT_LABELS,
        output_dict=True
    )

    # Display results
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    
    print(f"\n📊 OVERALL ACCURACY: {accuracy*100:.2f}%")
    
    print("\n📈 METRICS PER CLASS:")
    print("-" * 60)
    print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 60)
    for i, label in enumerate(SENTIMENT_LABELS):
        support = cm[i].sum()
        print(f"{label:<12} {precision[i]*100:>10.2f}% {recall[i]*100:>10.2f}% {f1[i]*100:>10.2f}% {support:>10}")
    
    print("-" * 60)
    print(f"{'Macro Avg':<12} {precision_macro*100:>10.2f}% {recall_macro*100:>10.2f}% {f1_macro*100:>10.2f}%")
    print(f"{'Weighted Avg':<12} {precision_weighted*100:>10.2f}% {recall_weighted*100:>10.2f}% {f1_weighted*100:>10.2f}%")
    
    print("\n📋 DETAILED CLASSIFICATION REPORT:")
    print("-" * 60)
    print(classification_report(
        y_test_le,
        nb_pred,
        target_names=SENTIMENT_LABELS
    ))

    # Plot confusion matrix
    print("\n📊 Generating confusion matrix visualization...")
    plot_confusion_matrix(y_test_le, nb_pred, "Naive Bayes Confusion Matrix")
    
    # Compile metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': {
            'per_class': {label: float(precision[i]) for i, label in enumerate(SENTIMENT_LABELS)},
            'macro': float(precision_macro),
            'weighted': float(precision_weighted)
        },
        'recall': {
            'per_class': {label: float(recall[i]) for i, label in enumerate(SENTIMENT_LABELS)},
            'macro': float(recall_macro),
            'weighted': float(recall_weighted)
        },
        'f1_score': {
            'per_class': {label: float(f1[i]) for i, label in enumerate(SENTIMENT_LABELS)},
            'macro': float(f1_macro),
            'weighted': float(f1_weighted)
        },
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)

    return nb_clf, clf, tf_transformer, metrics

