# -*- coding: utf-8 -*-
"""
Text cleaning module
Contains functions for cleaning and preprocessing text data
"""

import re
import string
from typing import List


def strip_emoji(text: str) -> str:
    """Remove emojis from text."""
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # simbol & piktogram
                               u"\U0001F680-\U0001F6FF"  # transportasi & simbol map
                               u"\U0001F1E0-\U0001F1FF"  # bendera (iOS)
                               u"\U00002500-\U00002BEF"  # CJK Ext A
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u200d"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def strip_all_entities(text: str) -> str:
    """Remove punctuations, links, mentions and newline characters."""
    text = text.replace('\r', '').replace('\n', ' ').lower()
    text = re.sub(r"(?:\@|https?\://)\S+", "", text)  # remove links and mentions
    text = re.sub(r'[^\x00-\x7f]', r'', text)  # remove non utf8/ascii characters
    banned_list = string.punctuation + 'Ã' + '±' + 'ã' + '¼' + 'â' + '»' + '§'
    table = str.maketrans('', '', banned_list)
    text = text.translate(table)
    return text


def clean_hashtags(tweet: str) -> str:
    """Clean hashtags at the end of sentence, keep those in middle by removing # symbol."""
    new_tweet = " ".join(word.strip() for word in re.split(
        '#(?!(?:hashtag)\b)[\w-]+(?=(?:\s+#[\w-]+)*\s*$)', tweet))
    new_tweet2 = " ".join(word.strip() for word in re.split('#|_', new_tweet))
    return new_tweet2


def filter_chars(text: str) -> str:
    """Filter special characters such as & and $ present in some words."""
    sent = []
    for word in text.split(' '):
        if ('$' in word) | ('&' in word):
            sent.append('')
        else:
            sent.append(word)
    return ' '.join(sent)


def remove_mult_spaces(text: str) -> str:
    """Remove multiple spaces."""
    return re.sub("\s\s+", " ", text)


def clean_text(text: str) -> str:
    """Apply all cleaning functions to text."""
    cleaned = strip_emoji(text)
    cleaned = strip_all_entities(cleaned)
    cleaned = clean_hashtags(cleaned)
    cleaned = filter_chars(cleaned)
    cleaned = remove_mult_spaces(cleaned)
    return cleaned


def clean_texts(texts: List[str]) -> List[str]:
    """Clean a list of texts."""
    return [clean_text(text) for text in texts]

