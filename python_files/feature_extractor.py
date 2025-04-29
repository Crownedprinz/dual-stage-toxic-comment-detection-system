import numpy as np
import string
import re
from typing import List, Dict, Union
import spacy

class FeatureExtractor:
    def __init__(self, glove_model):
        self.glove = glove_model
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])
        
    def clean_text(self, text: str) -> str:
        """Clean text using same preprocessing as training"""
        pattern = f"[{re.escape(string.punctuation)}]"
        return text.lower().replace(pattern, '', regex=True)
    
    def get_engineered_features(self, text: str) -> np.ndarray:
        """Extract the 8 engineered features used in training"""
        # Word count
        words = str(text).split()
        word_count = len(words)
        
        # Character count
        char_count = len(str(text))
        
        # Average word length
        avg_word_length = char_count / max(word_count, 1)
        
        # Punctuation features
        punct_count = sum(1 for char in str(text) if char in string.punctuation)
        punct_percent = punct_count * 100 / max(word_count, 1)
        
        # Case features
        uppercase_count = sum(1 for word in words if word.isupper())
        titlecase_count = sum(1 for word in words if word.istitle())
        
        # Unique words
        unique_words = len(set(words))
        word_unique_percent = unique_words * 100 / max(word_count, 1)
        
        return np.array([
            word_count,
            char_count,
            avg_word_length,
            punct_count,
            uppercase_count,
            titlecase_count,
            word_unique_percent,
            punct_percent
        ])
    
    def get_glove_embedding(self, text: str) -> np.ndarray:
        """Get averaged GloVe embedding for text"""
        words = [w for w in str(text).split() if w in self.glove]
        if not words:
            return np.zeros(300)
        return np.mean([self.glove[w] for w in words], axis=0)
    
    def get_combined_features(self, text: str) -> np.ndarray:
        """Get combined 308-dim feature vector"""
        glove_features = self.get_glove_embedding(text)
        engineered_features = self.get_engineered_features(text)
        return np.concatenate([glove_features, engineered_features])