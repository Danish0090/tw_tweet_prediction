import pandas as pd
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

class DataLoader:
    def __init__(self, filepath="data/Tweets.csv", is_train=True):
        self.filepath = filepath
        if is_train:
            self.load_data()
        self.vectorizer = None
        self.encoder = None

    def load_data(self):
        """Loads data from a CSV file."""
        self.data = pd.read_csv(self.filepath)  #<--Fix this
        return self.data                        #<--Fix this

    @staticmethod
    def remove_characters(text: str) -> str:
        """Remove non-letters from a given string"""
        remove_chars = string.punctuation
        translator = str.maketrans('', '', remove_chars)
        return text.translate(translator)

    def clean_text(self, text: str) -> str:
        """Keep only retain words in a given string"""
        text = self.remove_characters(text)
        return text.strip()

    def vectorize_text(self, tweets: list[str], fit=True):
        if fit or self.vectorizer is None:                                                 #<--Fix this
            self.vectorizer = TfidfVectorizer(max_features=2500, min_df=1, max_df=0.8)
            return self.vectorizer.fit_transform(tweets).toarray()
        else:
            if self.vectorizer is None:
                raise ValueError("Vectorizer is not fitted. Call with fit=True first!!")                    
            return self.vectorizer.transform(tweets).toarray()

    def label_encoder(self, parties):
        self.encoder = LabelEncoder()
        return self.encoder.fit_transform(parties)

    def preprocess_tweets(self):
        self.data.Tweet = self.data.Tweet.apply(self.clean_text)
        return self.vectorize_text(self.data.Tweet.values)

    def preprocess_parties(self):
        self.data.Party = self.data.Party.apply(self.clean_text)
        return self.label_encoder(self.data.Party.values)

