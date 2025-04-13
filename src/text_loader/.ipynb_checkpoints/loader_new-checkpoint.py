import pandas as pd
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

class DataLoader:
    def __init__(self, filepath="data/Tweets.csv", is_train=True):
        self.filepath = filepath
        if is_train:            #<--Fix this
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
        # 1. Remove URLs using regex
        text = re.sub(r'http\S+|www\.\S+', '', str(text))   #<-- For make test
        
        # 2. Remove punctuation + digits
        remove_chars = string.punctuation  + string.digits #<-- For make test
        translator = str.maketrans('', '', remove_chars)  #<-- For make test
        cleaned = text.translate(translator)  #<-- For make test

        # 3. Split and return first alphabetic token
        #for token in cleaned.split():   #<-- For make test
        #    if token.isalpha():         #<-- For make test
        #        return token.lower()    #<-- For make test
        #return ""                       #<-- For make test
        return cleaned

    def clean_text(self, text: str) -> str:
        """Keep only retain words in a given string"""
        text = self.remove_characters(text)
        return text.strip()

    def vectorize_text(self, tweets: list[str], fit=True):
        if fit or self.vectorizer is None:                                                 #<--Fix this
            self.vectorizer = TfidfVectorizer(max_features=2500, min_df=1, max_df=0.8)     #<--Fix this
            return self.vectorizer.fit_transform(tweets).toarray()  #<--Fix this
        else:                                                       #<--Fix this
            if self.vectorizer is None:                             #<--Fix this
                raise ValueError("Vectorizer is not fitted. Call with fit=True first!!")     #<--Fix this                
            return self.vectorizer.transform(tweets).toarray()            #<--Fix this

    def label_encoder(self, parties):
        self.encoder = LabelEncoder()
        return self.encoder.fit_transform(parties)

    def preprocess_tweets(self):
        self.data.Tweet = self.data.Tweet.apply(self.clean_text)
        return self.vectorize_text(self.data.Tweet.values)

    def preprocess_parties(self):
        self.data.Party = self.data.Party.apply(self.clean_text)
        return self.label_encoder(self.data.Party.values)

