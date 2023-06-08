import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from transformers import BertModel, BertTokenizer
import numpy as np
import pickle

from data_handling.text_column_preprocessor import TextPreprocessor

class SentenceEmbedder:
    """
    A module to generate sentence embeddings using different embedding techniques.
    """

    def __init__(self):
        """
        Initialize the SentenceEmbedder.
        """
        self.count_vectorizer = CountVectorizer()
        self.tfidf_vectorizer = TfidfVectorizer()
        self.doc2vec_model = None
        self.bert_model = None
        self.bert_tokenizer = None

    def train_doc2vec_model(self, texts):
        """
        Train the Doc2Vec model on the given texts.

        Args:
            texts (pandas.Series): Pandas Series containing input texts.
        """
        tagged_data = [TaggedDocument(words=text.split(), tags=[str(i)]) for i, text in enumerate(texts)]
        self.doc2vec_model = Doc2Vec(tagged_data, vector_size=100, min_count=2, epochs=10)

    def train_bert_model(self, texts):
        """
        Train the BERT model on the given texts.

        Args:
            texts (pandas.Series): Pandas Series containing input texts.
        """
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')

        inputs = self.bert_tokenizer(list(texts), padding=True, truncation=True, return_tensors='pt')
        outputs = self.bert_model(**inputs)

        self.bert_embeddings = outputs.pooler_output

    def generate_count_vectors(self, texts):
        """
        Generate count vectors using CountVectorizer.

        Args:
            texts (pandas.Series): Pandas Series containing input texts.

        Returns:
            count_vectors (sparse matrix): Count vectors.
        """
        count_vectors = self.count_vectorizer.fit_transform(texts)
        return count_vectors

    def generate_tfidf_vectors(self, texts):
        """
        Generate TF-IDF vectors using TfidfVectorizer.

        Args:
            texts (pandas.Series): Pandas Series containing input texts.

        Returns:
            tfidf_vectors (sparse matrix): TF-IDF vectors.
        """
        preprocessor=TextPreprocessor(texts)
        preprocessed_texts=preprocessor.preprocess_text()
        tfidf_vectors = self.tfidf_vectorizer.fit_transform(preprocessed_texts)
        return tfidf_vectors

    def generate_doc2vec_vectors(self, texts):
        """
        Generate Doc2Vec vectors.

        Args:
            texts (pandas.Series): Pandas Series containing input texts.

        Returns:
            doc2vec_vectors (list): Doc2Vec vectors.
        """
        doc2vec_vectors = [self.doc2vec_model.infer_vector(text.split()) for text in texts]
        return doc2vec_vectors

    def generate_bert_vectors(self, texts):
        """
        Generate BERT vectors.

        Args:
            texts (pandas.Series): Pandas Series containing input texts.

        Returns:
            bert_vectors (tensor): BERT vectors.
        """
        inputs = self.bert_tokenizer(list(texts), padding=True, truncation=True, return_tensors='pt')
        outputs = self.bert_model(**inputs)

        bert_vectors = outputs.last_hidden_state
        return bert_vectors

    def save_embeddings(self, embeddings, filename):
        """
        Save the embeddings to a file using pickle.

        Args:
            embeddings: Embeddings to be saved.
            filename (str): Name of the file to save the embeddings.
        """
        with open(filename, 'wb') as file:
            pickle.dump(embeddings, file)

    def load_embeddings(self, filename):
        """
        Load the embeddings from a file.

        Args:
            filename (str): Name of the file containing the embeddings.

        Returns:
            embeddings: Loaded embeddings.
        """
        with open(filename, 'rb') as file:
            embeddings = pickle.load(file)
        return embeddings
