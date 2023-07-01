import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from transformers import BertModel, BertTokenizer, BertForSequenceClassification, BertConfig
import numpy as np
import pickle
import torch
from scipy.sparse import csr_matrix

from data_handling.text_column_preprocessor import TextPreprocessor

class SentenceEmbedder:
    """
    A module to generate sentence embeddings using different embedding techniques.
    """

    def __init__(self):
        """
        Initialize the SentenceEmbedder.
        """
        
        self.tfidf_vectorizer = TfidfVectorizer()
        self.doc2vec_model = None
        self.bert_model = None
        self.bert_tokenizer = None


    def train_tfidf_vectorizer(self, texts):
        """
        Train the TF-IDF vectorizer on the given texts.

        Args:
            texts (pandas.Series): Pandas Series containing input texts.
        """
        preprocessor = TextPreprocessor(texts)
        preprocessed_texts = preprocessor.preprocess_text()
        self.tfidf_vectorizer.fit(preprocessed_texts)

    def train_doc2vec_model(self, texts):
        """
        Train the Doc2Vec model on the given texts.

        Args:
            texts (pandas.Series): Pandas Series containing input texts.
        """
        preprocessor = TextPreprocessor(texts)
        preprocessed_texts = preprocessor.preprocess_text()
        tagged_data = [TaggedDocument(words=text.split(), tags=[str(i)]) for i, text in enumerate(preprocessed_texts)]
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


    def generate_tfidf_vectors(self, texts):
        """
        Generate TF-IDF vectors using TfidfVectorizer.

        Args:
            texts (pandas.Series): Pandas Series containing input texts.

        Returns:
            tfidf_vectors (sparse matrix): TF-IDF vectors.
        """
        preprocessor = TextPreprocessor(texts)
        preprocessed_texts = preprocessor.preprocess_text()
        tfidf_vectors = self.tfidf_vectorizer.transform(preprocessed_texts)
        return tfidf_vectors

    def generate_doc2vec_vectors(self, texts):
        """
        Generate Doc2Vec vectors.

        Args:
            texts (pandas.Series): Pandas Series containing input texts.

        Returns:
            doc2vec_vectors (list): Doc2Vec vectors.
        """
        doc2vec_vectors = np.array([self.doc2vec_model.infer_vector(text.split()) for text in texts])
        return doc2vec_vectors

    # def generate_bert_vectors(self, texts):
    #     """
    #     Generate BERT vectors.

    #     Args:
    #         texts (pandas.Series): Pandas Series containing input texts.

    #     Returns:
    #         bert_vectors (tensor): BERT vectors.
    #     """
    #     # inputs = self.bert_tokenizer(list(texts), padding=True, truncation=True, return_tensors='pt')
    #     # outputs = self.bert_model(**inputs,)

    #     # # Access the BERT vectors based on the model's output structure
    #     # if isinstance(outputs, tuple):
    #     #     # For models like BERTForSequenceClassification
    #     #     bert_vectors = outputs[0]
    #     # else:
    #     #     # For models like BertModel
    #     #     bert_vectors = outputs.hidden_states[-1]
    #     #     # bert_vectors = outputs.pooler_output

    #     bert_vectors = []
    #     for text in texts:
    #         inputs = self.bert_tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='pt')
    #         with torch.no_grad():
    #             outputs = self.bert_model(**inputs)
    #         bert_vector = outputs.pooler_output  # Use pooler_output as the representation
    #         bert_vectors.append(bert_vector)

    #     return bert_vectors
    
    def generate_bert_vectors(self, texts):
        """
        Generate BERT vectors.

        Args:
            texts (pandas.Series): Pandas Series containing input texts.

        Returns:
            bert_vectors (numpy.ndarray): BERT vectors.
        """
        input_ids = []
        attention_masks = []

        # Tokenize and encode the texts
        for text in texts:
            encoded_dict = self.bert_tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=500,
                pad_to_max_length=True,
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )

            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])

        # Convert the lists into tensors
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)

        # Pass the inputs to the BERT model
        outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_masks)

        # Extract the last hidden state from the model's output
        last_hidden_state = outputs.hidden_states[-1]
        print(last_hidden_state.shape)
        # Apply mean pooling to get a single vector representation for each sentence
        # bert_vectors = torch.mean(last_hidden_state, dim=1)

        # Extract the embeddings for the [CLS] token from the model's output
        cls_embeddings = last_hidden_state[:, 0, :]
        bert_vectors = cls_embeddings
        print(bert_vectors.shape)

        # Extract the pooler output from the hidden states
        # bert_vectors = outputs.pooler_output

        # Convert the tensor to numpy ndarray
        bert_vectors = bert_vectors.detach().numpy()

        return bert_vectors


    def save_model(self, model, filename):
        """
        Save the embedding model to a file using pickle.

        Args:
            model: Embedding model to be saved.
            filename (str): Name of the file to save the model.
        """
        with open(filename, 'wb') as file:
            pickle.dump(model, file)

    def load_tfidf_vectorizer(self, filename):
        """
        Load the embedding model from a file.

        Args:
            filename (str): Name of the file containing the model.
        """
        with open(filename, 'rb') as file:
            model = pickle.load(file)
            self.tfidf_vectorizer=model

    def load_doc2vec_model(self, filename):
        """
        Load the embedding model from a file.

        Args:
            filename (str): Name of the file containing the model.
        """
        with open(filename, 'rb') as file:
            model = pickle.load(file)
            self.doc2vec_model=model

    def load_bert_model(self, filename):
        """
        Load the embedding model from a file.

        Args:
            filename (str): Name of the file containing the model.
        """
        # with open(filename, 'rb') as file:
        #     model = pickle.load(file)
        #     self.bert_model=model

        config = BertConfig.from_pretrained('/Users/vinayak/Development/nlp_drift_paper_code/saved_models/bert/config.json')
        config.output_hidden_states = True
        tokenizer=BertTokenizer.from_pretrained(filename, config=config)
        model = BertForSequenceClassification.from_pretrained(filename, config=config)
        model.eval()
        
        
        self.bert_model = model
        self.bert_tokenizer = tokenizer


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
    
    def save_embeddings_to_csv(self, embeddings, filename):
        """
        Save the embeddings to a CSV file.

        Args:
            embeddings: Embeddings to be saved.
            filename (str): Name of the CSV file to save the embeddings.
        """
        if isinstance(embeddings, csr_matrix):
            embeddings = embeddings.toarray()

        df = pd.DataFrame(embeddings)
        df.to_csv(filename, index=False)

    def load_embeddings_from_csv(self, filename):
        """
        Load the embeddings from a CSV file.

        Args:
            filename (str): Name of the CSV file containing the embeddings.

        Returns:
            embeddings (numpy.ndarray): Loaded embeddings.
        """
        df = pd.read_csv(filename)
        embeddings = df.values
        return embeddings



"""

#Example usage
# Train and save the TF-IDF vectorizer
texts = pd.Series(["This is sentence 1.", "Another sentence.", "Yet another sentence."])
embedder = SentenceEmbedder()
embedder.train_tfidf_vectorizer(texts)
embedder.save_model(embedder.tfidf_vectorizer, "tfidf_vectorizer.pkl")

# Load the TF-IDF vectorizer
loaded_vectorizer = embedder.load_model("tfidf_vectorizer.pkl")

# Generate TF-IDF vectors using the loaded vectorizer
new_texts = pd.Series(["New sentence 1.", "Another new sentence."])
tfidf_vectors = embedder.generate_tfidf_vectors(new_texts)

# Save the TF-IDF vectors
embedder.save_embeddings(tfidf_vectors, "new_tfidf_vectors.pkl")

# Load the TF-IDF vectors
loaded_tfidf_vectors = embedder.load_embeddings("new_tfidf_vectors.pkl")

# Print the loaded TF-IDF vectors
print(loaded_tfidf_vectors)

"""

"""
#Example usage

# Train and save the Doc2Vec model
texts = pd.Series(["This is sentence 1.", "Another sentence.", "Yet another sentence."])
embedder = SentenceEmbedder()
embedder.train_doc2vec_model(texts)
embedder.save_model(embedder.doc2vec_model, "doc2vec_model.pkl")

# Load the Doc2Vec model
loaded_model = embedder.load_model("doc2vec_model.pkl")

# Generate embeddings using the loaded model
new_texts = pd.Series(["New sentence 1.", "Another new sentence."])
embeddings = embedder.generate_doc2vec_vectors(new_texts)

# Save the embeddings
embedder.save_embeddings(embeddings, "new_embeddings.pkl")

# Load the embeddings
loaded_embeddings = embedder.load_embeddings("new_embeddings.pkl")

# Print the loaded embeddings
print(loaded_embeddings)

"""