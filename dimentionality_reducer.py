import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.random_projection import SparseRandomProjection
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
import pickle




class DimensionalityReducer:
    """
    A module to perform dimensionality reduction on saved embeddings.
    """

    def __init__(self):
        """
        Initialize the DimensionalityReducer.
        """
        self.pca_model = None
        self.lsa_model = None
        self.autoencoder_model = None
        self.srp_model = None

    def fit_pca(self, embeddings, n_components):
        """
        Fit PCA model on the embeddings.

        Args:
            embeddings (numpy.ndarray): Embeddings array of shape (num_samples, embedding_dim).
            n_components (int): Number of components for PCA.

        Returns:
            None
        """
        scaler = StandardScaler()
        pca = PCA(n_components=n_components)
        pipeline = Pipeline([('scaler', scaler), ('pca', pca)])
        self.pca_model = pipeline.fit(embeddings)

    def fit_lsa(self, embeddings, n_components):
        """
        Fit LSA model on the embeddings.

        Args:
            embeddings (numpy.ndarray): Embeddings array of shape (num_samples, embedding_dim).
            n_components (int): Number of components for LSA.

        Returns:
            None
        """
        scaler = StandardScaler()
        lsa = TruncatedSVD(n_components=n_components)
        pipeline = Pipeline([('scaler', scaler), ('lsa', lsa)])
        self.lsa_model = pipeline.fit(embeddings)

    def fit_autoencoder(self, embeddings, hidden_dim):
        """
        Fit autoencoder model on the embeddings.

        Args:
            embeddings (numpy.ndarray): Embeddings array of shape (num_samples, embedding_dim).
            hidden_dim (int): Dimension of the hidden layer in the autoencoder.

        Returns:
            None
        """
        scaler = StandardScaler()
        autoencoder = MLPRegressor(hidden_layer_sizes=(hidden_dim,), activation='relu', random_state=42)
        pipeline = Pipeline([('scaler', scaler), ('autoencoder', autoencoder)])
        self.autoencoder_model = pipeline.fit(embeddings)

    def fit_srp(self, embeddings, n_components):
        """
        Fit SRP model on the embeddings.

        Args:
            embeddings (numpy.ndarray): Embeddings array of shape (num_samples, embedding_dim).
            n_components (int): Number of components for SRP.

        Returns:
            None
        """
        srp = SparseRandomProjection(n_components=n_components, random_state=42)
        self.srp_model = srp.fit(embeddings)

    def reduce_dimension(self, embeddings):
        """
        Reduce the dimension of embeddings using the fitted dimensionality reduction models.

        Args:
            embeddings (numpy.ndarray): Embeddings array of shape (num_samples, embedding_dim).

        Returns:
            numpy.ndarray: Reduced dimension embeddings array of shape (num_samples, n_components).
        """
        if self.pca_model is None and self.lsa_model is None and self.autoencoder_model is None and self.srp_model is None:
            raise ValueError("No dimensionality reduction models fitted. Please fit the models first.")

        reduced_embeddings = []

        if self.pca_model is not None:
            reduced_embeddings.append(self.pca_model.transform(embeddings))

        if self.lsa_model is not None:
            reduced_embeddings.append(self.lsa_model.transform(embeddings))

        if self.autoencoder_model is not None:
            reduced_embeddings.append(self.autoencoder_model.transform(embeddings))

        if self.srp_model is not None:
            reduced_embeddings.append(self.srp_model.transform(embeddings))

        if len(reduced_embeddings) > 0:
            return np.concatenate(reduced_embeddings, axis=1)
        else:
            return embeddings

    def save_models(self, file_path):
        """
        Save the dimensionality reduction models to a pickle file.

        Args:
            file_path (str): File path to save the models.

        Returns:
            None
        """
        models = {
            'pca_model': self.pca_model,
            'lsa_model': self.lsa_model,
            'autoencoder_model': self.autoencoder_model,
            'srp_model': self.srp_model
        }

        with open(file_path, 'wb') as f:
            pickle.dump(models, f)

    def load_models(self, file_path):
        """
        Load the dimensionality reduction models from a pickle file.

        Args:
            file_path (str): File path to load the models from.

        Returns:
            None
        """
        with open(file_path, 'rb') as f:
            models = pickle.load(f)

        self.pca_model = models['pca_model']
        self.lsa_model = models['lsa_model']
        self.autoencoder_model = models['autoencoder_model']
        self.srp_model = models['srp_model']
