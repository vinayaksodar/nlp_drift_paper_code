import pickle
import numpy as np

class EmbeddingLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_embeddings(self):
        """
        Load embeddings from a pickle file and convert them to a NumPy array.

        Returns:
            numpy.ndarray: Array containing the embeddings.
        """
        with open(self.file_path, 'rb') as f:
            embeddings = pickle.load(f)

        # embeddings_array = np.array(embeddings)
        # return embeddings_array
        return embeddings

"""
#Example usage
from embedding_loader import EmbeddingLoader

# Provide the path to the pickle file
file_path = 'embeddings.pickle'

# Create an instance of the EmbeddingLoader class
loader = EmbeddingLoader(file_path)

# Load embeddings as a NumPy array
embeddings_array = loader.load_embeddings()

# Use the embeddings array as needed
print(embeddings_array.shape)
"""