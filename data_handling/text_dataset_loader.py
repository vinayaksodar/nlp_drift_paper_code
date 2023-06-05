"""Example usage
# Load data from a CSV file
loader = TextDatasetLoader(file_path='data.csv')
loader.load_data()

# Load data from a TFDS dataset
loader = TextDatasetLoader(tfds_dataset='imdb_reviews')
loader.load_data()

# Load data from a Hugging Face dataset
loader = TextDatasetLoader(huggingface_dataset='username/dataset_name')
loader.load_data()

# Load data from a Kaggle dataset
loader = TextDatasetLoader(kaggle_dataset='username/dataset_name')
loader.load_data()
"""

import logging
import pandas as pd
import tensorflow_datasets as tfds
from kaggle.api.kaggle_api_extended import KaggleApi
from huggingface_hub import Repository
import re

class TextDatasetLoader:
    """
    A data loader class to load text data from various sources.
    Supports loading data from CSV files, TFDS datasets, Hugging Face datasets, and Kaggle datasets.
    """

    def __init__(self, file_path=None, has_header=True, has_index=False, tfds_dataset=None,
                 huggingface_dataset=None, kaggle_dataset=None):
        """
        Initialize the TextDatasetLoader.

        Args:
            file_path (str): Path to the CSV file (default: None).
            has_header (bool): Whether the CSV file has a header (default: True).
            has_index (bool): Whether the CSV file has an index column (default: False).
            tfds_dataset (str): Name of the TFDS dataset (default: None).
            huggingface_dataset (str): Hugging Face model name (username/dataset) (default: None).
            kaggle_dataset (str): Kaggle dataset name (username/dataset) (default: None).
        """
        self.file_path = file_path
        self.tfds_dataset = tfds_dataset
        self.huggingface_dataset = huggingface_dataset
        self.kaggle_dataset = kaggle_dataset
        self.data = None
        self.has_header = has_header
        self.has_index = has_index
        self.logger = logging.getLogger(__name__)

    def load_data(self):
        """
        Load the text data based on the specified data source.
        """
        if self.file_path:
            self.load_csv_data()
        elif self.tfds_dataset:
            self.load_tfds_data()
        elif self.huggingface_dataset:
            self.load_huggingface_data()
        elif self.kaggle_dataset:
            self.load_kaggle_data()
        else:
            self.logger.error("Either file path, dataset name, Hugging Face model, or Kaggle dataset must be provided.")

    def load_csv_data(self):
        """
        Load text data from a CSV file.
        """
        try:
            # Load the text data from file
            if self.has_header:
                if self.has_index:
                    self.data = pd.read_csv(self.file_path,index_col=0)
                else:
                    pd.read_csv(self.file_path)
            else:
                if self.has_index:
                    self.data = pd.read_csv(self.file_path, header=None, index_col=0)
                else:
                    self.data= pd.read_csv(self.file_path,index_col=False)

            # Check if the index column is present
            if self.has_index:
                if 'index' in self.data.columns:
                    self.data = self.data.drop('index', axis=1)
                else:
                    self.logger.warning("Index column not found in data.")

            self.logger.info("Data loaded successfully.")
        except Exception as e:
            self.logger.error("Error loading data from CSV file: %s", str(e))

    def load_tfds_data(self):
        """
        Load text data from a TFDS dataset.
        """
        try:
            # Load the TFDS dataset
            dataset = tfds.load(self.tfds_dataset, split=tfds.Split.TRAIN)

            # Convert the dataset to a pandas DataFrame
            self.data = tfds.as_dataframe(dataset)

            self.logger.info("Data loaded successfully.")
        except Exception as e:
            self.logger.error("Error loading data from TFDS: %s", str(e))

    def load_huggingface_data(self):
        """
        Load text data from a Hugging Face dataset.
        """
        try:
            # Load the Hugging Face dataset
            repo = Repository(self.huggingface_dataset)
            dataset = repo.load_dataset()

            # Convert the dataset to a pandas DataFrame
            self.data = pd.DataFrame(dataset)

            self.logger.info("Data loaded successfully.")
        except Exception as e:
            self.logger.error("Error loading data from Hugging Face dataset: %s", str(e))

    def load_kaggle_data(self):
        """
        Load text data from a Kaggle dataset.
        """
        try:
            # Authenticate with the Kaggle API
            api = KaggleApi()
            api.authenticate()

            # Download the Kaggle dataset
            api.dataset_download_files(self.kaggle_dataset, path='./', unzip=True)

            # Load the text data from the downloaded CSV file
            self.data = pd.read_csv(f"{self.kaggle_dataset.split('/')[1]}.csv", header=None)

            # Check if the index column is present
            if self.has_index:
                if 'index' in self.data.columns:
                    self.data = self.data.drop('index', axis=1)
                else:
                    self.logger.warning("Index column not found in data.")

            self.logger.info("Data loaded successfully.")
        except Exception as e:
            self.logger.error("Error loading data from Kaggle: %s", str(e))



    
    def perform_initial_checks(self):
        if self.data is None:
            self.logger.warning("No data loaded. Please load the data first.")
            return
        
        # Perform initial checks on the data
        self.logger.info("Performing initial checks on the data...")
        # Example: Check for missing values
        missing_values = self.data.isnull().sum()
        if missing_values.any():
            self.logger.warning("Warning: Missing values found in the data.")
        
        # Add more checks as per your requirements
    
    def clean_data(self):
        if self.data is None:
            self.logger.warning("No data loaded. Please load the data first.")
            return
        
        # Clean the data
        self.logger.info("Cleaning the data...")
        # Example: Remove special characters and digits from text
        self.data['text'] = self.data['text'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))
        
        # Add more data cleaning steps as per your requirements
        
        self.logger.info("Data cleaning complete.")
    
    def preprocess_data(self):
        self.load_data()
        self.perform_initial_checks()
        self.clean_data()
        # Add any additional preprocessing steps
    
    def get_data(self):
        return self.data
