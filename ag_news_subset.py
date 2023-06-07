import pandas as pd
import numpy as np

from data_handling.text_dataset_loader import TextDatasetLoader

#Load the ag_news_subset from tensorflow_datasets
data_loader = TextDatasetLoader(tfds_dataset='ag_news_subset')
data_loader.load_data()
data_loader.convert_bytes_to_string() #eill convert text columns stored as bytes to strings
data= data_loader.get_data()
data.to_csv('/Users/vinayak/Development/nlp_drift_paper_code/datasets/ag_news_subset_test.csv',header=True, index=False)

#Get the embeddings
data = pd.read_csv('/Users/vinayak/Development/nlp_drift_paper_code/datasets/ag_news_subset_test.csv')


