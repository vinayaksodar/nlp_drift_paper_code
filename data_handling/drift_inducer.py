import re
import numpy as np
import random

class DriftInducer:
    """
    A class for inducing drift in the data
    """

    def __init__(self, data, target, class_label_d):
        """
        Initialize the drift inducer.

        Args:
            data (pandas.DataFrame): The pandas dataframe having the data needs to be drifted
            class_label_d (string): class label to induce
            target (string): target column
        """
        self.data = data
        self.class_label_d = class_label_d
        self.target = target

    def adding_drift(self, n_sample_datasets, per_samp, sample_dataset_size=1000):
        """
        Initialize the drift inducer.

        Args:
            n_sample_datasets (int): number of sample datasets to be created.
            per_samp (float): percentage of the data needs to be perturbed with the other label

        Returns:
            df_samples(List[Pandas.DataFrames]) : A list of randomly sampled datasets with per_samp of induced drift 
        """

        df = self.data
        target = self.target
        label = self.class_label_d

        df_orig = df[df[target] != label]
        drift_indx = df[df[target] == label].index.tolist()
        df_samples = []

        for i in range(n_sample_datasets):
            seed = i
            np.random.seed(i)
            df_sampled = df_orig.sample(n=sample_dataset_size, random_state=seed)

            #Remove the percentage of data specified by user
            num_values_to_remove = int(per_samp * len(df_sampled))
            indices_to_remove = np.random.choice(df_sampled.index, num_values_to_remove, replace=False)
            df_sampled = df_sampled.drop(indices_to_remove)

            #Add the samme percentage of data with different label
            np.random.seed(i+100)
            indices_to_add = np.random.choice(drift_indx, num_values_to_remove, replace= False)
            df_ind = df.loc[indices_to_add]
            df_ind = df_sampled.append(df_ind)

            df_samples.append(df_ind)

        return df_samples
