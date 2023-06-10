import re
import numpy as np
import random

class DriftInducer:
    """
    A class for inducing drift in the data
    """

    def __init__(self,data,target,class_label_d) :
        """
        Initialize the drift inducer.

        Args:
            data (pandas.DataFrame): The pandas dataframe having the data needs to be drifted
            class_label_d(string): class label to induce
            target(string): target column
        """
        self.data = data
        self.class_label_d = class_label_d
        self.target=target
        

    def adding_drift(self,n_samples,per_samp):
         """
        Initialize the drift inducer.

        Args:
            n_samples (int): number of samples to take for averaging the results
            per_samp(float): percentage of the data needs to perturbed with other label
        """
         
         df = self.data
         target = self.target
         label = self.class_label_d

         df_orig=df[df[target]!=label]
         drift_indx = df[df[target]==label].index.tolist()
         df_samples=[]


         for i in range(n_samples):
             k=int(per_samp*len(drift_indx))
             random.seed(i)
             smp_list = random.sample(drift_indx,k)

             df_ind = df.iloc[smp_list,:]
             df_ind = df_orig.append(df_ind)

             df_samples.append(df_ind)
         return df_samples