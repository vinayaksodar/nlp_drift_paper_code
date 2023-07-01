import pandas as pd
import numpy as np
import os
from data_handling import text_column_preprocessor
from data_handling import drift_inducer
from sentence_embedder import SentenceEmbedder
from shift_detector import DriftDetector
from dimentionality_reducer import DimensionalityReducer

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="No GPU detected, fall back on CPU.")
def main():
    
    #baseline file does not have class label =1 which is the noise/drift we will add
    baseline=pd.read_csv('/Users/vinayak/Development/nlp_drift_paper_code/datasets/df_drift_train.csv',usecols=['description','label','title'])
    
    scoring=pd.read_csv('/Users/vinayak/Development/nlp_drift_paper_code/datasets/df_drift_test.csv',usecols=['description','label','title'])
    baseline=baseline[:50]
    scoring=scoring[:50]

    print('shape of baseline data',baseline.shape)
    

    input_col = 'description'
    output_col = 'label'
    

    #Drift inducing strategy - 1. Adding percentage drift
    ### defining col label to drift , number of samples, percentage samples
    class_label_d = 1
    n_samples = 5
    inducer = drift_inducer.DriftInducer(scoring,output_col,class_label_d)
    perc_sampl_list =[0,0.1,0.25,0.5,0.75,1]

    # defining baseline data
    df_orig = baseline.copy()
    emb_obj = SentenceEmbedder()
    #getting doc2vec embeddings
    print('getting doc2vec for baseline...')
    emb_obj.train_doc2vec_model(df_orig.description)
    baseline_emb = emb_obj.generate_doc2vec_vectors(df_orig[input_col])
    print('Doc2vec embeddings shape',baseline_emb.shape)
    #getting PCA
    print('getting PCA for baseline...')
    dim_obj = DimensionalityReducer()
    
    dim_pca = dim_obj.fit_pca(baseline_emb)
    print(baseline_emb.shape[1])
    baseline_pca = dim_obj.reduce_dimension(baseline_emb)
    

    # defining drift detector with baseline embeddings
    print('setting drift detector...')
    drift_obj = DriftDetector()
    drift_obj.set_reference_embedding(baseline_pca)


    results={'KS_mean':[],'KS_stddev':[],'MMD_mean':[],'MMD_stddev':[],'perc_smpl':[],'model':[]}
    print('Getting samples...')
    for i in perc_sampl_list:
        #Only 10 samples in each sample dataset for testing
        df_d = inducer.adding_drift(n_samples,i,10)
        drift_val_ks=[]
        drift_val_mmd=[]
        for df in df_d:
            #getting embeddings for drifted data
            emb_df_d = emb_obj.generate_doc2vec_vectors(df[input_col])
            pca_df_d = dim_obj.reduce_dimension(emb_df_d)

            drift_ks = drift_obj.ks_test(pca_df_d)
            drift_mmd = drift_obj.mmd_test(pca_df_d)
            # drift_mmd=1

            drift_val_ks.append(drift_ks)
            drift_val_mmd.append(drift_mmd)
    
        results['KS_mean'].append(np.mean(drift_val_ks))
        results['KS_stddev'].append(np.std(drift_val_ks))
        results['MMD_mean'].append(np.mean(drift_val_mmd))
        results['MMD_stddev'].append(np.std(drift_val_mmd))
        results['perc_smpl'].append(i)
        results['model'].append('doc2vec-pca')
    
    df_agnews = pd.DataFrame(results)
    print('results of pca-doc2vec embeddings',df_agnews)
    output_path='/Users/vinayak/Development/nlp_drift_paper_code/results/doc2vec-pca.csv'
    if os.path.exists(output_path):
        df_agnews.to_csv(output_path,mode='a',index=False,header=False)
    else:
        df_agnews.to_csv('/Users/vinayak/Development/nlp_drift_paper_code/results/doc2vec-pca.csv',index=False)



if __name__ == "__main__":
    main()
