import pandas as pd
import numpy as np
import os
from data_handling import drift_inducer
from sentence_embedder import SentenceEmbedder
from shift_detector import DriftDetector
from dimentionality_reducer import DimensionalityReducer

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="No GPU detected, fall back on CPU.")
def main():
    
    #baseline file does not have class label =1 which is the noise/drift we will add
    baseline=pd.read_csv('/content/gdrive/MyDrive/nlp_data_drift/nlp_data_drift_paper_code/df_drift_train.csv',usecols=['description','label','title'])
    
    scoring=pd.read_csv('/content/gdrive/MyDrive/nlp_data_drift/nlp_data_drift_paper_code/df_drift_test.csv',usecols=['description','label','title'])
    baseline=baseline.sample(15000,random_state=42)
    # baseline=baseline[:500]
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
    #getting bert embeddings
    bert_model_path ='/content/gdrive/MyDrive/nlp_data_drift/nlp_data_drift_paper_code/bert'
    # emb_obj.load_bert_model(bert_model_path)
    # print('loaded bert model')
    # baseline_emb = emb_obj.generate_bert_vectors(df_orig[input_col])
    # print('bert embeddings shape',baseline_emb.shape)
    # emb_obj.save_embeddings_to_csv(baseline_emb,'/content/gdrive/MyDrive/nlp_data_drift/nlp_data_drift_paper_code/results/bert_baseline_emb.csv')
    baseline_emb=emb_obj.load_embeddings_from_csv('/content/gdrive/MyDrive/nlp_data_drift/nlp_data_drift_paper_code/results/bert_baseline_emb.csv')


    #getting PCA
    # print('getting PCA for baseline...')
    # dim_obj = DimensionalityReducer()
    
    # dim_pca = dim_obj.fit_pca(baseline_emb)
    # print(baseline_emb.shape[1])
    # baseline_pca = dim_obj.reduce_dimension(baseline_emb)
    

    # defining drift detector with baseline embeddings
    print('setting drift detector...')
    drift_obj = DriftDetector()
    # drift_obj.set_reference_embedding(baseline_pca)
    drift_obj.set_reference_embedding(baseline_emb)


    results={'KS_mean':[],'KS_stddev':[],'MMD_mean':[],'MMD_stddev':[],'perc_smpl':[],'model':[]}
    print('Getting samples...')
    for i in perc_sampl_list:
        #each test dataset will have 5000 samples
        df_d = inducer.adding_drift(n_samples,i,5000)
        drift_val_ks=[]
        drift_val_mmd=[]
        k=1
        for df in df_d:
            #getting embeddings for drifted data
            # emb_df_d = emb_obj.generate_bert_vectors(df[input_col])
            # emb_obj.save_embeddings_to_csv(emb_df_d,'/content/gdrive/MyDrive/nlp_data_drift/nlp_data_drift_paper_code/results/bert_emb_df_d'+ '_'+str(i)+'_'+str(k)+'.csv')
            emb_df_d=emb_obj.load_embeddings_from_csv('/content/gdrive/MyDrive/nlp_data_drift/nlp_data_drift_paper_code/results/bert_emb_df_d'+ '_'+str(i)+'_'+str(k)+'.csv')
            k=k+1
            # pca_df_d = dim_obj.reduce_dimension(emb_df_d)

            # drift_ks = drift_obj.ks_test(pca_df_d)
            # drift_mmd = drift_obj.mmd_test(pca_df_d)

            drift_ks=drift_obj.ks_test(emb_df_d)
            drift_mmd=drift_obj.mmd_test(emb_df_d)

            drift_val_ks.append(drift_ks)
            drift_val_mmd.append(drift_mmd)
    
        results['KS_mean'].append(np.mean(drift_val_ks))
        results['KS_stddev'].append(np.std(drift_val_ks))
        results['MMD_mean'].append(np.mean(drift_val_mmd))
        results['MMD_stddev'].append(np.std(drift_val_mmd))
        results['perc_smpl'].append(i)
        results['model'].append('bert-pca')
    
    df_agnews = pd.DataFrame(results)
    print('results of pca-bert embeddings',df_agnews)
    output_path='/content/gdrive/MyDrive/nlp_data_drift/nlp_data_drift_paper_code/results/bert.csv'
    if os.path.exists(output_path):
        df_agnews.to_csv(output_path,mode='a',index=False,header=False)
    else:
        df_agnews.to_csv('/content/gdrive/MyDrive/nlp_data_drift/nlp_data_drift_paper_code/results/bert.csv',index=False)



if __name__ == "__main__":
    main()