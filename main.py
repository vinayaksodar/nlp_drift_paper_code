# main.py
# import config
import pandas as pd
import numpy as np
import os
# from data_handling.text_dataset_loader import DataLoader
from data_handling import text_column_preprocessor
from data_handling import drift_inducer
from sentence_embedder import SentenceEmbedder
from shift_detector import DriftDetector
from dimentionality_reducer import DimensionalityReducer
# from dimensionality_reduction.tfidf import TFIDFVectorizer
# from drift_detection.ks_test import KSTest
# from evaluation_reporting.metrics import calculate_accuracy
# from utils.file_io import load_data, save_model

def main():
    # Load data
    # data_loader = DataLoader(config.DATA_FILE)
    # data = data_loader.load_data()
    #baseline file does not have class label =1 which is the noise/drift we will add
    baseline=pd.read_csv('/Users/ankitsekseria/Desktop/AIML Projects/nlp_drift_paper_code/datasets/df_drift_train.csv',usecols=['description','label','title'])
    scoring=pd.read_csv('/Users/ankitsekseria/Desktop/AIML Projects/nlp_drift_paper_code/datasets/df_drift_test.csv',usecols=['description','label','title'])
    print('shape of data',baseline.shape)
    # preprocessed_data = data_loader.preprocess_data()

    #preprocessing 

    input_col = 'description'
    op_col = 'label'
    # col_preprocess = text_column_preprocessor.TextPreprocessor(data['input_col'])
    # clean_data = col_preprocess.preprocess_text()

    # clean_data[op_col]=data[op_col]

    # # Split data
    # splitter = Splitter()
    # train_data, test_data = splitter.split_data(preprocessed_data, config.TEST_SIZE)

    #Drift inducing strategy - 1. Adding percentage drift
    ### defining col label to drift , number of samples, percentage samples
    class_label_d = 1
    n_samples = 5
    inducer = drift_inducer.DriftInducer(scoring,op_col,class_label_d)
    perc_sampl_list =[0,0.1,0.25,0.5,0.75,1]

    # defining baseline data
    df_orig = baseline.copy()
    emb_obj = SentenceEmbedder()
    #getting tfidf embeddings
    print('getting tfidft for baseline...')
    emb_obj.train_tfidf_vectorizer(df_orig.description)
    baseline_emb = emb_obj.generate_tfidf_vectors(df_orig[input_col])
    print('TFIDF embeddings shape',baseline_emb.shape)
    #getting LSA
    print('getting LSA for baseline...')
    dim_obj = DimensionalityReducer()
    dim_lsa = dim_obj.fit_lsa(baseline_emb,1000)
    baseline_lsa = dim_obj.reduce_dimension(baseline_emb)
    # path_lsa='/Users/ankitsekseria/Desktop/AIML Projects/nlp_drift_paper_code/artefacts/'
    # dim_obj.save_models(path_lsa)

    # defining drift detector with baseline embeddings
    print('setting drift detector...')
    drift_obj = DriftDetector()
    drift_obj.set_reference_embedding(baseline_lsa)


    results={'KS_mean':[],'KS_stddev':[],'MMD_mean':[],'MMD_stddev':[],'perc_smpl':[],'model':[]}
    print('Getting samples...')
    for i in perc_sampl_list:
        df_d = inducer.adding_drift(n_samples,i)
        drift_val_ks=[]
        drift_val_mmd=[]
        for df in df_d:
            #getting embeddings for drifted data
            emb_df_d = emb_obj.generate_tfidf_vectors(df[input_col])
            lsa_df_d = dim_obj.reduce_dimension(emb_df_d)

            drift_ks = drift_obj.ks_test(lsa_df_d)
            drift_mmd = drift_obj.mmd_test(lsa_df_d)
            # drift_mmd=1

            drift_val_ks.append(drift_ks)
            drift_val_mmd.append(drift_mmd)
    
        results['KS_mean'].append(np.mean(drift_val_ks))
        results['KS_stddev'].append(np.std(drift_val_ks))
        results['MMD_mean'].append(np.mean(drift_val_mmd))
        results['MMD_stddev'].append(np.std(drift_val_mmd))
        results['perc_smpl'].append(i)
        results['model'].append('TFIDF-LSA')
    
    df_agnews = pd.DataFrame(results)
    print('results of LSA-TFIDF embeddings',df_agnews)
    op_path='/Users/ankitsekseria/Desktop/AIML Projects/nlp_drift_paper_code/outputs/agnews_op.csv'
    if os.path.exists(op_path):
        df_agnews.to_csv(op_path,mode='a',index=False,header=False)
    else:
        df_agnews.to_csv('/Users/ankitsekseria/Desktop/AIML Projects/nlp_drift_paper_code/outputs/agnews_op.csv',index=False)
























    # Perform dimensionality reduction
    # tfidf_vectorizer = TFIDFVectorizer()
    # train_features = tfidf_vectorizer.fit_transform(train_data)
    # test_features = tfidf_vectorizer.transform(test_data)

    # # Train and evaluate model
    # model = MyModel()
    # model.train(train_features, config.TRAIN_LABELS)
    # predictions = model.predict(test_features)
    # accuracy = calculate_accuracy(predictions, config.TEST_LABELS)
    # print(f"Accuracy: {accuracy}")

    # # Perform drift detection
    # drift_detector = KSTest()
    # drift_detected = drift_detector.detect_drift(train_features, test_features)
    # if drift_detected:
    #     print("Data drift detected!")

    # # Save model
    # save_model(model, config.MODEL_FILE)

if __name__ == "__main__":
    main()
