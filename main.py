# main.py
# import config
import pandas as pd
import numpy as np
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
    data=pd.read_csv('/Users/ankitsekseria/Desktop/AIML Projects/nlp_drift_paper_code/datasets/ag_news_subset_test.csv')
    print('shape of data',data.shape)
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
    inducer = drift_inducer.DriftInducer(data,op_col,class_label_d)
    perc_sampl_list =[0.1,0.25,0.5,0.75,1]

    # defining baseline data
    df_orig = data[data.label!=class_label_d]
    emb_obj = SentenceEmbedder()
    #getting tfidf embeddings
    emb_obj.train_tfidf_vectorizer(df_orig.description)
    baseline_emb = emb_obj.generate_tfidf_vectors(df_orig[input_col])
    #getting LSA
    dim_obj = DimensionalityReducer()
    dim_lsa = dim_obj.fit_lsa(baseline_emb,baseline_emb.shape[1]-1)
    baseline_lsa = dim_obj.reduce_dimension(baseline_emb)

    # defining drift detector with baseline embeddings
    drift_obj = DriftDetector()
    drift_obj.set_reference_embedding(baseline_lsa)


    results={'mean':[],'stddev':[],'perc_smpl':[],'test':[]}
    for i in perc_sampl_list:
        df_d = inducer.adding_drift(n_samples,i)
        drift_val=[]
        for df in df_d:
            #getting embeddings for drifted data
            emb_df_d = emb_obj.generate_tfidf_vectors(df[input_col])
            lsa_df_d = dim_obj.reduce_dimension(emb_df_d)

            drift_q = drift_obj.ks_test(lsa_df_d)
            # drift_q = drift_detector.detect_drift(baseline_lsa,lsa_df_d)
            # drift_q=1

            drift_val.append(drift_q)
    
        results['mean'].append(np.mean(drift_val))
        results['stddev'].append(np.std(drift_val))
        results['perc_smpl'].append(i)
        results['test'].append('KS_Test')
    
    print('results of ks test',results)
























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
