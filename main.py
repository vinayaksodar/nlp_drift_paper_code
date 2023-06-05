# main.py
import config
from data_handling.text_dataset_loader import DataLoader
from dimensionality_reduction.tfidf import TFIDFVectorizer
from drift_detection.ks_test import KSTest
from evaluation_reporting.metrics import calculate_accuracy
from utils.file_io import load_data, save_model

def main():
    # Load data
    data_loader = DataLoader(config.DATA_FILE)
    data = data_loader.load_data()
    preprocessed_data = data_loader.preprocess_data()

    # Split data
    splitter = Splitter()
    train_data, test_data = splitter.split_data(preprocessed_data, config.TEST_SIZE)

    # Perform dimensionality reduction
    tfidf_vectorizer = TFIDFVectorizer()
    train_features = tfidf_vectorizer.fit_transform(train_data)
    test_features = tfidf_vectorizer.transform(test_data)

    # Train and evaluate model
    model = MyModel()
    model.train(train_features, config.TRAIN_LABELS)
    predictions = model.predict(test_features)
    accuracy = calculate_accuracy(predictions, config.TEST_LABELS)
    print(f"Accuracy: {accuracy}")

    # Perform drift detection
    drift_detector = KSTest()
    drift_detected = drift_detector.detect_drift(train_features, test_features)
    if drift_detected:
        print("Data drift detected!")

    # Save model
    save_model(model, config.MODEL_FILE)

if __name__ == "__main__":
    main()
