import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

class TextPreprocessor:
    """
    A class for preprocessing text data.
    """

    def __init__(self, column):
        """
        Initialize the TextPreprocessor.

        Args:
            column (pandas.Series): The pandas column containing text data to be preprocessed.
        """
        self.column = column
        self.stopwords = set(stopwords.words("english"))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    def preprocess_text(self):
        """
        Preprocess the text data in the pandas column by performing various steps.

        Returns:
            pandas.Series: The preprocessed text data as a pandas column.
        """
        # Perform text preprocessing steps
        preprocessed_column = self.column.apply(lambda text: self.preprocess_single_text(text))

        return preprocessed_column

    def preprocess_single_text(self, text):
        """
        Preprocess a single text by performing various steps.

        Args:
            text (str): The input text.

        Returns:
            str: The preprocessed text.
        """
        # Perform text preprocessing steps
        preprocessed_text = text.lower()  # Convert text to lowercase
        preprocessed_text = self.remove_special_characters(preprocessed_text)  # Remove special characters
        preprocessed_text = self.remove_extra_whitespace(preprocessed_text)  # Remove extra whitespaces
        preprocessed_text = self.remove_stopwords(preprocessed_text)  # Remove stopwords
        preprocessed_text = self.stem_text(preprocessed_text)  # Stem text
        preprocessed_text = self.lemmatize_text(preprocessed_text)  # Lemmatize text

        return preprocessed_text


    def remove_special_characters(self, text):
        """
        Remove special characters from the text.

        Args:
            text (str): The input text.

        Returns:
            str: The text with special characters removed.
        """
        special_chars = re.escape(string.punctuation)
        text = re.sub(f"[{special_chars}]", "", text)
        return text

    def remove_extra_whitespace(self, text):
        """
        Remove extra whitespaces from the text.

        Args:
            text (str): The input text.

        Returns:
            str: The text with extra whitespaces removed.
        """
        text = re.sub("\s+", " ", text)
        return text
    
    def remove_stopwords(self, text):
        """
        Remove stopwords from the text.

        Args:
            text (str): The input text.

        Returns:
            str: The text with stopwords removed.
        """
        words = text.split()
        filtered_words = [word for word in words if word not in self.stopwords]
        text = " ".join(filtered_words)
        return text
    
    def stem_text(self, text):
        """
        Perform stemming on the text.

        Args:
            text (str): The input text.

        Returns:
            str: The stemmed text.
        """
        words = text.split()
        stemmed_words = [self.stemmer.stem(word) for word in words]
        stemmed_text = " ".join(stemmed_words)
        return stemmed_text

    def lemmatize_text(self, text):
        """
        Perform lemmatization on the text.

        Args:
            text (str): The input text.

        Returns:
            str: The lemmatized text.
        """
        words = text.split()
        lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
        lemmatized_text = " ".join(lemmatized_words)
        return lemmatized_text
    
"""
# Example usage of TextPreprocessor
# Assuming `text_column` is the pandas column containing text data
text_column = pd.Series(["This is some text data with special characters!", "Another text.", "More text!"])
preprocessor = TextPreprocessor(text_column)
preprocessed_column = preprocessor.preprocess_text()

print("Preprocessed Text Column:")
print(preprocessed_column)
"""