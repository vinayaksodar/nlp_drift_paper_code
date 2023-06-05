import re
import string


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

"""
# Example usage of TextPreprocessor
# Assuming `text_column` is the pandas column containing text data
text_column = pd.Series(["This is some text data with special characters!", "Another text.", "More text!"])
preprocessor = TextPreprocessor(text_column)
preprocessed_column = preprocessor.preprocess_text()

print("Preprocessed Text Column:")
print(preprocessed_column)
"""