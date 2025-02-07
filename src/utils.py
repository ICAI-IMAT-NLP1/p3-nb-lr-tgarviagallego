from typing import List, Dict
import torch
import re
import string


def remove_punctuations(input_col):
    """To remove all the punctuations present in the text.Input the text column"""
    table = str.maketrans("", "", string.punctuation)
    return input_col.translate(table)


# Tokenizes a input_string. Takes a input_string (a sentence), splits out punctuation and contractions, and returns a list of
# strings, with each input_string being a token.
def tokenize(input_string):
    input_string = remove_punctuations(input_string)
    input_string = re.sub(r"[^A-Za-z0-9(),.!?\'`\-\"]", " ", input_string)
    input_string = re.sub(r"\'s", " 's", input_string)
    input_string = re.sub(r"\'ve", " 've", input_string)
    input_string = re.sub(r"n\'t", " n't", input_string)
    input_string = re.sub(r"\'re", " 're", input_string)
    input_string = re.sub(r"\'d", " 'd", input_string)
    input_string = re.sub(r"\'ll", " 'll", input_string)
    input_string = re.sub(r"\.", " . ", input_string)
    input_string = re.sub(r",", " , ", input_string)
    input_string = re.sub(r"!", " ! ", input_string)
    input_string = re.sub(r"\?", " ? ", input_string)
    input_string = re.sub(r"\(", " ( ", input_string)
    input_string = re.sub(r"\)", " ) ", input_string)
    input_string = re.sub(r"\-", " - ", input_string)
    input_string = re.sub(r"\"", ' " ', input_string)
    # We may have introduced double spaces, so collapse these down
    input_string = re.sub(r"\s{2,}", " ", input_string)
    return list(filter(lambda x: len(x) > 0, input_string.split(" ")))


class SentimentExample:
    """
    Data wrapper for a single example for sentiment analysis.

    Attributes:
        words (List[str]): List of words.
        label (int): Sentiment label (0 for negative, 1 for positive).
    """

    def __init__(self, words: List[str], label: int):
        self._words = words
        self._label = label

    def __repr__(self) -> str:
        if self.label is not None:
            return f"{self.words}; label={self.label}"
        else:
            return f"{self.words}, no label"

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other) -> bool:
        if not isinstance(other, SentimentExample):
            # don't attempt to compare against unrelated types
            return NotImplemented
        return self.words == other.words and self.label == other.label

    @property
    def words(self):
        return self._words

    @words.setter
    def words(self, value):
        self._words.append(value)

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, value):
        self._label = value


def evaluate_classification(predictions: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """
    Evaluate classification metrics including accuracy, precision, recall, and F1-score.

    Args:
        predictions (torch.Tensor): Predictions from the model.
        labels (torch.Tensor): Actual ground truth labels.

    Returns:
        dict: A dictionary containing the calculated metrics.
    """
    metrics: Dict[str, float] = {}
    
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(predictions.size(0)):
        if predictions[i] == labels[i]:
            if predictions[i] == 1:
                tp += 1
            else:
                tn += 1
        else:
            if predictions[i] == 1 and labels[i] == 0:
                fp += 1
            elif predictions[i] == 0 and labels[i] == 1:
                fn += 1
    
    metrics["accuracy"] = (tp+tn)/(tp+fn+tn+fp)
    metrics["precision"] = tp/(tp+fp) if (tp + fp) > 0 else 0.0
    metrics["recall"] = tp/(tp+fn) if (tp + fn) > 0 else 0.0
    metrics["f1_score"] = (2*metrics["precision"]*metrics["recall"])/(metrics["precision"]+metrics["recall"]) if (metrics["precision"]+metrics["recall"]) > 0 else 0.0

    return metrics
