"""Caption normalization applied to every caption during training."""

from re import sub


def preprocess_caption(sentence: str) -> str:
    """Lower-case the caption and replace punctuation with spaces.

    This mirrors the preprocessing of the training data loader, so text queries
    at inference time should go through it as well.
    """
    sentence = sentence.lower()
    # remove any forgotten space before punctuation and double space
    sentence = sub(r'\s([,.!?;:"](?:\s|$))', r"\1", sentence).replace("  ", " ")
    # remove punctuations
    sentence = sub('[(,.!?;:|*")]', " ", sentence).replace("  ", " ")
    return sentence
