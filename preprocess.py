from nltk.stem.snowball import EnglishStemmer
from nltk.tokenize.treebank import TreebankWordTokenizer
import string


def tokenize(review: str) -> list:
    """Tokenize string based on NLTK TreebankWordTokenizer.

    Args:
        review: The raw review content.

    Returns:
        A list of tokens found by the NLTK tokenizer.
    """
    tokenizer = TreebankWordTokenizer()
    return tokenizer.tokenize(review)


def isfloat(s: str) -> bool:
    """Evaluate if string can be cast to a float.

    Args:
        s: The string to test.

    Returns:
        True if s can be cast to a float. False otherwise.
    """
    try:
        float(s)
        return True
    except BaseException:
        return False


def normalize(review: list) -> list:
    """Remove punctuation, cast to lower case, and replace all numbers with NUM for each token.

    Args:
        review: The tokenized review.

    Returns:
        A list of normalized tokens.
    """
    no_punctuation_review = [
        s.translate(
            s.maketrans(
                '',
                '',
                string.punctuation)) for s in review]

    lowercase_and_no_punct_review = [s.lower()
                                     for s in no_punctuation_review if s]
    # map numbers to NUM symbol
    # misses cases like (1) where the main idea of the token is just numeric
    # but there are non-numeric characters
    normalized = ['NUM' if isfloat(
        s) else s for s in lowercase_and_no_punct_review]
    return normalized


def stem(review: list) -> list:
    """Stem each token down to root using the NLTK EnglishStemmer.

    Args:
        review: The normalized tokens of the review.

    Returns:
        A list of stemmed tokens.
    """
    stemmer = EnglishStemmer()
    stemmed_review = [stemmer.stem(s) for s in review]
    return stemmed_review


def get_stopwords() -> list:
    """Retrieve stopwords from predefined text file.

    Returns:
        A list of stopwords.
    """
    with open('stopwords.txt', 'r') as f:
        return f.read().split('\n')


def remove_stopwords(review: list, stopwords: list) -> list:
    """Remove N-grams from stopwords.

    Args:
        review: The list of N-grams.
        stopwords: The list of stopwords.

    Returns:
        A list of non-stopword-only N-grams.
    """
    return [ngram for ngram in review if not all(
        [s in stopwords for s in ngram.split('#')])]


def construct_Ngrams(review: list, N: int = 1) -> list:
    """Construct N-grams from tokens.

    Args:
        review: The list of stemmed tokens.
        N: The size of each N-gram.

    Returns:
        A list of N-grams.
    """
    return ['#'.join(review[i:i + N]) for i in range(len(review) - N + 1)]


def pipe(review: str, stopwords: list, N: int = 1) -> list:
    """Send review through preprocessing pipeline.

    Args:
        review: The raw review content.
        stopwords: The list of stopwords.
        N: The size of each N-gram.

    Returns:
        A list of N-grams after stemming, normalizing, and stopword removing.
    """
    return remove_stopwords(
        construct_Ngrams(
            stem(
                normalize(
                    tokenize(review))),
            N),
        stem(normalize(stopwords)))
