from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#from sklearn.ensemble import RandomForestClassifier
import nltk
nltk.download(['punkt',
               'wordnet',
               'averaged_perceptron_tagger',
               "stopwords",
               "omw-1.4"])


def tokenize(text):
    """
        tokenize the text and returns list of token
        - tokenize
        - lemmatize
        - normalize
        - stop words filtering
        - punctuation filtering
    """
    tokens = word_tokenize(text)
    stops = set(stopwords.words('english'))
    tokens = [tok for tok in tokens if tok not in stops]
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens