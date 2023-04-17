"""
train model on the responses
"""
import sys
sys.path.append(".")
import pickle
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
from app import tokenz
#from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.tree import DecisionTreeClassifier

#from sklearn.ensemble import RandomForestClassifier
import nltk
nltk.download(['punkt',
               'wordnet',
               'averaged_perceptron_tagger',
               "stopwords",
               "omw-1.4"])


def load_data(database_filepath):
    """
    load sqlite database from database_filepath
    and make dataframe from `Message` table

    input:
    database_filepath: directory of database

    returns:
    X: message text list
    Y: corresponding category list
    columns: categorical labels of Y
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql("YourTableName", con=engine)
  #  print(df.count())
 #   print(df[df.iloc[:, 4:].astype(int).any(axis=1)==False].tail(10))
    df = df[df.iloc[:, 4:].astype(int).any(axis=1)]
   # print(len(df))
    X = df["message"]
    Y = df.iloc[:, 5:].astype(int)
    print((Y.sum(axis=1)))
    print(Y[(Y["aid_related"]==1)])
    # Z = Y[~((Y["aid_related"]==1) & (Y.sum(axis=1)==1))]
    # print(Z.sum())
#    df = df[Y]
 #   print(len(df))
    for column in Y.columns:
        Y[column] = Y[column].apply(lambda x: 1 if x > 1 else x)
        if Y[column].sum(axis=0) == 0:
            Y = Y.drop(column, axis=1)
    category_names = Y.columns
    Y = np.array(Y)
    X = np.array(X)
    X = X.reshape(len(X), 1)
    X = X.flatten()
    return X, Y, category_names



def build_model():
    """
    Build a count_vectorizer,multi-output logistic regression model as a pipeline

    returns:
    cv : a grid_searched optimised model

    """
    model = Pipeline(steps=[("vect", CountVectorizer(tokenizer=tokenz.tokenize)),
                            ("transformer",TfidfTransformer(smooth_idf=False)),
                            ("estimator", MultiOutputClassifier(
                                LogisticRegression(max_iter=1000)))
                            ])

    parameters = {
        'estimator__estimator__C': [0.5, 0.75, 1.0],
        'vect__ngram_range': [(1, 1), (2, 2)]
    }

    cv = GridSearchCV(model, param_grid=parameters, refit=True)

    return cv



def evaluate_model(model, X_test, Y_test, category_names):
    """
        evaluate the model by X_test, Y_Test dataset
        print the classification report(f1-score, recall, precision) of each column(category)
    """
    y_pred_sub_cv = model.predict(X_test)
    score = []
    for name_id, name in zip(range(len(category_names)), category_names):
        print(f"------------------{name}-------------------")
        print(classification_report(Y_test[:, name_id], y_pred_sub_cv[:, name_id]))
    print(accuracy_score(Y_test, y_pred_sub_cv))
    return


def save_model(model, model_filepath):
    """
        Save the model as a pickle file to model_filepath
    """
    with open(model_filepath, "wb") as f:
        pickle.dump(model, f)

def main():
    """
        main
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.3, random_state=42)


        print('Building model...')
        model = build_model()
        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')
        # print(Y[0:10].any(axis=1).sum())
        # print(Y[0:10].any(axis=1))
        # print(Y[0:10])


    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
