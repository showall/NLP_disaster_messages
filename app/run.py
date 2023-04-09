"""
app run file
"""
import sys
sys.path.append(".")
import json
import plotly
import pandas as pd
from app import tokenz
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)


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


YourDatabaseName = "DisasterResponse"
your_model_name = "classifier"
# load data
engine = create_engine(f'sqlite:///data/{YourDatabaseName}.db')
df = pd.read_sql("SELECT * FROM YourTableName", con=engine)

# load model
model = joblib.load(f"models/{your_model_name}.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """
    index
    """
    # extract data needed for visuals
    # Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    categories = df.drop(
        ["id", "message", "original", "genre","related"], axis=1).columns
    categories_count = []
    for i in categories:
        if i != "related":
            categories_count.append(df[i].astype("int").sum())
            df_response = pd.DataFrame(list(zip(categories, categories_count)), columns=[
                                "category", "no_of_responses"]).sort_values(
                                    "no_of_responses", ascending=False)
            category_series = list(df_response["category"][0:10])
            response_series = list(df_response["no_of_responses"][0:10])
        else:
            continue

    # create visuals
    # Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_series,
                    y=response_series
                )
            ],

            'layout': {
                'title': 'Top 10 Responses',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Response"
                }
            }
        }

    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    """
    """
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]

    classification_results = dict(zip(df.drop(["child_alone"], axis=1).columns[5:], classification_labels))
    print(len(df.columns[5:]))
    print(len(classification_labels))
    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    """
    """
    app.run(port=3000, debug=True)


if __name__ == '__main__':
    main()
