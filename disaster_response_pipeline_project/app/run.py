import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Heatmap
#from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine



app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('tweets', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    # Creating counts by each labels
    y = df.drop(['id','categories','message','original','genre'], axis=1)
    y0=y.melt()
    total_counts = y0.groupby('variable').sum()['value']
    total_names = list(total_counts.index)
    # Instead of doing correlation, we will be doing Baye's metrics
    # Essention it given a label , how many times other label has occured
    # This gives joint probabilty for, say given a tweet is marked Shelter, what is the probablity it is also marked as food.
    # This is useful as ML algorithms may not label the Shelter tweet also as Food but a human may. 
    cr=y.corr()
    cols=cr.columns.tolist()
    for a,col in enumerate(cr.columns):  
        for i, row in enumerate(cr[col]):
            #print(col,cols[i])
            same = y[col][(y[col]==1) & (y[cols[i]]==1)].sum()
            total =y[col][(y[col]==1)].sum()
            occur = round(same/total,2)
            cr.iloc[a,i]=occur 
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
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
                    x=total_names,
                    y=total_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Messages',
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
            Heatmap( z=cr.values.tolist(),
                   x= cols,
                   y= cols, colorscale='Viridis')
            ],

            'layout': {
                'title': 'Joint Probabilty Heatmap of Message Labels'
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
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[5:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    #app.run(host='0.0.0.0', port=3001, debug=True)
    app.run(debug=True,use_reloader=False)


if __name__ == '__main__':
    main()