import json
import plotly
import pandas as pd
import numpy as np
import os
import sys


import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
sno = nltk.stem.SnowballStemmer('english')

from flask import Flask, abort
from flask import render_template, request, jsonify
from flask_cors import CORS
from plotly.graph_objs import Bar
try:
    from sklearn.externals import joblib
except:
    import joblib

from sqlalchemy import create_engine

from plotly.subplots import make_subplots
import plotly.graph_objects as goplot

app = Flask(__name__)

####################################
#
#   DATA PROCESSING
#
####################################
def get_col_sample(df, samplen):
    '''
    INPUT: 
    dataframe, number of samples
    
    OUTPUT: 
    a bootstrapped sample of the df of row size samplen
    '''
    return(df.sample(n=samplen, replace=True, random_state=1).reset_index(drop = True))

def tokenize(txt):  
    '''
    INPUT: 
    a line of text
    
    OUTPUT: 
    that same text cleaned:  
    
    STEPS:
        1. creates empty string to fill
        2. tokenizes and pos_tags
        3. lemmatizes the text
        4. doesn't lemmatize adj/adv bc they don't change
        5. only appends and lemmatizes longer verbs (theory that shorter verb forms are less regular and therefore more              likely to be common words/stop words)
    
    '''
    new_txt = ""
    tokens = word_tokenize(txt)
    pos_tagged = pos_tag(tokens)
    for z in pos_tagged:
        if (('NN' in z[1])):
            lem = lemma.lemmatize(z[0])
            new_txt= new_txt + " " + str(lem.lower())
        elif ('JJ' in z[1]):
            new_txt= new_txt + " " + str(z[0].lower())
        elif ('VB' in z[1]) and (len(z[0]) > 3):
            lem = lemma.lemmatize(z[0], 'v')
            new_txt= new_txt + " " + str(lem.lower())
    return(new_txt)
    pass

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('data/DisasterResponse.db', engine)

####################################
#
#   DATA PROCESSING ENDS
#
####################################

# load model
model = joblib.load("../models/classifier.pkl")
balanced_df = pd.read_csv('../data/balanced_df.csv', index_col = 0)

####################################
#
#   FLASK APP ROUTES
#
####################################
# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    '''
    STEPS:
        1. create genre counts and get names (for pie plot)
        2. create category names and category counts (for bar plot)
        3. get row sums of Y (for histogram)
        4. create pie chart (g1) and bar chart (g2)
        5. create layout for this fig
        6. create a balanced df to produce bar plot and histogram
        7. set layout for this fig
        8. render in json
    '''
    # create visuals for the raw data
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    not_y = ['index','id', 'message', 'original', 'genre']
    Y = df.drop(not_y, axis = 1)
    category_names = list(Y.columns)
    category_counts = list(Y.sum(axis = 0).values)
    
    agg = Y
    agg_sum = list(agg.sum(axis = 1).values)
    
    number_samples = int(np.median(category_counts))
    
    g1 = {      "type": "pie",
                "domain": {
                    "x": [0,1],
                    "y": [0,1]
                    },
                "marker": {
                    "colors": genre_counts
                    },
                "hoverinfo": "label+value",
                "labels": genre_names,
                "values": genre_counts
           }

        
    
    g2 =  {     'type': 'bar',
                 'x':category_names,
                 'y': category_counts,
                "hoverinfo": "x+y",
                 "marker": {
                    "color": category_counts,
                    "colorscale":"Viridis"
                    }
           
              
        }
  
    
    fig = make_subplots(rows=2, cols=2,
          specs=[[{"type": "pie"}, {"type": "histogram"}] ,[{"type": "bar", "colspan": 2},None]],
          subplot_titles=("Genre Breakdown (Raw)","Number of Categories per ID (Raw)", "Category Counts (Raw)"))
    
    fig.add_trace(goplot.Pie(g1), row=1, col=1)
    fig.add_trace(goplot.Histogram(x = agg_sum), row=1, col=2)
    fig.add_trace(goplot.Bar(g2), row=2, col=1)
    
    fig.update_layout(width = 1000, height = 1000, margin=dict(l=20, r=20, b=20, t=100),  showlegend = False)

    # create parallel visuals for the bootstrapped/rebalanced data
    
    agg_bal = balanced_df.drop(not_y, axis = 1)
    agg_bal_sum = list(agg_bal.sum(axis = 1).values)
    
    Y_bal = balanced_df.drop(not_y, axis = 1)
    X_bal = balanced_df['message']
    
    category_names_bal = list(Y_bal.columns)
    category_counts_bal = list(Y_bal.sum(axis = 0).values) 
        
    g3 =  {     'type': 'bar',
                 'x':category_names_bal,
                 'y': category_counts_bal,
                "hoverinfo": "x+y",
                 "marker": {
                    "color": category_counts_bal,
                    "colorscale":"blackbody"
                    }
           
              
        }
    
    
    fig1 = make_subplots(rows=1, cols=2,
          specs=[[{"type": "histogram"},{"type": "bar"}]],
          subplot_titles=( "Category Counts (Balanced)","Number of Categories per ID (Balanced)"))
    
    fig1.add_trace(goplot.Histogram(x = agg_bal_sum), row=1, col=2)
    fig1.add_trace(goplot.Bar(g3), row=1, col=1)
    
    fig1.update_layout(width = 1000, height = 500, margin=dict(l=20, r=20, b=20, t=100),  showlegend = False)

    graphs = [
        fig1,
        fig
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/classify')
def go():
    '''
    takes user input and returns predictions
    '''
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[5:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'classify.html',
        query=query,
        classification_result=classification_results
    )

####################################
#
#   FLAS APP END
#
####################################


####################################
#
#   API DECLARATION
#
####################################
@app.route('/classify', methods=['POST'])
def api_classify():

    query = request.get_json()

    if not query:
        abort(422, {'message': 'Unprocessable Data in Request'})
    
    inp = query.get('query', ' ')

    if not inp:
        abort(422, {'message': '{} cannot be blank'.format('Input message')})

    try:
        # use model to predict classification for query
        classification_labels = model.predict([inp])[0]
        classification_results = dict(zip(df.columns[5:], classification_labels))
        result = []
        for category, classification in classification_results.items():
            if classification == 1 or classification == 1.0:
                result.append(category.replace('_', ' ').title())

        return jsonify({
            "success": True,
            "query": inp,
            "classification_result": result
        })
    except Exception:
        abort(400, description={'message': 'Sorry! Our application Failed to classify message'})
####################################
#
#   END API DECLARATION
#
####################################


####################################
#
#   API ERROR HANDLING BEGINS
#
####################################
'''
    404 ERROR
'''
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 404,
        'message': get_error_message(error, 'Resource Not Found')
    }), 404

'''
    422 ERROR
'''
@app.errorhandler(422)
def unprocessable(error):
    return jsonify({
        'success': False,
        'error': 422,
        'message': get_error_message(error, 'Unprocessable Data')
    }), 422

'''
    400 ERROR
'''
@app.errorhandler(400)
def badrequest(error):
    return jsonify({
        'success': False,
        'error': 400,
        'message': get_error_message(error, 'Sorry! Our application Failed to classify message. Please check your input')
    }), 400


def get_error_message(error, default_message):
    '''
    Returns if there is any error message provided in
    error.description.message else default_message
    This can be passed by calling
    abort(404, description={'message': 'your message'})
    Parameters:
    error (werkzeug.exceptions.NotFound): error object
    default_message (str): default message if custom message not available
    Returns:
    str: Custom error message or default error message
    '''
    try:
        return error.description['message']
    except TypeError:
        return default_message

####################################
#
#   API ERROR HANDLING ENDS
#
####################################

def main():
    '''
    as the main function this runs whenever the file is called
    
    it sets the port and then runs the app through the desired port
    '''
    
    if len(sys.argv) == 2:
        app.run(host='0.0.0.0', port=int(sys.argv[1]), debug=True)
    else:
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port)
  


if __name__ == '__main__':
    main()
