import sys

# Import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# ML libraries
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

# NLP libraries
import re
import nltk 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download(['punkt', 'wordnet'])
nltk.download('stopwords')


def load_data(database_filepath):
    '''
    INPUT - path to a database
    OUTPUT - data, labels and categories
    '''
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df.message
    y = df.iloc[:,4:]
    category_names = y.columns.tolist()
    return X,y,category_names
    

def tokenize(text):
    '''
    INPUT - raw text
    OUTPUT - cleaned text
           1. removes numbers, punctuation and tokenizes text
           2. performs lemmatization
           3. remove stopwords
    '''
    
    # remove numbers, punctuation and tokenize text
    
    tokens = word_tokenize(re.sub(r"[^a-zA-Z0-9]", " ", text.lower()))
    
    # lemmatization 
    lemma = WordNetLemmatizer()
    
    # save cleaned tokens
    clean_tokens = [lemma.lemmatize(tok).strip() for tok in tokens]
    
    # remove stopwords
    english_stopwords = list(set(stopwords.words('english')))
    clean_tokens = [token for token in clean_tokens if token not in english_stopwords]
    
    return clean_tokens
    


def build_model():
    """
    INPUT - none
    OUTPUT - ML model
        builds an NLP pipeline,  tf-idf(message transformation), RandomForest+MultiOutputClassifier as 
        classification model , grid search the best parameters
    """   
    # NLP pipeline 
    pipeline = Pipeline([
        ('vect',CountVectorizer(tokenizer = tokenize)),
        ('tfidf',TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    # parameters GridSearchCV
    
    parameters = {
    'tfidf__norm':['l2','l1'],
    'clf__estimator__min_samples_split':[2,3],
    }

    # Instantiate GridSearchCV
    model = GridSearchCV(pipeline, param_grid=parameters)

    return model
   


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    for i, col in enumerate(category_names): 
        print('***********',col,'***********')
        print(classification_report(Y_test.iloc[:,i], y_pred[:,i]))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()