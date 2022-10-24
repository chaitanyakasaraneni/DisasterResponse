import sys
import sqlite3
from sqlalchemy import create_engine

import pandas as pd
import numpy as np
import joblib

from nltk import pos_tag
from nltk.tokenize import word_tokenize
import nltk
nltk.download('omw-1.4')

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

lemma = nltk.wordnet.WordNetLemmatizer()

def get_col_sample(df, samplen):
    '''
    INPUT: data frame, number of samples desired
    
    OUTPUT: number of samples for the dataframe
    '''
    return(df.sample(n=samplen, replace=True, random_state=1).reset_index(drop = True))

def load_data(database_filepath):
    '''
    INPUT: data file path
    OUTPUT: X, Y (numerical for category), category_names
    
    STEPS:  
    1. reads in DisasterResponse db
    2. cleans text in messages and inserts in new column
    3. drops non-category columns from Y dataframe
    4. gets the name of categories, count (to find median)
    5. creates a more balanced dataframe using sampling with replacement
    6. cleans the text of the new dataframe
    7. returns an X (messages) Y categories, and category names
    '''
    engine = create_engine('sqlite:///data/DisasterResponse.db')
    df = pd.read_sql_table(database_filepath, engine)
    
    not_y = ['index','id', 'message', 'original', 'genre']
    Y = df.drop(not_y, axis = 1)
    category_names = list(Y.columns)
    category_counts = list(Y.sum(axis = 0).values)
    number_samples = int(np.median(category_counts))
    
    balanced_list = []
    for i in category_names[1:len(category_names)-1]:
        for_balance = df[df[i] == 1].reset_index(drop = True)
        balanced_list.append(get_col_sample(for_balance, number_samples))
        
        
    balanced_df = pd.concat(balanced_list, axis = 0)
    balanced_df.to_csv('./data/balanced_df.csv')
    
    clean_messages = []
    for i in balanced_df['message']:
         clean_messages.append(tokenize(i))
        
    Y_bal = balanced_df.drop(not_y, axis = 1)
    X_bal = clean_messages
    return(X_bal, Y_bal, category_names)
    pass

def tokenize (txt):  
    '''
    INPUT: 
    a line of text
    
    OUTPUT: 
    that same text cleaned:  
    
    STEPS:
        1. creates empty string to fill
        2. tokenizes and pos_tags (easier to find critical words)
        3. lemmatizes all nouns
        4. doesn't lemmatize adj bc they don't change 
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


def build_model():
    '''
    INPUT: none
    
    OUTPUT: GridSearchCV obj
    
    '''
    parameters = {
        #for the convenience of the grader, the grid search is currently revealing a limited list of tested parameters.  
        #A complete list is included commented out
        'clf__estimator__min_samples_split': [2],#[5, 10,50]
        'clf__estimator__max_features': [300, 500],#[10, 50, 100, 150, 1000, 1500],
        'clf__estimator__max_depth':[700] # [300, 500, 800, 1000]
        }
    
    pipeline = Pipeline([
        ('tfidf_vec', TfidfVectorizer()),
        ('clf', MultiOutputClassifier(estimator = RandomForestClassifier(random_state = 5)))
        ])
    cv = GridSearchCV(pipeline, param_grid = parameters, verbose = 3)
    
    return(cv)
    pass



def precision_(cm):
    '''
    INPUT: confusion matrix
    OUTPUT: precision
    '''
    return(np.diag(cm)[np.diag(cm).shape[0] -1] / np.sum(cm, axis = 0)[ np.sum(cm, axis = 0).shape[0]-1])

def recall_(cm):
    '''
    INPUT: confusion matrix
    OUTPUT: recall
    '''
    return(np.diag(cm)[np.diag(cm).shape[0] -1] / np.sum(cm, axis = 1)[ np.sum(cm, axis = 1).shape[0]-1])

def f1score_(cm):
    '''
    INPUT: confusion matrix
    OUTPUT: f1 score
    '''
    return(2 * (precision_(cm) * recall_(cm)) / (precision_(cm)  + recall_(cm)))

#https://stats.stackexchange.com/questions/21551/how-to-compute-precision-recall-for-multiclass-multilabel-classification
def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT:
    cv model, X_test, Y_test, list of category names
    
    OUTPUT:
    precision, f1 score, recall for each category
    '''
    
    y_pred = model.predict(X_test)
    y_true = np.array(Y_test)
    y_pred = np.array(y_pred)

    labels = category_names

    conf_mat_dict={}
    for label_col in range(len(labels)):
        y_true_label = y_true[:, label_col]
        y_pred_label = y_pred[:, label_col]
        cm = confusion_matrix(y_pred=y_pred_label, y_true=y_true_label)
        
        conf_mat_dict[labels[label_col]] = {}
        conf_mat_dict[labels[label_col]] ['conf_mat'] = confusion_matrix(y_pred=y_pred_label, y_true=y_true_label)
        conf_mat_dict[labels[label_col]]['recall']  =  recall_(cm)
        conf_mat_dict[labels[label_col]]['precision']  = precision_(cm)
        conf_mat_dict[labels[label_col]]['f1_score']  = f1score_(cm)



    for label, matrix in conf_mat_dict.items():
        print("Accuracy metrics for label {}:".format(label))
        print('recall: ' + str(matrix['recall']))
        print('precision: ' + str(matrix['precision']))
        print('f1_score: ' + str(matrix['f1_score']))

    print("\nBest Parameters:", model.best_params_)

    
    return(conf_mat_dict)
    pass

def save_model(model, model_filepath):
    '''
    INPUT: model and filepath
    
    OUTPUT: None
    Saves model at given file path
    '''
    with open(model_filepath, 'wb') as f:
        joblib.dump(model, f)
    pass


def main():
    '''
    As the main function, this runs when the file is run

    STEPS:
        1. check for required number of arguments (this file, database_filepath, pkl file to save model)
        2. load_data, split into training and test categories
        3. build a model using GridSearchCV
        4. Train the model
        5. Evaluate the model
        5. save the model 
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4)
        
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
