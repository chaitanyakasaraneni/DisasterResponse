import sys
import pandas as pd
import numpy as np
from sqlalchemy.engine import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    INPUT - messages csv file name and file path
            categories csv file name and file path
    OUTPUT - pandas dataframe 
           1. read the message file into a pandas dataframe
           2. read the categories file into a pandas dataframe
           3. merge the messages dataframe and catergories dataframe
           4. return the merged dataframe
    '''

    # load messages and categories
    messages=pd.read_csv(messages_filepath)
    categories=pd.read_csv(categories_filepath)
    
    # merge dataframes
    df = pd.merge(messages, categories, on = 'id')
    
    return df


def clean_data(df):
    '''
    INPUT - pandas  dataframe
    OUTPUT - pandas dataframe with cleaned data
        1. create categories dataframe by spliting the categories column by ';'
        2. rename the new columns created by splitting the categories with the category values.
        3. Convert category values to just numbers 0 or 1.
        4. merge the input dataframe and the message categories split columnn
        5. remove any duplicate messages
    '''

    # Split the values in the categories column on the ;
    categories = df['categories'].str.split(';', expand=True)

    # Rename columns of categories with new column names.
    row = categories.iloc[1]
    category_colnames = row.apply(lambda x: x.split('-')[0])
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].apply(lambda x: int(x.split('-')[1]))

    # Replace categories column in df with new category columns.
    df.drop('categories', axis = 1, inplace = True)
    df = df.join(categories)

    # drop duplicates
    df = df.drop_duplicates()
    return df


def save_data(df, database_filename):
    '''
    INPUT - panda dataframe , database file name 
    OUTPUT - pandas dataframe data stored to the database file 
        1. Create a SQLlite database using SQLAlchamey packages
        2. Load the input dataframe to the the SQL database.
    '''
    # Create database engine object
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponse', engine, if_exists='replace', index=False)
    engine.dispose()
   
def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()