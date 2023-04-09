import sys
import pandas as pd
from sqlalchemy.engine import create_engine

def clean_cat_col(col_name, categories):
    '''
    INPUT: 
    series of columns where there is a regex leading a value that should be the title
    
    OUTPUT: 
    renamed column with 0,1s recast as floats
    '''
    new_name = categories[col_name][0].split('-')[0]
    categories = categories.rename({col_name: new_name}, axis='columns')
    categories[new_name] = [float(i.replace(new_name + '-', '')) for i in categories[new_name]]
    return(categories)
    pass 



def load_data(messages_filepath, categories_filepath):
    '''
    INPUT: two filepaths
    
    OUTPUT: data
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    data = messages.merge(categories, left_on = 'id', right_on = 'id', how = 'inner')
    return(data)
    pass


def clean_data(df):
    '''
    INPUT: df to clean    
    
    OUTPUT: clean df
    
    STEPS: 
        1. drop any rows with no message
        2. transform the categories column into a usable dataframe full of dummy vars
        3. replace original categories with series of columns representing values
        4. clean the messages in question
        5. for col related, as it has some mistyped 2s, replace with column mode; if any columns are completely
            uniform, it also drops
    '''
    fillmode = lambda col : col.fillna(col.mode()[0])
    
    df = df.dropna(subset = ['message'], axis = 0, how = 'any')
    df = df.drop_duplicates(subset = ['message'], keep = 'first')


    categories = df.categories.str.split(';', expand=True)
    for i in categories.columns:
        categories = clean_cat_col(i, categories)
    categories['id'] = df['id']
    df = df.drop('categories', axis = 1)
    df = df.merge(categories, left_on = 'id', right_on = 'id', how = 'inner')
    
    
    df = df[[i for i in list(df) if len(df[i].unique()) > 1]]   
    df.loc[(df.related == 2),'related'] = df['related'].mode()[0]
    df = df.apply(fillmode)
    return(df)
    pass


def save_data(df, database_filename):
    '''
    INPUT: dataframe, the desired name of the file

    OUTPUT: none

    saves file as sql
    '''
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
