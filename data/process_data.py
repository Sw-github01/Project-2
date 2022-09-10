import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
        # merge datasets
    df = messages.merge(categories,how='outer',on=['id'])  
    return df
    #pass


def clean_data(df):
                             
    '''                         
    INPUT
    dataframe
    
    OUTPUT
    dataframe
    
    This function cleans df using the following steps to produce a dataframe:
    1. Merge datasets
    2. Split categories into separate category columns
    3. Convert category values to just numbers 0 or 1
    4. Replace categories column in df with new category columns
    5. Remove duplicates
    '''


    # Split categories into separate category columns
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x:x[0:-2])
    # rename the columns of `categories`
    categories.columns = category_colnames

    #Convert category values to just numbers 0 or 1
    for column in categories.columns:
        # set each value to be the last character of the string
        for i in range(len(categories[column].values)):
                categories[column].values[i]=categories[column].values[i][-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # drop original categories column                         
    new_df=df.drop(['categories'],axis=1)
    #concatenate the original dataframe with the new categories
    new_df = pd.concat([new_df,categories],axis=1,sort=False) 

    # Remove duplicates
    new_df.drop_duplicates(inplace=True)
    new_df.related.replace(2,1,inplace = True)
    
    return new_df
    # pass


def save_data(df, database_filename):
    # create engine
    engine = create_engine('sqlite:///{}'.format(database_filename))
    # load data to sql
    df.to_sql('ETL01', engine, index=False,if_exists='replace')
    # pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        #df = load_data(messages_filepath, categories_filepath)
        df=load_data('data/disaster_messages.csv', 'data/disaster_categories.csv')
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