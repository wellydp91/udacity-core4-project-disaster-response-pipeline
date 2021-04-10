import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    input:
        messages_filepath: File path of messages dataset.
        categories_filepath: File path of categories dataset.
    output:
        df: Dataframe of the merged datasets. 
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = messages.merge(categories, how='inner', on='id')
    
    return df


def clean_data(df):
    """
    input:
        df: The dataset that needs to be cleaned.
    output:
        cleaned_df: Cleaned dataset.
    """
    categories = df['categories'].str.split(";",expand=True)
    
    row = categories.loc[0]
    
    category_colnames =  row.apply(lambda x: x[:-2])
    
    categories.columns = category_colnames
    
    categories.related.loc[categories.related == 'related-2'] = 'related-1'
    
    for column in categories:
        categories[column] = categories[column].str[-1:]
        categories[column] = categories[column].astype(int)
        
    cleaned_df = df.copy()
        
    cleaned_df.drop(columns=['categories'],inplace=True)
    cleaned_df = cleaned_df.join(categories)
    cleaned_df.drop_duplicates(inplace=True)
    
    return cleaned_df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('DisasterResponseTbl', engine, index=False)  


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