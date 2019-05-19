import sys
import pandas as pd
import sqlalchemy as db


def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, left_on='id', right_on='id', how='inner')
    return df


def clean_data(df):
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';',expand=True)
    category_colnames=categories.iloc[0].str.split(pat='-',expand=True)[0].tolist()
    # rename the columns of `categories`
    categories.columns = category_colnames

    categories=categories.applymap(lambda x: x.split("-",1)[1])
    categories=categories.applymap(lambda x: int(x))

    # concatenate the original dataframe with the new `categories` dataframe
    df = df.join(categories)
    # drop duplicates
    df.drop_duplicates(subset=None, keep='first', inplace=True)
    return df


def save_data(df, database_filename):
    engine = db.create_engine('sqlite:///'+database_filename)
    df.to_sql('tweets', engine, index=False, if_exists='replace') 


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