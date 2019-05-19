import sys
# import libraries
import pandas as pd
import sqlalchemy as db
import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


import numpy as np

import warnings
warnings.filterwarnings('ignore')
import pickle


from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

def load_data(database_filepath):
    engine = db.create_engine('sqlite:///'+database_filepath)
    df=pd.read_sql_query("SELECT * FROM tweets;", engine)
    X = df['message']
    y = df.drop(['id','categories','message','original','genre'], axis=1)
    y_cols= y.columns.tolist()
    return X,y,y_cols


def tokenize(example_sent):
    example_sent = example_sent.lower()
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(example_sent)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    wordnet_lemmatizer = WordNetLemmatizer()
    filtered_Lemm_sentence = [wordnet_lemmatizer.lemmatize(w).strip() for w in filtered_sentence]
    return filtered_Lemm_sentence
    


def build_model():
    RndmFrst_pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(),n_jobs=20))
    ])    
    return RndmFrst_pipeline
       


def evaluate_model(model, X_test, y_test, category_names):
    prediction = model.predict(X_test)
    for i, cat in enumerate(category_names):
        print(i, cat)
        print('------')
        print(classification_report(y_true= y_test[cat].values.reshape(-1,1), y_pred=prediction[:,i].reshape(-1,1)))
        print('------')
        pass


def save_model(model, model_filepath):
    filehandler = open(model_filepath,"wb")
    pickle.dump(model,filehandler)


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