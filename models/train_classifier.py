import sys
from sqlalchemy import create_engine
import pandas as pd

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

from sklearn.externals import joblib


def load_data(database_filepath):
    """
    input:
        database_filepath: File path of the database.
    output:
        X: Features.
        y: Target.
        category_names: List of category names.
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table("DisasterResponseTbl",con=engine)
    X =  df['message']
    y = df.drop(columns=['id','message','original','genre'])
    
    category_names = list(df.columns[4:])
    
    return X, y, category_names


def tokenize(text):
    """
    input:
        text: Message data for tokenization.
    output:
        clean_tokens: Result list after tokenization.
    """
    text = text.lower() 

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(estimator=RandomForestClassifier()))
    ])

    parameters = {
        'clf__estimator__n_estimators': [100, 150],
        'clf__estimator__min_samples_split': [3, 4]
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=1)
    
    return cv



def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    for idx,column in enumerate(Y_test.columns):
        print(column)
        print("---")
        print(classification_report(Y_test[column], Y_pred[:,idx]))


def save_model(model, model_filepath):
    joblib.dump(model, model_filepath)



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