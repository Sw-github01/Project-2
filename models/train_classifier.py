import sys
# download necessary NLTK data
import nltk
nltk.download(['punkt', 'wordnet'])
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table(
        'ETL01',
        con=engine
    )
    df.drop('original', axis=1, inplace=True)    
    X = df['message'].values
    y = df[df.columns[4:]]
    category_names = y.columns.tolist()                       
    return X, y, category_names


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    # build pipeline
    pipeline = Pipeline([('Vec', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultiOutputClassifier(RandomForestClassifier()))])

      
    # choose parameters
    parameters = {'clf__estimator__n_estimators': [50]}
    
    # create grid search object
    model = GridSearchCV(pipeline, param_grid=parameters, scoring='recall_micro', cv=4)
   
    return model 


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Function to evaluate a model and return the classification and accurancy score.
    Inputs: Model, X_test, y_test, Catgegory_names
    Outputs: Prints the Classification report & Accuracy Score
    '''
    # Predicted value of test data   
    y_pred = model.predict(X_test)
  
    for i in range(len(category_names)):
        print('Category: {} '.format(category_names[i]))
        print(classification_report(Y_test.iloc[:, i].values, y_pred[:, i]))
        print('Accuracy {}\n\n'.format(accuracy_score(Y_test.iloc[:, i].values, y_pred[:, i])))


def save_model(model, model_filepath):
    '''
    Parameters
    model : ML model
        trained and ready to be deployed to production.
    model_filepath : string
        distination to be saved.
    '''
    # Dumping the created model to a file
    pickle.dump(model,open(model_filepath,'wb'))
    

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

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