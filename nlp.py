import pandas as pd
import re
import nltk
import sklearn
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Define the pipeline globally
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', LogisticRegression())
])

def preprocess_text(text: str) -> str:
    """
    Preprocesses the text by removing special characters, converting to lowercase,
    and lemmatizing the words.
    
    Args:
        text (str): The input text to be preprocessed.
    
    Returns:
        str: The preprocessed text.
    """
    lemma = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    msgs = re.sub('[^a-zA-Z]', ' ', text).lower().split()  # removing special characters and converting to lowercase
    msgs = [lemma.lemmatize(word) for word in msgs if word not in stop_words]  # lemmatizing the words and removing stopwords
    msgs = ' '.join(msgs)  # joining the words again to form a sentence 
    
    return msgs

def train_and_evaluate_model(df: pd.DataFrame) -> float:
    """
    Trains and evaluates a logistic regression model on the given dataframe.
    
    Args:
        df (pd.DataFrame): The input dataframe containing the dataset.
    
    Returns:
        float: The accuracy score of the model.
    """
    df['Spam'] = pd.get_dummies(df['Category'], drop_first=True)  # encoding the target variable
    y = df['Spam']  # target variable
    messages = df['Message']  # feature / input variable
    corpus = [preprocess_text(i) for i in messages]  # preprocessing the input text
    
    x_train, x_test, y_train, y_test = train_test_split(corpus, y, test_size=0.2, random_state=42)  # splitting the dataset into train and test
    pipeline.fit(x_train, y_train)
    pred = pipeline.predict(x_test)
    acc_score = accuracy_score(y_test, pred)
    
    return acc_score

# Reading the dataset:
df = pd.read_csv('spam.csv')

# Train and evaluate the model:
accuracy = train_and_evaluate_model(df)
print(f"Accuracy Score: {accuracy}")

# Reading the User email as input:    
user_input = input("Enter your email: ")
user_email = preprocess_text(user_input)
prediction = pipeline.predict([user_email])
if prediction[0] == 0:
    print("Your email is Not Spam")
else:
    print("Your email is Spam! Hope you don't get it")