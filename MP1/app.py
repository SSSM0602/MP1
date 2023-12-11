from flask import Flask, request, render_template, redirect, url_for
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
import os
import joblib

app = Flask(__name__)

# Define the SQLite database file path
db_path = "layering.db"

# Create a route to train the model
@app.route('/train', methods=['GET', 'POST'])
def train_model():
    if request.method == 'POST':
        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Load features and labels from the database
        data = pd.read_sql('SELECT Test1, Test2, layer FROM circles', conn)
        
        conn.close()

        # Split the data into training and testing sets
        data = data.dropna()
        data = data[data['Test2'] != 'NULL']
        data = data[data['Test1'] != 'NULL']
        X = data[['Test1', 'Test2']]
        y = data['layer']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create a preprocessing and model pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPClassifier(hidden_layer_sizes=(100,50), activation='relu', max_iter=1000))
        ])

        # Fit the model
        pipeline.fit(X_train, y_train)

        # Calculate and store the model accuracy
        accuracy = pipeline.score(X_test, y_test)

        # Save the model using joblib
        model_path = "model.pkl"
        joblib.dump(pipeline, model_path)

        return render_template('accuracy.html', accuracy=accuracy)

    return render_template('train.html')

# Route for predicting new inputs
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Load the saved model
        model_path = "model.pkl"
        pipeline = joblib.load(model_path)

        # Get input values from the HTML form
        feature1 = float(request.form['feature1'])
        feature2 = float(request.form['feature2'])

        # Make a prediction
        prediction = pipeline.predict([[feature1, feature2]])[0]

        # Store the user input and prediction in the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

         # Create the "predictions" table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY,
                feature1 REAL,
                feature2 REAL,
                prediction INTEGER
            )
        ''')
        conn.commit()

        cursor.execute("INSERT INTO predictions (feature1, feature2, prediction) VALUES (?, ?, ?)",
                       (feature1, feature2, prediction))
        conn.commit()
        conn.close()

        return render_template('prediction.html', prediction=prediction)

    return render_template('predict.html')

# Home page
@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)


'''
Responses
a. What was the fun part and what were the challenges you faced?
I had fun writing the code to train and test the MLP model, and I found reading and writing to the database with SQL more challenging. 

b. Which algorithm did you choose and how or why did you come to that
conclusion?
I chose the MLP model because it resulted in a higher accuracy for the test data. 

c. How did you go about testing your project at different stages of development?
For the ML side of the project, I used a separate ipynb file to determine where I had errors in my process if any. In using flask to create the web app,
I first tried basic commands for the routing functions to test that the base functionality was working. 
'''