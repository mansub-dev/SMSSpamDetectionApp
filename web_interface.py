from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained model and vectorizer
with open('model.pkl', 'rb') as model_file, open('vectorizer.pkl', 'rb') as vectorizer_file:
    model = pickle.load(model_file)
    vectorizer = pickle.load(vectorizer_file)

# Render the home page
@app.route('/')
def home():
    return render_template('index.html')

# Handle the prediction request
@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    message_vectorized = vectorizer.transform([message])
    prediction = model.predict(message_vectorized)[0]
    return render_template('index.html', message=message, prediction=('spam' if prediction == 1 else 'not spam'))

if __name__ == '__main__':
    app.run(debug=True)