from preprocess import preprocess_dataset
from train_evaluate import train_and_evaluate_model
import pickle

# Load and preprocess dataset
df = preprocess_dataset('SMSSpamCollection')

# Train and evaluate model
model, vectorizer = train_and_evaluate_model(df)

# Save model and vectorizer
with open('model.pkl', 'wb') as model_file, open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(model, model_file)
    pickle.dump(vectorizer, vectorizer_file)

print("Model and vectorizer saved successfully.")