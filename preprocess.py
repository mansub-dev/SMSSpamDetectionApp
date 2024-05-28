
import pandas as pd

def preprocess_dataset(filepath):
    df = pd.read_csv(filepath, sep='\t', header=None, names=['label', 'message'])
    # Convert labels to binary (1 for spam, 0 for not spam)
    df['label'] = df['label'].map({'spam': 1, 'ham': 0})
    return df
    