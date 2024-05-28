import seaborn as sns
import matplotlib.pyplot as plt

def visualize_data(df):
    sns.countplot(df['label'])
    plt.title('Distribution of Spam vs. Not Spam')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.show()