from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import imdb
import LogisticRegression


m = 5000 #Most used words 
n = 10 #Top n most used words
k = 0 #Least used words

(X_train, y_train), (X_test, y_test) = imdb.load_data(
    path="imdb.npz",
    num_words=m,
    skip_top=n,
    maxlen=None,
    seed=113,
    start_char=1,
    oov_char=2,
    index_from=3
)

def vectorize_sequences(sequences, dimension=m):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))

    for i, sequence in enumerate(sequences):
        # Sets specific indices of results[i] to 1s
        results[i, sequence] = 1

    return results


X_train = vectorize_sequences(X_train)
X_test = vectorize_sequences(X_test)    

clf = LogisticRegression()
clf.fit(X_train, y_train)

print(classification_report(y_train, clf.predict(X_train), zero_division=1))
print(classification_report(y_test, clf.predict(X_test), zero_division=1))

# Define the scoring metrics
scorings = ['accuracy', 'f1', 'precision', 'recall']

# Create and plot learning curves for each metric
for scoring in scorings:
    plt.figure(figsize=(10, 6))
    train_sizes, train_scores, test_scores = learning_curve(LogisticRegression(), X_train, y_train, cv=5, scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    
    plt.plot(train_sizes, train_scores_mean, label=f'Training {scoring.capitalize()}')
    plt.plot(train_sizes, test_scores_mean, label=f'Testing {scoring.capitalize()}')
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.title(f'Learning Curve for Logistic Regression ({scoring.capitalize()} Scoring)')
    plt.legend()
    plt.grid()
    plt.show()
