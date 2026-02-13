import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

def load_and_preprocess(file_path):
    """Loads and cleans the Titanic dataset."""
    df = pd.read_csv(r'D:\Sem 4\ML\Lab\ML-lab\Titanic-Dataset.csv')
    
    # Selecting relevant features and dropping identifiers
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    
    print(df['Survived'].value_counts())

    # Imputing missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    

    # Encoding categorical features
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])
    df['Embarked'] = le.fit_transform(df['Embarked'])
    
    return df

def perform_knn_analysis(X_train, X_test, y_train, y_test, k_value):
    """Trains KNN and returns evaluation metrics for both sets."""
    knn = KNeighborsClassifier(n_neighbors=k_value)
    knn.fit(X_train, y_train)
    
    sets = {'train': (X_train, y_train), 'test': (X_test, y_test)}
    evaluation = {}
    
    for label, (X, y_true) in sets.items():
        y_pred = knn.predict(X)
        evaluation[label] = {
            'cm': confusion_matrix(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)
        }
    return evaluation, knn

def find_optimal_k(X_train, y_train):
    """Uses GridSearchCV to find the best hyperparameter k."""
    param_grid = {'n_neighbors': np.arange(1, 31)}
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_['n_neighbors']

# --- MAIN PROGRAM ---
if __name__ == "__main__":
    # 1. Data Preparation
    data = load_and_preprocess('Titanic-Dataset.csv')
    X = data.drop('Survived', axis=1)
    y = data['Survived']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Feature Scaling (Critical for KNN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Hyperparameter Tuning
    ideal_k = find_optimal_k(X_train_scaled, y_train)
    print(f"Optimal K-value found via GridSearchCV: {ideal_k}")
    
    # 4. Evaluation
    metrics, final_model = perform_knn_analysis(X_train_scaled, X_test_scaled, y_train, y_test, ideal_k)
    
    for dataset in ['train', 'test']:
        print(f"\n--- {dataset.upper()} DATA PERFORMANCE ---")
        print(f"Confusion Matrix:\n{metrics[dataset]['cm']}")
        print(f"Precision: {metrics[dataset]['precision']:.4f}")
        print(f"Recall: {metrics[dataset]['recall']:.4f}")
        print(f"F1-Score: {metrics[dataset]['f1']:.4f}")



from sklearn.metrics import accuracy_score

k_range = range(1, 101)
train_accuracies = []
test_accuracies = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    
    train_accuracies.append(accuracy_score(y_train, knn.predict(X_train_scaled)))
    test_accuracies.append(accuracy_score(y_test, knn.predict(X_test_scaled)))


plt.figure(figsize=(12, 6))
plt.plot(k_range, train_accuracies, label='Train Accuracy', color='blue', linestyle='--')
plt.plot(k_range, test_accuracies, label='Test Accuracy', color='red', linewidth=2)
plt.title('KNN: Accuracy vs. Number of Neighbors (K)')
plt.xlabel('Value of K')
plt.ylabel('Accuracy Score')
plt.legend()
plt.grid(True)
plt.show()