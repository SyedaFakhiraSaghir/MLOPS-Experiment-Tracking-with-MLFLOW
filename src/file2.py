import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize DagsHub
import dagshub
dagshub.init(repo_owner='SyedaFakhiraSaghir', 
             repo_name='MLOPS-Experiment-Tracking-with-MLFLOW', 
             mlflow=True)

# Load data
wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.10, random_state=42)

# Set experiment
mlflow.set_experiment('YT-MLOPS-Exp3')

with mlflow.start_run():
    # Train model
    rf = RandomForestClassifier(max_depth=10, n_estimators=6, random_state=42)
    rf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Log parameters and metrics
    mlflow.log_params({
        'max_depth': 10,
        'n_estimators': 6
    })
    mlflow.log_metric('accuracy', accuracy)
    
    # Log confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=wine.target_names, 
                yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    



# Key Changes:
# Removed model logging with mlflow.sklearn - This was causing the endpoint error

# Used pickle to save the model - Simpler and more compatible

# Logged the pickle file as an artifact - Still preserves the model

# Simplified parameter logging - Using a single log_params call

    # Save model using pickle format instead of MLflow's format
    import pickle
    with open("model.pkl", "wb") as f:
        pickle.dump(rf, f)
    mlflow.log_artifact("model.pkl")
    
    # Set tags
    mlflow.set_tags({
        "Author": "Vikash",
        "Project": "Wine Classification"
    })
    
    print(f"Model accuracy: {accuracy:.4f}")