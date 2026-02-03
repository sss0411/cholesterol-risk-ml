import joblib
import numpy as np

MODEL_PATH = "rf_cholesterol_model.pkl"

def load_model(path=MODEL_PATH):
    return joblib.load(path)

def predict_risk(model, X):
    return model.predict_proba(X)[:, 1]

if __name__ == "__main__":
    model = load_model()
    # пример входных данных
    sample = np.array([[55, 1, 0, 0, 27.5, 92, 1]])
    risk = predict_risk(model, sample)
    print("Predicted cholesterol risk:", risk[0])
