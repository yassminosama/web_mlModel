from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
import joblib

def train_and_save_model():
    
    X, y = make_regression(n_samples=100, n_features=7, n_targets=2, noise=0.1, random_state=42)

    # Train
    model = LinearRegression()
    model.fit(X, y)

    # Save
    joblib.dump(model, "dummy_model.pkl")

if __name__ == "__main__":
    train_and_save_model()
