from flask import Flask, request, render_template
import joblib


app = Flask(__name__, template_folder='templates')

@app.route("/")
def home():
    return render_template("index.html")  

@app.route("/predict", methods=["POST"])
def predict():
    
    features = [
        float(request.form[f"feature{i}"]) for i in range(1, 8)
    ]  
    
    
    model = joblib.load("dummy_model.pkl")
    prediction = model.predict([features])

    predicted_values = {
        "output1": round(prediction[0][0], 2),
        "output2": round(prediction[0][1], 2)
    }

    return render_template("result.html", prediction=predicted_values)


if __name__ == "__main__":
    app.run(debug=True)