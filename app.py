from flask import Flask, render_template, request
import pickle
from pathlib import Path


# Ruta base del proyecto
BASE_DIR = Path(__file__).resolve().parent


# Cargar CountVectorizer
with open(BASE_DIR / "models" / "cv.pkl", "rb") as file:
    tokenizer = pickle.load(file)


# Cargar modelo clasificador
with open(BASE_DIR / "models" / "clf.pkl", "rb") as file:
    model = pickle.load(file)


app = Flask(__name__)


@app.route("/")
def home():
    return render_template(
        "index.html",
        prediction=None,
        email_text=""
    )


@app.route("/predict", methods=["POST"])
def predict():

    # Obtener texto escrito en el formulario
    email_text = request.form.get("email-content", "")

    # Vectorizar el email
    tokenized_email = tokenizer.transform([email_text])

    # Realizar predicción
    prediction = model.predict(tokenized_email)[0]

    # Convertir resultado
    prediction = "Spam" if prediction == 1 else "Not Spam"

    return render_template(
        "index.html",
        prediction=prediction,
        email_text=email_text
    )


if __name__ == "__main__":
    app.run(debug=True)