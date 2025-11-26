from flask import Flask, render_template, request
import os
from predict import predict_image

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Debug print
    print("FILES RECEIVED:", request.files)

    if "image" not in request.files:
        return "No image uploaded", 400

    file = request.files["image"]

    if file.filename == "":
        return "No selected file", 400

    save_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(save_path)

    # Predict
    result, accuracy = predict_image(save_path)

    return render_template(
        "result.html",
        result=result,
        accuracy=round(accuracy, 2),
        image_path="/" + save_path
    )

if __name__ == "__main__":
    app.run(debug=True)
