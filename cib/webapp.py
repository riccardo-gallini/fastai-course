from flask import Flask, jsonify, request
import base64
from predict import Predict
from io import BytesIO

app = Flask(__name__)

predictor = Predict()


@app.route("/")
def home():
    return app.send_static_file("app.html")


@app.route("/predict_ws", methods=['POST'])
def predict_ws():
        data_url = request.values["imageData"]
        content = data_url.split(';')[1]
        image_encoded = content.split(',')[1]
        bytes = base64.decodebytes(image_encoded.encode('utf-8'))

        preds = do_predict(bytes)

        return jsonify(preds)


def do_predict(bytes):
    df = predictor.predict(BytesIO(bytes))
    
    labels = list(df["LABELS"])
    probs = list(df["PROBS"])
    
    pred = { 'labels': labels, 'probs': probs}
    return pred

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)



