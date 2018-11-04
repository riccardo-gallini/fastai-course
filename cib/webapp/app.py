from flask import Flask, jsonify, request
import base64

app = Flask(__name__)

@app.route("/")
def home():
    return app.send_static_file("app.html")

@app.route("/predict_ws", methods=['POST'])
def predict_ws():
        data_url = request.values["imageData"]
        content = data_url.split(';')[1]
        image_encoded = content.split(',')[1]
        body = base64.decodebytes(image_encoded.encode('utf-8'))

        preds = do_predict(body)

        return jsonify(preds)


def do_predict(body):
    labels = ['POMODORO POMODORO POX', 'CASSERUOLA', 'PENTOLICCHIA', 'CACCAMO']
    probs = [31., 6., .5, .1]

    pred = { 'labels': labels, 'probs': probs}
    
    return pred

if __name__ == "__main__":
    app.run(debug=True)



