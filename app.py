from flask import Flask, request, jsonify
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://localhost:5000") # cambiar en función de su servidor
## http://127.0.0.1:5000/#/models/clase06/versions/1
model = mlflow.sklearn.load_model("models:/clase06/1") # cambiar en función de su modelo

app = Flask(__name__)

@app.route("/predecir", methods=["POST"])
def predecir():
    datos = request.get_json()
    texto = datos.get("texto")
    if not texto:
        return jsonify({"error": "No se envió texto"})

    pred = model.predict([texto])[0]

    if pred == 1:
        resultado = "positivo"
    else:
        resultado = "negativo"

    return jsonify({"texto": texto, "sentimiento": resultado})

@app.route("/predecir/dos", methods=["GET"])
def predecir_02():
    texto = request.args.get("texto")
    if not texto:
        return jsonify({"error": "No enviaste texto"})

    pred = model.predict([texto])[0]

    if pred == 1:
        resultado = "positivo"
    else:
        resultado = "negativo"

    return jsonify({"texto": texto, "sentimiento": resultado})


if __name__ == "__main__":
    app.run(debug=True, port=5050)
