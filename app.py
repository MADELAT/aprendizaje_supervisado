from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Cargar el modelo pickle al iniciar la app
MODEL_PATH = "modelo.pkl"
try:
    with open(MODEL_PATH, "rb") as f:
        modelo = pickle.load(f)
    print(f"Modelo cargado desde {MODEL_PATH}")
    # Confirmar n_features_in_ si existe
    if hasattr(modelo, "n_features_in_"):
        print("n_features_in_:", modelo.n_features_in_)
    else:
        print("Advertencia: el modelo no tiene atributo n_features_in_")
except Exception as e:
    modelo = None
    print(f"No se pudo cargar {MODEL_PATH}: {e}")


@app.route("/predecir", methods=["POST"])
def predecir():
    if modelo is None:
        return jsonify({"error": "Modelo no disponible"}), 500
    data = request.get_json(force=True)
    if not data or "input" not in data:
        return (
            jsonify(
                {
                    "error": 'JSON inválido. Se requiere key "input" con lista de características.'
                }
            ),
            400,
        )
    try:
        arr = np.array(data["input"], dtype=float)
        arr = arr.reshape(1, -1)
        # Validar que haya exactamente 7 features entrantes (ya procesadas)
        if arr.shape[1] != 7:
            return (
                jsonify({"error": f"input must have 7 features, got {arr.shape[1]}"}),
                400,
            )
        pred = modelo.predict(arr)
        # Asegurar entero 0/1
        val = int(pred[0])
        return jsonify({"prediccion": val})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Ejecutar en 127.0.0.1:5000 según la sección 2.5
    print("Iniciando servidor Flask en http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000)
