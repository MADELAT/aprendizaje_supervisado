from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Intentar cargar pipeline completo (preprocesamiento + escalador + modelo)
PIPELINE_PATH = "pipeline.pkl"
modelo = None
pipeline = None
try:
    with open(PIPELINE_PATH, "rb") as f:
        pipeline = pickle.load(f)
    print(f"Pipeline cargado desde {PIPELINE_PATH}")
    try:
        print("pipeline steps:", pipeline.named_steps.keys())
    except Exception:
        pass
except Exception as e:
    pipeline = None
    print(f"No se pudo cargar {PIPELINE_PATH}: {e}")


@app.route("/predecir", methods=["POST"])
def predecir():
    # Preferir pipeline (acepta JSON humano con las 7 features en columnas)
    data = request.get_json(force=True)
    if pipeline is None:
        return jsonify({"error": "Pipeline no disponible"}), 500

    # columnas esperadas en el JSON humano
    expected = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    # Aceptar dos formatos: diccionario con keys, o {'input':[...]} lista en orden esperada
    try:
        if isinstance(data, dict) and all(k in data for k in expected):
            df = pd.DataFrame([{k: data[k] for k in expected}])
        elif isinstance(data, dict) and "input" in data:
            arr = np.array(
                data["input"]
            )  # no forzamos float here to preserve categorical strings
            arr = arr.reshape(1, -1)
            if arr.shape[1] != len(expected):
                return (
                    jsonify(
                        {
                            "error": f"input must have {len(expected)} features, got {arr.shape[1]}"
                        }
                    ),
                    400,
                )
            df = pd.DataFrame(arr, columns=expected)
        else:
            return (
                jsonify(
                    {
                        "error": f"JSON inválido. Envíe un dict con keys {expected} o 'input' lista."
                    }
                ),
                400,
            )

        pred = pipeline.predict(df)
        return jsonify({"Survived": int(pred[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Ejecutar en 127.0.0.1:5000 según la sección 2.5
    print("Iniciando servidor Flask en http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000)
