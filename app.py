from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

MODEL_PATH = "models/sales_model.pkl"

bundle = None

if os.path.exists(MODEL_PATH):
    bundle = joblib.load(MODEL_PATH)


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Inventory page
@app.route("/inventory")
def inventory():
    return render_template("inventory.html")


# Models page
@app.route("/models")
def models():
    return render_template("models.html")


# Upload page
@app.route("/upload")
def upload():
    return render_template("upload.html")


# Sales prediction API
@app.route("/predict", methods=["POST"])
def predict():

    if bundle is None:
        return jsonify({
            "error": "Model not found. Please train the model first."
        }), 400

    data = request.get_json(silent=True) or {}

    try:
        price = float(data.get("price", 1000))
        units = float(data.get("units", 10))
        discount = float(data.get("discount", 5))
        marketing = float(data.get("marketing", 100))
        month = int(data.get("month", 1))
        day = int(data.get("day", 1))
        day_of_week = int(data.get("day_of_week", 0))

        if (
            price <= 0
            or units < 0
            or discount < 0
            or discount > 100
            or marketing < 0
            or month < 1
            or month > 12
            or day < 1
            or day > 31
            or day_of_week < 0
            or day_of_week > 6
        ):
            raise ValueError("Input values are outside the allowed range.")

        row = {
            "Price": price,
            "Units_Sold": units,
            "Discount": discount,
            "Marketing_Spend": marketing,
            "Month": month,
            "Day": day,
            "DayOfWeek": day_of_week
        }

        features = bundle["features"]

        # Add categorical feature columns.
        for feature in features:
            if feature.startswith("Store_ID_"):
                row[feature] = int(
                    feature == "Store_ID_Store_1"
                )

            elif feature.startswith("Category_"):
                row[feature] = int(
                    feature == "Category_Electronics"
                )

        input_df = pd.DataFrame([row])
        input_df = input_df.reindex(
            columns=features,
            fill_value=0
        )

        prediction = bundle["model"].predict(input_df)[0]

        return jsonify({
            "predicted_sales": round(max(0, float(prediction)), 2)
        })

    except (ValueError, TypeError):
        return jsonify({
            "error": "Please enter valid numeric values."
        }), 400


# CSV upload
@app.route("/api/dashboard")
def dashboard_api():
    import os
    import pandas as pd

    paths = [
        "data/retail_sales.csv",
        "dataset/retail_sales.csv"
    ]

    csv_path = next(
        (path for path in paths if os.path.exists(path)),
        None
    )

    if csv_path is None:
        return jsonify({
            "total_sales": 0,
            "average_sales": 0,
            "records": 0,
            "rows": []
        })

    df = pd.read_csv(csv_path)

    if "Sales" not in df.columns:
        return jsonify({
            "total_sales": 0,
            "average_sales": 0,
            "records": len(df),
            "rows": []
        })

    df["Sales"] = pd.to_numeric(
        df["Sales"],
        errors="coerce"
    )

    df = df.dropna(subset=["Sales"])

    total_sales = float(df["Sales"].sum())
    average_sales = float(df["Sales"].mean())

    rows = []

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce"
        )

        df = df.dropna(subset=["Date"])

        df["Period"] = df["Date"].dt.to_period("M").astype(str)

        monthly = (
            df.groupby("Period")["Sales"]
            .sum()
            .tail(12)
        )

        rows = [
            {
                "period": str(period),
                "sales": float(sales)
            }
            for period, sales in monthly.items()
        ]

    return jsonify({
        "total_sales": total_sales,
        "average_sales": average_sales,
        "records": len(df),
        "rows": rows
    })


if __name__ == "__main__":
    app.run(debug=True)