const form = document.getElementById("prediction-form");
const result = document.getElementById("result");

form.addEventListener("submit", async function (event) {
    event.preventDefault();

    result.textContent = "Calculating prediction...";

    const data = {
        price: Number(document.getElementById("price").value),
        units: Number(document.getElementById("units").value),
        discount: Number(document.getElementById("discount").value),
        marketing: Number(document.getElementById("marketing").value),
        month: Number(document.getElementById("month").value),
        day: Number(document.getElementById("day").value),
        day_of_week: Number(
            document.getElementById("day_of_week").value
        )
    };

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        });

        const output = await response.json();

        if (!response.ok) {
            throw new Error(output.error || "Prediction failed.");
        }

        result.textContent =
            "Predicted Sales: ₹" +
            output.predicted_sales.toLocaleString("en-IN");

    } catch (error) {
        result.textContent = error.message;
    }
});