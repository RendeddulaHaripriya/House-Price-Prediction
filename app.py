from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load dataset
data = pd.read_csv("house.csv")

X = data[['area', 'bedrooms', 'bathrooms']]
y = data['price']

# Train model
model = LinearRegression()
model.fit(X, y)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        area = int(request.form['area'])
        bedrooms = int(request.form['bedrooms'])
        bathrooms = int(request.form['bathrooms'])

        price = model.predict([[area, bedrooms, bathrooms]])
        prediction = int(price[0])

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
