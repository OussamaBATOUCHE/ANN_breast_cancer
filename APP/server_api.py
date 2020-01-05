from flask import Flask, request, render_template, Response
from tensorflow import keras
import json

app = Flask(__name__, template_folder="template")

# Load the model
model = keras.models.load_model("model/best_model.h5")

@app.route('/', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        # Get data
        data = [[int(request.form.get('demo-age')),int(request.form.get('demo-year'))-1900,int(request.form.get('demo-axi'))]]
        print("[Data from post] ",data)
        # Make prediction
        print(model.summary())
        pred = model.predict(data)
        result = pred[0][0]
        print(result)
        return render_template('index.html', sentiment=round((result*100),2))
    return render_template('index.html', sentiment='')
    
if __name__ == '__main__':
    app.run(port=3000, debug=True)