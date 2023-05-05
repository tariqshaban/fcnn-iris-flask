from flask import Flask, request, render_template
from keras import models

LABEL_MAPPING = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

app = Flask(__name__)


def get_prediction(sepal_length, sepal_width, petal_length, petal_width):
    model = models.load_model('./out/iris.h5')
    return model.predict([[sepal_length, sepal_width, petal_length, petal_width]]).argmax(axis=1)[0]


@app.route('/', methods=['GET'])
def index():
    return render_template('./index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    sepal_length = float(request.values.get('sepalLength'))
    sepal_width = float(request.values.get('sepalWidth'))
    petal_length = float(request.values.get('petalLength'))
    petal_width = float(request.values.get('petalWidth'))

    prediction = get_prediction(sepal_length, sepal_width, petal_length, petal_width)

    if request.method == 'GET':
        return {'prediction': LABEL_MAPPING[prediction]}
    else:
        return render_template('result.html', prediction=LABEL_MAPPING[prediction])


if __name__ == '__main__':
    app.run()
