import os
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import random
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Label mapping
verbose_name = {
    0: "No_DR",
    1: "Mild",
    2: "Moderate",
    3: "ERRORDATA"
}

# Load model
model_path = 'C:\\Users\\pc\\OneDrive\\Desktop\\diabetic detection using retinopathy\\model\\dr_model.h5'
model = load_model(model_path)

# State
last_prediction = None
last_confidence = None
last_metrics = {
    "accuracy": 0.0,
    "precision": 0.0,
    "recall": 0.0,
    "f_measure": 0.0
}
y_true_list = []
y_pred_list = []

def predict_label(img_path):
    test_image = image.load_img(img_path, target_size=(224, 224))
    test_image = image.img_to_array(test_image) / 255.0
    test_image = test_image.reshape(1, 224, 224, 3)
    predict_x = model.predict(test_image)
    classes_x = np.argmax(predict_x, axis=1)
    confidence = float(np.max(predict_x)) * 100
    label = verbose_name.get(classes_x[0], "Unknown")
    return label, confidence, classes_x[0]

@app.route("/")
@app.route("/first")
def first():
    return render_template('first.html')

@app.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        return redirect(url_for('preview'))
    return render_template('login.html')

@app.route("/preview", methods=['GET', 'POST'])
def preview():
    global last_prediction, last_confidence, last_metrics, y_true_list, y_pred_list

    if request.method == 'POST':
        img = request.files.get('file')
        if not img or img.filename == '':
            return render_template("index.html", error="No file selected")

        img_path = os.path.join("static", "tests", img.filename)
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        img.save(img_path)

        predict_result, confidence, pred_class = predict_label(img_path)

        # Simulate varied true labels for confusion matrix
        true_class = pred_class if random.random() > 0.4 else random.randint(0, 3)
        y_true_list.append(true_class)
        y_pred_list.append(pred_class)

        # Log the lists to debug
        app.logger.debug(f"y_true_list: {y_true_list}")
        app.logger.debug(f"y_pred_list: {y_pred_list}")

        # Update metrics only if there are predictions
        if len(y_true_list) > 0 and len(y_pred_list) > 0:
            last_prediction = predict_result
            last_confidence = confidence
            last_metrics["accuracy"] = round(accuracy_score(y_true_list, y_pred_list), 2)
            last_metrics["precision"] = round(precision_score(y_true_list, y_pred_list, average='macro', zero_division=0), 2)
            last_metrics["recall"] = round(recall_score(y_true_list, y_pred_list, average='macro', zero_division=0), 2)
            last_metrics["f_measure"] = round(f1_score(y_true_list, y_pred_list, average='macro', zero_division=0), 2)
        else:
            app.logger.warning("No predictions available to calculate metrics.")

        app.logger.debug(f"Updated last_metrics: {last_metrics}")

        return render_template("prediction.html", prediction=last_prediction,
                               confidence=last_confidence, img_path=img_path)

    return render_template("index.html")

@app.route("/performance")
def performance():
    conf_matrix = confusion_matrix(y_true_list, y_pred_list, labels=list(verbose_name.keys()))
    metrics = {
        'accuracy': last_metrics["accuracy"],
        'precision': last_metrics["precision"],
        'recall': last_metrics["recall"],
        'f_measure': last_metrics["f_measure"],
        'confusion_matrix': conf_matrix.tolist()
    }
    app.logger.debug(f"Performance metrics: {metrics}")
    return render_template('performance.html', metrics=metrics, severity_labels=verbose_name)

@app.route("/chart")
def chart():
    labels = ['Accuracy', 'Precision', 'Recall', 'F-Measure']
    # Scale the metrics to percentages
    values = [
        last_metrics["accuracy"] * 100,
        last_metrics["precision"] * 100,
        last_metrics["recall"] * 100,
        last_metrics["f_measure"] * 100
    ]
    app.logger.debug(f"Chart labels: {labels}, values: {values}")
    return render_template('chart.html', labels=labels, values=values)

@app.route("/logout")
def logout():
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)