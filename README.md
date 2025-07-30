# Detection-of-Diabetic-Retinopathy-Using-Machine-Learning 🔗 [Live Demo](https://polapranithkumarreddy.github.io/Detection-of-Diabetic-Retinopathy-Using-Machine-Learning/) 

This project is a Flask-based web application that uses a trained deep learning model to detect the severity of Diabetic Retinopathy (DR) from retinal images. The system provides prediction results along with performance metrics like accuracy, precision, recall, and F1-score.

Features
Upload a retinal image and get a diagnosis instantly
View prediction confidence score

Real-time computation of:
Accuracy
Precision
Recall
F1-Score
Confusion Matrix
Visualization of performance metrics using charts
Simple login/logout interface
Logging enabled for debugging and performance tracking

Model Info
Model: Pre-trained Keras model (dr_model.h5)

Input: Retinal image resized to 224x224

Output: DR severity classification:

No_DR (0)

Mild (1)

Moderate (2)

ERRORDATA (3 – used for unexpected or test scenarios)

📁 File Structure
bash
Copy code
project/
│
├── app.py                    # Main Flask backend
├── templates/                # HTML templates for UI
│   ├── first.html
│   ├── login.html
│   ├── index.html
│   ├── prediction.html
│   ├── performance.html
│   └── chart.html
├── static/
│   └── tests/                # Uploaded test images
├── model/
│   └── dr_model.h5           # Trained model file
🛠 How It Works
Start the server:

bash
Copy code
python app.py
Navigate to the app:
Open your browser and go to: http://127.0.0.1:5000

Upload Image:
Use the upload interface to test the model.

View Results:
Get prediction with confidence and check performance metrics from /performance and /chart.

📉 Performance Metrics
Metrics are calculated using sklearn based on prediction history.

Confusion matrix and other stats are updated dynamically with every new prediction.

🧪 Technologies Used
Flask - Web framework

TensorFlow / Keras - Deep learning model

scikit-learn - Metrics calculation

HTML/CSS (Jinja2) - Frontend templates

Logging - For monitoring and debugging

Notes
This project uses simulated true labels for metrics evaluation (randomized for testing purposes).
The model path should be updated as per your environment:
python
Copy code
model_path = 'C:\\...\\model\\dr_model.h5'


