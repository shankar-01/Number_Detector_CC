from flask import Flask, request, jsonify
import numpy as np
import joblib
import tensorflow as tf
app = Flask(__name__)

# Load trained model
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("data : ", data)
        image = np.array(data['image'])
        #store in S3 storage and get url / id
        
        
        # if you saved with model.save('mnist_cnn_model.h5'):
        model = tf.keras.models.load_model('mnist_cnn_model.h5')
        
        batch = image.reshape(1, 28, 28, 1)
        probs = model.predict(batch)
        digit = int(np.argmax(probs, axis=1)) # gives 0–9
        print(f"Predicted digit: {digit}")
        # prediction = model.predict(image)
        
        # add id / url and digit in DB
        
        # print("prediction : ", prediction)
        return jsonify({'prediction': int(digit)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400
@app.route('/')
def index():
    return app.send_static_file('index.html')  # if saving in static folder
    # or: return render_template('index.html') if saved in templates/
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
