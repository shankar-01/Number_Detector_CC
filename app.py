from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import boto3
from datetime import datetime
import io
from dotenv import load_dotenv
import os
from PIL import Image

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model('mnist_cnn_model.h5')

# AWS configuration (credentials loaded from env)
s3 = boto3.client('s3', region_name='eu-north-1')
dynamodb = boto3.resource('dynamodb', region_name='eu-north-1')
bucket_name = 'ai-inference-image-storage'
table = dynamodb.Table('mnist_predictions')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_array = np.array(data['image'])

        # Prepare image
        batch = image_array.reshape(1, 28, 28, 1)
        probs = model.predict(batch)
        digit = int(np.argmax(probs, axis=1))
        print("Digit:", digit)
        # Save image as PNG file
        filename = f"digit_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}.jpg"
        image_uint8 = (image_array * 255).astype(np.uint8)
        img = Image.fromarray(image_uint8.reshape(28, 28), mode='L')
        print("Image:", img)
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        # s3.upload_fileobj(Bucket=bucket_name, Key=f"inputs/{filename}", Fileobj=img_byte_arr)
        # Upload to S3
        s3.put_object(Bucket=bucket_name, Key=f"inputs/{filename}", Body=img_byte_arr.getvalue(), ContentType='image/png')
        #save image byte array
        
        s3_uri = f"https://{bucket_name}.s3.amazonaws.com/inputs/{filename}"

        # Save metadata in DynamoDB
        table.put_item(Item={
            'timestamp': datetime.utcnow().isoformat(),
            's3_uri': s3_uri,
            'prediction': str(digit)
        })

        return jsonify({'prediction': digit, 's3_uri': s3_uri})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predictions')
def get_predictions():
    try:
        response = table.scan()
        items = response['Items']
        items.sort(key=lambda x: x['timestamp'], reverse=True)
        return jsonify(items[:5])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
