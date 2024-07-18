from flask import Flask, request, jsonify
import os
import json
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from test_nlp_claims_project import SequentialModel, EncoderDecoderBERT

app = Flask(__name__)

# Set environment variables for AWS
os.environ['AWS_ACCESS_KEY_ID'] = 'AKIAW3MECQ67O4AGEJO7'
os.environ['AWS_SECRET_ACCESS_KEY'] = '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'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-2'

# Initialize BERT model and tokenizer
bert_model = TFBertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Initialize the SequentialModel
output_dim = 1
model = SequentialModel(output_dim)

# Load model weights from S3
s3 = boto3.client('s3')
with open('/tmp/model_weights.h5', 'wb') as f:
    s3.download_fileobj('claimssumbul', 'model_weights.h5', f)
model.load_weights('/tmp/model_weights.h5')

# Tokenize the string inputs using BERT tokenizer
def tokenize_inputs(texts):
    tokenized = tokenizer(texts, padding=True, truncation=True, return_tensors='tf')
    return {'input_ids': tokenized['input_ids'], 'attention_mask': tokenized['attention_mask']}

def preprocess_input(float_features, string_features, bert_model):
    tdp, upcs = float_features
    products, category, manufacturer, time = string_features

    tensor_Products = tokenize_inputs([products])
    tensor_Category = tokenize_inputs([category])
    tensor_Manufacturer = tokenize_inputs([manufacturer])
    tensor_Time = tokenize_inputs([time])

    bert_output_product = bert_model(tensor_Products)[1]
    bert_output_category = bert_model(tensor_Category)[1]
    bert_output_manufacturer = bert_model(tensor_Manufacturer)[1]
    bert_output_time = bert_model(tensor_Time)[1]

    concatenate_string_inputs = tf.concat([bert_output_product, bert_output_category, bert_output_manufacturer, bert_output_time], axis=-1)

    tensor_TDP = tf.convert_to_tensor([tdp], dtype=tf.float32)
    tensor_TDP = tf.reshape(tensor_TDP, (tensor_TDP.shape[0], 1))

    tensor_UPCs = tf.convert_to_tensor([upcs], dtype=tf.float32)
    tensor_UPCs = tf.reshape(tensor_UPCs, (tensor_UPCs.shape[0], 1))

    return [concatenate_string_inputs, tensor_TDP, tensor_UPCs]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        float_features = [data['TDP'], data['UPCs']]
        string_features = [data['Products'], data['Category'], data['Manufacturer'], data['Time']]
        
        processed_input = preprocess_input(float_features, string_features, bert_model)

        prediction = model.predict(processed_input)
        return jsonify({'predicted_sales': prediction[0].tolist()})
    
    except KeyError as e:
        return jsonify({'error': f'Missing required parameter: {e}'}), 400

if __name__ == '__main__':
    app.run(debug=True)
