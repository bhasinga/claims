from flask import Flask, request, jsonify
import tensorflow as tf
import boto3
from transformers import BertTokenizer, TFBertModel
from test_nlp_claims_project import EncoderDecoderBERT, SequentialModel
import os

app = Flask(__name__)

os.environ['AWS_ACCESS_KEY_ID'] = 'AKIAW3MECQ67O4AGEJO7'
os.environ['AWS_SECRET_ACCESS_KEY'] = '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'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-2'

def create_model():
    bert_model = TFBertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_embeddings = EncoderDecoderBERT(bert_model, tokenizer)
    output_dim = 1
    model = SequentialModel(output_dim)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), loss='mse')
    return model, bert_embeddings, tokenizer

model, bert_embeddings, tokenizer = create_model()
model.load_weights('model_weights.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        float_features = [data['TDP'], data['UPCs']]
        string_features = [data['Products'], data['Category'], data['Manufacturer'], data['Time']]

        processed_input = preprocess_input(float_features, string_features, bert_embeddings)

        prediction = model.predict(processed_input)
        return jsonify({'predicted_sales': prediction[0].tolist()})
    
    except KeyError as e:
        return jsonify({'error': f'Missing required parameter: {e}'}), 400

def preprocess_input(float_features, string_features, bert_embeddings):
    tensor_TDP = tf.convert_to_tensor([float_features[0]], dtype=tf.float32)
    tensor_TDP = tf.reshape(tensor_TDP, (tensor_TDP.shape[0], 1))

    tensor_UPCs = tf.convert_to_tensor([float_features[1]], dtype=tf.float32)
    tensor_UPCs = tf.reshape(tensor_UPCs, (tensor_UPCs.shape[0], 1))

    bert_output_product, bert_output_category, bert_output_manufacturer, bert_output_time = bert_embeddings.call((
        [string_features[0]], [string_features[1]], [string_features[2]], [string_features[3]]
    ))

    concatenate_string_inputs = tf.concat([bert_output_product, bert_output_category, bert_output_manufacturer, bert_output_time], axis=-1)
    return (concatenate_string_inputs, tensor_TDP, tensor_UPCs)

if __name__ == '__main__':
    app.run(debug=True)
