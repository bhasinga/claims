import os
import json
import boto3
import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

from io import StringIO

os.environ['AWS_ACCESS_KEY_ID'] = 'AKIAW3MECQ67O4AGEJO7'
os.environ['AWS_SECRET_ACCESS_KEY'] = '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'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-2'

# Read CSV from S3
def read_csv_from_s3(bucket, file_key, dtype_dict):
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket, Key=file_key)
    csv_content = response['Body'].read().decode('utf-8')
    return pd.read_csv(StringIO(csv_content), dtype=dtype_dict)

# S3 bucket and file keys
bucket_name = 'claimssumbul'
file_key = 'Claims_w_Sales_txt.csv'

# Data types dictionary
dtype_dict = {
    'Products': 'str',
    'CATEGORY': 'str',
    'TDP' : 'float', 
    'SUB_CATEGORY': 'str',
    'SEGMENT' : 'str',
    'MANUFACTURER': 'str',
    'SALES': 'float',
    'TIME': 'str',
    'UPCs' : 'int'
}



# Read CSV from S3
df_new = read_csv_from_s3(bucket_name, file_key, dtype_dict)
df_new = df_new[['Products', 'CATEGORY', 'TDP', 'SUB_CATEGORY', 'SEGMENT', 'MANUFACTURER', 'TIME', 'UPCs', 'SALES']].dropna()

# Normalize numerical data
min_max_scaler = MinMaxScaler()
df_new[['TDP', 'UPCs']] = min_max_scaler.fit_transform(df_new[['TDP', 'UPCs']])

import tensorflow as tf
import tensorflow as tf
min_max_scaler = MinMaxScaler()
df_new[['SALES']] = min_max_scaler.fit_transform(df_new[['SALES']])

# Extract SALES column as a TensorFlow tensor of type float32
y_sales = tf.convert_to_tensor(df_new['SALES'].values, dtype=tf.float32)

# Reshape the tensor if needed
y_sales = tf.reshape(y_sales, (y_sales.shape[0], 1))

print("Tensor shape:", y_sales.shape)


print("Tensor shape:", y_sales.shape)

Products = df_new['Products'].tolist()
Category = df_new['CATEGORY'].tolist()
Manufacturer = df_new['MANUFACTURER'].tolist()
Time = df_new['TIME'].tolist()

class EncoderDecoderBERT:
    def __init__(self, bert_model, tokenizer):
        self.bert = bert_model
        self.tokenizer = tokenizer

    def tokenize_inputs(self, texts):
        tokenized = self.tokenizer(texts, padding=True, truncation=True, return_tensors='tf')
        return {'input_ids': tokenized['input_ids'], 'attention_mask': tokenized['attention_mask']}

    def call(self, inputs):
        Products, Category, Manufacturer, Time = inputs
        tensor_Products = self.tokenize_inputs(Products)
        tensor_Category = self.tokenize_inputs(Category)
        tensor_Manufacturer = self.tokenize_inputs(Manufacturer)
        tensor_Time = self.tokenize_inputs(Time)

        bert_output_product = self.bert(tensor_Products)[1]
        bert_output_category = self.bert(tensor_Category)[1]
        bert_output_manufacturer = self.bert(tensor_Manufacturer)[1]
        bert_output_time = self.bert(tensor_Time)[1]

        return bert_output_product, bert_output_category, bert_output_manufacturer, bert_output_time

# Initialize BERT model and tokenizer
bert_model = TFBertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Initialize the model
bert_embeddings = EncoderDecoderBERT(bert_model, tokenizer)
bert_output_product, bert_output_category, bert_output_manufacturer, bert_output_time = bert_embeddings.call((Products, Category, Manufacturer, Time))
concatenate_string_inputs = tf.concat([bert_output_product, bert_output_category, bert_output_manufacturer, bert_output_time], axis=-1)

tensor_TDP = tf.convert_to_tensor(df_new['TDP'].values, dtype=tf.float32)
tensor_TDP = tf.reshape(tensor_TDP, (tensor_TDP.shape[0], 1))

tensor_UPCs = tf.convert_to_tensor(df_new['UPCs'].values, dtype=tf.float32)
tensor_UPCs = tf.reshape(tensor_UPCs, (tensor_UPCs.shape[0], 1))

class SequentialModel(tf.keras.Model):
    def __init__(self, output_dim):
        super(SequentialModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.dense2 = tf.keras.layers.Dense(16, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.output_layer = tf.keras.layers.Dense(output_dim)

    def call(self, inputs):
        concatenate_string_inputs, tensor_TDP, tensor_UPCs = inputs
        concatenated_inputs = tf.concat([concatenate_string_inputs, tensor_TDP, tensor_UPCs], axis=-1)
        x = self.dense1(concatenated_inputs)
        x = self.dense2(x)
        x = self.dropout(x)
        output = self.output_layer(x)
        return output

    
        return self.output_layer(x)

# Initialize the SequentialModel
output_dim = 1
model = SequentialModel(output_dim)
model.compile(optimizer=Adam(learning_rate=0.00001), loss='mse')

# Implement gradient clipping
optimizer = Adam(learning_rate=0.00001, clipvalue=1.0)
model.compile(optimizer=optimizer, loss='mse')

# Fit the model
model.fit(
    (concatenate_string_inputs, tensor_TDP, tensor_UPCs),
    y_sales,
    epochs=150,
    batch_size=32,
    validation_split=0.2
)

# Predictions
y_pred = model.predict((concatenate_string_inputs, tensor_TDP, tensor_UPCs))
y_pred_np = np.array(y_pred).flatten()
y_true_np = np.array(y_sales).flatten()




# Calculate metrics
mse = mean_squared_error(y_true_np, y_pred_np)
mae = mean_absolute_error(y_true_np, y_pred_np)
rmse = np.sqrt(mse)
r2 = r2_score(y_true_np, y_pred_np)

print(f'Mean Squared Error (MSE): {mse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'R2 Score: {r2}')

import tempfile

# Function to upload a file to S3
def upload_to_s3(file_name, bucket, object_name=None):
    s3_client = boto3.client('s3')
    try:
        s3_client.upload_file(file_name, bucket, object_name or file_name)
    except Exception as e:
        print(f"Error uploading file to S3: {e}")
        return False
    return True

# Save model weights to a temporary file
with tempfile.NamedTemporaryFile(suffix=".h5") as temp_file:
    model.save_weights(temp_file.name)

    # Set your AWS S3 bucket and weights file name
    s3_weights_key = 'model_weights.h5'

    # Upload weights file to S3
    upload_to_s3(temp_file.name, bucket_name, s3_weights_key)
