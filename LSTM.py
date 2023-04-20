import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense

# Load the cleaned data
cleaned_data = pd.read_csv('Cleaned_EN_PL_dataset.csv')

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Tokenize the source and target sentences
source_tokenizer = Tokenizer(filters='')
source_tokenizer.fit_on_texts(cleaned_data['source'])
source_sequences = source_tokenizer.texts_to_sequences(cleaned_data['source'])

target_tokenizer = Tokenizer(filters='')
target_tokenizer.fit_on_texts(cleaned_data['target'])
target_sequences = target_tokenizer.texts_to_sequences(cleaned_data['target'])

# Pad the sequences
max_source_seq_length = max([len(seq) for seq in source_sequences])
max_target_seq_length = max([len(seq) for seq in target_sequences])

source_data = pad_sequences(source_sequences, maxlen=max_source_seq_length, padding='post')
target_data = pad_sequences(target_sequences, maxlen=max_target_seq_length, padding='post')

source_train, source_val, target_train, target_val = train_test_split(source_data, target_data, test_size=0.2, random_state=42)

from tensorflow.keras.layers import Embedding

embedding_dim = 256
latent_dim = 1024

# Encoder
encoder_inputs = Input(shape=(None,))
enc_emb = Embedding(len(source_tokenizer.word_index) + 1, embedding_dim, mask_zero=True)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,))
dec_emb_layer = Embedding(len(target_tokenizer.word_index) + 1, embedding_dim, mask_zero=True)
dec_emb = dec_emb_layer(decoder_inputs)

decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)

decoder_dense = Dense(len(target_tokenizer.word_index) + 1, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

batch_size = 64
epochs = 30
history = model.fit([source_train, target_train], np.expand_dims(target_train, -1),
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=([source_val, target_val], np.expand_dims(target_val, -1)))
