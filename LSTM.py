import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class Seq2SeqModel:
    
    def __init__(self, data_path, embedding_dim=256, latent_dim=1024, batch_size=64, epochs=30, test_size=0.2, random_state=42):
        self.data_path = data_path
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.test_size = test_size
        self.random_state = random_state
        
    def load_data(self):
        # Load the cleaned data
        cleaned_data = pd.read_csv(self.data_path)
        
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
        
        # Split the data into training and validation sets
        source_train, source_val, target_train, target_val = train_test_split(source_data, target_data, test_size=self.test_size, random_state=self.random_state)
        
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.max_source_seq_length = max_source_seq_length
        self.max_target_seq_length = max_target_seq_length
        self.source_train = source_train
        self.source_val = source_val
        self.target_train = target_train
        self.target_val = target_val
    
    def build_model(self):
        # Encoder
        encoder_inputs = Input(shape=(None,))
        enc_emb = Embedding(len(self.source_tokenizer.word_index) + 1, self.embedding_dim, mask_zero=True)(encoder_inputs)
        encoder_lstm = LSTM(self.latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
        encoder_states = [state_h, state_c]

        # Decoder
        decoder_inputs = Input(shape=(None,))
        dec_emb_layer = Embedding(len(self.target_tokenizer.word_index) + 1, self.embedding_dim, mask_zero=True)
        dec_emb = dec_emb_layer(decoder_inputs)

        decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)

        decoder_dense = Dense(len(self.target_tokenizer.word_index) + 1, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Model
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        self.model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
    
    def train(self):
        history = self.model.fit([self.source_train, self.target_train], np.expand_dims(self.target_train, -1),
                                 batch_size=self.batch_size,epochs=self.epochs,validation_data=([self.source_val, self.target_val], 
                                                                                                np.expand_dims(self.target_val, -1)))
        self.history = history

model = Seq2SeqModel('Cleaned_EN_PL_dataset.csv')
model.load_data()
model.build_model()
model.train()

