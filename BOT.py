import random
import string
import pandas as pd
import json
import tensorflow as tf
import keras
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
from keras.preprocessing.text import Tokenizer
from keras.layers import Input, Embedding, LSTM , Dense,GlobalMaxPooling1D,Flatten
from keras.models import Model


with open('content.json') as content:
  data1 = json.load(content)


tags = []
inputs = []
responses={}
for intent in data1['intents']:
  responses[intent['tag']]=intent['responses']
  for lines in intent['input']:
    inputs.append(lines)
    tags.append(intent['tag'])

data = pd.DataFrame({"inputs":inputs,
                     "tags":tags})

data = data.sample(frac=1)

data['inputs'] = data['inputs'].apply(lambda wrd:[ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
data['inputs'] = data['inputs'].apply(lambda wrd: ''.join(wrd))

tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(data['inputs'])
train = tokenizer.texts_to_sequences(data['inputs'])
#apply padding

x_train = pad_sequences(train)

#encoding the outputs

le = LabelEncoder()
y_train = le.fit_transform(data['tags'])

input_shape = x_train.shape[1]
vocabulary = len(tokenizer.word_index)
print("number of unique words : ",vocabulary)
output_length = le.classes_.shape[0]
print("output length: ",output_length)

i = Input(shape=(input_shape,))
x = Embedding(vocabulary+1,10)(i)
x = LSTM(10,return_sequences=True)(x)
x = Flatten()(x)
x = Dense(output_length,activation="softmax")(x)


class Bot():
    def __init__(self):
        self.model = Model(i,x)
    def train(self):
        self.model.compile(loss="sparse_categorical_crossentropy",optimizer='adam',metrics=['accuracy'])

        train = self.model.fit(x_train,y_train,epochs=200)
        self.model.save('my_model',train)
    def load_model(self):
        my_model = keras.models.load_model("my_model")

        self.model = my_model
        loss, accuracy = self.model.evaluate(x_train, y_train)
        print(f"accuracy: {accuracy * 100:.2f}%")
    def response(self,user_input):
        texts_p = []
        prediction_input = user_input

        #removing punctuation and converting to lowercase
        prediction_input = [letters.lower() for letters in prediction_input if letters not in string.punctuation]
        prediction_input = ''.join(prediction_input)
        texts_p.append(prediction_input)

        #tokenizing and padding
        prediction_input = tokenizer.texts_to_sequences(texts_p)
        prediction_input = np.array(prediction_input).reshape(-1)
        prediction_input = pad_sequences([prediction_input],input_shape)

        #getting output from model
        output = self.model.predict(prediction_input)
        output = output.argmax()

        #finding the right tag and predicting
        response_tag = le.inverse_transform([output])[0]
        res = random.choice(responses[response_tag])
        return res

