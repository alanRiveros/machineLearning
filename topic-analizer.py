import sys
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

text = np.array(["this is a nice son of a bitch"])
model = load_model("model.h5")

# loading
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

list_tokenized_test = tokenizer.texts_to_sequences(text)
print(list_tokenized_test)
maxlen = 200
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)

prediction = model.predict(X_te)
print(prediction)