import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder    

#switch between part 1 and part 3
#(also switch below) falls out of scope otherwise would just use the same variable for everything
part = int(input("load data for part [1] or [3]"))

# load datasets and tokenize features
if part == 1:
    data = pd.read_csv('Sentiment.csv')
    
    # Keeping only the neccessary columns
    data = data[['text','sentiment']]
    data = data[data.sentiment != 'Neutral']

    for idx, row in data.iterrows():
        row[0] = row[0].replace('rt', ' ')

    data = data[data.sentiment != "Neutral"]
    data['text'] = data['text'].apply(lambda x: x.lower())
    data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

    max_fatures = 2000
    tokenizer = Tokenizer(num_words=max_fatures, split=' ')
    tokenizer.fit_on_texts(data['text'].values)
    X = tokenizer.texts_to_sequences(data['text'].values)
    X = pad_sequences(X)

    labelencoder = LabelEncoder()
    integer_encoded = labelencoder.fit_transform(data['sentiment'])
    y = to_categorical(integer_encoded)
    X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.33, random_state = 42)


if part == 3:
    data = pd.read_csv('spam.csv', encoding="ISO-8859-1")
    data['v2'] = data['v2'].apply(lambda x: x.lower())
    data['v2'] = data['v2'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

    for idx, row in data.iterrows():
        row[0] = row[0].replace('rt', ' ')

    max_fatures = 2000
    tokenizer = Tokenizer(num_words=max_fatures, split=' ')
    tokenizer.fit_on_texts(data['v2'].values)
    X = tokenizer.texts_to_sequences(data['v2'].values)
    X = pad_sequences(X)

    labelencoder = LabelEncoder()
    integer_encoded = labelencoder.fit_transform(data['v1'])
    y = to_categorical(integer_encoded)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)

embed_dim = 128
lstm_out = 196
def createmodel():
    model = Sequential()
    model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    return model

def predict(text,tokenizer,model,maxlength):
    text = text.lower()
    text = re.sub('[^a-zA-z0-9\s]', '', text)
    l = [text]
    tokenizer.fit_on_texts(l)
    l = tokenizer.texts_to_sequences(l)
    l = pad_sequences(l,maxlen=maxlength)
    p = model.predict(l)
    pv = np.argmax(p)
    #print(p,pv)
    if pv == 0:
        print("positive")
    else:
        print("negative")

#try to load the model from part 1/3 if it exists (delete to retrain)
try:
    
    
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("model.h5")
    print("loaded")

    part = int(input("test part [1] or [3]"))
    
    if part == 1:
        #run prediction until user enters "quit"
        while True:
            text = input("input tweet or quit: ")
            if text.lower() == "quit":
                break
            predict(text,tokenizer,model,28)

    elif part == 3:
        #run prediction until user enters "quit"
        while True:
            text = input("input text or quit: ")
            if text.lower() == "quit":
                break
            predict(text,tokenizer,model,152)

#if the model doesn't exist, train either part 1 or 3.
except Exception as e:
    print(e)
    input("press enter to train...")

##    # Uncomment for part 2
##    # Part 2
##    model = KerasClassifier(build_fn=createmodel, verbose=0)
##    batch_size = [10, 20, 40]
##    epochs = [1, 2]
##    param_grid = dict(batch_size=batch_size, epochs=epochs)
##    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
##    grid_result = grid.fit(X_train, Y_train)
##    # summarize results
##    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    #switch between part 1 and part 3
    part = int(input("train for part [1] or [3]"))

    if part == 1:
        batch_size = 40
        model = createmodel()
        model.fit(X_train, Y_train, epochs = 3, batch_size=batch_size, verbose = 2)
        score,acc = model.evaluate(X_test,Y_test,verbose=2,batch_size=batch_size)
        print(score)
        print(acc)

        # save model
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("model.h5")
        print("Saved model to disk")

    elif part == 3:
        batch_size = 10
        model = createmodel()
        model.fit(X_train, Y_train, epochs = 2, batch_size=batch_size, verbose = 2)
        score,acc = model.evaluate(X_test,Y_test,verbose=2,batch_size=batch_size)
        print(score)
        print(acc)

        # save model
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("model.h5")
        print("Saved model to disk")


