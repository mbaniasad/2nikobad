import tflearn
import numpy as np
from  LSTMconfig import LSTMconfig 
cl = LSTMconfig("LSTM","aclImdb",10000,0.8,12,'categorical_crossentropy','sgd')



trainX= np.zeros((3000,1000))
print trainX.shape
trainY= np.zeros(3000)
testX= np.zeros((3000,1000))
testY=np.zeros(3000)

# Data preprocessing
# Sequence padding
# trainX = pad_sequences(trainX, maxlen=100, value=0.)
# testX = pad_sequences(testX, maxlen=100, value=0.)
# Converting labels to binary vectors
# trainY = to_categorical(trainY, nb_classes=2)
# testY = to_categorical(testY, nb_classes=2)

# Network building
net = tflearn.input_data([None, 1000])
# net = tflearn.embedding(net, input_dim=cl.number_of_words_used_in_embedding, output_dim=128)
net = tflearn.lstm(net, 128, dropout=cl.dropout)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer=cl.optimizer, learning_rate=0.001,
                         loss=cl.loss)

# Training
model = tflearn.DNN(net, tensorboard_verbose=0, tensorboard_dir='./tflearn_logs/'+cl.setting_name())
model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,
          batch_size=10, n_epoch=cl.n_epoch)

# model.save("./5-SAVED_MODELS/"+cl.setting_name())
