# -*- coding: utf-8 -*-
import tflearn
from gensim.models import Doc2Vec
import matplotlib.pyplot as plt

net = tflearn.input_data([None, 100])
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='Momentum', learning_rate=0.001, loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0, tensorboard_dir='./tflearn_logs/')
model.load("./5-SAVED_MODELS/doc2vecds-GAR-cls-acl10optimizer-Momentumloss-categorical_crossentropy")

gensimModel = Doc2Vec.load("./4-PKLED/GAR-cls-acl10.d2v")


def doc2vec_online_prediction(sentence):
	global gensimModel, model
	sentence_array = sentence.split()
	vectors = []
	for i in range(len(sentence_array)):
		sentence = sentence_array[0:i+1]
		# print sentence
		vector= gensimModel.infer_vector(sentence, alpha=0.1, min_alpha=0.0001, steps=5)
	 	vectors.append(vector)
	predictions = model.predict(vectors)
	return predictions
		



predictions= doc2vec_online_prediction("die beste musik immer")
# print predictions




plt.plot(predictions)
plt.ylabel('polarity')
plt.show()
