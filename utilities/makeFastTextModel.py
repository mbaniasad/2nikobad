from __future__ import division
import fasttext

epoch = 10 #number of epochs
ws=5 #size of context window
word_ngrams=3 #ngfram

datasetName = "aclImdb"
datasetDir = "../3-PREPROCESSED/LOWER-NOPUNC/"+datasetName+"/"
train_file = datasetDir+"fastTextFomatTrain.txt"
# Skipgram model
model = fasttext.supervised(train_file, datasetName+"-fasttextModel-skipGram-ws"+str(ws)+"-epoch"+str(epoch)+"-word_ngrams"+str(word_ngrams),ws=5,epoch = epoch,silent=0)
model = fasttext.load_model(datasetName+"-fasttextModel-skipGram-ws"+str(ws)+"-epoch"+str(epoch)+"-word_ngrams"+str(word_ngrams)+".bin")


with open("../3-PREPROCESSED/LOWER-NOPUNC/"+datasetName+"/fastTextFomatTest.txt") as f:
	texts=f.readlines()
sentences = []
labels = []	
count = 0
for txt in texts:
	count +=1
	sentences.append( txt[:txt.find("__label__")])
	lblindex = txt.find("__label__")
	labels.append(txt[lblindex:lblindex+10])

predicted_labels = model.predict(sentences)


i =correct=unknown= 0
for i  in range(len(predicted_labels)-1):
	# print predicted_labels[i]
	# print i
	try:
		if predicted_labels[i][0]== labels[i] :
			correct+=1
	except:
		unknown+=1

print correct, i+1 , (correct/(i+1)), ["unknow-->",unknown]
# CBOW model
# model = fasttext.supervised(train_file, datasetName+"-fasttextModel-cbow")



