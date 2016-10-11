# nikobad
this is a set of python codes for experiments on different sentiment classification algorithms
#### running different configuration of lstm and dlstm
##### prerequisits
not all the prerquisits are needed for all experiments
1. tensorflow
2. tflearn
3. NLTK
4. sklearn
##### creating PKL file
 - use `utilities/aclimdb_tflearnstyle_preprocess.py` to create the pkl file. change the dataset name and path. the result would be saved in `4-PKLED/` under the name of the `datasetName.pkl`
##### use runner to run lstm and dlstm variation of datsets. 
 - this file can run different configuration on different variations of datasets 
 - open the file and use the section marked by `configuration variations` to set different configurations. possible configurations are commented in the file. you can use the [http://tflearn.org/](http://tflearn.org/) to check the possible variations as well.
 - runner would save each of the model variation under the folder `5-SAVED_MODELS` and will not overwrite the model. if you need to redo the training on one model remove the coresponding files from the folder
 - each run's result would be saved under `./tflearn_logs/` with a special directory name format:
```
    <classifier>'-DS'<dataset_name>'-EMBEDDING'<number_of_words_used_in_embedding>'-DROPOUT'<dropout>'-N_EPOCH'<n_epoch>'-LOSS'<loss>'-OPTIMIZER'<optimizer>
```

##### bidirectional  RNN
- to run  different variations of  use ```bidirectional_lstm.py``` file. due to some problems with tflearn running consecutive Bidirectional RNN was not possible

#### logistic regression on doc2vec representation of words
- to run a logistic regression on the results of doc2vec processing of a dataset you should first change the format of the dataset to a doc2vec compatible format using `utilities/convertFromDirToSingleFile.py`. this would add .txt files in the same address
- use `utilities/makeDoc2VecPkl.py` to create a doc2vec pkl file. this would take some time and the result would be saved in 4-PKLED as `<datasetName>.d2v`. you can also change the properties of doc2vec learner to get better results for word vectors.
- use   `doc2vec-logistic-regression.py` for executing logistic regression. the results would be logged in tensorboard readable style under  `'doc2vecds-'<dataset>'optimizer-'<optimizer>'loss-'<loss_function>`
#### classification using fasttext
1. use `convertFromDirToFastTextFormat.py` to convert datset dirs to a fasttext compatible format
2. use `makeFastTextModel.py` to make fasttext model.  number of epochs and window size and word ngrams are customizable. this would save the fasttext model in the curret directory
3. use `utilities/makeFastTextpklFile.py` to create pkl file suited for fast text. this would save a pkl file under 4-PKLED folder.
4. use `classification-example.sh` file from [fasttext](https://github.com/facebookresearch/fastText) for classification

#### classification using naive bayes
1. use `mobayes/naiveBayes.py` for executing the bayesclassifier on a dataset with the regular dir format
