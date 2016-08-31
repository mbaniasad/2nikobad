class LSTMconfig:
    
    def __init__(self,classifier,dataset_name,number_of_words_used_in_embedding,dropout,n_epoch,loss,optimizer):
    	self.dataset_name=dataset_name
        self.dataset_path = './4-PKLED/'+dataset_name + '.pkl'
        self.number_of_words_used_in_embedding  = int(number_of_words_used_in_embedding)
        self.dropout = float(dropout)
        self.n_epoch = int(n_epoch)
        self.loss = loss
        self.optimizer = optimizer
        self.classifier=classifier

    def setting_name(self):
        return  (self.classifier+
        		'-DS'+self.dataset_name +
        		'-EMBEDDING'+str(self.number_of_words_used_in_embedding)+
        		'-DROPOUT'+str(self.dropout)+
        		'-N_EPOCH'+str(self.n_epoch)+
        		'-LOSS'+ self.loss+
        		'-OPTIMIZER'+self.optimizer)
