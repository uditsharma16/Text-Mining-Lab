import nltk
from nltk.corpus import brown
from pickle import dump
from nltk.tag import CRFTagger
from nltk.tag import tnt
from pickle import load
############################################################################################
brown_tagged_sents = brown.tagged_sents(categories='news')
#Spliting the dataset into 80-20
size = int(len(brown_tagged_sents) * 0.8)
# print(len(brown_tagged_sents))
#print(size)
train_sents = brown_tagged_sents[:size]
test_sents = brown_tagged_sents[size:]
#############################################################################################
#Intiating the Targgers
unigram_tagger = nltk.UnigramTagger(train_sents)#Unigram tagger and training
tnt_tagger = tnt.TnT() #Tnt
perceptron_tagger=nltk.PerceptronTagger(train_sents)#perceptron
crf_tagger=CRFTagger()#crf

#Training the Taggers
tnt_tagger.train(train_sents)
# tnt_tagger_preds = tnt_tagger.tag_sents([[word for word,_ in test_sents] for test_sent in train_sents])
# print(tnt_tagger.tag(test_sents))
# perceptron_tagger.train(train_sents)
# crf_tagger.train(train_sents)
# crf_tagger.train(train_sents,'/tmp/crf_tagger')
# crf_tagger.train(train_sents,'model.crf.tagger')

##############################################################################################
#Creating the Pickle files
# output = open('ugTagger.pkl', 'wb')#Unigram Tagger
# dump(unigram_tagger, output, -1)
# output.close()
#
# output = open('tntTagger.pkl', 'wb')#Tnt Tagger
# dump(tnt_tagger, output,-1)
# output.close()
# #
# output = open('pnTagger.pkl', 'wb')#Perceptron Tagger
# dump(perceptron_tagger, output, -1)
# output.close()
#
# output = open('crfTagger.pkl', 'wb')#Crf Tagger
# dump(crf_tagger, output, -1)
# output.close()

##################################################################################################
# #Importing the Taggers
# input = open('ugTagger.pkl', 'rb')#Unigram
# ug_impTagger = load(input)
# input.close()
#
# input = open('tntTagger.pkl', 'rb')#Tnt
# tnt_impTagger = load(input)
# input.close()
#
# input = open('pnTaggerTagger.pkl', 'rb')#Percentron Tagger
# pn_impTagger = load(input)
# input.close()
#

# input = open('crfTaggerTagger.pkl', 'rb')#Percentron Tagger
# crf_impTagger = load(input)
# input.close()
#


















