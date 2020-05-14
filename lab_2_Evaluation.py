import lab_2_InstantiateAndTrainingTaggers
import nltk
import sklearn
from newspaper import Article
from pickle import load
import newspaper
import pickle
test_data=lab_2_InstantiateAndTrainingTaggers.test_sents
#Importing the Taggers
input = open('ugTagger.pkl', 'rb')#Unigram
ug_tagger = load(input)

input.close()


input1 = open('tntTagger.pkl', 'rb',)#Tnt
tnt_tagger = load(input1)
input1.close()

input2 = open('pnTagger.pkl', 'rb')#Percentron Tagger
pn_tagger = load(input2)
input2.close()

# #####################################################################################
# #Print the accuracy of taggers
# print('Accuracy of Unigram Tagger is:',ug_tagger.evaluate(test_data))
# print('Accuracy of Tnt Tagger is:',tnt_tagger.evaluate(test_data))
# print('Accuracy of Perceptron Tagger is:',pn_tagger.evaluate(test_data))
#
# ########################################################################################
# #Metrics Printing of all taggers
# tagged_test_sentences = ug_tagger.tag_sents([[token for token,tag in sent] for sent in test_data])
# # print(tagged_test_sentences)
# gold = [str(tag) for sentence in test_data for token,tag in sentence]
# pred = [str(tag) for sentence in tagged_test_sentences for token,tag in sentence]
# from sklearn import metrics
# print('Metric Value for Unigram Tagger is:-')
# print(metrics.classification_report(gold, pred,zero_division=0))
#
# tagged_test_sentences = tnt_tagger.tag_sents([[token for token,tag in sent] for sent in test_data])
# # print(tagged_test_sentences)
# gold = [str(tag) for sentence in test_data for token,tag in sentence]
# pred = [str(tag) for sentence in tagged_test_sentences for token,tag in sentence]
# from sklearn import metrics
# print('Metric Value for TNT Tagger is:-')
# print(metrics.classification_report(gold, pred,zero_division=0))
#
# tagged_test_sentences = pn_tagger.tag_sents([[token for token,tag in sent] for sent in test_data])
# # print(tagged_test_sentences)
# gold = [str(tag) for sentence in test_data for token,tag in sentence]
# pred = [str(tag) for sentence in tagged_test_sentences for token,tag in sentence]
# print('Metric Value for Perceptron Tagger is:-')
# print(metrics.classification_report(gold, pred,zero_division=0))

#########################################################################################

def basic_sent_chop(data, raw=True):
    """
    Basic method for tokenizing input into sentences
    for this tagger:

    :param data: list of tokens (words or (word, tag) tuples)
    :type data: str or tuple(str, str)
    :param raw: boolean flag marking the input data
                as a list of words or a list of tagged words
    :type raw: bool
    :return: list of sentences
             sentences are a list of tokens
             tokens are the same as the input

    Function takes a list of tokens and separates the tokens into lists
    where each list represents a sentence fragment
    This function can separate both tagged and raw sequences into
    basic sentences.

    Sentence markers are the set of [,.!?]

    This is a simple method which enhances the performance of the TnT
    tagger. Better sentence tokenization will further enhance the results.
    """

    new_data = []
    curr_sent = []
    sent_mark = [",", ".", "?", "!"]

    if raw:
        for word in data:
            if word in sent_mark:
                curr_sent.append(word)
                new_data.append(curr_sent)
                curr_sent = []
            else:
                curr_sent.append(word)

    else:
        for (word, tag) in data:
            if word in sent_mark:
                curr_sent.append((word, tag))
                new_data.append(curr_sent)
                curr_sent = []
            else:
                curr_sent.append((word, tag))
    return new_data


#######################################################################################
#Download 10 articles
setT=['virus','pandemic','government']
countT=[0,0,0]
countNoun=0

url=["https://www.nzherald.co.nz/nz/news/article.cfm?c_id=1&objectid=12317543",
'https://www.nytimes.com/2020/03/20/world/coronavirus-news-usa-world.html',
     'https://www.washingtonpost.com/health/2020/03/20/emily-landon-coronavirus/',
     'https://www.theguardian.com/world/2020/mar/20/behave-or-face-strict-coronavirus-lockdown-germans-told',
     'https://www.cnbc.com/2020/03/20/italy-conte-calls-for-eu-crisis-fund-as-coronavirus-death-toll-rises.html',
     'https://www.ft.com/content/c0755b30-69bb-11ea-800d-da70cff6e4d3',
     'https://www.washingtonpost.com/world/2020/03/19/coronavirus-latest-news/',
     'https://www.aljazeera.com/news/2020/03/germany-authorities-crack-corona-parties-200319205701825.html',
     'https://edition.cnn.com/2020/03/19/europe/prince-albert-monaco-coronavirus-intl-scli/index.html',
     'https://www.cnbc.com/2020/03/19/coronavirus-live-updates.html']
word=[]
tags=[]
for i in range(10):
 # print(url[i])
 article=Article(url[i])
 article.download()
 article.parse()
 sent1=nltk.word_tokenize(article.text)
 tagged_test_sentences1 = tnt_tagger.tagdata(basic_sent_chop(sent1,raw=True))
 word1 = [str(token) for sentence in tagged_test_sentences1 for token, tag in sentence]
 tags1 = [str(tag) for sentence in tagged_test_sentences1 for token, tag in sentence]
 tagged_test_sentences1.append(tagged_test_sentences1)
 word.extend(word1)
 tags.extend(tags1)
 # print(tags)


for i in range(len(word)):
 if(word[i].lower()==setT[0]):
     countT[0]+=1
 if (word[i].lower() == setT[1]):
         countT[1] += 1

 if (word[i].lower() == setT[2]):
         countT[2] += 1
 if(tags[i]=='NN' or tags[i]=='NNS' or tags[i]=='NNP' or tags[i]=='NNPS'):
     countNoun+=1

for i in range(3):
 print('The count for ',setT[i],':',countT[i])
 print('Percentage for ',setT[i],' is:',countT[i]*100/countNoun)
print('The total percentage of setT agaist total number of noun is:',(countT[0]+countT[1]+countT[2])*100/countNoun)
