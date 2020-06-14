import newspaper
from newspaper import Article
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
import nltk
def get_continuous_chunks(text):
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    # print(chunked)
    prev = None
    continuous_chunk = []
    current_chunk = []
    #print(chunked)
    for i in chunked:
        if type(i) == Tree:
            current_chunk.append(" ".join([token for token, pos in i.leaves()]))
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
        else:
            continue

    if continuous_chunk:
        named_entity = " ".join(current_chunk)
        if named_entity not in continuous_chunk:
            continuous_chunk.append(named_entity)

    return continuous_chunk

url='https://www.nzherald.co.nz/nz/news/article.cfm?c_id=1&objectid=12317543'
article=Article(url)
article.download()
# article.html()
article.parse() #parsing the article
txt=article.text


# print(get_continuous_chunks(txt))


def find_word_sentence_num(text):
 word2sentence={}
 for i,sentence in enumerate(text.split('\n')):
    for word in nltk.word_tokenize(sentence):
     word2sentence[word]=i+1
 return word2sentence

word2sentence=find_word_sentence_num(txt)
# print(word2sentence['INZ'])
t_lab=['GPE','LOCATION','PERSON','NONE']
i=0
true_labels=[0,0,3,3,2,3,0,2,0,0,0,0,3,2,3,0,0,2,0,0,0,0,3,0,3,0,3,3,2,3,3,2,3,2,3]
cal_labels=[]
for sent in nltk.sent_tokenize(txt):
   for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
      if hasattr(chunk, 'label'):
          if(chunk.label() in ('GPE','PERSON' ,'LOCATION')):
           print(chunk.label(), ' '.join(c[0] for c in chunk),word2sentence[chunk[0][0]])
           i+=1

           cal_labels.append(chunk.label())
from sklearn.metrics import recall_score,precision_score

c_label=[]
for label in cal_labels:
    if(label==t_lab[0]):
        c_label.append(0)
    if (label == t_lab[1]):
            c_label.append(1)
    if (label == t_lab[2]):
        c_label.append(2)
    if (label == t_lab[3]):
        c_label.append(3)
def cal_recall(true_labels,cal_labels,entity):
    tp=0
    fp=0
    fn=0
    tn=0
    for i,lbl in enumerate(true_labels):
        # print(lbl,true_labels[i])
        if(lbl==entity and cal_labels[i]==entity):
            tp+=1
        if (lbl == entity and cal_labels[i] != entity):
            fp+=1
        if (lbl != entity and cal_labels[i] == entity):
            fn+=1
        if (lbl != entity and cal_labels[i] != entity):
            tn+=1
    # print(tp,fp,tn,fn)
    return float(tp/(tp+fn))
def cal_precision(true_labels,cal_labels,entity):
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for i, lbl in enumerate(true_labels):
        if (lbl == entity and cal_labels[i] == entity):
            tp += 1
        if (lbl == entity and cal_labels[i] != entity):
            fp += 1
        if (lbl != entity and cal_labels[i] == entity):
            fn += 1
        if (lbl != entity and cal_labels[i] != entity):
            tn += 1

    return float(tp / (tp + fp))
p_recall=cal_recall(true_labels,c_label,2)
p_precision=cal_precision(true_labels,c_label,2)
l_recall=cal_recall(true_labels,c_label,0)
l_precision=cal_precision(true_labels,c_label,0)
print('Recall for Location:',l_recall,'Precision for Location:',l_precision)
print('Recall for Person:',p_recall,'Precision for Person:',p_precision)
print('Average Recall:',((l_recall+p_recall)/2),'Average Precision:',(l_precision+p_precision)/2)

##############################################################################
#Results:
# GPE New Zealand 89
# GPE South East Asia 11
# GPE New 89
# PERSON Immigration NZ 23
# PERSON Stephen Vaughan 15
# PERSON Focus 81
# GPE New Zealand 89
# PERSON Health Dr Ashley Bloomfield 77
# GPE Wellington 29
# PERSON Dunedin 29
# GPE Wellington 29
# PERSON Dunedin 29
# GPE Coronavirus 35
# PERSON Logan Park 31
# GPE Coronavirus 35
# PERSON Melville High School 35
# GPE New Zealand 89
# PERSON Vaughan 39
# GPE New Zealand 89
# GPE New Zealand 89
# GPE New Zealand 89
# GPE New Zealand 89
# GPE Tourist 47
# GPE New Zealand 89
# GPE Rucksacker 57
# GPE Christchurch 53
# GPE Rucksacker 57
# GPE Chch 57
# PERSON Chris Lynch 57
# GPE Health 77
# GPE Focus 81
# PERSON Health Dr Ashley Bloomfield 77
# PERSON Focus 81
# PERSON Bloomfield 85
# GPE New 89
# Recall for Location: 0.5454545454545454 Precision for Location: 0.8
# Recall for Person: 0.5384615384615384 Precision for Person: 1.0
# Average Recall: 0.5419580419580419 Average Precision: 0.9
#
# Process finished with exit code 0
