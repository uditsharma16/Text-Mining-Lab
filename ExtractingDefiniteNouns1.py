import nltk
import newspaper
from newspaper import Article

#-----part of test runs

# sent = "I saw a boy in the park with a telescope and the boy was sitting under an apple tree."
# taggedS = nltk.pos_tag(nltk.word_tokenize(sent))
#
#
# sentences = nltk.sent_tokenize(sent) #tokenize sentences
#
# n=len(taggedS)
#
# i=0
# while(i<n):
#  if(taggedS[i][1]=='DT' and taggedS[i][0]=='the' and taggedS[i+1][1]=='NN'):
#    print(taggedS[i+1][0])
#
#  i+=1



#using the newspaper3k package to import articles from Nz herald

url='https://www.nzherald.co.nz/nz/news/article.cfm?c_id=1&objectid=12317543'
article=Article(url)
article.download()
# article.html()
article.parse() #parsing the article
sent=article.text
print(nltk.word_tokenize(sent))
taggedS = nltk.pos_tag(nltk.word_tokenize(sent))
print(taggedS)
count=0
# sentences = nltk.sent_tokenize(sent) #tokenize sentences

n=len(taggedS)
print('The definite noun sequence is:-')
i=0
d_nouns=[]

#LOOP to find out the definite nouns
while(i<n):
 if(taggedS[i][1]=='DT' and (taggedS[i][0]=='the' or taggedS[i][0]=='The' ) and taggedS[i+1][1]=='NN'):
   d_nouns.append(taggedS[i + 1][0])
   # print(count + 1, '. ', taggedS[i + 1][0])
 count+=1
 i+=1
d_nouns.sort()

#printing definite nouns and their count
print(d_nouns)
print('The number of definite noun is:',count)

