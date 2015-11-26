
# coding: utf-8

# In[496]:

import numpy as np
import email
import os
from textblob import TextBlob
from textblob import Word
import json
import time 
import datetime 
from datetime import date
from datetime import timedelta
import re


# In[273]:

stop_w1 = ["'m", "am", "all","she'll", "don't", 'being', 'over', 'through', 'during', 'its', 'before', "he's", "when's", "we've", 'had', 'should', "he'd", 'to', 'only', 'does', "here's", 'under', 'ours', 'has', "haven't", 'then', 'them', 'his', 'above', 'very', "who's", "they'd", 'cannot', "you've", 'they', 'not', 'yourselves', 'him', 'nor', "we'll", 'did', "they've", 'these', 'she', 'each', "won't", 'where', "mustn't", "isn't", "i'll", "why's", 'because', "you'd", 'doing', 'some', "hasn't", "we'd", 'further', 'ourselves', "shan't", 'what', 'for', 'herself', 'below', "there's", "shouldn't", "they'll", 'between', 'be', 'we', 'who', "doesn't", 'of', 'here', "hadn't", "aren't", 'by', 'both', 'about', 'her', 'theirs', "wouldn't", 'against', "i'd", "weren't", "i'm", 'or', "can't", 'this', 'own', 'into', 'yourself', 'down', 'hers', "couldn't", 'your', "you're", 'from', "how's", 'would', 'whom', "it's", 'there', 'been', "he'll", 'their', "we're", 'themselves', 'was', 'until', 'too', 'himself', 'that', "didn't", "what's", 'but', 'it', 'with', 'than', 'those', 'he', 'me', "they're", 'myself', "wasn't", 'up', 'while', 'ought', 'were', 'more', 'my', 'could', 'are', 'and', 'do', 'is', 'am', 'few', 'an', 'as', 'itself', 'at', 'have', 'in', 'any', 'if', 'again', "let's", 'no', "i've", 'when', 'same', 'how', 'other', 'which', 'you', 'out', 'our', 'after', "where's", 'most', 'such', 'on', 'why', 'a', 'off', 'i', "she'd", 'having', "you'll", 'so', "she's", 'the', 'once', 'yours', "that's"]
stop_w2 = [i.capitalize() for i in stop_w1]
stop_w = stop_w2 + stop_w1
confidence = ["just","sorry","hopefully","actually","kind of"]
positive = ['absolutely', 'adorable', 'accepted', 'acclaimed', 'accomplish', 'accomplishment', 'achievement', 'action', 'active', 'admire', 'adventure', 'affirmative', 'affluent', 'agree', 'agreeable', 'amazing', 'angelic', 'appealing', 'approve', 'aptitude', 'attractive', 'awesome', 'beaming', 'beautiful', 'believe', 'beneficial', 'bliss', 'bountiful', 'bounty', 'brave', 'bravo', 'brilliant', 'bubbly', 'calm', 'celebrated', 'certain', 'champ', 'champion', 'charming', 'cheery', 'choice', 'classic', 'classical', 'clean', 'commend', 'composed', 'congratulation', 'constant', 'cool', 'courageous', 'creative', 'cute', 'dazzling', 'delight', 'delightful', 'distinguished', 'divine', 'earnest', 'easy', 'ecstatic', 'effective', 'effervescent', 'efficient', 'effortless', 'electrifying', 'elegant', 'enchanting', 'encouraging', 'endorsed', 'energetic', 'energized', 'engaging', 'enthusiastic', 'essential', 'esteemed', 'ethical', 'excellent', 'exciting', 'exquisite', 'fabulous', 'fair', 'familiar', 'famous', 'fantastic', 'favorable', 'fetching', 'fine', 'fitting', 'flourishing', 'fortunate', 'free', 'fresh', 'friendly', 'fun', 'funny', 'generous', 'genius', 'genuine', 'giving', 'glamorous', 'glowing', 'good', 'gorgeous', 'graceful', 'great', 'green', 'grin', 'growing', 'handsome', 'happy', 'harmonious', 'healing', 'healthy', 'hearty', 'heavenly', 'honest', 'honorable', 'honored', 'hug', 'idea', 'ideal', 'imaginative', 'imagine', 'impressive', 'independent', 'innovate', 'innovative', 'instant', 'instantaneous', 'instinctive', 'intuitive', 'intellectual', 'intelligent', 'inventive', 'jovial', 'joy', 'jubilant', 'keen', 'kind', 'knowing', 'knowledgeable', 'laugh', 'legendary', 'light', 'learned', 'lively', 'lovely', 'lucid', 'lucky', 'luminous', 'marvelous', 'masterful', 'meaningful', 'merit', 'meritorious', 'miraculous', 'motivating', 'moving', 'natural', 'nice', 'novel', 'now', 'nurturing', 'nutritious', 'okay', 'one', 'one-hundred percent', 'open', 'optimistic', 'paradise', 'perfect', 'phenomenal', 'pleasurable', 'plentiful', 'pleasant', 'poised', 'polished', 'polishe', 'powerful', 'prepared', 'pretty', 'principled', 'productive', 'progress', 'prominent', 'protected', 'proud', 'quality', 'quick', 'quiet', 'ready', 'reassuring', 'refined', 'refreshing', 'rejoice', 'reliable', 'remarkable', 'resounding', 'respected', 'restored', 'reward', 'rewarding', 'right', 'robust', 'safe', 'satisfactory', 'secure', 'seemly', 'simple', 'skilled', 'skillful', 'smile', 'soulful', 'sparkling', 'special', 'spirited', 'spiritual', 'stirring', 'stupendous', 'stunning', 'success', 'successful', 'sunny', 'super', 'superb', 'supporting', 'surprising', 'terrific', 'thorough', 'thrilling', 'transforming', 'transformative', 'trusting', 'truthful', 'unreal', 'unreal', 'upbeat', 'upright', 'upstanding', 'valued', 'vibrant', 'victorious', 'victory', 'vigorous', 'wealthy', 'welcome', 'well', 'whole', 'wholesome', 'willing', 'wonderful', 'worthy', 'wow', 'yes', 'yummy', 'zeal', 'zealous']
negative = ['abysmal', 'adverse', 'alarming', 'angry', 'annoy', 'anxious', 'apathy', 'appalling', 'atrocious', 'awful', 'bad', 'banal', 'barbed', 'belligerent', 'bemoan', 'beneath', 'boring', 'broken', 'callous', "can't", 'clumsy', 'coarse', 'cold', 'cold-hearted', 'collapse', 'confused', 'contradictory', 'contrary', 'corrosive', 'corrupt', 'crazy', 'creepy', 'criminal', 'cruel', 'cry', 'cutting', 'dead', 'decaying', 'damage', 'damaging', 'dastardly', 'deplorable', 'depressed', 'deprived', 'deformed', 'deny', 'despicable', 'detrimental', 'dirty', 'disease', 'disgusting', 'disheveled', 'dishonest', 'dishonorable', 'dismal', 'distress', "don't", 'dreadful', 'dreary', 'enraged', 'eroding', 'evil', 'fail', 'faulty', 'fear', 'feeble', 'fight', 'filthy', 'foul', 'frighten', 'frightful', 'gawky', 'ghastly', 'grave', 'greed', 'grim', 'grimace', 'gross', 'grotesque', 'gruesome', 'guilty', 'haggard', 'hard', 'hard-hearted', 'harmful', 'hate', 'hideous', 'homely', 'horrendous', 'horrible', 'hostile', 'hurt', 'hurtful', 'icky', 'ignore', 'ignorant', 'ill', 'immature', 'imperfect', 'impossible', 'inane', 'inelegant', 'infernal', 'injure', 'injurious', 'insane', 'insidious', 'insipid', 'jealous', 'junky', 'lose', 'lousy', 'lumpy', 'malicious', 'mean', 'menacing', 'messy', 'misshapen', 'missing', 'misunderstood', 'moan', 'moldy', 'monstrous', 'naive', 'nasty', 'naughty', 'negate', 'negative', 'never', 'no', 'nobody', 'nondescript', 'nonsense', 'not', 'noxious', 'objectionable', 'odious', 'offensive', 'old', 'oppressive', 'pain', 'perturb', 'pessimistic', 'petty', 'plain', 'poisonous', 'poor', 'prejudice', 'questionable', 'quirky', 'quit', 'reject', 'renege', 'repellant', 'reptilian', 'repulsive', 'repugnant', 'revenge', 'revolting', 'rocky', 'rotten', 'rude', 'ruthless', 'sad', 'savage', 'scare', 'scary', 'scream', 'severe', 'shoddy', 'shocking', 'sick', 'sickening', 'sinister', 'slimy', 'smelly', 'sobbing', 'sorry', 'spiteful', 'sticky', 'stinky', 'stormy', 'stressful', 'stuck', 'stupid', 'substandard', 'suspect', 'suspicious', 'tense', 'terrible', 'terrifying', 'threatening', 'ugly', 'undermine', 'unfair', 'unfavorable', 'unhappy', 'unhealthy', 'unjust', 'unlucky', 'unpleasant', 'upset', 'unsatisfactory', 'unsightly', 'untoward', 'unwanted', 'unwelcome', 'unwholesome', 'unwieldy', 'unwise', 'upset', 'vice', 'vicious', 'vile', 'villainous', 'vindictive', 'weary', 'wicked', 'woeful', 'worthless', 'wound', 'yell', 'yucky', 'zero', 'wary']
urgent_words = ["time is running out" ,"last chance" ,"up to", "until", "deadline",
          "hurry", "quick","ASAP","fast","as soon as possible"]


# In[ ]:




# In[35]:

path = '/Users/shani/Documents/private/hack/datahack/maildir/'

def aharon():
    emails = []
    counter = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            if counter > 10:
                break
            if f.find('.DS_Store') == -1:
                counter = counter + 1
                ff = open(root + '/' + f)
                print "FILE NAME: " , root + '/' + f
                s = email.message_from_file(ff)
                emails.append({"message":s.get_payload(),"data":zip(s.keys(), s.values())})
    return emails


# In[268]:

ss = TextBlob(s)
print ss.sentiment
print ss.noun_phrases
#print ss.sentences


# In[100]:

#s = "I plan on calling my all-- star selections tonight but I'm not sure what to \ntell them as far as practices are concerned.\n\nalso: do the Packers still get 8 plus 1 alternate?"


# In[288]:

def noun_phrases(txt_b):
    return len(txt_b.noun_phrases)

def polarity(txt_b):
    return txt_b.sentiment.polarity

def subjectivity(txt_b):
    return txt_b.sentiment.subjectivity

def polarity(txt_b):
    return txt_b.sentiment.polarity

def has_questions(txt):
    return txt.find("?")!=-1
        
def question_related(txt):
    return txt.find("question")!=-1 
    
def kria(txt):
    return txt.find("!")!=-1 

def count_dollar(txt):
    return txt.find("$")!=-1

def count_mess(txt_words):
    return len(txt_words)

def count_stop_words(txt_words):
    return len([i for i in txt_words if i in stop_w])

def count_not_stop_words(txt_words_filtered):
    return len(txt_words_filtered)

def confident(txt_words):
    return len([i for i in txt_words if i in confidence])

def negative_kw(txt_words):
    return len([i for i in txt_words if i in negative])

def positive_kw(txt_words):
    return len([i for i in txt_words if i in positive])

def num_sentences(txt_b):
    return len(txt_b.sentences)

def num_spelling_error(txt_words):
    t = [i for i in txt_words if i not in stop_w] # shouldn't be here..
    return len([len(j) for j in [i.spellcheck() for i in t if not i[0].isupper()] if len(j) > 1])

def check_urgency(txt):
    # gets string and finds occurences of words from urgent dictionary
    idx = [txt.find(word) for word in urgent_words]
    return len([i for i in idx if i!=-1])


# In[289]:

def get_body_features(txt):
    txt_b = TextBlob(txt)
    txt_words = txt_b.words
    txt_words_filtered = [i for i in [i.lower() for i in txt_words] if i not in stop_w1]
    return [noun_phrases(txt_b), polarity(txt_b),subjectivity(txt_b), has_questions(txt), question_related(txt), kria(txt), count_dollar(txt), count_mess(txt_words),
            count_stop_words(txt_words),count_not_stop_words(txt_words_filtered), confident(txt_words_filtered),
            negative_kw(txt_words_filtered), positive_kw(txt_words_filtered), num_sentences(txt_b),
            num_spelling_error(txt_words),check_urgency(txt)]


# In[ ]:

filename = "/Users/shani/Documents/private/hack/msg_txt0.json"
with open(filename) as data_file:    
    data = json.load(data_file)


# In[325]:




# In[302]:

len(data)


# In[ ]:

cont = [i['Content'].split("\nFrom:") for i in data]


# In[466]:

cont


# In[494]:


#print len([j for j in [i['Content'].split("------------- Forwarded by") for i in data] if len(j)>1])


[i['Content'] for i in data if "Bryan Hull" in i['Content']]


# In[409]:

#dd = [email.message_from_string("from" + j) for j in g]

email.message_from_string("From:"+g[1]).keys()


# In[503]:

# for one perseon..
   
re.split("\nFrom: | \tFrom:",  "n\n\nFrom:  Bryan Hull  andsdkfjdlskaj \tFrom: it is liskeltsfkjdvc")


# In[517]:

def addDate(matchobj):
    return '\nDate: ' + matchobj.group(0)

a = []
for l in data:
    c = l['Content']
    c = re.sub('\d\d/\d\d/\d\d\d\d \d\d:\d\d',addDate,c)
    ll = re.split('\nFrom: | \tFrom:',c)
    if len(ll)>1:
        for i in ll[1:]:
            e = email.message_from_string("From:" + i)
            #if "Bryan Hull" in c:
                #print "WTFFF  : ", ll
            print "this is what printed : " , e['From']
            print "and tha dates is: " , e['Date']
        a.append({"date": l['Date'], "content": ll})


# In[486]:

a[0]['content'][1]
a[0]['date']


# In[487]:

a[0]['content'][1]


# In[ ]:




# In[ ]:

all_emails_with_from = [j for j in [] if len(j)>1]
for one_email in all_emails_with_from:
    for return_email in one_email[1:]:  
        e = email.message_from_string("From:" + return_email)
        print "this is" + e['From']


# In[498]:

a='Beautiful, is; better*than\nugly'

['Beautiful', 'is', 'better', 'than', 'ugly']


# In[474]:


all_emails_with_from1 = [j for j in [i['Content'].split("\n\t\n\t\n\tFrom:") for i in data] if len(j)>1]
len(all_emails_with_from1)
len(all_emails_with_from)


# In[464]:

import datetime

#datetime.datetime.strptime("10/25/2000 02:42 PM", '%m/%d/%y %YT%H:%M:%SZ')
datetime.datetime.strptime("10/25/2000", '%m/%d/%Y ')

#fromtimestamp("10/24/2000 06:25 PM")


# In[361]:

dd.keys()


# In[362]:

dd['from']


# In[ ]:



