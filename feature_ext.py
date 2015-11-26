
# coding: utf-8

# In[77]:

import numpy as np
import email
import os
from textblob import TextBlob
from textblob import Word


# In[170]:

stop_w1 = ["'m", "am", "all","she'll", "don't", 'being', 'over', 'through', 'during', 'its', 'before', "he's", "when's", "we've", 'had', 'should', "he'd", 'to', 'only', 'does', "here's", 'under', 'ours', 'has', "haven't", 'then', 'them', 'his', 'above', 'very', "who's", "they'd", 'cannot', "you've", 'they', 'not', 'yourselves', 'him', 'nor', "we'll", 'did', "they've", 'these', 'she', 'each', "won't", 'where', "mustn't", "isn't", "i'll", "why's", 'because', "you'd", 'doing', 'some', "hasn't", "we'd", 'further', 'ourselves', "shan't", 'what', 'for', 'herself', 'below', "there's", "shouldn't", "they'll", 'between', 'be', 'we', 'who', "doesn't", 'of', 'here', "hadn't", "aren't", 'by', 'both', 'about', 'her', 'theirs', "wouldn't", 'against', "i'd", "weren't", "i'm", 'or', "can't", 'this', 'own', 'into', 'yourself', 'down', 'hers', "couldn't", 'your', "you're", 'from', "how's", 'would', 'whom', "it's", 'there', 'been', "he'll", 'their', "we're", 'themselves', 'was', 'until', 'too', 'himself', 'that', "didn't", "what's", 'but', 'it', 'with', 'than', 'those', 'he', 'me', "they're", 'myself', "wasn't", 'up', 'while', 'ought', 'were', 'more', 'my', 'could', 'are', 'and', 'do', 'is', 'am', 'few', 'an', 'as', 'itself', 'at', 'have', 'in', 'any', 'if', 'again', "let's", 'no', "i've", 'when', 'same', 'how', 'other', 'which', 'you', 'out', 'our', 'after', "where's", 'most', 'such', 'on', 'why', 'a', 'off', 'i', "she'd", 'having', "you'll", 'so', "she's", 'the', 'once', 'yours', "that's"]
stop_w2 = [i.capitalize() for i in stop_w1]
stop_w = stop_w2 + stop_w1
confidence = ["just","sorry","hopefully","actually","kind of"]
positive = ['absolutely', 'adorable', 'accepted', 'acclaimed', 'accomplish', 'accomplishment', 'achievement', 'action', 'active', 'admire', 'adventure', 'affirmative', 'affluent', 'agree', 'agreeable', 'amazing', 'angelic', 'appealing', 'approve', 'aptitude', 'attractive', 'awesome', 'beaming', 'beautiful', 'believe', 'beneficial', 'bliss', 'bountiful', 'bounty', 'brave', 'bravo', 'brilliant', 'bubbly', 'calm', 'celebrated', 'certain', 'champ', 'champion', 'charming', 'cheery', 'choice', 'classic', 'classical', 'clean', 'commend', 'composed', 'congratulation', 'constant', 'cool', 'courageous', 'creative', 'cute', 'dazzling', 'delight', 'delightful', 'distinguished', 'divine', 'earnest', 'easy', 'ecstatic', 'effective', 'effervescent', 'efficient', 'effortless', 'electrifying', 'elegant', 'enchanting', 'encouraging', 'endorsed', 'energetic', 'energized', 'engaging', 'enthusiastic', 'essential', 'esteemed', 'ethical', 'excellent', 'exciting', 'exquisite', 'fabulous', 'fair', 'familiar', 'famous', 'fantastic', 'favorable', 'fetching', 'fine', 'fitting', 'flourishing', 'fortunate', 'free', 'fresh', 'friendly', 'fun', 'funny', 'generous', 'genius', 'genuine', 'giving', 'glamorous', 'glowing', 'good', 'gorgeous', 'graceful', 'great', 'green', 'grin', 'growing', 'handsome', 'happy', 'harmonious', 'healing', 'healthy', 'hearty', 'heavenly', 'honest', 'honorable', 'honored', 'hug', 'idea', 'ideal', 'imaginative', 'imagine', 'impressive', 'independent', 'innovate', 'innovative', 'instant', 'instantaneous', 'instinctive', 'intuitive', 'intellectual', 'intelligent', 'inventive', 'jovial', 'joy', 'jubilant', 'keen', 'kind', 'knowing', 'knowledgeable', 'laugh', 'legendary', 'light', 'learned', 'lively', 'lovely', 'lucid', 'lucky', 'luminous', 'marvelous', 'masterful', 'meaningful', 'merit', 'meritorious', 'miraculous', 'motivating', 'moving', 'natural', 'nice', 'novel', 'now', 'nurturing', 'nutritious', 'okay', 'one', 'one-hundred percent', 'open', 'optimistic', 'paradise', 'perfect', 'phenomenal', 'pleasurable', 'plentiful', 'pleasant', 'poised', 'polished', 'polishe', 'powerful', 'prepared', 'pretty', 'principled', 'productive', 'progress', 'prominent', 'protected', 'proud', 'quality', 'quick', 'quiet', 'ready', 'reassuring', 'refined', 'refreshing', 'rejoice', 'reliable', 'remarkable', 'resounding', 'respected', 'restored', 'reward', 'rewarding', 'right', 'robust', 'safe', 'satisfactory', 'secure', 'seemly', 'simple', 'skilled', 'skillful', 'smile', 'soulful', 'sparkling', 'special', 'spirited', 'spiritual', 'stirring', 'stupendous', 'stunning', 'success', 'successful', 'sunny', 'super', 'superb', 'supporting', 'surprising', 'terrific', 'thorough', 'thrilling', 'transforming', 'transformative', 'trusting', 'truthful', 'unreal', 'unreal', 'upbeat', 'upright', 'upstanding', 'valued', 'vibrant', 'victorious', 'victory', 'vigorous', 'wealthy', 'welcome', 'well', 'whole', 'wholesome', 'willing', 'wonderful', 'worthy', 'wow', 'yes', 'yummy', 'zeal', 'zealous']

fatNeg = "cannot, damage, do not, error, fail, impossible, little value, loss, mistake, not, problem, refuse, stop, unable to, unfortunately"
fatNeg = fatNeg.split(', ')
fatPos = "benefit, it is best to, issue, matter, progress, success, unfortunate, valuable"
fatPos = fatPos.split(', ')

negative = ['abysmal', 'adverse', 'alarming', 'angry', 'annoy', 'anxious', 'apathy', 'appalling', 'atrocious', 'awful', 'bad', 'banal', 'barbed', 'belligerent', 'bemoan', 'beneath', 'boring', 'broken', 'callous', "can't", 'clumsy', 'coarse', 'cold', 'cold-hearted', 'collapse', 'confused', 'contradictory', 'contrary', 'corrosive', 'corrupt', 'crazy', 'creepy', 'criminal', 'cruel', 'cry', 'cutting', 'dead', 'decaying', 'damage', 'damaging', 'dastardly', 'deplorable', 'depressed', 'deprived', 'deformed', 'deny', 'despicable', 'detrimental', 'dirty', 'disease', 'disgusting', 'disheveled', 'dishonest', 'dishonorable', 'dismal', 'distress', "don't", 'dreadful', 'dreary', 'enraged', 'eroding', 'evil', 'fail', 'faulty', 'fear', 'feeble', 'fight', 'filthy', 'foul', 'frighten', 'frightful', 'gawky', 'ghastly', 'grave', 'greed', 'grim', 'grimace', 'gross', 'grotesque', 'gruesome', 'guilty', 'haggard', 'hard', 'hard-hearted', 'harmful', 'hate', 'hideous', 'homely', 'horrendous', 'horrible', 'hostile', 'hurt', 'hurtful', 'icky', 'ignore', 'ignorant', 'ill', 'immature', 'imperfect', 'impossible', 'inane', 'inelegant', 'infernal', 'injure', 'injurious', 'insane', 'insidious', 'insipid', 'jealous', 'junky', 'lose', 'lousy', 'lumpy', 'malicious', 'mean', 'menacing', 'messy', 'misshapen', 'missing', 'misunderstood', 'moan', 'moldy', 'monstrous', 'naive', 'nasty', 'naughty', 'negate', 'negative', 'never', 'no', 'nobody', 'nondescript', 'nonsense', 'not', 'noxious', 'objectionable', 'odious', 'offensive', 'old', 'oppressive', 'pain', 'perturb', 'pessimistic', 'petty', 'plain', 'poisonous', 'poor', 'prejudice', 'questionable', 'quirky', 'quit', 'reject', 'renege', 'repellant', 'reptilian', 'repulsive', 'repugnant', 'revenge', 'revolting', 'rocky', 'rotten', 'rude', 'ruthless', 'sad', 'savage', 'scare', 'scary', 'scream', 'severe', 'shoddy', 'shocking', 'sick', 'sickening', 'sinister', 'slimy', 'smelly', 'sobbing', 'sorry', 'spiteful', 'sticky', 'stinky', 'stormy', 'stressful', 'stuck', 'stupid', 'substandard', 'suspect', 'suspicious', 'tense', 'terrible', 'terrifying', 'threatening', 'ugly', 'undermine', 'unfair', 'unfavorable', 'unhappy', 'unhealthy', 'unjust', 'unlucky', 'unpleasant', 'upset', 'unsatisfactory', 'unsightly', 'untoward', 'unwanted', 'unwelcome', 'unwholesome', 'unwieldy', 'unwise', 'upset', 'vice', 'vicious', 'vile', 'villainous', 'vindictive', 'weary', 'wicked', 'woeful', 'worthless', 'wound', 'yell', 'yucky', 'zero', 'wary']

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


# In[79]:

ss = TextBlob(s)
print ss.sentiment
print ss.noun_phrases
#print ss.sentences
ss.tags


# In[100]:

#s = "I plan on calling my all-- star selections tonight but I'm not sure what to \ntell them as far as practices are concerned.\n\nalso: do the Packers still get 8 plus 1 alternate?"


# In[192]:

txt = "this is my my"
txt_b = TextBlob(txt)
txt_words = txt_b.words

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

def count_not_stop_words(txt_words):
    return len([i for i in txt_words if i not in stop_w])

def confident(txt_words):
    return len([i for i in txt_words if i in confidence])

def negative_kw(txt_words):
    return len([i for i in txt_words if i in negative])

def positive_kw(txt_words):
    return len([i for i in txt_words if i in positive])

def num_sent(txt_b):
    return len(txt_b.sentences)

def num_spelling_error(txt_words):
    t = [i for i in txt_words if i not in stop_w] # shouldn't be here..
    return len([len(j) for j in [i.spellcheck() for i in t if not i[0].isupper()] if len(j) > 1])


# In[193]:

def test():
    s = "theres a spleaing mistakas here ,I plan on calling my all-- star selections tonight but I'm not sure what to \ntell them as far as practices are concerned.\n\nalso: do the Packers still get 8 plus 1 alternate?"
    kria(m)
    ss = TextBlob(s)
    w = ss.words
    print count_not_stop_words(w)
    print positive_kw(w)
    print negative_kw(w)
    print count_stop_words(w)
    print count_mess(w)
    print count_dollar(s)
    print num_sent(ss)
    print num_spelling_error(w) 


# In[196]:




# In[ ]:



