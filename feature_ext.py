
# coding: utf-8

# In[52]:

import numpy as np
import email
import os

stop_w = ["all", "she'll", "don't", 'being', 'over', 'through', 'during', 'its', 'before', "he's", "when's", "we've", 'had', 'should', "he'd", 'to', 'only', 'does', "here's", 'under', 'ours', 'has', "haven't", 'then', 'them', 'his', 'above', 'very', "who's", "they'd", 'cannot', "you've", 'they', 'not', 'yourselves', 'him', 'nor', "we'll", 'did', "they've", 'these', 'she', 'each', "won't", 'where', "mustn't", "isn't", "i'll", "why's", 'because', "you'd", 'doing', 'some', "hasn't", "we'd", 'further', 'ourselves', "shan't", 'what', 'for', 'herself', 'below', "there's", "shouldn't", "they'll", 'between', 'be', 'we', 'who', "doesn't", 'of', 'here', "hadn't", "aren't", 'by', 'both', 'about', 'her', 'theirs', "wouldn't", 'against', "i'd", "weren't", "i'm", 'or', "can't", 'this', 'own', 'into', 'yourself', 'down', 'hers', "couldn't", 'your', "you're", 'from', "how's", 'would', 'whom', "it's", 'there', 'been', "he'll", 'their', "we're", 'themselves', 'was', 'until', 'too', 'himself', 'that', "didn't", "what's", 'but', 'it', 'with', 'than', 'those', 'he', 'me', "they're", 'myself', "wasn't", 'up', 'while', 'ought', 'were', 'more', 'my', 'could', 'are', 'and', 'do', 'is', 'am', 'few', 'an', 'as', 'itself', 'at', 'have', 'in', 'any', 'if', 'again', "let's", 'no', "i've", 'when', 'same', 'how', 'other', 'which', 'you', 'out', 'our', 'after', "where's", 'most', 'such', 'on', 'why', 'a', 'off', 'i', "she'd", 'having', "you'll", 'so', "she's", 'the', 'once', 'yours', "that's"]
confidence = ["just","sorry","hopefully","actually","kind of"]
positive = ['absolutely', 'adorable', 'accepted', 'acclaimed', 'accomplish', 'accomplishment', 'achievement', 'action', 'active', 'admire', 'adventure', 'affirmative', 'affluent', 'agree', 'agreeable', 'amazing', 'angelic', 'appealing', 'approve', 'aptitude', 'attractive', 'awesome', 'beaming', 'beautiful', 'believe', 'beneficial', 'bliss', 'bountiful', 'bounty', 'brave', 'bravo', 'brilliant', 'bubbly', 'calm', 'celebrated', 'certain', 'champ', 'champion', 'charming', 'cheery', 'choice', 'classic', 'classical', 'clean', 'commend', 'composed', 'congratulation', 'constant', 'cool', 'courageous', 'creative', 'cute', 'dazzling', 'delight', 'delightful', 'distinguished', 'divine', 'earnest', 'easy', 'ecstatic', 'effective', 'effervescent', 'efficient', 'effortless', 'electrifying', 'elegant', 'enchanting', 'encouraging', 'endorsed', 'energetic', 'energized', 'engaging', 'enthusiastic', 'essential', 'esteemed', 'ethical', 'excellent', 'exciting', 'exquisite', 'fabulous', 'fair', 'familiar', 'famous', 'fantastic', 'favorable', 'fetching', 'fine', 'fitting', 'flourishing', 'fortunate', 'free', 'fresh', 'friendly', 'fun', 'funny', 'generous', 'genius', 'genuine', 'giving', 'glamorous', 'glowing', 'good', 'gorgeous', 'graceful', 'great', 'green', 'grin', 'growing', 'handsome', 'happy', 'harmonious', 'healing', 'healthy', 'hearty', 'heavenly', 'honest', 'honorable', 'honored', 'hug', 'idea', 'ideal', 'imaginative', 'imagine', 'impressive', 'independent', 'innovate', 'innovative', 'instant', 'instantaneous', 'instinctive', 'intuitive', 'intellectual', 'intelligent', 'inventive', 'jovial', 'joy', 'jubilant', 'keen', 'kind', 'knowing', 'knowledgeable', 'laugh', 'legendary', 'light', 'learned', 'lively', 'lovely', 'lucid', 'lucky', 'luminous', 'marvelous', 'masterful', 'meaningful', 'merit', 'meritorious', 'miraculous', 'motivating', 'moving', 'natural', 'nice', 'novel', 'now', 'nurturing', 'nutritious', 'okay', 'one', 'one-hundred percent', 'open', 'optimistic', 'paradise', 'perfect', 'phenomenal', 'pleasurable', 'plentiful', 'pleasant', 'poised', 'polished', 'polishe', 'powerful', 'prepared', 'pretty', 'principled', 'productive', 'progress', 'prominent', 'protected', 'proud', 'quality', 'quick', 'quiet', 'ready', 'reassuring', 'refined', 'refreshing', 'rejoice', 'reliable', 'remarkable', 'resounding', 'respected', 'restored', 'reward', 'rewarding', 'right', 'robust', 'safe', 'satisfactory', 'secure', 'seemly', 'simple', 'skilled', 'skillful', 'smile', 'soulful', 'sparkling', 'special', 'spirited', 'spiritual', 'stirring', 'stupendous', 'stunning', 'success', 'successful', 'sunny', 'super', 'superb', 'supporting', 'surprising', 'terrific', 'thorough', 'thrilling', 'transforming', 'transformative', 'trusting', 'truthful', 'unreal', 'unreal', 'upbeat', 'upright', 'upstanding', 'valued', 'vibrant', 'victorious', 'victory', 'vigorous', 'wealthy', 'welcome', 'well', 'whole', 'wholesome', 'willing', 'wonderful', 'worthy', 'wow', 'yes', 'yummy', 'zeal', 'zealous']
fatNeg = "cannot, damage, do not, error, fail, impossible, little value, loss, mistake, not, problem, refuse, stop, unable to, unfortunately"
fatNeg = fatNeg.split(', ')
fatPos = "benefit, it is best to, issue, matter, progress, success, unfortunate, valuable"
fatPos = fatPos.split(', ')
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


# In[36]:

a = aharon()


# In[34]:

s = "I plan on calling my all-- star selections tonight but I'm not sure what to \ntell them as far as practices are concerned.\n\nalso: do the Packers still get 8 plus 1 alternate?"
ss = [i.lower() for i in s.split(" ")]


# In[44]:

def has_questions(messg):
    return "?" in messg
        
def question_related(messg):
    return "question" in messg
    
def kria(messg):
    return "!" in messg

def count_mess(messg):
    return len(messg)

def count_stop_words(messg,stop_w):
    return len([i for i in messg if i in stop_w])

def count_not_stop_words(messg,stop_w):
    return len([i for i in messg if i not in stop_w])

def count_dollar(messg):
    return "$" in messg

def confident(messg):
    return len([i for i in confidence if i in messg])

    


# In[46]:

count_stop_words(ss,stop_w)


# In[50]:

absolutely
adorable
accepted
acclaimed
accomplish
accomplishment
achievement
action
active
admire
adventure
affirmative
affluent
agree
agreeable
amazing
angelic
appealing
approve
aptitude
attractive
awesome
beaming
beautiful
believe
beneficial
bliss
bountiful
bounty
brave
bravo
brilliant
bubbly
calm
celebrated
certain
champ
champion
charming
cheery
choice
classic
classical
clean
commend
composed
congratulation
constant
cool
courageous
creative
cute
dazzling
delight
delightful
distinguished
divine
earnest
easy
ecstatic
effective
effervescent
efficient
effortless
electrifying
elegant
enchanting
encouraging
endorsed
energetic
energized
engaging
enthusiastic
essential
esteemed
ethical
excellent
exciting
exquisite
fabulous
fair
familiar
famous
fantastic
favorable
fetching
fine
fitting
flourishing
fortunate
free
fresh
friendly
fun
funny
generous
genius
genuine
giving
glamorous
glowing
good
gorgeous
graceful
great
green
grin
growing
handsome
happy
harmonious
healing
healthy
hearty
heavenly
honest
honorable
honored
hug
idea
ideal
imaginative
imagine
impressive
independent
innovate
innovative
instant
instantaneous
instinctive
intuitive
intellectual
intelligent
inventive
jovial
joy
jubilant
keen
kind
knowing
knowledgeable
laugh
legendary
light
learned
lively
lovely
lucid
lucky
luminous
marvelous
masterful
meaningful
merit
meritorious
miraculous
motivating
moving
natural
nice
novel
now
nurturing
nutritious
okay
one
one-hundred percent
open
optimistic
paradise
perfect
phenomenal
pleasurable
plentiful
pleasant
poised
polished
popular
positive
powerful
prepared
pretty
principled
productive
progress
prominent
protected
proud
quality
quick
quiet
ready
reassuring
refined
refreshing
rejoice
reliable
remarkable
resounding
respected
restored
reward
rewarding
right
robust
safe
satisfactory
secure
seemly
simple
skilled
skillful
smile
soulful
sparkling
special
spirited
spiritual
stirring
stupendous
stunning
success
successful
sunny
super
superb
supporting
surprising
terrific
thorough
thrilling
thriving
tops
tranquil
transforming
transformative
trusting
truthful
unreal
unwavering
up
upbeat
upright
upstanding
valued
vibrant
victorious
victory
vigorous
virtuous
vital
vivacious
wealthy
welcome
well
whole
wholesome
willing
wonderful
wondrous
worthy
wow
yes
yummy
zeal
zealous


# In[28]:

d


# In[ ]:



