import os
import json
import time 
import datetime 
from datetime import date
from datetime import timedelta
import re

#filename = 'C:/Enron/git/output/msg_txt6 .json'
path = 'C:/Enron/git/output/'

data = []
for root, dirs, files in os.walk(path):
	for filename in files:
		print filename
		with open(path+filename) as data_file:    
		    data.append(json.load(data_file))
data = [item for sublist in data for item in sublist]

len([i['Content'] for i in data if "Original Message" in i['Content']])
len(data)

def addDate(matchobj):
    return '\nDate: ' + matchobj.group(0)

a = []
for l in data:
	c = l['Content']
	if("Original Message" not in c):		
	    c1 = re.sub('\d\d/\d\d/\d\d\d\d \d\d:\d\d',addDate,c)
	    ll = re.split('\nFrom: | \tFrom:',c1)
	    if len(ll)>1:
	        for i in ll[1:]:
	            e = email.message_from_string("From:" + i)
	            #print "this is what printed : " , e['From']
	            #print "and tha dates is: " , e['Date']
	            if(e['Date'] is None):
					print 'FUCK WE HAVE A NONE'
					stophere =1
	        a.append({"date": l['Date'], "content": ll})