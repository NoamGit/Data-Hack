import os
import json
import time 
import datetime 
from datetime import date
from datetime import timedelta
import re
import email

#filename = 'C:/Enron/git/output/msg_txt6 .json'
path = 'C:/Enron/git/output/'

if False:
	data = []
	for root, dirs, files in os.walk(path):
		for filename in files:
			print filename
			with open(path+filename) as data_file:    
			    data.append(json.load(data_file))
	data = [item for sublist in data for item in sublist]


def addDate(matchobj):
    return '\nDate: ' + matchobj.group(0)

a = []
counter = 0
A = 0
B = 0
C = 0
for row in data:
	c =row['Content']
	cont = c
	if("Original Message" not in c):
		
		c1 = re.sub('\d\d/\d\d/\d\d\d\d \d\d:\d\d',addDate,c)
		ll = re.split('\nFrom: | \tFrom:',c1)
		if len(ll)>1:
			counter += 1
			#print l['Date'][:-11].strip()
			replyTime = datetime.datetime.strptime(row['Date'][:-11].strip(), '%a, %d %b %Y %H:%M:%S')#  %z (%Z)')
			ok = 1
			for i in ll[1:]:
				e = email.message_from_string("From:" + i)
				if e['Date'] is None and e['When'] is None:
					m = i.replace('\n\n', '\n')
					e = email.message_from_string("From:" + m)
				if(e['Date'] is None):
					if(e['When'] is None):
						print 'FUCK WE HAVE A NONE'
						A += 1
						ok = 0
					else:
						dtstr = e['When'].strip();
				else:
					dtstr = e['Date'].strip();
				msg_text = e.get_payload()
				if type(msg_text) != type([]):
					q = re.findall('Subject.*\n\n(.+)+', msg_text)
					if len(q) > 0:
						cont_next = q[-1]
					else:
						cont_next = msg_text
						msg_text
						C += 1
				if(ok):
					try:
						data_type = 3
						if('am' in dtstr.lower() or 'pm' in dtstr.lower()):
							data_type = 1
						if(len(re.findall('[a-zA-Z]+', dtstr.lower())) == 0):							
							data_type = 2
							
						if data_type == 1:
							sendTime = datetime.datetime.strptime(dtstr, '%m/%d/%Y %I:%M %p')
						if data_type == 2:
							sendTime = datetime.datetime.strptime(dtstr, '%m/%d/%Y %I:%M')
						if data_type == 3:
							sendTime = datetime.datetime.strptime(dtstr[:-6], '%a, %d %b %Y %H:%M:%S')
						dt =  replyTime - sendTime;					
						print dt
						replyTime = sendTime
						label = abs(dt.total_seconds()) / 3600 < 24
						a.append({"label": label, "total_time": dt.total_seconds(), "content" : cont})
						cont = cont_next
					except:
						ok = 0
						B+=1
						
							
		#a.append({"date": row['Date'], "content": e['Content']})