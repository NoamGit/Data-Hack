import email, os, csv
import nltk, re, pprint
from nltk import word_tokenize

path = '/Users/aharon/Downloads/maildir'

counter = 0
for root, dirs, files in os.walk(path):
	print dirs, files
	for file_name in files:
		print root, dirs, file_name
		if file_name.find('DS_Store') != -1:
			continue
		f = open(root + '/' + file_name)
		e = email.message_from_file(f)
		counter = counter + 1
		meta_dat = [e[x] for x in e.keys()]
		body = getMailBody(e)
		msg_txt = [word.lower() for word in word_tokenize(body)]
	if counter > 1:
			break



def getMailBody(e):
	if e.is_multipart():
	    for payload in e.get_payload():
	        # if payload.is_multipart(): ...
	        print payload.get_payload()
	else:
	    return e.get_payload()