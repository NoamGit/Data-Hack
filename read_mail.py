import email, os
import json
import nltk, re, pprint
from nltk import word_tokenize

#path = 'C:\Enron\enron_mail_20150507\maildir'
path = '/Users/aharon/Downloads/maildir'

outPath = '../output/'
rel_array = []

def getMailBody(e):
	if e.is_multipart():
	    for payload in e.get_payload():
	        # if payload.is_multipart(): ...
	        print payload.get_payload()
	else:
	    return e.get_payload()

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
		email_dat = {}
		for x in e.items():
			email_dat[x[0]] = x[1]
		body = getMailBody(e)
		msg_txt = [word.lower() for word in word_tokenize(body)]
		email_dat['Content'] = body
		email_dat['NumLines'] = body.count('\n')
		json_content = json.dumps(email_dat)
		rel_file = root[len(path)+1:] + '/' + file_name
		rel_file = rel_file.replace('/', '_')
		rel_file = rel_file.replace('\\', '_')
		email_dat['fileName'] = rel_file
		rel_array.append(email_dat)
		#with open(outPath + rel_file, 'w') as outfile:
		#	json.dump(email_dat, outfile)
	if counter > 1000:
		break

with open('msg_txt.json', 'w') as outfile:
	json.dump(rel_array, outfile)

