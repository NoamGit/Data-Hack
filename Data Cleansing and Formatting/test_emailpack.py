import email, os

path = 'C:\Users\Noam\Documents\Data Hack\Data Set\Enron\maildir'

counter = 0
for root, dirs, files in os.walk(path):
	print dirs, files
	for file in files:
		print root, dirs, file
		if file == '.DS_Store':
			pass
		counter = counter + 1
		f = open(root + '/' + file)
		e = email.message_from_file(f)
	if counter > 10:
			break