import os
import sys
import random

label_count = -1
for subdir, dirs, files in os.walk("/home/users/saman/today/VGG19-TF/256_ObjectCategories"):
	if subdir == "/home/users/saman/today/VGG19-TF/256_ObjectCategories/":
		continue
	for file in files:
		print("%d,%s" % (label_count, os.path.join(subdir, file)))	
	label_count = label_count + 1
