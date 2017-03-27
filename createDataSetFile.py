import os
import sys
import random

label_count = -1
for subdir, dirs, files in os.walk("/home/saman/VGG19-Scratch/101_ObjectCategories"):
	if subdir == "/home/saman/VGG19-Scratch/101_ObjectCategories/":
		continue
	for file in files:
		print("%s,%d" % (os.path.join(subdir, file), label_count))	
	label_count = label_count + 1