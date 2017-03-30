f = open('dataset.txt','r')
filedata = f.read()
f.close()

newdata = filedata.replace("/home/saman/VGG19-Scratch/","/home/visa/vgg19/VGG19-TF/")

f = open('dataset.txt','w')
f.write(newdata)
f.close()
