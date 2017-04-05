import numpy as np
def read_input(root_dir):
    images = []
    with open("./dataset-shuffle.txt") as f:
        content = f.readlines()
    np.random.shuffle(content)
    #print content
    list_size = len(content)
    print list_size
    print(type(list_size))
    train_size = list_size * (80)
    train_size = train_size // 100
    #print train_size
    test_size = list_size - train_size
    train_images = content[:train_size]
    test_images = content[train_size:]
    #print train_images
    
    with open("./dataset-train.txt" , 'w') as f:
        for i in train_images:
            f.write(i)

    with open("./dataset-test.txt", 'w') as f:
        for i in test_images:
            f.write(i)
    

if __name__=='__main__':
    read_input(".")
