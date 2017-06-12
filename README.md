Teacher.py - Trains a network which consists of 6 convolutional and 2 fully connected layers. This architecture is shallow but wider.

Student.py - Trains a network which consists of 17 convolutional layers and 2 fully connected. This architecture has a larger depth and lesser width(Thin and Deep)

main.py - Runs Teacher and Student with various flag combinations.

Command to run main.py
1. python main.py --teacher True --batch_size 50 --learning_rate 0.003
2. python main.py --student True --batch_size 128 --learning_rate 0.0001

If you achieve a very high accuracy in the very early stages of training, this indicates the model is memorizing rather than learning.

Below flags KD and HT explanation.
FLAGS- KD- Knowledge Distillation; When you train student network by taking the softmax layer loss(only the last layer) from teacher.
FLAGS- HT- Hind Based Training; When you train student network by taking middle layer losses from teacher.
 

