noisyteacher.py - Trains a network which consists of 9 convolutional and Global Average Pooling layer as last layer.

noisystudent.py - Trains a network which consists of 2 convolutional layers and 2fully connected. 

main-noisy.py - Runs NoisyTeacher and NoisyStudent with various flag combinations.

Command to run main.py
1. python main.py --teacher True --batch_size 64 --learning_rate 0.001
2. python main.py --student True --batch_size 64 --learning_rate 0.001
3. python main.py --student True --HT True --batch_size 64 --learning_rate 0.001
4. python main.py --student True --KD True --batch_size 64 --learning_rate 0.001

If you achieve a very high accuracy in the very early stages of training, this indicates the model is memorizing rather than learning.

Below flags KD and HT explanation.

FLAGS- KD- Knowledge Distillation; When you train student network by taking the softmax layer loss(only the last layer) from teacher.

FLAGS- HT- Hind Based Training; When you train student network by taking thelast layer losses from teacher but with some extra noise.
 

