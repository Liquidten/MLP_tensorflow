import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data", one_hot = True)
type(mnist)
sample = mnist.train.images[31].reshape(28,28)
#print(sample)

import matplotlib.pyplot as plt
plt.imshow(sample, cmap='Greys')

learning_rate = 0.001
training_epochs = 10
batch_size = 100

#n_classes = 10
#n_samples = mnist.train.num_examples
#print(n_samples)

n_input = 784
n_hidden_1 = 256
n_hidden_2 = 128
n_hidden_3 = 64
n_classes = 10
n_samples = mnist.train.num_examples


def multilayer_preceptron(x, weights, biases):
    '''
    x : Place Holder for Data Input
    weights: Dictionary of weights
    biases: Dicitionary of biases
    '''
    #first hidden layer with RELU activation
    # X * W + B
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # func(X * W + B) = RELU _>f(x) = max(0,x)
    layer_1 = tf.nn.relu(layer_1)
    
    #second hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # func(X * W + B) = RELU
    layer_2 = tf.nn.relu(layer_2)
    
     #third hidden layer with RELU activation
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    # func(X * W + B) = RELU
    layer_3 = tf.nn.relu(layer_3)
    
    #last output layer with linear activation
    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
    return out_layer

weights = {
        'h1' : tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2' : tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'h3' : tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
        'out': tf.Variable(tf.random_normal([n_hidden_3, n_classes]))
                           }
biases = {
        'b1' : tf.Variable(tf.random_normal([n_hidden_1])),
        'b2' : tf.Variable(tf.random_normal([n_hidden_2])),
        'b3' : tf.Variable(tf.random_normal([n_hidden_3])),
        'out': tf.Variable(tf.random_normal([n_classes]))}

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

#Construct model
pred = multilayer_preceptron(x, weights, biases)
print(pred)

#Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(pred, y))
optimizer = tf.train.AdadeltaOptimizer(learning_rate = learning_rate).minimize(cost)

#Initializing the variables
init = tf.initialize_all_variables()

'''
Xsample, ysample = mnist.train.next_batch(1)
plt.imshow(Xsample.reshape(28,28), cmap = 'Greys')
print(ysample)
'''

sess = tf.InteractiveSession()

#Intialize all the variables
sess.run(init)

# Training Epochs
# Essentially the max amount of loops possible before we stop
# May stop earlier if cost/loss limit was set
#15 loops
for epoch in range(training_epochs):
    
    #start with cost = 0.0
    avg_cost = 0.0
    
    #convert total number of batches to integer
    total_batch = int(n_samples/batch_size)
    
    #loop over all batches
    for i in range(total_batch):
        
        #grab the next batch of trainning data and labels
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        
        #feed dictionary for optimization and loss value
        #Returns a tuple, but we only need 'c' the cost
        #so we set an underscore as a 'throwaway'
        _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
        
        #compute average loss
        avg_cost += c / total_batch
    
    print("Epoch: {} cost={:.4f}".format(epoch+1, avg_cost))
    
print("Model has completed {} Epochs of Training".format(training_epochs))

#Model Evaluation

#Test Model
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y,1))
print(correct_prediction[0])

correct_prediction = tf.cast(correct_prediction, "float")
print(correct_prediction[0])

accuracy = tf.reduce_mean(correct_prediction)
type(accuracy)

mnist.test.labels
mnist.test.images

print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
  
