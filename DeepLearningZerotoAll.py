import tensorflow as tf

# Create a constant op
# This op is added as a node to the default graph
hello = tf.constant('Hello, TensorFlow!')

# start a TF session
sess = tf.Session()

# run the op and get result
print(sess.run(hello))

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)    # node3 = node1+node2

print("node1:",node1,"node2:",node2)
print("ndoe3",node3)
# 결과 값이 나오지 않음!

sess = tf.Session()
print("sess.run(node1,node2):",sess.run((node1,node2)))
print("sess.run(node3):",sess.run(node3))

# (1) Build graph (tensors) using TF operations
node1 = tf.constant(3.0,tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)
# (2) feed data and run graph(operation) sess.run(op
sess = tf.Session()
print("sess.run(nod1, node2):",sess.run((node1, node2)))
print("sess.run(node3):",sess.run(node3))

##############
# Placeholder: 그래프만 만들어놓고 값만 나중에 넣게 할 때#
##############
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a+b
print(sess.run(adder_node, feed_dict={a:3,b:4.5}))
print(sess.run(adder_node, feed_dict={a:[1,3],b:[2,4]}))

# Tensor Ranks: array 차원
# ex) 0: scalar, 1: vector, 2: Matrix, 3-Tensor:
# Shapes: 몇 개씩 들어가 있느냐. 제일 밖에서 안으로 순으로
# 0: [], 1: [D0], 2: [D0, D1]
# Types
# tf.float32, tf.int32 많이 쓴다

"""
Linear Regression
"""

# (1) Graph 구축
X_train = [1,2,3]
y_train = [1,2,3]

W = tf.Variable(tf.random_normal([1]),name='weight')    # TensorFlow가 이용하는 값! Trainable variable!!
b = tf.Variable(tf.random_normal([1]),name='bias')  # Trainable variable이다! shape 지정
hypothesis = X_train*W+b

cost = tf.reduce_mean(tf.square(hypothesis-y_train))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# (2)/(3) Run/Update graph and get results
sess = tf.Session()
sess.run(tf.global_variables_initializer()) ### 꼭 해야함 꼬꼬꼬꼬꼬꼮

for step in range(2001):
    sess.run(train)
    if step%20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))

# Placeholders
X = tf.placeholder(tf.float32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None])    # None --> 안에 있는 갯수에 제한 없음

W = tf.Variable(tf.random_normal([1]),name='weight')
b = tf.Variable(tf.random_normal([1]),name='bias')

hypothesis = X*W + b
cost = tf.reduce_mean(tf.square(y - hypothesis))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train],
                                         feed_dict={X:[1,2,3],y:[1,2,3]})
    if step%20==0:
        print(step, cost_val, W_val, b_val)

# Testing
print(sess.run(hypothesis, feed_dict={X:[5]}))
print(sess.run(hypothesis, feed_dict={X:[1.5,3.5]}))

"""
Linear Regression의 cost 최소화
"""

import matplotlib.pyplot as plt

X = [1,2,3]
y = [1,2,3]
W = tf.placeholder(tf.float32)
hypothesis = X*W

cost = tf.reduce_mean(tf.square(hypothesis-y))
sess = tf.Session()
sess.run(tf.global_variables_initializer())
W_val = []
cost_val = []

for i in range(-30, 50):
    feed_W = i*0.1
    curr_cost, curr_W = sess.run([cost, W], feed_dict={W: feed_W})
    W_val.append(curr_W)
    cost_val.append(curr_cost)

plt.plot(W_val, cost_val)
plt.show()

X_data = [1,2,3]
y_data = [1,2,3]

W = tf.Variable(tf.random_normal([1]),name='weight')
X = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

hypothesis = X*W

cost = tf.reduce_sum(tf.square(hypothesis-y))

learning_rate = 0.1
gradient = tf.reduce_mean((W*X-y)*X)
descent = W-learning_rate*gradient
update = W.assign(descent)  # Tensor는 equal로 assign 안되고 따로 assign을 해야 함!

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(21):
    sess.run(update, feed_dict={X:X_data,y:y_data})
    print(step, sess.run(cost, feed_dict={X:X_data,y:y_data}),sess.run(W))

# 미분 잘 안 될 때
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train],
                                         feed_dict={X:X_data,y:y_data})
    if step%20==0:
        print(step, cost_val, W_val, b_val)

### Parameter 조정

X = [1,2,3]
Y = [1,2,3]
W = tf.Variable(5.)
hypothesis = X*W
gradient = tf.reduce_mean((W*X-Y)*X)*2
cost = tf.reduce_mean(tf.square(hypothesis-Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

gvs = optimizer.compute_gradients(cost,[W])
apply_gradients = optimizer.apply_gradients(gvs)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(100):
    print(step, sess.run([gradient, W, gvs]))
    sess.run(apply_gradients)

"""
Multi-variable Linear Regression
"""

x1_data = [73.,93.,89.,96.,73.]
x2_data = [80.,88.,91.,98.,66.]
x3_data = [75.,93.,90.,100.,70.]
y_data = [152.,185.,180.,196.,142.]

X1 = tf.placeholder(tf.float32)
X2 = tf.placeholder(tf.float32)
X3 = tf.placeholder(tf.float32)

Y = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random_normal([1]),name='weight1')
w2 = tf.Variable(tf.random_normal([1]),name='weight2')
w3 = tf.Variable(tf.random_normal([1]),name='weight3')
b = tf.Variable(tf.random_normal([1]),name='bias')

hypothesis = X1*w1+X2*w2+X3*w3+b

cost = tf.reduce_mean(tf.square(hypothesis-Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                                   feed_dict = {X1:x1_data, X2:x2_data, X3:x3_data, Y:y_data})
    if step%10 ==0:
        print(step, 'Cost:',cost_val)
        print('Prediction:',hy_val)

X_data = [[73., 80., 75.],[93.,88.,93.],
          [89.,91.,90.],[96.,98.,100.],[73.,66.,70.]]
y_data = [[152.],[185.],[180.],[196.],[142.]]

X = tf.placeholder(tf.float32, shape=[None,3])
Y = tf.placeholder(tf.float32, shape=[None,1])

W = tf.Variable(tf.random_normal([3,1]),name='weight')
b = tf.Variable(tf.random_normal([1]),name='bias')

hypothesis = tf.matmul(X,W)+b
cost = tf.reduce_mean(tf.square(hypothesis-Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                                   feed_dict = {X:X_data, Y:y_data})
    if step%10 ==0:
        print(step, 'Cost:',cost_val)
        print('Prediction:',hy_val)

"""
Logistic Classification
"""
x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]
y_data = [[0],[0],[0],[1],[1],[1]]

X = tf.placeholder(tf.float32, shape=[None,2])
Y = tf.placeholder(tf.float32, shape=[None,1])
W = tf.Variable(tf.random_normal([2,1]),name='weight')  # [X의 개수, Y의 개수]
b = tf.Variable(tf.random_normal([1]),name='bias')      # [Y의 개수]

hypothesis = tf.sigmoid(tf.matmul(X,W)+b)
cost = -tf.reduce_mean(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

predicted = tf.cast(hypothesis>0.5,dtype=tf.float32)    # hypothesis가 0.5보다 크면 True=1, 작으면 False=0
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,Y),dtype=tf.float32))  # tf.equal: 예측한 값, 실제 값과 같은지=1, 다른지=0

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, _ = sess.run([cost,train],feed_dict = {X:x_data,Y:y_data})
        if step % 200 == 0:
            print(step, cost_val)
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:x_data, Y:y_data})
    print("\nHypothesis: ",h,"\nCorrect(Y): ", c, "\nAccuracy: ",a)

import numpy as np
xy = np.loadtxt('file.csv',delimiter=',',dtype=np.float32)  # data 못찾음. 이렇게 읽어들인다!
x_data = xy[:,0:-1]
y_data = xy[:,[-1]]

X = tf.placeholder(tf.float32, shape=[None,8])
Y = tf.placeholder(tf.float32, shape=[None,1])
W = tf.Variable(tf.random_normal([8,1]), name='weight')
b = tf.Variable(tf.random_nromal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(X,W)+b)
cost = -tf.reduce_mean(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis))
train = tf.train.GradientDescentOptimizer.minimize(cost)

predicted = tf.cast(hypothesis>0.5,dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,Y),dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(10001):
        sess.run(train, feed_dict={X:x_data,Y:y_data})
        if step % 20 == 0:
            print(step, sess.run(cost, feed_dict={X:x_data,Y:y_data}))
    h,c,a = sess.run([hypothesis,predicted,accuracy],feed_dict={X:x_data,Y:y_data})
    print("\nHypothesis: ",h,"\nCorrect(Y): ",c,"\nAccuracy: ",a)

"""
Softmax Classification
"""
x_data = [[1,2,1,1],[2,1,3,2],[3,1,3,4],[4,1,5,5],[1,7,5,5],[1,2,5,6],[1,6,6,6],[1,7,7,7]]
y_data = [[0,0,1],[0,0,1],[0,0,1],[0,1,0],[0,1,0],[0,1,0],[1,0,0],[1,0,0]]

X = tf.placeholder("float",[None,4])
Y = tf.placeholder("float",[None,3])
n_classes = 3
W = tf.Variable(tf.random_normal([4,n_classes]),name='weight')
b = tf.Variable(tf.random_normal([n_classes]),name='bias')

hypothesis = tf.nn.softmax(tf.matmul(X,W)+b)
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis),axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        sess.run(optimizer, feed_dict={X:x_data, Y:y_data})
        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}))

    a = sess.run(hypothesis,feed_dict={X:[[1,11,7,9]]})
    print(a, sess.run(tf.arg_max(a,1)))

    all = sess.run(hypothesis,feed_dict={X:[[1,11,7,9],
                                            [1,3,4,3],
                                            [1,1,0,1]]})
    print(all, sess.run(tf.arg_max(all,1)))     # tf.arg_max(arr, num): 각 row에서 상위 num의 index

"""
Fancy Softmax Classifier
"""
# logits = tf.matmul(X,W)+b
# hypothesis = tf.nn.softmax(logits)
# cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis),axis=1))
# cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
#                                                  labels = Y_one_hot)    #cross_entropy with logits!!!
# cost = tf.reduce_mean(cost_i)

xy = np.loadtxt("data-04=zoo.csv",delimiter=",",dtype=np.float32)
x_data = xy[:,0:-1]
y_data = xy[:,[-1]]

X = tf.placeholder(tf.float32,[None,16])
n_classes=7
Y = tf.placeholder(tf.int32,[None,n_classes])

Y_one_hot = tf.one_hot(Y,n_classes)
Y_one_hot = tf.reshape(Y_one_hot,shape=[-1,n_classes])

W = tf.Variable(tf.random_normal([16,n_classes]),name='weight')
b = tf.Variable(tf.random_normal([n_classes]),name='bias')
logits = tf.matmul(X,W)+b
hypothesis = tf.nn.softmax(logits)

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                 labels=Y_one_hot)
cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

xy = np.loadtxt('data-04-zoo.csv',delimiter=',',dtype=np.float32)
x_data = xy[:,0:-1]
y_data = xy[:,[-1]]

X = tf.placeholder(tf.float32, [None, 16])
Y = tf.placeholder(tf.int32,[None,1])       # shape: (?, 1)
n_classes = 7
Y_one_hot = tf.one_hot(Y, n_classes)    # [[0],[3]] --> [[[1,0,0,0]]],[[0,0,0,1]]]
                                        # shape: (?, 1, 7)
Y_one_hot = tf.reshape(Y_one_hot,[-1,n_classes])    # shape: (?, 7) 뒤엑 7이 되고 나머지는 알 바 X

W = tf.Variable(tf.random_normal([16,n_classes]),name='weight')
b = tf.Variable(tf.random_normal([n_classes]),name='bias')
logits = tf.matmul(X,W)+b
hypothesis = tf.nn.softmax(logits)
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=Y_one_hot)
cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.argmax(hypothesis,1)
correct_prediction = tf.equal(prediction,tf.argmax(Y_one_hot,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2000):
        sess.run(optimizer,feed_dict={X: x_data,Y: y_data})
        if step % 100 == 0:
            loss, acc = sess.run([cost,accuracy],feed_dict={X:x_data,Y:y_data})
            print("Step: {:5}\tLoss:{:.3f}\tAcc:{:.2%}".format(step,loss,acc))

    pred = sess.run(prediction,feed_dict={X:x_data})
    for p, y in zip(pred, y_data.flatten()):    # [[1],[0]] --> [1,0]
        print("[{}] Prediction: {} True Y: {}".format(p == int(y),p,int(y)))

"""
Training Testing
"""
x_data = [[1, 2, 1],[1, 3, 2],[1, 3, 4],[1, 5, 5],[1, 7, 5],[1, 2, 5],[1, 6, 6],[1, 7, 7]]
y_data = [[0, 0, 1],[0, 0, 1],[0, 0, 1],[0, 1, 0],[0, 1, 0],[0, 1, 0],[1, 0, 0],[1, 0, 0]]

x_test = [[2, 1, 1],[3, 1, 2],[3, 3, 4]]
y_test = [[0, 0, 1],[0, 0, 1],[0, 0, 1]]

X = tf.placeholder("float",[None,3])
Y = tf.placeholder("float",[None,3])
W = tf.Variable(tf.random_normal([3,3]))
b = tf.Variable(tf.random_normal([3]))
hypothesis = tf.nn.softmax(tf.matmul(X,W)+b)
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis),axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.arg_max(hypothesis,1)
is_correct = tf.equal(prediction,tf.arg_max(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(201):
        cost_val, W_val, _ = sess.run([cost,W,optimizer],feed_dict={X:x_data,Y:y_data})
        print(step, cost_val, W_val)
    print("Prediction: ",sess.run(prediction,feed_dict={X:x_test}))
    print("Accuracy: ",sess.run(accuracy, feed_dict={X:x_test,Y:y_test}))

# Normalization
xy = np.array([[828.659973, 833.450012, 908100, 828.349976, 831.659973],
               [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
               [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
               [816, 820.958984, 1008100, 815.48999, 819.23999],
               [819.359985, 823, 1188100, 818.469971, 818.97998],
               [819, 823, 1198100, 816, 820.450012],
               [811.700012, 815.25, 1098100, 809.780029, 813.669983],
               [809.51001, 816.659973, 1398100, 804.539978, 809.559998]])

x_data = xy[:, 0:-1]
y_data = xy[:,[-1]]

X = tf.placeholder(tf.float32,shape=[None,4])
Y = tf.placeholder(tf.float32,shape=[None,1])
W = tf.Variable(tf.random_normal([4,1]),name='weight')
b = tf.Variable(tf.random_normal([1]),name='bias')

hypothesis = tf.matmul(X,W)+b
cost = tf.reduce_mean(tf.square(hypothesis-Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        cost_val, hy_val, _ = sess.run([cost, hypothesis, optimizer],
                                       feed_dict={X:x_data,Y:y_data})
        print(step,"Cost: ",cost_val, "\nPrediction:|n",hy_val)

def MinMaxScaler(data):
    numerator = data - np.min(data,0)
    denominator = np.max(data,0) - np.min(data,0)
    return numerator/(denominator+1e-7)
xy = MinMaxScaler(xy)

"""
MNIST
"""
from tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(777)  # for reproducibility
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

nb_classes = 10
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, nb_classes])
W = tf.Variable(tf.random_normal([784, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))

hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

is_correct = tf.equal(tf.arg_max(hypothesis,1),tf.arg_max(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))

training_epochs = 15
# epoch: 전체 데이터셋을 한 번 학습시키는 것
batch_size = 100
# 너무 큰 데이터 --> 나눠서 올림
'''
epoch: one forward pass and one backward pass of all!!!! the training examples
batch size: the number of training exmapels in one forward/backward pass. The higher the batch size, 
the more memory size you need
# of iterations: number of passes, each pass using [batch size] number of examples
To be clear, oen pass = one forward pass + one backward pass
EX)
If you have 1000 training examples, and your batch size is 500
==> 2 iterations to complete 1 epoch!
'''
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        avg_cost = 0
        total_iterations = int(mnist.train.num_examples/batch_size)

        for i in range(total_iterations):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost,optimizer],feed_dict={X:batch_xs,Y:batch_ys})
            avg_cost += c/total_iterations
        print("Epoch:",'%04d'%(epoch+1),'cost =','{:.9f}'.format(avg_cost))
    print("Learning finished")

    print("Accuracy:",accuracy.eval(session=sess,feed_dict={X:mnist.test.images,
                                                            Y:mnist.test.labels}))

    import matplotlib.pyplot as plt
    import random
    r = random.randint(0,mnist.test.num_examples-1)
    print("Label:",sess.run(tf.argmax(mnist.test.labels[r:r+1],1)))
    print("Prediction:",sess.run(tf.argmax(hypothesis,1),
                                 feed_dict={X:mnist.test.images[r:r+1]}))

    plt.imshow(mnist.test.images[r:r+1].reshape(28,28),cmap='Greys',interpolation='nearest')
    plt.show()

"""
Tensor Manipulation
"""
t = tf.constant([1,2,3,4])
sess = tf.Session()
tf.shape(t).eval(session = sess)
t = tf.constant([[1,2],[3,4]])
tf.shape(t).eval(session=sess)
t = tf.constant(
    [
        [
            [
                [
                    [1,2,3,4],
                    [5,6,7,8],
                    [9,10,11,12]
                ],
                [
                    [13,14,15,16],
                    [17,18,19,20],
                    [21,22,23,24]
                ]
            ]
        ]
    ]
)
tf.shape(t).eval(session=sess)  #array([1, 1, 2, 3, 4])

# Matmul vs Multply
matrix1 = tf.constant([[1,2],[3,4]])
matrix2 = tf.constant([[1],[2]])
matrix1.shape
matrix2.shape
tf.matmul(matrix1,matrix2).eval(session=sess)
(matrix1*matrix2).eval(session=sess)

# Broadcasting: shape이 달라도 더하기 되게!
matrix1 = tf.constant([[1,2]])
matrix2 = tf.constant(3)
(matrix1+matrix2).eval(session=sess)
matrix1 = tf.constant([[1,2]])
matrix2 = tf.constant([[1],[2]])
(matrix1+matrix2).eval(session=sess)

# Reduce Mean
x = [[1.,2.],[3.,4.]]
tf.reduce_mean(x,axis=-1).eval(session=sess)    #가장 안 쪽에 있는 축으로!

# Argmax
x = [[0,1,2],[2,1,0]]
tf.argmax(x,axis=0).eval(session=sess)  # 위치 구하는 것!
tf.argmax(x,axis=1).eval(session=sess)
tf.argmax(x,axis=-1).eval(session=sess)

# Reshape
t = np.array([[[0,1,2],
               [3,4,5]],
              [[6,7,8],
               [9,10,11]]])
t.shape
tf.reshape(t,shape=[-1,3]).eval(session=sess)   # 보통 가장 안쪽에 있는 값을 중심으로 함
tf.reshape(t,shape=[-1,1,3]).eval(session=sess)

tf.squeeze([[0],[1],[2]]) #--> [0,1,2]
tf.expand_dims([0,1,2],1) #==> [[0],[1],[2]] 하나 더!

# One HOt
tf.one_hot([[0],[1],[2],[0]],depth=3) # --> expand 되어! reshape 필요!

# Casting: 원하는 형태의 datatype으로 바꾸려고 할때에!

# Stack: 쌓기! merge? concatenate랑 비슷

# Ones and zeros like: 모양이 똑같은 애들로 다 1이거나 0인 matrix

# Zip:

