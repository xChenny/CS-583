from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print('Shape of x_train: ' + str(x_train.shape))
print('Shape of x_test: ' + str(x_test.shape))
print('Shape of y_train: ' + str(y_train.shape)) 
print('Shape of y_test: ' + str(y_test.shape))

x_train_vec = x_train.reshape(60000, 784) 
x_test_vec = x_test.reshape(10000, 784)

print('Shape of x_train_vec is ' + str(x_train_vec.shape))
