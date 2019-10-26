import tensorflow as tf
import scipy
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics

class Neural_Net():
	def __init__(self, layers, loss_func=AdamOptimizer):
		self.layers = layers
		for layer in self.layers:
			self.params.append(layer.params)
		self.loss_func = loss_func

	def forward(self, X):
		for layer in self.layers:
			X = layer.forward(X)
		return X

	def backward(self, dout):


class RNN():
	#Data shape: batch size, timesteps, num_input


	def __init__(self, x, weights, biases, learning_rate):
		x = tf.unstack(x, timesteps, 1)

		lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

		self.outputs, self.states = rnn.static_rnn(lstm_cell, x, dtype = tf.float32)

		self.logits = tf.matmul(outputs[-1], weights['out']) + biases['out']

		self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
		self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
		self.train_op = optimizer.minimize(loss_op)

		self.init = tf.global_variables_initalizer()

	def make_prediction(self, x, weights, biases, logits = None):
		if logits == None:
			self.logits = tf.matmul(outputs[-1], weights['out']) + biases['out']
		else:
			return self.logits

	def run_session(X, y, test_size = 0.9, training_steps):
		for step in xrange(0, training_steps):
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = i)
			sess.run(X_train, feed_dict = {X: X_train, Y: y_train})

			if step % self.display_step == 0:
				loss, acc = sess.run([loss_op, accuracy], feed_dict = {X: X_train, Y: y_train})


		        test_score = 1 - metrics.accuracy_score(y_test, y_pred, normalize = True)
		        train_score = 1 - metrics.accuracy_score(y_train, y_pred_train, normalize = True)
		        '''
				print("Step " + str(step) + ", Minibatch Loss= " + \
	                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
	                  "{:.3f}".format(acc))
				'''
				print("Validation Score: " + str(test_score) + " Train Score: " + str(train_score))





def main():
	num_input = 1
	X = tf.placeholder("float", [None, timesteps, num_input])
	Y = tf.placeholder("float", [None, num_classes])
	NND = Neural_Net()

	#num_input = len(array_data)
	num_input = 5
	num_features = sample_length
	num_classes = 1

	weights = {'out': tf.Variable(tf.random_normal([num_features, num_classes]))}
	biases = {'out': tf.Variable(tf.random_normal([num_classes]))}

if __name__ == '__main__':
	try:
		main()
	except KeyvalueError:
		exit()