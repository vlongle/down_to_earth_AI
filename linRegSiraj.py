'''
This module is a practice in building ground-up linear regression model in Python

The file follows the video by Siraj Raval: How to Do Linear Regression using Gradient Descent

Link to video: https://www.youtube.com/watch?v=XdM6ER7zTLk
'''

import os
import csv # for loading dataset
import threading # for multi-threading
import numpy as np # for array manipulation
import matplotlib.pyplot as plt # for graphing
import matplotlib.animation as animation # for updating the graph
from collections import defaultdict # for efficient loading dataset
import time # for timing the task

'''
A thread that optimizes the slope -- m -- and bias -- b -- for the linear 
regression model. 
The arg (threading.Thread) passed to the class is like inheritance in Java
<< Optimize_Linear_Reg >> is a subclass of << threading.Thread >>
'''
class Optimize_Linear_Reg(threading.Thread):

	def __init__(self, param_path, predictor_name, output_name, LEARNING_RATE, N_EPOCHS):
		super().__init__() # initialize the thread

		# error checking
		if predictor_name == None or output_name == None or param_path == None:
			raise Exception('Please enter the predictor_name & output_name')

		self.predictor_name = predictor_name
		self.output_name = output_name
		self.param_path = param_path
		self.LEARNING_RATE = LEARNING_RATE
		self.N_EPOCHS = N_EPOCHS


	def run(self):
		'''
		This function trains the simple linear regression model on the following params
		
		Feel free to modify the hyper-parameters

		@params: predictor_name (x), output_name (y)
		
		@return: (m,b) with m -- the slope of the line, b -- the constant term (bias)
		
		'''

		print('Starting to optimize the parameters')
		# load the dataset
		data_lookup = self.load_csv('auto-mpg 2.csv', self.predictor_name, self.output_name)

		# make the dataset into a standard array form [[list of x], [list of y]]
		dataset = [None, None]
		dataset[0] = data_lookup[self.predictor_name]
		dataset[1] = data_lookup[self.output_name]

		dataset[0] = [float(x) for x in dataset[0]]
		dataset[1] = [float(x) for x in dataset[1]]


		#hyper-parameters
		LEARNING_RATE = self.LEARNING_RATE
		N_EPOCHS = self.N_EPOCHS

		initial_b = 0.0
		initial_m = 0.0

		print('Begin training.')
		m, b = self.train(dataset, initial_m, initial_b, LEARNING_RATE, N_EPOCHS, self.param_path)

		print('The final (m,b): ', (m,b))


	def load_csv(self, path, *args):
		'''
		This function loads the data from a csv file

		@param: path to the file, any attributes / column names we want to extract (*args)

		@return: a dict of the requested attributes
		'''
		with open(path, 'r') as file:
			print('Loading dataset with attributes: ', args)
			reader = csv.reader(file)
			# use iterator to loop through the dataset
			iterator = iter(reader)

			first_row = next(iterator) # first_row contains all the attributes name
			index_holder = {}		   # dict holding the indices in the dataset of our attributes
			requested_attrib = defaultdict(list)	# return this

			for attribute in args:
				try:
					index_holder[attribute] = first_row.index(attribute)
				except ValueError:
					print('Attribute <<',attribute, '>> does not exist')
			# loop through the dataset
			for row in reader:
				for key, value in index_holder.items():
					requested_attrib[key].append(row[value])

		print('Loading completed')
		return requested_attrib


	# muscle / training functions

	def compute_error(self, m_current, b_current, dataset):
		'''
		This function computes the error for one epoch.
		The loss function is simple mean squared residual
		'''
		predictors = dataset[0]
		outputs = dataset[1]

		if len(predictors) != len(outputs):
			raise Exception('The number of predictors must match that of the output')
		
		total_loss = 0
		N = len(predictors)

		for i in range(N):
			x = float(predictors[i])
			y = float(outputs[i])
			y_predicted = x*m_current+b_current
			loss = (y-y_predicted)**2
			total_loss += loss


		return total_loss/N


	def gradient_descent(self, m_current, b_current, dataset, LEARNING_RATE):
		'''
		This function computes the gradient descent for one epoch.
		The formula for m_gradient and b_gradient can easily be found online
		'''
		predictors = dataset[0]
		outputs = dataset[1]

		# error checking
		if len(predictors) != len(outputs):
			raise Exception('The number of predictors must match that of the output')
		
		b_gradient = 0.0
		m_gradient = 0.0
		N = len(predictors)

		# Looping through the dataset 
		for i in range(N):
			x = float(predictors[i])
			y = float(outputs[i])
			m_gradient += -(2/N)*(x*(y-(m_current*x+b_current)))
			b_gradient += -(2/N)*(y-(m_current*x+b_current))

		b_new = b_current - LEARNING_RATE*b_gradient
		m_new = m_current - LEARNING_RATE*m_gradient

		return [m_new, b_new]


	def train(self, dataset, starting_m, starting_b, LEARNING_RATE, N_EPOCHS, param_path):
		'''
		This function trains the simple regression by iterating throught the 
		entire dataset << N_EPOCHS >> times and calculate the gradient_descent and
		compute_error for each epoch. The m,b is then updated and written into a text file
		named param_path
		'''
		m = starting_m
		b = starting_b

		# looping and updating w,b into param.txt
		for i in range(N_EPOCHS):
			m,b = self.gradient_descent(m,b, dataset, LEARNING_RATE)
			if i % 500 == 0:
				print('The loss after', i, 'epochs is ', self.compute_error(m,b,dataset))

				with open(param_path, 'w') as file:
					file.write(str(m) + ',' + str(b))

		return [m,b]




'''
A Daemon thread that is automatically closed after all other threads have finished.
This thread just updates the graph in real-time while the linear model parameters are 
being optimized by @class Optimize_Linear_Reg
'''
class Visual_Thread(threading.Thread):


	def __init__(self, param_path, predictor_name, output_name):
		'''
		This function initializes Visual_Thread to a Daemon Thread
		'''
		super().__init__() # super.__init__(self) also works
		super().setDaemon(True)
		print('IS DAEMON _________', self.isDaemon())
		self.param_path = param_path
		self.predictor_name = predictor_name
		self.output_name = output_name


	def run(self):
		'''
		In a << threading.Thread >> subclass, the << run() >> function is always 
		called.
		This function calls animate_data()
		'''
		self.animate_data(self.param_path, self.predictor_name, self.output_name)



	def load_csv(self, path, *args):
		'''
		This function loads the data from a csv file

		@param: path to the file, any attributes / column names we want to extract (*args)

		@return: a dict of the requested attributes
		'''
		with open(path, 'r') as file:
			print('Loading dataset with attributes: ', args)
			reader = csv.reader(file)
			# use iterator to loop through the dataset
			iterator = iter(reader)

			first_row = next(iterator) # first_row contains all the attributes name
			index_holder = {}		   # dict holding the indices in the dataset of our attributes
			requested_attrib = defaultdict(list)	# return this

			for attribute in args:
				try:
					index_holder[attribute] = first_row.index(attribute)
				except ValueError:
					print('Attribute <<',attribute, '>> does not exist')
			# loop through the dataset
			for row in reader:
				for key, value in index_holder.items():
					requested_attrib[key].append(row[value])

		print('Loading completed')
		return requested_attrib


	# graphing / visualization functions
	def animate_data(self, param_path, predictor_name, output_name):
		'''
		This function sets up the dataset and the animation calls 
		to repetively display a graph through the function graph()
		'''

		# load the dataset
		data_lookup = self.load_csv('auto-mpg 2.csv', predictor_name, output_name)

		# make the dataset into a standard array form [[list of x], [list of y]]
		dataset = [None, None]
		dataset[0] = data_lookup[predictor_name]
		dataset[1] = data_lookup[output_name]

		dataset[0] = [float(x) for x in dataset[0]]
		dataset[1] = [float(x) for x in dataset[1]]

		# This is just routine procedure for drawing matplotlib
		fig = plt.figure()
		ax1 = fig.add_subplot(1,1,1) # axes 1
		ani = animation.FuncAnimation(fig, self.graph, fargs=(dataset, param_path, ax1),interval=1000)
		plt.show()


	def graph(self, dummy_i, dataset, param_path, axis):
		'''
		This function graphs the 2D plane based on the dataset passed to it
		and the line with m, b found in a text file in param_path
		'''
		file = open(param_path, 'r')
		parameters = file.read().split(',')

		# exception may occur as the << Optimize_Linear_Reg >> is in the
		# process of writing out to param_path. Making the param_path
		# file appears as empty
		try:
			m, b = float(parameters[0]), float(parameters[1])

			x_vals = np.arange(min(dataset[0]), max(dataset[0]), 3)
			y_vals = [b + m*x for x in x_vals]

			axis.clear()
			axis.set_title("Linear Regression")
			axis.set_xlabel('MPG')
			axis.set_ylabel('Engine displacement')
			# plot the blue line
			axis.plot(x_vals, y_vals, 'b--')
			# plot red data points
			axis.plot(dataset[0],dataset[1], 'ro')

		except:
			print('Not loading the parameters')



# main function
if __name__ == '__main__':

	# set up a clock to count the time it takes for our program to run
	start_time = time.time()

	# set up Visual_Thread with @params: param_path, predictor_name, output_name
	visual_thread = Visual_Thread('param.txt','mpg','displacement')
	visual_thread.start()


	# set up Optimize_Linear_Reg with @params: param_path, predictor_name, output_name
	# LEARNING_RATE, N_EPOCHS
	optimize_thread = Optimize_Linear_Reg('param.txt','mpg','displacement', 0.0001, 100000)
	optimize_thread.start()


	# tells the main thread not to run any code below until after the optimize_thread finishes
	optimize_thread.join()

	end_time = time.time()

	print("Finished after", end_time - start_time)

	# force the process to exit
	os._exit(0)


