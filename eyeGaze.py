#Author: Usama Munir Sheikh
	#Electrical and computer systems engineering student at RPI
#This is my code for the final project for the Intro to Deep Learning course at RPI
	#taught by Professor Qiang Ji in Spring 2017
#The following code implements a convolutional neural network for
	#eye tracking on a screen
#About two gigabytes of data from http://gazecapture.csail.mit.edu/ was used
	#for training and validation

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

print("Tensorflow Version: ")
print(tf.__version__)  

def main():
	time_initial = time.time() #To see how much time required for the entire code to run

	#Load Training and Validation Data
	npzfile = np.load("train_and_val.npz") #requires data file
	train_eye_left = npzfile["train_eye_left"]
	train_eye_right = npzfile["train_eye_right"]
	train_face = npzfile["train_face"]
	train_y = npzfile["train_y"]
	train_face_mask = npzfile["train_face_mask"]
	val_eye_left = npzfile["val_eye_left"]
	val_eye_right = npzfile["val_eye_right"]
	val_face = npzfile["val_face"]
	val_face_mask = npzfile["val_face_mask"]
	val_y = npzfile["val_y"]

	#Parameters
	N = 48000
	N_val = 5000;
	B = 128 #batch size
	VB = 100 #Test batch size
	K = 2 #number of outputs
	F = 512 #Size of fully connected layer
	M = 625 #Mask(625)
	Opt = 3*F + M +1024#A3_left(2592) + A3_right(2592) + A3_face(2592) + Mask(625)
	#Network Parameters
	eta = 0.001 #learning rate
	
	#Make Tensor Flow Variables and Placeholders
	X_left = tf.placeholder("float32",[None,64,64,3], name='Input-Left-Eye')
	X_right = tf.placeholder("float32",[None,64,64,3], name='Input-Right-Eye')
	X_face = tf.placeholder("float32",[None,64,64,3], name='Input-Face')
	Mask = tf.placeholder("float32",[None,25,25], name='Input-Mask') 
	Y = tf.placeholder("float32",[None,2], name='Actual-Output')

	with tf.variable_scope('Weights'):
		W1 = tf.get_variable("W1", shape=[11, 11, 3, 96], initializer=tf.contrib.layers.xavier_initializer_conv2d())
		b1 = tf.Variable(tf.zeros([96]))
		W2 = tf.get_variable("W2", shape=[5, 5, 96, 256], initializer=tf.contrib.layers.xavier_initializer_conv2d())
		b2 = tf.Variable(tf.zeros([256]))
		W3 = tf.get_variable("W3", shape=[3, 3, 256, 384], initializer=tf.contrib.layers.xavier_initializer_conv2d())
		b3 = tf.Variable(tf.zeros([384]))
		W4 = tf.get_variable("W4", shape=[1, 1, 384, 64], initializer=tf.contrib.layers.xavier_initializer_conv2d())
		b4 = tf.Variable(tf.zeros([64]))
		W1_face = tf.get_variable("W1_face", shape=[11, 11, 3, 96], initializer=tf.contrib.layers.xavier_initializer_conv2d())
		b1_face = tf.Variable(tf.zeros([96]))
		W2_face = tf.get_variable("W2_face", shape=[5, 5, 96, 256], initializer=tf.contrib.layers.xavier_initializer_conv2d())
		b2_face = tf.Variable(tf.zeros([256]))
		W3_face = tf.get_variable("W3_face", shape=[3, 3, 256, 384], initializer=tf.contrib.layers.xavier_initializer_conv2d())
		b3_face = tf.Variable(tf.zeros([384]))
		W4_face = tf.get_variable("W4_face", shape=[1, 1, 384, 64], initializer=tf.contrib.layers.xavier_initializer_conv2d())
		b4_face = tf.Variable(tf.zeros([64]))
		W5_face = tf.get_variable("W5_face", shape=[1, 1, 64, 128], initializer=tf.contrib.layers.xavier_initializer_conv2d())
		b5_face = tf.Variable(tf.zeros([128]))
		W_face = tf.get_variable("W_face", shape=[F, F], initializer=tf.contrib.layers.xavier_initializer())
		b_face = tf.Variable(tf.zeros([F]))
		W_mask = tf.get_variable("W_mask", shape=[M, M], initializer=tf.contrib.layers.xavier_initializer())
		b_mask = tf.Variable(tf.zeros([M]))
		Wout = tf.get_variable("Wout", shape=[Opt, K], initializer=tf.contrib.layers.xavier_initializer_conv2d())
		bout = tf.Variable(tf.zeros([K]))

	#Write Tensorflow equations and models
	'''
	C = Convolutional Layer
	A = Activation Layer
	pool = Pooling Layer
	'''
	#Forward Propagation Left Eye
	with tf.name_scope('conv1-eye-left'):
		conv1_left = tf.nn.conv2d(X_left, W1, [1, 1, 1, 1], padding = 'VALID')	
		C1_left = tf.nn.bias_add(conv1_left, b1)
		A1_left = tf.nn.relu(C1_left)
		pool1_left = tf.nn.max_pool(A1_left, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
	with tf.name_scope('conv2-eye-left'):
		conv2_left = tf.nn.conv2d(pool1_left, W2, [1, 1, 1, 1], padding = 'VALID')
		C2_left = tf.nn.bias_add(conv2_left, b2)
		A2_left = tf.nn.relu(C2_left)
		pool2_left = tf.nn.max_pool(A2_left, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
	with tf.name_scope('conv3-eye-left'):
		conv3_left = tf.nn.conv2d(pool2_left, W3, [1, 1, 1, 1], padding = 'VALID')
		C3_left = tf.nn.bias_add(conv3_left, b3)
		A3_left = tf.nn.relu(C3_left)
		pool3_left = tf.nn.max_pool(A3_left, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
	with tf.name_scope('conv4-eye-left'):
		conv4_left = tf.nn.conv2d(pool3_left, W4, [1, 1, 1, 1], padding = 'VALID')
		C4_left = tf.nn.bias_add(conv4_left, b4)
		A4_left = tf.nn.relu(C4_left)
	print(A4_left.get_shape())
	batch_size = tf.shape(A4_left)[0]
	with tf.name_scope('fc-eye-left'):
		A4_flattened_left = tf.reshape(A4_left,[batch_size, -1])

	#Forward Propagation right Eye
	with tf.name_scope('conv1-eye-right'):
		conv1_right = tf.nn.conv2d(X_right, W1, [1, 1, 1, 1], padding = 'VALID')
		C1_right = tf.nn.bias_add(conv1_right, b1)
		A1_right = tf.nn.relu(C1_right)
		pool1_right = tf.nn.max_pool(A1_right, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
	with tf.name_scope('conv2-eye-right'):
		conv2_right = tf.nn.conv2d(pool1_right, W2, [1, 1, 1, 1], padding = 'VALID')
		C2_right = tf.nn.bias_add(conv2_right, b2)
		A2_right = tf.nn.relu(C2_right)
		pool2_right = tf.nn.max_pool(A2_right, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
	with tf.name_scope('conv3-eye-right'):
		conv3_right = tf.nn.conv2d(pool2_right, W3, [1, 1, 1, 1], padding = 'VALID')
		C3_right = tf.nn.bias_add(conv3_right, b3)
		A3_right = tf.nn.relu(C3_right)
		pool3_right = tf.nn.max_pool(A3_right, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
	with tf.name_scope('conv4-eye-right'):
		conv4_right = tf.nn.conv2d(pool3_right, W4, [1, 1, 1, 1], padding = 'VALID')
		C4_right = tf.nn.bias_add(conv4_right, b4)
		A4_right = tf.nn.relu(C4_right)
	with tf.name_scope('fc-eye-right'):
		A4_flattened_right = tf.reshape(A4_right, [batch_size, -1])
		#print(A3_flattened_left.get_shape())

	#Forward Propagation Both Eyes
	with tf.name_scope('fc1_eye'):
		fc1_eye = tf.concat([A4_flattened_left, A4_flattened_left],1) 

	#Forward Propagation Face
	with tf.name_scope('conv1-face'):
		conv1_face = tf.nn.conv2d(X_face, W1_face, [1, 1, 1, 1], padding = 'VALID')
		C1_face = tf.nn.bias_add(conv1_face, b1_face)
		A1_face = tf.nn.relu(C1_face)
		pool1_face = tf.nn.max_pool(A1_face, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
	with tf.name_scope('conv2-face'):
		conv2_face = tf.nn.conv2d(pool1_face, W2_face, [1, 1, 1, 1], padding = 'VALID')
		C2_face = tf.nn.bias_add(conv2_face, b2_face)
		A2_face = tf.nn.relu(C2_face)
		pool2_face = tf.nn.max_pool(A2_face, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
	with tf.name_scope('conv3-face'):
		conv3_face = tf.nn.conv2d(pool2_face, W3_face, [1, 1, 1, 1], padding = 'VALID')
		C3_face = tf.nn.bias_add(conv3_face, b3_face)
		A3_face = tf.nn.relu(C3_face)
		pool3_face = tf.nn.max_pool(A3_face, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
	with tf.name_scope('conv4-face'):
		conv4_face = tf.nn.conv2d(pool3_face, W4_face, [1, 1, 1, 1], padding = 'VALID')
		C4_face = tf.nn.bias_add(conv4_face, b4_face)
		A4_face = tf.nn.relu(C4_face)
		pool4_face = tf.nn.max_pool(A4_face, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
	with tf.name_scope('conv5-face'):
		conv5_face = tf.nn.conv2d(pool4_face, W5_face, [1, 1, 1, 1], padding = 'VALID')
		C5_face = tf.nn.bias_add(conv5_face, b5_face)
		A5_face = tf.nn.relu(C5_face)
	print(A5_face.get_shape())
	with tf.name_scope('fc1-face'):
		A5_flattened_face = tf.reshape(A5_face, [batch_size, -1])
	with tf.name_scope('fc2-face'):
		fc2_face = tf.matmul(A5_flattened_face, W_face) + b_face

	#Forward Propagation Eyes + Face
	with tf.name_scope('fc1_face_eye'):
		fc1_face_eye = tf.concat([fc1_eye, fc2_face],1)
	
	#Face Mask
	with tf.name_scope('fc1_mask'):
		mask_flattened = tf.reshape(Mask, [batch_size, -1])
	with tf.name_scope('fc2_mask'):
		fc2_mask = tf.matmul(mask_flattened, W_mask) + b_mask

	#Forward Propagation Eyes + Face + Mask
	with tf.name_scope('fc1_face_mask_eye'):
		fc1_face_mask_eye = tf.concat([fc1_face_eye, fc2_mask],1)
	#with tf.name_scope('fc1_face_mask_eye_dropout'):
		#fc_final = tf.nn.dropout(fc1_face_mask_eye, 0.5)
	#Forward Propagation Output
	with tf.name_scope('Output'):
		output = tf.matmul(fc1_face_mask_eye, Wout) + bout
	#print(output.get_shape())

	#Back Propagation
	error = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(Y,output)),axis = 1))) #RMS Error
	cost = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(Y,output)),axis = 1)) #MS Cost
	optimizer_step = tf.train.AdamOptimizer(learning_rate=eta).minimize(cost) #run optimizer
	
	# Create the collection.
	tf.get_collection("validation_nodes")
	#Add stuff to the collection.
	tf.add_to_collection("validation_nodes", X_left)
	tf.add_to_collection("validation_nodes", X_right)
	tf.add_to_collection("validation_nodes", X_face)
	tf.add_to_collection("validation_nodes", Mask)
	tf.add_to_collection("validation_nodes", output)
	#Save Model
	saver = tf.train.Saver()
	
	#Create Empty Matrices to Save results
	loss_plot = []
	err_train_plot = []
	err_val_plot = []
	n = 500 #How many iterations before error calculations
	num_epochs =  1000 #number of epochs
	num_itr = np.divide(N,B) #number of iterations per epoch
	num_itr = num_itr.astype(np.int64)
	besterror = 20
	err_val_np = 0
	model = tf.global_variables_initializer()	
	with tf.Session() as session:
		session.run(model)
		#writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())
		writer = tf.summary.FileWriter('./graph_logs/', session.graph)
		for j in range(num_epochs):
			epoch_number = j+1
			for i in range(num_itr):
				itr_number = i+1
				#Pick Batch for Training
				indices = np.arange(B) + (i*B)
				data_X_left = np.divide(train_eye_left[indices],255.0)
				data_X_right = np.divide(train_eye_right[indices],255.0)
				data_X_face = np.divide(train_face[indices],255.0)
				data_Y = train_y[indices]
				mask = train_face_mask[indices]

				#Train
				optimizer_step.run(feed_dict={X_left: data_X_left, X_right: data_X_right, X_face: data_X_face, Y: data_Y, Mask: mask})

				#Calculate Accuracy/Errors #PrintValues # SaveResults #Every_n_Iterations
				if((itr_number % n == 0) or (itr_number == 1)):
					print('-------------------------')
					print('Epoch Number: ' + repr(epoch_number))
					print('Iteration Number: ' + repr(itr_number))
					print(' ')

					loss_np, yp_train = session.run([cost, output], feed_dict={X_left: data_X_left, X_right: data_X_right, X_face: data_X_face, Y: data_Y, Mask: mask})
					err_train_np = np.mean(np.sqrt(np.sum((yp_train - data_Y)**2, axis=1)))

					#Calculate Validation Error
					val_itr = int(5000/VB)
					for m in range(val_itr):
						indices_random = np.arange(VB) + (m*VB)
						val_X_left = np.divide(val_eye_left[indices_random],255.0)
						val_X_right = np.divide(val_eye_right[indices_random],255.0)
						val_X_face = np.divide(val_face[indices_random],255.0)
						val_Y = val_y[indices_random]
						val_mask = val_face_mask[indices_random]
						temp = error.eval(feed_dict={X_left: val_X_left, X_right: val_X_right, X_face: val_X_face, Y: val_Y, Mask: val_mask})
						err_val_np = err_val_np + temp
					err_val_np = err_val_np/50

					print('Loss: ' + repr(loss_np))
					print('Training Error: ' + repr(err_train_np))
					print('Validation Error: ' + repr(err_val_np))
					if besterror >= err_val_np:
						besterror = err_val_np
						if besterror < 2.2:
							n = 250
						if besterror < 2.0:
							save_path = saver.save(session, "my-model")
							n = 50
							break
					print('Best Val Error: ' + repr(besterror))

					err_train_plot.append(err_train_np)
					err_val_plot.append(err_val_np)
					loss_plot.append(loss_np)

					print(' ')
					print('------------------------')		
		
	session.close()

	#Print Elapsed Time
	print('------------------------')
	print('Optimization Finished')
	elapsed = time.time() - time_initial
	print('Time Elapsed: ' + repr(elapsed))

	#Plots
	itr_number = len(loss_plot)
	t = np.arange(itr_number)
	fig, ax1 = plt.subplots()
	ax1.plot(t,np.reshape(loss_plot,(itr_number,1)), 'b-')
	ax1.set_xlabel('Number of Iterations ')
	ax1.set_ylabel('Loss', color='b')
	ax1.tick_params('y', colors='b')
	
	ax2 = ax1.twinx()
	ax2.plot(t,np.reshape(err_train_plot,(itr_number,1)), 'r-')
	ax2.set_ylabel('Error ', color='k')
	ax2.tick_params('y', colors='k')

	ax2.plot(t,np.reshape(err_val_plot,(itr_number,1)), 'g-')
	
	fig.tight_layout()
	plt.show()
	
if __name__ == "__main__":
    main()
