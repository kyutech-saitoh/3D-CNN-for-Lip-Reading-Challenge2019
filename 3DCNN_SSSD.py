import argparse
import os
import cv2
import time

import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt
import numpy as np

from keras import optimizers
from keras.layers import (Activation, Conv3D, Dense, Dropout, Flatten, MaxPooling3D)
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model

from sklearn.metrics import classification_report, confusion_matrix

from tqdm import tqdm


# number of class (25 words)
WORD_NUM = 25
# image size (32 x 32) [pixel]
IMAGE_SIZE = 32
# color image
CHANNEL_NUM = 3
# frame number fo 3D-CNN [frame]
DEPTH_NUM = 25
# LFROI dir name 
DATA_PATH = "LFROI"
# learning rate
LEARNING_RATE = 0.0001
# batch size
BATCH_SIZE = 128
# epoch
EPOCH = 100
# work directory
WORK_DIR = "Challenge2019"


# generate learning curve
def plot_history(history, result_dir):
	acc_name = "graph_accuracy.png"
	loss_name = "graph_loss.png"

	# accuracy
	plt.plot(range(1, EPOCH+1), history.history["acc"], label="training", marker=".")
	plt.plot(range(1, EPOCH+1), history.history["val_acc"], label="test", marker=".")
	plt.title("model accuracy")
	plt.xlabel("epoch")
	plt.ylabel("accuracy")
	plt.grid()
	plt.legend(loc="lower right")
	plt.savefig(os.path.join(result_dir, acc_name))
	plt.close()

	# loss
	plt.plot(range(1, EPOCH+1), history.history["loss"], label="training", marker=".")
	plt.plot(range(1, EPOCH+1), history.history["val_loss"], label="test", marker=".")
	plt.title("model loss")
	plt.xlabel("epoch")
	plt.ylabel("loss")
	plt.grid()
	plt.legend(loc="upper right")
	plt.savefig(os.path.join(result_dir, loss_name))
	plt.close()

# save result
def save_history(history, result_dir):
	log_name = "log.txt"

	loss = history.history["loss"]
	acc = history.history["acc"]
	val_loss = history.history["val_loss"]
	val_acc = history.history["val_acc"]
	nb_epoch = len(acc)

	with open(os.path.join(result_dir, log_name), "w") as fp:
		fp.write("epoch\tloss\tacc\tval_loss\tval_acc\n")
		for i in range(nb_epoch):
			fp.write("{}\t{}\t{}\t{}\t{}\n".format(i, loss[i], acc[i], val_loss[i], val_acc[i]))

# load scene images
def load_images(dir_name):
	# get filename list
	file_list = os.listdir(dir_name)

	# frame number
	frame_num = len(file_list)

	if frame_num < DEPTH_NUM:
		dframe = DEPTH_NUM - frame_num
		iframe = round(dframe / 2)
		fframe = dframe - iframe
		
		iframes = [1 for x in range(iframe)]
		nframes = [x+1 for x in range(frame_num)]
		fframes = [frame_num for x in range(fframe)]
		frames = iframes + nframes + fframes
		
	else:
		frames = [round(x * frame_num / DEPTH_NUM + 1) for x in range(DEPTH_NUM)]

	frame_array = []

	for i in range(DEPTH_NUM):
		# image file name (00000.jpg)
		image_name = os.path.join(dir_name, str(frames[i]).zfill(5) + ".jpg")

		# load image
		img = cv2.imread(image_name)
		if img is None:
			print("ERROR: can not read image : ", image_name)
		else:
			# original image size --> IMAGE_SIZE x IMAGE_SIZE
			img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
			frame_array.append(img)

	return np.array(frame_array)

def load_data(list_file):
	file_num = sum(1 for line in open(list_file))

	X = []
	labels = []

	pbar = tqdm(total=file_num)

	for line in open(list_file, "r"):
		temp = line.split()
		file_name = temp[0]
		label = temp[1]
		pbar.update(1)
		dir_name = os.path.join(DATA_PATH, file_name)
		labels.append(int(label))
		X.append(load_images(dir_name))

	pbar.close()

	return np.array(X).transpose((0, 2, 3, 4, 1)), labels

def main():
	# load data
	print("loading training data...")
	x_train, y_train = load_data(args.train)
	print("loading test data...")
	x_test, y_test = load_data(args.test)

	X_train = x_train.reshape((x_train.shape[0], IMAGE_SIZE, IMAGE_SIZE, DEPTH_NUM, CHANNEL_NUM))
	X_test = x_test.reshape((x_test.shape[0], IMAGE_SIZE, IMAGE_SIZE, DEPTH_NUM, CHANNEL_NUM))

	Y_train = np_utils.to_categorical(y_train, WORD_NUM)
	Y_test = np_utils.to_categorical(y_test, WORD_NUM)

	X_train = X_train.astype("float32")
	X_test = X_test.astype("float32")
	
	print("X_shape:{}\nY_shape:{}".format(X_train.shape, Y_train.shape))
	print("X_shape:{}\nY_shape:{}".format(X_test.shape, Y_test.shape))

	# define model
	model = Sequential()
	model.add(Conv3D(32, kernel_size=(3, 3, 3), input_shape=(X_train.shape[1:]), padding="same"))
	model.add(Activation("relu"))
	model.add(Conv3D(32, kernel_size=(3, 3, 3), padding="same"))
	model.add(Activation("softmax"))
	model.add(MaxPooling3D(pool_size=(3, 3, 3), padding="same"))
	model.add(Dropout(0.25))
	model.add(Conv3D(64, kernel_size=(3, 3, 3), padding="same"))
	model.add(Activation("relu"))
	model.add(Conv3D(64, kernel_size=(3, 3, 3), padding="same"))
	model.add(Activation("softmax"))
	model.add(MaxPooling3D(pool_size=(3, 3, 3), padding="same"))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(512, activation="sigmoid"))
	model.add(Dropout(0.5))
	model.add(Dense(WORD_NUM, activation="softmax"))

	# optimizer
	opt = optimizers.Adam(lr=LEARNING_RATE)

	#
	model.compile(loss=categorical_crossentropy, optimizer=opt, metrics=["accuracy"])
	model.summary()

	plot_model(model, show_shapes=True, to_file=os.path.join(WORK_DIR, "model.png"))

	# training
	history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCH, validation_data=(X_test, Y_test))

	weight_name = "weight.h5"
	cm_name = "confusion_matrix.csv"
	result_name = "result.txt"

	# learning curve
	plot_history(history, WORK_DIR)
	# result
	save_history(history, WORK_DIR)

	# save weight (h5)
	model.save_weights(os.path.join(WORK_DIR, weight_name))

	# evaluation
	model.evaluate(X_test, Y_test, verbose=0)

	test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)

	print("Test loss: ", test_loss)
	print("Test accuracy: ", test_acc)

	pred_prob = model.predict(X_test)
	pred_class = np.argmax(pred_prob, axis=1)

	Y_test = np.argmax(Y_test, axis=1)

	cr = classification_report(Y_test, pred_class)
	print(cr)

	# confusion matrix
	cmLabel = sorted(list(set(Y_test)))
	cm = confusion_matrix(Y_test, pred_class, labels=cmLabel)
	print(cm)

	np.savetxt(os.path.join(WORK_DIR, cm_name), cm, fmt="%8d", delimiter=",")

	with open(os.path.join(WORK_DIR, result_name), "a") as fp:
		fp.write(cr)
		fp.write("\n")
		fp.write("Test loss: " + str(test_loss) + "\n")
		fp.write("Test accuracy: " + str(test_acc) + "\n")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="3DCNN for SSSD")
	parser.add_argument("--train", type=str, default="sample_training.txt")
	parser.add_argument("--test", type=str, default="sample_test.txt")
	args = parser.parse_args()

	result_name = "result.txt"

	with open(os.path.join(WORK_DIR, result_name), "w") as fp:
		fp.write("depth: " + str(DEPTH_NUM) +"\n")
		fp.write("optimizer: Adam\n")
		fp.write("learning_rate: " + str(LEARNING_RATE) +"\n")
		fp.write("batch_size: " + str(BATCH_SIZE) +"\n")
		fp.write("epoch: " + str(EPOCH) +"\n")
		fp.write("train: " + str(args.train) +"\n")
		fp.write("test: " + str(args.test) +"\n")
		fp.write("work_dir: " + WORK_DIR +"\n")
		fp.write("\n")

	start = time.time()

	main()

	elapsedTime = time.time() - start

	with open(os.path.join(WORK_DIR, result_name), "a") as fp:
		fp.write("\n")
		fp.write("elapsed_time: {0}".format(elapsedTime) + " [sec] = {0}".format(elapsedTime/60) + " [min]\n")

	print ("elapsed_time: {0}".format(elapsedTime) + " [sec] = {0}".format(elapsedTime/60) + " [min]")

