CC=g++
CFLAGS=-I

nn_cc: neural_network.cc
	$(CC) -o neural_network neural_network.cc
