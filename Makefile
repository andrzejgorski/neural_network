CC=g++
CFLAGS=--std=c++11

nn_cc: neural_network.cc
	$(CC) -o neural_network neural_network.cc $(CFLAGS)
