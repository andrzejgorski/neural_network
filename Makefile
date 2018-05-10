CC=nvcc
CFLAGS=--std=c++11
objects = matrix_mult.o neural_network.o

# all: $(objects)
# 	$(CC) -arch=sm_20 $(objects) -o neural_network $(CFLAGS)

nn_cc: neural_network.cc
	$(CC) -o neural_network matrix_mult.o neural_network.cc $(CFLAGS)

%.o: %.cu
	$(CC) -x cu -arch=sm_20 -I. -dc $< -o $@ $(CFLAGS)

clean:
	rm -f *.o neural_network
