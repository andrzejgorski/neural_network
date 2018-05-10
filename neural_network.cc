#include <random>
#include <sstream>
#include <vector>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <cmath>
#include <iomanip>
#include <string.h>
#include "matrix_mult.h"


using namespace std;


const int INPUT_SIZE = 4096;
const int OUTPUT_SIZE = 62;
const int INPUTS = 4574;

typedef vector< double > matrix_column;
typedef vector< matrix_column > matrix;


matrix create_matrix(int rows, int columns, double default_value){
    return matrix(rows, matrix_column(columns, default_value));
}

matrix create_matrix(int rows, int columns){
    return create_matrix(rows, columns, 0);
}

matrix to_matrix(matrix_column input) {
    matrix result = create_matrix(input.size(), 1);
    for (int i = 0; i < input.size(); i++ ) {
        result[i][0] = input[i];
    }
    return result;
}

void print_m(matrix mat) {
    cout << "----- Printing matrix -----" << endl;
    cout << "Rows: " << mat.size() << endl;
    cout << "Columns: " << mat[0].size() << endl;
    for (int i = 0; i < mat.size(); i++ ) {
        for (int j = 0; j < mat[0].size(); j++ ) {
            cout << mat[i][j] << " ";
        }
        cout << endl;
    }
}


void print_m_t(matrix mat) {
    cout << "----- Printing matrix -----" << endl;
    cout << "Rows: " << mat.size() << endl;
    cout << "Columns: " << mat[0].size() << endl;
    for (int i = 0; i < mat[0].size(); i++ ) {
        for (int j = 0; j < mat.size(); j++ ) {
            cout << mat[j][i] << " ";
        }
        cout << endl;
    }
}


void matrix_copy(double *c_like_matrix, matrix& matrix, int rows, int columns) {
    double (*matrix_ptr)[rows][columns] = (
        double(*)[rows][columns]) c_like_matrix;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            (*matrix_ptr)[i][j] = matrix[i][j];
        }
    }
}


void matrix_copy(matrix& matrix, double *c_like_matrix, int rows, int columns) {
    double (*matrix_ptr)[rows][columns] = (
        double(*)[rows][columns]) c_like_matrix;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            matrix[i][j] = (*matrix_ptr)[i][j];
        }
    }
}


void matrix_mult(matrix& first, matrix& second, matrix& out,
                 int first_r, int second_c, int first_c, bool gpu) {
    double *first_ptr = (double*) malloc(first_r * first_c * sizeof(double));
    double *second_ptr = (double*) malloc(first_c * second_c * sizeof(double));
    double *out_ptr = (double*) malloc(first_r * second_c * sizeof(double));

    matrix_copy(first_ptr, first, first_r, first_c);
    matrix_copy(second_ptr, second, first_c, second_c);

    if (gpu) {
        matrix_mult_gpu(first_ptr, second_ptr, out_ptr, first_r, second_c, first_c);
    } else {
        matrix_mult(first_ptr, second_ptr, out_ptr, first_r, second_c, first_c);
    }

    matrix_copy(out, out_ptr, first_r, second_c);

    free(first_ptr);
    free(second_ptr);
    free(out_ptr);
}


void matrix_mult_tf_to_ss(matrix& first, matrix& second, matrix& out,
                          int first_r, int second_c, int first_c, bool gpu) {
    double *first_ptr = (double*) malloc(first_r * first_c * sizeof(double));
    double *second_ptr = (double*) malloc(first_c * second_c * sizeof(double));
    double *out_ptr = (double*) malloc(first_r * second_c * sizeof(double));

    matrix_copy(first_ptr, first, first_r, first_c);
    matrix_copy(second_ptr, second, first_c, second_c);

    if (gpu) {
        matrix_mult_gpu_tf_to(first_ptr, second_ptr, out_ptr, first_r, second_c, first_c);
    } else {
        matrix_mult_tf_to(first_ptr, second_ptr, out_ptr, first_r, second_c, first_c);
    }
    matrix_copy(out, out_ptr, first_r, second_c);

    free(first_ptr);
    free(second_ptr);
    free(out_ptr);
}


void matrix_mult_ts(matrix& first, matrix& second, matrix& out,
                    int first_r, int second_c, int first_c, bool gpu) {
    double *first_ptr = (double*) malloc(first_r * first_c * sizeof(double));
    double *second_ptr = (double*) malloc(first_c * second_c * sizeof(double));
    double *out_ptr = (double*) malloc(first_r * second_c * sizeof(double));

    matrix_copy(first_ptr, first, first_r, first_c);
    matrix_copy(second_ptr, second, first_c, second_c);
    if (gpu) {
        matrix_mult_gpu_ts(first_ptr, second_ptr, out_ptr, first_r, second_c, first_c);
    } else {
        matrix_mult_ts(first_ptr, second_ptr, out_ptr, first_r, second_c, first_c);
    }
    matrix_copy(out, out_ptr, first_r, second_c);

    free(first_ptr);
    free(second_ptr);
    free(out_ptr);
}


bool matrix_same(matrix first, matrix second) {
    for (int i = 0; i < first.size(); i++) {
        for (int j = 0; j < first[0].size(); j++) {
            if (first[i][j] != second[i][j]) {
                printf("values: at index %d %d are different %f %f\n",
                    i, j, first[i][j], second[i][j]);
                return false;
            }
        }
    }
    return true;
}


double fRand(double fMin, double fMax) {
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}


class Layer {
    public:
    bool GPU;
    int input_size, output_size;
    double learning_rate;
    matrix mid_value, mid_error;
    matrix gradients;

    matrix output_value, output_error;
    matrix weights;
    Layer (int i_size, int o_size, double rate):
            input_size{i_size}, output_size{o_size}, learning_rate{rate} {
        mid_value = create_matrix(output_size, 1);
        mid_error = create_matrix(output_size, 1);
        output_value = create_matrix(output_size + 1, 1, 1);
        output_error = create_matrix(output_size, 1);
        weights = create_matrix(output_size, input_size + 1, 1);
        gradients = create_matrix(output_size, input_size + 1);
    }
    Layer (int i_size, int o_size, double rate, bool gpu):
        Layer(i_size, o_size, rate)
    {
        GPU = gpu;
    }
    Layer () {};
    virtual matrix calc() {};
    virtual void print();
    void set_correct_output(matrix outputs) {
        for (int i = 0; i < output_size; i++) {
            double tmp = outputs[i][0] - output_value[i][0];
            output_error[i][0] = tmp * tmp;
        }
    };
    void set_random_weights() {
        for (int i = 0; i < weights.size(); i++) {
            for (int j = 0; j < weights[i].size(); j++) {
                weights[i][j] = fRand(0, 1);
            }
        }
    }
};


void Layer::print() {
    cout << "mid_value: " << endl;
    for (int i = 0; i < output_size; i ++) {
        cout << mid_value[i][0] << " ";
    }
    cout << endl;
    cout << "mid_error: " << endl;
    for (int i = 0; i < output_size; i ++) {
        cout << mid_error[i][0] << " ";
    }
    cout << endl;
    cout << "out_value: " << endl;
    for (int i = 0; i < output_size; i ++) {
        cout << output_value[i][0] << " ";
    }
    cout << endl;
    cout << "out_error: " << endl;
    for (int i = 0; i < output_size; i ++) {
        cout << output_error[i][0] << " ";
    }
    cout << endl;
    cout << "output_size: " << output_size << " ";
    cout << "input_size: " << input_size << endl;
    cout << "weights: " << endl;
    for (int i = 0; i < output_size; i ++) {
        for (int j = 0; j < input_size + 1; j ++) {
            cout << weights[i][j] << " ";
        }
        cout << endl;
    }
    cout << "gradients: " << endl;
    for (int i = 0; i < output_size; i ++) {
        for (int j = 0; j < input_size + 1; j ++) {
            cout << gradients[i][j] << " ";
        }
        cout << endl;
    }
}


class FirstLayer:public Layer {
    public:
    matrix input_value;
    FirstLayer (int, int, double, bool);
    void print ();
    matrix calc ();
    void init_input(matrix input) {
        for (int i = 0; i < input_size; i++) {
            input_value[i][0] = input[i][0];
        }
    }
    void backpropagation ();
};


matrix FirstLayer::calc() {
    matrix_mult(
        weights,
        input_value,
        mid_value,
        output_size,
        1,
        input_size + 1,
        GPU
    );

    for (int i = 0; i < output_size; i++) {
        if (mid_value[i][0] > 0) {
            output_value[i][0] = mid_value[i][0];
        } else {
            output_value[i][0] = 0;
        }
    }
    return output_value;
}


void FirstLayer::backpropagation() {
    for (int i = 0; i < output_size; i++) {
        if (mid_value[i][0] >= 0) {
            mid_error[i][0] = output_error[i][0];
        } else {
            mid_error[i][0] = 0;
        }
    }
    matrix_mult_ts(
        mid_error,
        input_value,
        gradients,
        output_size,
        input_size + 1,
        1,
        GPU
    );
    for (int i = 0; i < output_size; i ++) {
        for (int j = 0; j < input_size + 1; j ++) {
            weights[i][j] -= learning_rate * gradients[i][j];
        }
    }
}


FirstLayer::FirstLayer (int i_size, int o_size, double rate, bool gpu):
        Layer (i_size, o_size, rate, gpu) {
    input_value = create_matrix(input_size + 1, 1, 1);
}


void FirstLayer::print() {
    cout << "input_value: " << endl;
    for (int i = 0; i < input_size + 1; i ++) {
        cout << input_value[i][0] << " ";
    }
    cout << endl;
    Layer::print();
}


// MidLayer


class MidLayer:public Layer {
    public:
    Layer *previous;
    MidLayer (int i_size, int o_size, double rate, Layer &prev, bool gpu):
        Layer(i_size, o_size, rate, gpu), previous{&prev} {};
    MidLayer () {};
    matrix calc();
    void backpropagation();
};


matrix MidLayer::calc() {
    matrix_mult(
        weights,
        previous->output_value,
        mid_value,
        output_size,
        1,
        input_size + 1,
        GPU
    );

    for (int i = 0; i < output_size; i++) {
        if (mid_value[i][0] > 0) {
            output_value[i][0] = mid_value[i][0];
        } else {
            output_value[i][0] = 0;
        }
    }
    return output_value;
}


void MidLayer::backpropagation() {
    for (int i = 0; i < output_size; i++) {
        if (mid_value[i][0] >= 0) {
            mid_error[i][0] = output_error[i][0];
        } else {
            mid_error[i][0] = 0;
        }
    }
    matrix_mult_ts(
        mid_error,
        previous->output_value,
        gradients,
        output_size,
        input_size + 1,
        1,
        GPU
    );

    matrix_mult_tf_to_ss(
        mid_error,
        weights,
        previous->output_error,
        1,
        input_size,
        output_size,
        GPU
    );

    for (int i = 0; i < output_size; i ++) {
        for (int j = 0; j < input_size + 1; j ++) {
            weights[i][j] -= learning_rate * gradients[i][j];
        }
    }
}


class LastLayer:public Layer {
    public:
    Layer *previous;
    LastLayer (int i_size, int o_size, double rate, Layer &prev, bool gpu):
        Layer(i_size, o_size, rate, gpu), previous{&prev}
    {
        output_value = create_matrix(output_size, 1, 1);
    };
    matrix calc();
    void backpropagation();
};


matrix LastLayer::calc() {
    matrix_mult(
        weights,
        previous->output_value,
        mid_value,
        output_size,
        1,
        input_size + 1,
        GPU
    );

    // Softmax stable
    double max = mid_value[0][0];
    double sum = 0;
    matrix exponents = create_matrix(output_size, 1, 0);
    for (int i = 0; i < output_size; i++) {
        if (mid_value[i][0] > max) {
            max = mid_value[i][0];
        }
    }
    for (int i = 0; i < output_size; i++) {
        exponents[i][0] = exp(mid_value[i][0] - max);
        sum += exponents[i][0];
    }

    for (int i = 0; i < output_size; i++) {
        output_value[i][0] = (exponents[i][0]) / sum;
    }
    return output_value;
}


void LastLayer::backpropagation() {
    // Put here softmax derivatives
    matrix derivatives = create_matrix(output_size, output_size, 0);
    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < output_size; j++) {
            if (i != j) {
                derivatives[j][i] = - mid_value[i][0] * mid_value[j][0];
            } else {
                derivatives[i][i] = mid_value[i][0] * (1 - mid_value[i][0]);
            }
        }
    }

    matrix_mult(
        derivatives,
        output_error,
        mid_error,
        output_size,
        1,
        output_size,
        GPU
    );

    matrix_mult_ts(
        mid_error,
        previous->output_value,
        gradients,
        output_size,
        input_size + 1,
        1,
        GPU
    );

    matrix_mult_tf_to_ss(
        mid_error,
        weights,
        previous->output_error,
        1,
        input_size,
        output_size,
        GPU
    );

    for (int i = 0; i < output_size; i ++) {
        for (int j = 0; j < input_size + 1; j ++) {
            weights[i][j] -= learning_rate * gradients[i][j];
        }
    }

}


class NeuralNetwork {
    public:
    bool GPU;
    FirstLayer *first;
    vector< MidLayer > mids;
    LastLayer *last;

    matrix calc(matrix);
    void set_random_weights();
    matrix backpropagation(matrix, matrix);
    NeuralNetwork(vector <int>, double, bool);
    ~NeuralNetwork(){
        if (first != NULL) {
            free(first);
        }
        if (last != NULL) {
            free(last);
        }
    }
    void print();
};


NeuralNetwork::NeuralNetwork (vector <int> layers, double learning_rate, bool gpu)
{
    GPU = gpu;
    first = new FirstLayer(layers[0], layers[1], learning_rate, gpu);
    mids = vector< MidLayer > (layers.size() - 3);
    mids[0] = MidLayer(layers[1], layers[2], learning_rate, *first, gpu);
    for (int i = 1; i < mids.size(); i ++) {
        mids[i] = MidLayer(
            layers[i + 1], layers[i + 2], learning_rate, mids[i - 1], gpu);
    }
    last = new LastLayer(
            layers[layers.size() - 2],
            layers[layers.size() - 1],
            learning_rate,
            mids[mids.size() - 1],
            gpu
        );
}


// DEBUG FUNCTION
void NeuralNetwork::print () {
    cout << "======== First Layer ========" << endl;
    first->print();
    for (int i = 0; i < mids.size(); i ++) {
        cout << "======== Mid Layer no " << i + 1 << " ======== " << endl;
        mids[i].print();
    }
    cout << "======== Last Layer ========" << endl;
    last->print();
}


matrix NeuralNetwork::calc(matrix input) {
    first->init_input(input);
    first->calc();
    for (int i = 0; i < mids.size(); i ++) {
        mids[i].calc();
    }
    return last->calc();
}


matrix NeuralNetwork::backpropagation(matrix input, matrix proper_output) {
    matrix result = calc(input);
    last->set_correct_output(proper_output);
    last->backpropagation();
    for (int i = mids.size() - 1; i >= 0; i --) {
        mids[i].backpropagation();
    }
    first->backpropagation();
    return result;
}


void NeuralNetwork::set_random_weights() {
    first->set_random_weights();
    for (auto &layer: mids) {
        layer.set_random_weights();
    }
    last->set_random_weights();
}


struct neural_network_file {
    matrix input;
    matrix output;
};


neural_network_file load_input_file(ifstream& file_stream) {
    neural_network_file file;
    double dd;
    file.input = create_matrix(INPUT_SIZE, 1, 0);
    file.output = create_matrix(OUTPUT_SIZE, 1, 0);
    for (int i = 0; i < INPUT_SIZE; i ++){
        file_stream >> file.input[i][0];
    }
    for (int i = 0; i < OUTPUT_SIZE; i ++){
        file_stream >> file.output[i][0];
    }
    return file;
}


neural_network_file load_input_file(string filename) {
    ifstream file_stream;
    file_stream.open(filename);
    neural_network_file file = load_input_file(file_stream);
    file_stream.close();
    return file;
}


vector< neural_network_file > load_inputs() {
    vector< neural_network_file > inputs;
    stringstream ss;
    string str;
    for (int i = 0; i < INPUTS; i ++){
        ss << std::setw(4) << std::setfill('0') << i;
        str = ss.str();
        inputs.push_back(load_input_file("inputs/" + str + ".in"));
    }
    return inputs;
}


int main(int argc, char **argv)
{
    srand(123);
    if (argc < 3) {
        fprintf(stderr, "Error: Program require more arguments\n");
        return -1;
    }
    double epsilon = stof(argv[1]);
    double learning_rate = stof(argv[2]);

    int max_epochs = stoi(argv[3]);
    bool random_weights;
    if (argv[4] == "false" || argv[4] == "False")
    {
        random_weights = false;
    }
    else
    {
        random_weights = true;
    }

    vector< neural_network_file > inputs = load_inputs();
    NeuralNetwork neural_network = NeuralNetwork(
         {4096, 8192, 6144, 3072, 1024, 62}, learning_rate, false);
         // {10, 10, 10, 3}, learning_rate, false);

    NeuralNetwork gpu_neural_network = NeuralNetwork(
         {4096, 8192, 6144, 3072, 1024, 62}, learning_rate, true);
         // {10, 10, 10, 3}, learning_rate, true);

    if (random_weights) {
        neural_network.set_random_weights();
        gpu_neural_network.set_random_weights();
    }

    for (int epoch = 0; epoch < max_epochs; epoch++){
        int counter = 0;
        for (int i = 0; i < INPUTS; i++){
            matrix result = neural_network.backpropagation(
                inputs[i].input, inputs[i].output);

            matrix gpu_result = gpu_neural_network.backpropagation(
                inputs[i].input, inputs[i].output);

            if (result[0][0] != result[0][0] or gpu_result[0][0] != gpu_result[0][0]) {
                break;
            }

            if (!matrix_same(result, gpu_result)) {
                puts("matrix_different");
                break;
            }

            if (matrix_same(result, inputs[i].output)) {
                counter++;
            }
        }
        if ((float)counter / INPUTS > epsilon) {
            break;
        }
    }
    // neural_network.print();
    return 0;
}
