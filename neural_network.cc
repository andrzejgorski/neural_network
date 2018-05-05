#include <random>
#include <sstream>
#include <vector>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <cmath>
#include <iomanip>
#include <string.h>


using namespace std;


const int INPUT_SIZE = 4096;
const int OUTPUT_SIZE = 62;
// const int INPUTS = 4574;
const int INPUTS = 1;


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


void matrix_mult(matrix& first, matrix& second, matrix& out,
                 int first_r, int second_c, int first_c) {
    for (int b = 0; b < first_r; b++) {
        for (int c = 0; c < second_c; c++) {
            out[b][c] = 0;
            for (int a = 0; a < first_c; a++) {
                out[b][c] += first[b][a] * second[a][c];
            }
        }
    }
}


void matrix_mult_tf_to_ss(matrix& first, matrix& second, matrix& out,
                          int first_r, int second_c, int first_c) {
    for (int b = 0; b < first_r; b++) {
        for (int c = 0; c < second_c; c++) {
            out[c][b] = 0;
            for (int a = 0; a < first_c; a++) {
                out[c][b] += first[a][b] * second[a][c];
            }
        }
    }
}


double fRand(double fMin, double fMax) {
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}


void matrix_mult_ts(matrix& first, matrix& second, matrix& out,
                    int first_r, int second_c, int first_c) {
    for (int b = 0; b < first_r; b++) {
        for (int c = 0; c < second_c; c++) {
            out[b][c] = 0;
            for (int a = 0; a < first_c; a++) {
                out[b][c] += first[b][a] * second[c][a];
            }
        }
    }
}


class Layer {
    public:
    int input_size, output_size;
    double alpha;
    matrix mid_value, mid_error;
    matrix gradients;

    matrix output_value, output_error;
    matrix weights;
    Layer (int i_size, int o_size): input_size{i_size}, output_size{o_size} {
        mid_value = create_matrix(output_size, 1);
        mid_error = create_matrix(output_size, 1);
        output_value = create_matrix(output_size + 1, 1, 1);
        output_error = create_matrix(output_size, 1);
        weights = create_matrix(output_size, input_size + 1);
        gradients = create_matrix(output_size, input_size + 1);
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
    FirstLayer (int, int);
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
        input_size + 1
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
        1
    );
    for (int i = 0; i < output_size; i ++) {
        for (int j = 0; j < input_size + 1; j ++) {
            weights[i][j] -= 0.1 * gradients[i][j];
        }
    }
}


FirstLayer::FirstLayer (int i_size, int o_size):
        Layer (i_size, o_size) {
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
    MidLayer (int i_size, int o_size, Layer &prev): Layer(i_size, o_size), previous{&prev} {};
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
        input_size + 1
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
        1
    );

    matrix_mult_tf_to_ss(
        mid_error,
        weights,
        previous->output_error,
        1,
        input_size,
        output_size
    );

    for (int i = 0; i < output_size; i ++) {
        for (int j = 0; j < input_size + 1; j ++) {
            weights[i][j] -= 0.1 * gradients[i][j];
        }
    }
}


class LastLayer:public Layer {
    public:
    Layer *previous;
    LastLayer (int i_size, int o_size, Layer &prev):
        Layer(i_size, o_size), previous{&prev} {};
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
        input_size + 1
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
        mid_value[i][0] = mid_value[i][0] - max;
        sum += exp(mid_value[i][0]);
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
        output_size
    );

    matrix_mult_ts(
        mid_error,
        previous->output_value,
        gradients,
        output_size,
        input_size + 1,
        1
    );

    matrix_mult_tf_to_ss(
        mid_error,
        weights,
        previous->output_error,
        1,
        input_size,
        output_size
    );

    for (int i = 0; i < output_size; i ++) {
        for (int j = 0; j < input_size + 1; j ++) {
            weights[i][j] -= 0.1 * gradients[i][j];
        }
    }

}


class NeuralNetwork {
    public:
    FirstLayer first;
    vector< MidLayer > mids;
    vector< LastLayer > _last;

    matrix calc(matrix);
    void set_random_weights();
    void backpropagation(matrix, matrix);
    NeuralNetwork(vector <int>);
    void print();
};


NeuralNetwork::NeuralNetwork (vector <int> layers):
    first{FirstLayer(layers[0], layers[1])} {
    // last{_last[0]} {
    mids = vector< MidLayer > (layers.size() - 3);
    mids[0] = MidLayer(layers[1], layers[2], first);
    for (int i = 1; i < mids.size(); i ++) {
        mids[i] = MidLayer(layers[i + 1], layers[i + 2], mids[i - 1]);
    }
    _last.push_back(
        LastLayer(
            layers[layers.size() - 2],
            layers[layers.size() - 1],
            mids[mids.size() - 1]
        )
    );
}


void NeuralNetwork::print () {
    cout << "======== First Layer ========" << endl;
    first.print();
    for (int i = 0; i < mids.size(); i ++) {
        cout << "======== Mid Layer no " << i + 1 << " ======== " << endl;
        mids[i].print();
    }
    cout << "======== Last Layer ========" << endl;
    _last[0].print();
}


matrix NeuralNetwork::calc(matrix input) {
    first.init_input(input);
    first.calc();
    for (int i = 0; i < mids.size(); i ++) {
        mids[i].calc();
    }
    return _last[0].calc();
}


void NeuralNetwork::backpropagation(matrix input, matrix proper_output) {
    matrix result = calc(input);
    _last[0].set_correct_output(proper_output);
    _last[0].backpropagation();
    for (int i = mids.size() - 1; i >= 0; i --) {
        mids[i].backpropagation();
    }
    first.backpropagation();
}


void NeuralNetwork::set_random_weights() {
    first.set_random_weights();
    for (auto &layer: mids) {
        layer.set_random_weights();
    }
    _last[0].set_random_weights();
}


// Loading function


void read_weigts (ifstream& file_stream, Layer& layer, int rows, int columns) {
    for (int i = 0; i < rows; i ++) {
        for (int j = 0; j < columns; j ++) {
            file_stream >> layer.weights[i][j];
        }
    }
}


LastLayer load_last (ifstream& file_stream, Layer &prev) {
    int rows, columns;
    file_stream >> rows;
    file_stream >> columns;
    LastLayer new_layer = LastLayer(columns, rows, prev);
    read_weigts(file_stream, new_layer, rows, columns);
    return new_layer;
}


MidLayer load (ifstream& file_stream, Layer &prev) {
    int rows, columns;
    file_stream >> rows;
    file_stream >> columns;
    MidLayer new_layer = MidLayer(columns, rows, prev);
    read_weigts(file_stream, new_layer, rows, columns);
    return new_layer;
}


FirstLayer load (ifstream& file_stream) {
    int rows, columns;
    file_stream >> rows;
    file_stream >> columns;
    FirstLayer new_layer = FirstLayer(columns, rows);
    read_weigts(file_stream, new_layer, rows, columns);
    return new_layer;
}


void read_layer_weights (ifstream& file_stream, Layer &layer) {
    int rows, columns;
    file_stream >> rows;
    file_stream >> columns;
    read_weigts(file_stream, layer, rows, columns);
}


NeuralNetwork load_neural_network(ifstream& file_stream) {
    int layers;
    file_stream >> layers;
    vector< int > layers_shape = vector< int > (layers, 0);
    for (int i = 0; i < layers; i++) {
        file_stream >> layers_shape[i];
    }
    NeuralNetwork neural_network = NeuralNetwork(layers_shape);
    read_layer_weights(file_stream, neural_network.first);
    for (int i = 0; i < neural_network.mids.size(); i++) {
        read_layer_weights(file_stream, neural_network.mids[i]);
    }
    read_layer_weights(file_stream, neural_network._last[0]);
    return neural_network;
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


int main()
{
    srand(123);
    vector< neural_network_file > inputs = load_inputs();
    NeuralNetwork neural_network = NeuralNetwork(
         {4096, 8192, 6144, 3072, 1024, 62});
        // {4096, 16, 16, 8, 8, 62});
    neural_network.set_random_weights();

    for (int i = 0; i < 1; i ++){
        // neural_network.calc(inputs[0].input);
        neural_network.backpropagation(inputs[0].input, inputs[0].output);
    }
    print_m_t(neural_network._last[0].mid_value);
    // neural_network.print();
    return 0;
}
