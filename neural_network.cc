#include <vector>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <cmath>
using namespace std;


typedef vector< double > matrix_column;
typedef vector< matrix_column > matrix;

matrix create_matrix(int rows, int columns, double default_value){
    return matrix(rows, matrix_column(columns, default_value));
}

matrix create_matrix(int rows, int columns){
    return create_matrix(rows, columns, 0);
}


void matrix_mult(matrix& first, matrix& second, matrix& out,
                 int first_r, int second_c, int first_c) {
    for (int b = 0; b < first_r; b++) {
        for (int c = 0; c < second_c; c++) {
            out[b][c] = 0;
            for (int a = 0; a < first_c; a++) {
                out[b][c] = first[b][a] * second[a][c];
            }
        }
    }
}


void matrix_mult_ts(matrix& first, matrix& second, matrix& out,
                    int first_r, int second_c, int first_c) {
    for (int b = 0; b < first_r; b++) {
        for (int c = 0; c < second_c; c++) {
            out[b][c] = 0;
            for (int a = 0; a < first_c; a++) {
                out[b][c] = first[b][a] * second[c][a];
            }
        }
    }
}


class Layer {
    protected:
    int input_size, output_size;
    double alpha;
    matrix mid_value, mid_error;
    matrix gradients;

    virtual void activation_function() {};

    public:
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
    virtual void calc() {};
    virtual void print();
    void set_correct_output(matrix outputs) {
        for (int i = 0; i < output_size; i++) {
            output_error[i][0] = abs(outputs[i][0] - output_value[i][0]);
        }
    };
};


void Layer::print() {
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
}


class FirstLayer:public Layer {
    protected:
    matrix input_value;
    public:
    FirstLayer (int, int);
    void print ();
    void calc ();
    void init_input(matrix input) {
        input_value = input;
    }
    void backpropagation ();
};


void FirstLayer::calc() {
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
}


void FirstLayer::backpropagation() {
    for (int i = 0; i < output_size; i++) {
        if (mid_value[i][0] >= 0) {
            mid_error[i][0] = mid_value[i][0] * output_error[i][0];
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
    Layer::print();
    cout << "input_value: " << endl;
    for (int i = 0; i < input_size + 1; i ++) {
        cout << input_value[i][0] << " ";
    }
    cout << endl;
}


// MidLayer


class MidLayer:public Layer {
    protected:
    Layer &previous;
    public:
    MidLayer (int i_size, int o_size, Layer &prev): Layer(i_size, o_size), previous{prev} {};
    void calc();
    void backpropagation();
};


void MidLayer::calc() {
    matrix_mult(
        weights,
        previous.output_value,
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
}


void MidLayer::backpropagation() {
    for (int i = 0; i < output_size; i++) {
        if (mid_value[i][0] >= 0) {
            mid_error[i][0] = mid_value[i][0] * output_error[i][0];
        } else {
            mid_error[i][0] = 0;
        }
    }
    matrix_mult_ts(
        mid_error,
        previous.output_value,
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


// Loading function


void read_weigts(ifstream& file_stream, Layer& layer, int rows, int columns) {
    for (int i = 0; i < rows; i ++) {
        for (int j = 0; j < columns + 1; j ++) {
            file_stream >> layer.weights[i][j];
        }
    }
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


int main()
{
    ifstream in_stream;
    in_stream.open("test.in");
    FirstLayer first_layer = load(in_stream);
    // first_layer.init_input(create_matrix(4, 1, -1));
    first_layer.calc();
    first_layer.print();
    first_layer.set_correct_output(create_matrix(3, 1, 1));
    first_layer.backpropagation();

    ifstream mid_stream;
    mid_stream.open("test_mid.in");
    MidLayer mid_layer = load(mid_stream, first_layer);
    mid_layer.calc();
    mid_layer.backpropagation();
    mid_layer.print();
    in_stream.close();
    mid_stream.close();
    return 0;
}
