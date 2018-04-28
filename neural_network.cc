#include <vector>
#include <iostream>
#include <fstream>
using namespace std;


typedef vector< double > matrix_column;
typedef vector< matrix_column > matrix;

matrix create_matrix(int rows, int columns, double default_value){
    return matrix(columns, matrix_column(rows, default_value));
}

matrix create_matrix(int rows, int columns){
    return create_matrix(rows, columns, 0);
}

matrix* new_matrix(int rows, int columns, double default_value){
    return new matrix(columns, matrix_column(rows, default_value));
}

matrix* new_matrix(int rows, int columns){
    return new_matrix(rows, columns, 0);
}


void matrix_mult(matrix& first, matrix& second, matrix& out, int first_r,
                 int second_c, int first_c) {
                // t_first=False, t_second=False, t_out=False,
                // shift_second=False):
    for (int b = 0; b < first_r; b++) {
        for (int c = 0; c < first_r; c++) {
            out[b][c] = 0;
            for (int a = 0; a < first_r; a++) {
                out[b][c] = first[b][a] * second[a][b];
            }
        }
    }
}


class Layer {
    protected:
    int input_size, output_size;
    double alpha;
    Layer *previous;
    matrix *input_value, *input_error;
    matrix mid_value, mid_error, output_value, output_error;
    matrix gradients;

    void init ();
    virtual void activation_function() {};

    public:
    matrix weights;
    Layer (int i_size, int o_size): Layer(i_size, o_size, 0.1) {};
    Layer (int i_size, int o_size, Layer& prev):
        Layer(i_size, o_size, 0.1, prev) {};
    Layer (int, int, double);
    Layer (int, int, double, Layer&);
    virtual ~Layer ();
    virtual void calc ();
};


Layer::Layer (int i_size, int o_size, double alp):
        input_size{i_size}, output_size{o_size}, alpha{alp} {
    previous = NULL;
    input_value = new_matrix(input_size + 1, 1, 1);
    input_error = new_matrix(input_size + 1, 1, 1);
    init();
}


Layer::Layer (int i_size, int o_size, double alp, Layer& prev):
        input_size{i_size}, output_size{o_size}, alpha{alp}, previous{&prev} {
    input_value = &(previous->output_value);
    input_error = &(previous->output_error);
    init();
}


void Layer::calc() {
    matrix_mult(
        weights,
        *input_value,
        mid_value,
        output_size,
        1,
        input_size + 1
    );
}


Layer::~Layer () {
    if (previous == NULL) {
        free(input_value);
        free(input_error);
    }
}


void Layer::init () {
    mid_value = create_matrix(output_size, 1);
    mid_error = create_matrix(output_size, 1);
    output_value = create_matrix(output_size + 1, 1, 1);
    output_error = create_matrix(output_size, 1);
    weights = create_matrix(output_size, input_size + 1);
    gradients = create_matrix(output_size, input_size + 1);
}


void read_weigts(ifstream& file_stream, Layer& layer, int rows, int columns) {
    for (int i = 0; i < rows; i ++) {
        for (int j = 0; j < columns; j ++) {
            file_stream >> layer.weights[i][j];
        }
    }
}


class ReLULayer:public Layer {
    public:
    void activation_function() {
        for (int i = 0; i < output_size; i ++) {
            double tmp = mid_value[0][i];
            if (tmp < 0) {
                tmp = 0;
            }
            output_value[0][i] = tmp;
        }
    };
    ReLULayer(int i_size, int o_size): Layer(i_size, o_size) {};
    ReLULayer(int i_size, int o_size, Layer& prev): Layer(i_size, o_size, prev) {};
};


Layer load(ifstream& file_stream, Layer& prev_layer) {
    int rows, columns;
    file_stream >> rows;
    file_stream >> columns;
    Layer new_layer = ReLULayer(rows, columns, prev_layer);
    read_weigts(file_stream, new_layer, rows, columns);
    return new_layer;
}


Layer load (ifstream& file_stream) {
    int rows, columns;
    file_stream >> rows;
    file_stream >> columns;
    Layer new_layer = ReLULayer(rows, columns);
    read_weigts(file_stream, new_layer, rows, columns);
    return new_layer;
}


int main()
{
    ifstream file_stream;
    file_stream.open("test.in");
    Layer ll = load(file_stream);
    file_stream.close();
    return 0;
}
