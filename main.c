#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

#define SIZE_L1 2
#define SIZE_L2 2
#define SIZE_OF_TEST 4
#define SIZE_OF_TRAIN 4
#define L1_WEIGHTS 4
#define L2_WEIGHTS 2

//TODO make this a struct or move to c++ and make it a class
double train_input[8] = {0, 0, 0, 1, 1, 0, 1, 1}; //input is first layer of training
double train_answr_key[4] = {0, 1, 1, 0};//storage for train solution
double test_input[8] = {0, 0, 0, 1, 1, 0, 1, 1}; //input is first layer of test
double test_answr_key[4] = {0, 1, 1, 0}; //storage for test image solution
double L2[2]; //space for second layer results
double L3[1]; //space for third layer results
double sigL2[2]; //space for second layer results
double sigL3[1]; //space for third layer results
double L1_to_L2_weights[4]; //input weights
double L2_to_L3_weights[2]; //weights from first hidden layer (second overall) to next hidden layer
double *input_ptr = NULL; //pointer to set in input or test array
double learning_rate = 0.8; //learning rate

//vars used temporarily during backpropagation
double L3_der_err_der_y[1]; //output value derivatives
double L3_der_err_der_x[1]; //derivatives of the input to the final layer
double L2_der_err_der_w[2]; //derivatives of the weights from L2 to L3
double L2_suggested_weight_changes[2];
double L2_der_err_der_y[2];
double L2_der_err_der_x[2];
double L1_der_err_der_w[4]; //derivatives of the weights from L1 to L2
double L1_suggested_weight_changes[4];

double rand_doubles(double min, double max);
void fill_hyperparams_with_rand(void);
void L1_weight_updater(void);
void L2_weight_updater(void);
void feed_forward(void);
double sigmoid(double x);
void backprop(void);
void test(void);
void reset_nn(void);

void test(void)
{
    input_ptr = test_input;

    for (int i = 0; i < SIZE_OF_TEST; i++)
    {
        feed_forward();
        printf("%lf %lf\n", sigL3[0], test_answr_key[i]);
        reset_nn();
        input_ptr += 2;
    }
}

void reset_nn(void) //reset_neurons
{
    for(int i = 0; i < L1_WEIGHTS; i++)
    {
        if(i < L2_WEIGHTS)
            L2[i] = 0;
    }
    L3[0] = 0;
}

void backprop(void) //aka chain rule time boiz
{
    int i = 0;
    int j = 0;

    for (int epochs = 0; epochs < 1000000; epochs++)
    {
        input_ptr = train_input;

        for (int train_number = 0; train_number < SIZE_OF_TRAIN; train_number++)
        {
            feed_forward(); //run through the network

            //find derivatives of final output, (Y_output - Y_expected)
            L3_der_err_der_y[0] = sigL3[0] - train_answr_key[train_number];

            //find derivatives of the input to the final layer
            L3_der_err_der_x[0] = (sigL3[0] * (1 - sigL3[0])) *
                                  L3_der_err_der_y[0]; //sigL3[0]*(1-sigL3[0])) is derivative of the sigmoid


            //derivative of weights from L2 to L3
            for (i = 0; i < SIZE_L2; i++)
                L2_der_err_der_w[i] = sigL2[i] * L3_der_err_der_x[0];

            L2_weight_updater();

            //derivative of the output of second layer
            for (i = 0; i < SIZE_L2; i++)
                L2_der_err_der_y[i] = L2_to_L3_weights[i] * L3_der_err_der_x[0];

            //derivative of the input to the third layer
            for (i = 0; i < SIZE_L2; i++)
                L2_der_err_der_x[i] = (sigL2[i] * (1 - sigL2[i])) * L2_der_err_der_y[i];


            //derivative of the weights from L1 to L2
            for (i = 0; i < SIZE_L1; i++)
                for (j = 0; j < SIZE_L2; j++)
                    L1_der_err_der_w[(i * 2) + j] = input_ptr[i] * L2_der_err_der_x[j];

            L1_weight_updater();

            input_ptr += 2; //move the pointer to the next set

            for (i = 0; i < L1_WEIGHTS; i++)
            {

                L1_der_err_der_w[i] = 0;

                if (i < L2_WEIGHTS)
                {
                    L2_der_err_der_y[i] = 0;
                    L2_der_err_der_x[i] = 0;
                    L2_der_err_der_w[i] = 0;
                }
            }
            L3_der_err_der_y[0] = 0;
            L3_der_err_der_x[0] = 0;

            reset_nn();
        }

        for (i = 0; i < L1_WEIGHTS; i++)
        {
            L1_to_L2_weights[i] += L1_suggested_weight_changes[i];
            L1_suggested_weight_changes[i] = 0;

            if (i < L2_WEIGHTS)
            {
                L2_to_L3_weights[i] += L2_suggested_weight_changes[i];
                L2_suggested_weight_changes[i] = 0;
            }
        }
    }
}


void L1_weight_updater(void)
{
    for (int i = 0; i < L1_WEIGHTS; i++)
        L1_suggested_weight_changes[i] += -1 * learning_rate * L1_der_err_der_w[i];
}

void L2_weight_updater(void)
{
    for (int i = 0; i < L2_WEIGHTS; i++)
        L2_suggested_weight_changes[i] += -1 * learning_rate * L2_der_err_der_w[i];
}

double sigmoid(double x)
{
    return 1 / (1 + exp(-1.0 * x));
}

void feed_forward(void) //matrix multiplication
{
    int i = 0;
    int j = 0;

    for (i = 0; i < SIZE_L1; i++)
    {
        for (j = 0; j < SIZE_L2; j++)
            L2[i] += L1_to_L2_weights[(j * 2) + i] * input_ptr[j];

        sigL2[i] = sigmoid(L2[i]);
    }

    for (i = 0; i < SIZE_L2; i++)
        L3[0] += L2_to_L3_weights[i] * sigL2[i];

    sigL3[0] = sigmoid(L3[0]);
}

double rand_doubles(const double min, const double max)
{
    if (min < max)
        return (max - min) * ((double) rand() / RAND_MAX) + min;

    // return 0 if min > max or min = max, which shouldn't ever happen as we control the args
    return 0;
}

void fill_hyperparams_with_rand(void) //this works so DON'T TOUCH
{
    for (int i = 0; i < L1_WEIGHTS; i++)
        L1_to_L2_weights[i] = rand_doubles(-0.5, 0.5);

    for (int i = 0; i < L2_WEIGHTS; i++)
        L2_to_L3_weights[i] = rand_doubles(-0.5, 0.5);
}

int main(void)
{
    //randomizer
    srand((unsigned) 0); //seed with 0 for consistency
    fill_hyperparams_with_rand();

    backprop();

    test();

    putchar('\n');
    for(int i = 0; i < L1_WEIGHTS; i++)
    {
        printf("%lf\n", L1_to_L2_weights[i]);
    }
    putchar('\n');
    for(int i = 0; i < L2_WEIGHTS; i++)
    {
        printf("%lf\n", L2_to_L3_weights[i]);
    }

    return 0;
}
