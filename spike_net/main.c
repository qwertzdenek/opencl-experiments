// main.c
// Zdeněk Janeček 2015

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <signal.h>

#include "target_opencl.c"

#define BLOCK_SIZE 100

float *net = NULL;   // adjacency matrix

void print_net(float *net, int size)
{
    int i, j;

    for (i = 0; i < size; i++)
    {
        for (j = 0; j < size; j++)
        {
            printf("%d ", (int) net[i * size + j]);
        }
        printf("\n");
    }
}

void print_vec(float *array, int size)
{
    int i;

    for (i = 0; i < size; i++)
    {
        printf("%.2f ", array[i]);
    }
    printf("\n");
}

int initialize_net_weights(float **net, int num_input, int num_blocks)
{
    int i, j, k;
    int test;
    int prefix;
    int block_tl = RAND_MAX / 4;
    int block_tm = RAND_MAX / 2;
    int block_th = block_tm + block_tl; // interval division for random
    int net_size = num_input + num_blocks; // input + output layer neurons
    float *net_ptr;

    int block_sizes[num_blocks];
    int block_sizes_sums[num_blocks];

    // generate block sizes
    for (i = 0; i < num_blocks; i++)
    {
        test = random();
        if (test < block_tl)
            block_sizes[i] = BLOCK_SIZE - 20;
        else if (test < block_tm)
            block_sizes[i] = BLOCK_SIZE - 10;
        else if (test < block_th)
            block_sizes[i] = BLOCK_SIZE + 10;
        else
            block_sizes[i] = BLOCK_SIZE + 20;

        net_size += block_sizes[i];
    }

    // allocate network
    *net = (float *) malloc(sizeof(float) * net_size * net_size);
    net_ptr = *net;
    memset(net_ptr, 0, sizeof(float) * net_size * net_size);

    // initialize input layer
    for (i = 0; i < num_input; i++)
    {
        for (j = num_input; j < net_size - num_blocks; j++)
        {
            if (random() < RAND_MAX / 8)
                net_ptr[i * net_size + j] = 1.0f;
        }
    }

    //initialize output layer
    prefix = num_input;
    for (i = 0; i < num_blocks; i++)
    {
        test = block_sizes[i] / 2;
        // set output neurons from this block
        net_ptr[(prefix + test - 30) * net_size + (net_size - num_blocks) + i] = 1.0f;
        net_ptr[(prefix + test - 25) * net_size + (net_size - num_blocks) + i] = 1.0f;
        net_ptr[(prefix + test - 20) * net_size + (net_size - num_blocks) + i] = 1.0f;
        net_ptr[(prefix + test - 10) * net_size + (net_size - num_blocks) + i] = 1.0f;
        net_ptr[(prefix + test) * net_size + (net_size - num_blocks) + i] = 1.0f;
        net_ptr[(prefix + test + 10) * net_size + (net_size - num_blocks) + i] = 1.0f;
        net_ptr[(prefix + test + 20) * net_size + (net_size - num_blocks) + i] = 1.0f;
        net_ptr[(prefix + test + 25) * net_size + (net_size - num_blocks) + i] = 1.0f;
        net_ptr[(prefix + test + 30) * net_size + (net_size - num_blocks) + i] = 1.0f;
        block_sizes_sums[i] = prefix;
        prefix += block_sizes[i];
    }

    // initialize middle layer
    for (i = num_input; i < net_size - num_blocks; i++)
    {
        for (j = num_input; j < net_size - num_blocks; j++)
        {
            if (random() < RAND_MAX / 64)
                net_ptr[i * net_size + j] = 1.0f;
        }
    }

    // and main components with complete graph
    for (i = 0; i < num_blocks; i++)
    {
        // main diagonal for full graph
        for (j = 0; j < block_sizes[i]; j++)
        {
            for (k = 0; k < j; k++)
            {
                if (random() < RAND_MAX / 2)
                    net_ptr[block_sizes_sums[i] * net_size + j * net_size + block_sizes_sums[i] + k] = 3.0f;
                if (random() < RAND_MAX / 2)
                    net_ptr[block_sizes_sums[i] * net_size + k * net_size + block_sizes_sums[i] + j] = 4.0f;
            }
            net_ptr[block_sizes_sums[i] * net_size + block_sizes_sums[i] + j * net_size + j] = 0.0f;
        }
    }

    return net_size;
}

int main()
{
    int num_input = 0;
    int num_blocks = 0;
    int net_size; // count of all neurons
    char buffer[16];

    srand(time(NULL));
    signal(SIGINT, stop);

    // don't buff stdout
    setbuf(stdout, NULL);

    do
    {
        puts("Give input layer size: ");
        fgets((char *) buffer, 8, stdin);
        num_input = strtol(buffer, NULL, 10);
    }
    while (num_input <= 0);

    do
    {
        puts("Give count of hidden blocks (at least 3): ");
        fgets((char *) buffer, 8, stdin);
        num_blocks = strtol(buffer, NULL, 10);
    }
    while (num_blocks < 3);

    net_size = initialize_net_weights(&net, num_input, num_blocks);

    if (cl_init(net, net_size) == 0)
    {
        free(net);
        cl_simulate(num_blocks, num_input);
    }

    cl_cleanup();

    printf("\nBye.\n");
    return 0;
}
