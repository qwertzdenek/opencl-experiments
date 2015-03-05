// target_opencl.c
// Zdeněk Janeček 2015

#include <stdlib.h>
#include <math.h>

#define THRESHOLD 30
#define local_dim 8

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue command_queue;
cl_program program;
cl_kernel kernel_spike;

cl_mem buf_net;
cl_mem buf_acts_net;
cl_mem buf_acts_buff;

size_t global_size[1] = {0};
size_t local_size[1] = {local_dim};

int net_size;
int running = 1;

/**
 * Reads CL file
 */
static char* read_source_file(const char *filename)
{
    long int
    size = 0,
    res  = 0;

    char *src = NULL;

    FILE *file = fopen(filename, "rb");

    if (!file)  return NULL;

    if (fseek(file, 0, SEEK_END))
    {
        fclose(file);
        return NULL;
    }

    size = ftell(file);
    if (size == 0)
    {
        fclose(file);
        return NULL;
    }

    rewind(file);

    src = (char *)calloc(size + 1, sizeof(char));
    if (!src)
    {
        src = NULL;
        fclose(file);
        return src;
    }

    res = fread(src, 1, sizeof(char) * size, file);
    if (res != sizeof(char) * size)
    {
        fclose(file);
        free(src);

        return src;
    }

    src[size] = '\0'; /* NULL terminated */
    fclose(file);

    return src;
}

/**
 * initialize OpenCL device
 */
int cl_init(float *net, int size)
{
    cl_uint platformCount = 0;
    char string_one[64];
    char string_two[64];
    char string[128];

    net_size = size;

    float init_acts[net_size];
    memset(init_acts, 0, sizeof(float) * net_size);

    const char *source = NULL;
    cl_int err;
    cl_context_properties properties[3];

    size_t num_local_groups = ceil((float) net_size / local_size[0]);
    global_size[0] = local_size[0] * num_local_groups;

    // Probe platforms
    clGetPlatformIDs(0, NULL, &platformCount);
    if (platformCount < 1)
    {
        printf("OpenCL platform not found!\n");
        return -1;
    }

    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);

    clGetDeviceInfo(device, CL_DEVICE_NAME, 64, string_one, NULL);
    clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, 64, string_two, NULL);

    sprintf(string, "%s (version %s)", string_one, string_two);
    printf("running on device %s\n", string);

    // read kernel
    source = read_source_file("kernel_spike.cl");
    if (source == NULL)
        return -1;

    // context properties list - must be terminated with 0
    properties[0]= CL_CONTEXT_PLATFORM; // specifies the platform to use
    properties[1]= (cl_context_properties) platform;
    properties[2]= 0;

    // create context
    context = clCreateContext(properties,1,&device,NULL,NULL,&err);
    if (err != CL_SUCCESS)
    {
        printf("chyba ve vytváření kontextu %d\n", err);
    }

    // create command queue
    command_queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
    if (err != CL_SUCCESS)
    {
        printf("chyba ve vytváření fronty úloh %d\n", err);
    }

    program = clCreateProgramWithSource(context, 1, &source, 0, &err);

    free((void *) source);

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    if (err != CL_SUCCESS)
    {
        // Determine the size of the log
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        // Allocate memory for the log
        char *log = (char *) malloc(log_size);

        // Get the log
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

        // Print the log
        printf("%s\n", log);
        free(log);
        clReleaseCommandQueue(command_queue);
        clReleaseContext(context);
        clReleaseProgram(program);
        return 1;
    }

    // specify which kernel from the program to execute
    kernel_spike = clCreateKernel(program, "kernel_spike", &err);

    buf_net = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * net_size * net_size, net, NULL);
    buf_acts_net = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * net_size, init_acts, NULL);
    buf_acts_buff = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * net_size, init_acts, NULL);

    // set the argument list for the kernel command
    clSetKernelArg(kernel_spike, 0, sizeof(cl_mem), &buf_net);
    clSetKernelArg(kernel_spike, 3, sizeof(cl_mem), &buf_acts_net);
    clSetKernelArg(kernel_spike, 4, sizeof(cl_mem), &buf_acts_buff);

    return 0;
}

void cl_simulate(int num_blocks, int num_input)
{
    int i, gactivated, nactivated;
    float res[net_size];
    float init_acts[net_size];
    memset(init_acts, 0, sizeof(float) * net_size);
    time_t t;
    float prob;

    clSetKernelArg(kernel_spike, 1, sizeof(cl_int), &net_size);
    clSetKernelArg(kernel_spike, 2, sizeof(cl_int), &num_input);

    while (running)
    {
        // cleanup buffer
        clEnqueueWriteBuffer(command_queue, buf_acts_buff, CL_TRUE, 0, sizeof(float) * num_input, init_acts, 0, NULL, NULL);

        clEnqueueNDRangeKernel(command_queue, kernel_spike, 1, NULL, global_size, local_size, 0, NULL, NULL);
        clFinish(command_queue);

        clEnqueueReadBuffer(command_queue, buf_acts_net, CL_TRUE, 0, sizeof(float) * net_size, res, 0, NULL, NULL);

        // Do some statistics
        gactivated = 0;
        nactivated = 0;
        for (i = 0; i < num_blocks; i++)
            if (res[net_size - i - 1] > 0.0001f)
                gactivated++;

        for (i = 0; i < net_size; i++)
            if (res[i] > 0.01)
                nactivated++;

        printf("\rActivated groups %d, activated neurons %.6f %%", gactivated, 100.0f * nactivated / net_size);

        // random activations
        t = time(NULL) % 16;
        prob = 512 * sinf(2.0f * 3.14f * t / 16.0f) + 860.0f;
        for (i = 0; i < num_input; i++)
        {
            if (random() < RAND_MAX / prob)
                res[i] += 20.0f;
        }

        clEnqueueWriteBuffer(command_queue, buf_acts_net, CL_TRUE, 0, sizeof(float) * num_input, res, 0, NULL, NULL);
    }
}

void cl_cleanup()
{
    clFinish(command_queue);

    // clean CL objects
    clReleaseMemObject(buf_net);
    clReleaseMemObject(buf_acts_net);
    clReleaseMemObject(buf_acts_buff);
    clReleaseProgram(program);
    clReleaseKernel(kernel_spike);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);
}

void stop(int signo)
{
    if (signo == SIGINT)
        running = 0;
}
