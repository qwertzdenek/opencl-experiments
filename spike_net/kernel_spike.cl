// kernel_spike.cl
// Zdeněk Janeček 2015

#define THRESHOLD 30
#define local_dim 8

inline void atomic_fadd(volatile __global float *source, const float operand)
{
    union
    {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union
    {
        unsigned int intVal;
        float floatVal;
    } oldVal;

    do
    {
        oldVal.floatVal = *source;
        newVal.floatVal = oldVal.floatVal + operand;
    } while (atomic_cmpxchg((volatile __global unsigned int *) source, oldVal.intVal, newVal.intVal) != oldVal.intVal);
}

__kernel void kernel_spike(__global float *net, const int net_size, const int num_input, __global float *acts_real, __global float *acts_diff)
{
    // private
    float activ;
    int j;

    const int neuron = get_global_id(0);

    if (neuron >= net_size)
        return;

    activ = acts_real[neuron] - THRESHOLD;
    barrier(CLK_GLOBAL_MEM_FENCE);

    activ = 1 / (1 + exp(-activ));
    if (activ > 0.5f)
    {
        acts_real[neuron] = 0.0f;
        for (j = 0 ; j < net_size; j++)
        {
            atomic_fadd(acts_diff + j, net[neuron * net_size + j]);
        }
    }

    acts_real[neuron] += acts_diff[neuron];
    acts_real[neuron] *= 0.85f;
}
