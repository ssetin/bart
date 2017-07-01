#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

/*!
    vectors for processing in gpu memory
*/
namespace cuda_data{
    double *dev_w=NULL;
    double *dev_x=NULL;
}

/*!
    CUDA Kernel Device code
    for(int col=0;col<sizex;col++){
        w[row][col]+=d*x[col];
    }
*/
__global__ void vectorCorrect(double *w, const double *x, const int sizex, const int sizey, const double d, const int row){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < sizex)
    {
        w[row*sizey+i]+=d*x[i];
    }
}


/*!
    Allocate cuda_data
*/
extern "C" void allocateobjects_cuda(const int sizex, const int sizey, double** w){
    size_t sx = sizex * sizeof(double);
    size_t sy = sizey * sizeof(double);
    cudaMalloc((void**)&cuda_data::dev_w, sx * sy);
    cudaMalloc((void**)&cuda_data::dev_x, sx);

    cudaMemcpy(cuda_data::dev_w, w, sx*sy, cudaMemcpyHostToDevice);
}

/*!
    Free memory
*/
extern "C" void freeobjects_cuda(){
    if(cuda_data::dev_x!=NULL)
        cudaFree(cuda_data::dev_x);
    if(cuda_data::dev_w!=NULL)
        cudaFree(cuda_data::dev_w);
}


/*!
    Prepare data to calculate on GPU
*/
extern "C" void correctweight_cuda(double **w,const double *x,const int sizex, const int sizey,const int row,const double d){
    cudaError_t err = cudaSuccess;
    size_t size = sizex * sizeof(double);

    cudaMemcpy(cuda_data::dev_x, x, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid =(sizex + threadsPerBlock - 1) / threadsPerBlock;

    vectorCorrect<<<blocksPerGrid, threadsPerBlock>>>(cuda_data::dev_w, cuda_data::dev_x, sizex, sizey, d, row);

    err = cudaGetLastError();
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to launch vectorCorrect kernel (error code %s)!\n", cudaGetErrorString(err));
        cudaFree(cuda_data::dev_x);
        cudaFree(cuda_data::dev_w);
        exit(EXIT_FAILURE);
    }    

    cudaMemcpy(w[row], &cuda_data::dev_w[row*sizey], size, cudaMemcpyDeviceToHost);

}
