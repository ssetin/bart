#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>


const int BLOCK_SIZE=256;

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
        w[row][col]=w[row][col] + d*x[col];
    }
    w[row*sizey+idx]+=d*x[idx];
*/
__global__ void vectorCorrect(double *w, const double *x, const int sizex, const int sizey,const double d, const int row){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ double shared_w[BLOCK_SIZE];
    __shared__ double shared_x[BLOCK_SIZE];

    if(idx<sizex){

        shared_x[threadIdx.x]=x[idx];
        shared_w[threadIdx.x]=w[row*sizey+idx];

        __syncthreads();

        shared_w[threadIdx.x]+=d*shared_x[threadIdx.x];

        __syncthreads();

        w[row*sizey+idx]=shared_w[threadIdx.x];
    }


}

/*!
    Free memory
*/
extern "C" bool freeobjects_cuda(){
    if(cuda_data::dev_x!=NULL)
        cudaFree(cuda_data::dev_x);
    if(cuda_data::dev_w!=NULL)
        cudaFree(cuda_data::dev_w);
    return true;
}

/*!
    Allocate cuda_data
*/
extern "C" bool allocateobjects_cuda(const int sizex, const int sizey, double *w){
    cudaError_t err = cudaSuccess;

    err = cudaMalloc((void**)&cuda_data::dev_w,  sizex*sizey*sizeof(double));
    if (err != cudaSuccess){
        fprintf(stderr, "allocateobjects_cuda/cudaMalloc dev_w returned error %s (code %d), line(%d)\n", cudaGetErrorString(err), err, __LINE__);
        freeobjects_cuda();
        return false;
    }

    err = cudaMalloc((void**)&cuda_data::dev_x, sizex * sizeof(double));
    if (err != cudaSuccess){
        fprintf(stderr, "allocateobjects_cuda/cudaMalloc dev_x returned error %s (code %d), line(%d)\n", cudaGetErrorString(err), err, __LINE__);
        freeobjects_cuda();
        return false;
    }

    err=cudaMemcpy(cuda_data::dev_w, w, sizex *sizey * sizeof(double) , cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        fprintf(stderr, "allocateobjects_cuda/cudaMemcpy dev_w returned error %s (code %d), line(%d)\n", cudaGetErrorString(err), err, __LINE__);
        freeobjects_cuda();
        return false;
    }

    return true;
}


extern "C" bool setx_cuda(const int sizex, double *x){
    cudaError_t err = cudaSuccess;
    err=cudaMemcpy(cuda_data::dev_x, x, sizex * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        fprintf(stderr, "setx_cuda/cudaMemcpy dev_x returned error %s (code %d), line(%d)\n", cudaGetErrorString(err), err, __LINE__);
        freeobjects_cuda();
        return false;
    }
    return true;
}



/*!
    Prepare data to calculate on GPU
*/
extern "C" bool correctweight_cuda(double *w,const int sizex, const int sizey,const int row,const double d){

    cudaError_t err = cudaSuccess;
    size_t size = sizex * sizeof(double);    

    dim3 threadsPerBlock(BLOCK_SIZE, 1);
    dim3 blocksPerGrid(sizex / BLOCK_SIZE +1, 1);

    vectorCorrect<<<blocksPerGrid, threadsPerBlock>>>(cuda_data::dev_w, cuda_data::dev_x, sizex, sizey, d, row);

    err = cudaGetLastError();
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to launch vectorCorrect kernel (error code %s)!\n", cudaGetErrorString(err));
        freeobjects_cuda();
        return false;
    }    

    cudaThreadSynchronize();


    err=cudaMemcpyAsync(&w[row*sizey], &cuda_data::dev_w[row*sizey], size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess){
        fprintf(stderr, "correctweight_cuda/cudaMemcpy dev_w returned error %s (code %d), line(%d)\n", cudaGetErrorString(err), err, __LINE__);
        freeobjects_cuda();
        return false;
    }


    return true;

}
