#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

/*!
    vectors for processing in gpu memory
*/
namespace cuda_data{
    //double *dev_w=NULL;
    double *dev_roww=NULL;
    double *dev_x=NULL;
}

/*!
    CUDA Kernel Device code
    for(int col=0;col<sizex;col++){
        w[row][col]=w[row][col] + d*x[col];
    }
    w[row*sizey+i]+=d*x[i];
    w[idx]+=d*x[i];
*/
__global__ void vectorCorrect(double *w, const double *x, const int sizex, const int sizey, const double d, const int row){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if(idx<sizex){
        w[idx]+=d*x[idx];
    }

}

/*!
    Free memory
*/
extern "C" bool freeobjects_cuda(){
    if(cuda_data::dev_x!=NULL)
        cudaFree(cuda_data::dev_x);
    if(cuda_data::dev_roww!=NULL)
        cudaFree(cuda_data::dev_roww);
    return true;
}

/*!
    Allocate cuda_data
*/
extern "C" bool allocateobjects_cuda(const int sizex, const int sizey, double** w){

    cudaError_t err = cudaSuccess;

    err = cudaMalloc((void**)&cuda_data::dev_roww, sizex*sizeof(double));
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

    return true;
}



/*!
    Prepare data to calculate on GPU
*/
extern "C" bool correctweight_cuda(double **w,const double *x,const int sizex, const int sizey,const int row,const double d){

    cudaError_t err = cudaSuccess;
    size_t size = sizex * sizeof(double);        

    err=cudaMemcpy(cuda_data::dev_roww, w[row], size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        fprintf(stderr, "correctweight_cuda/cudaMemcpy dev_roww returned error %s (code %d), line(%d)\n", cudaGetErrorString(err), err, __LINE__);
        freeobjects_cuda();
        return false;
    }

    err=cudaMemcpy(cuda_data::dev_x, x, size, cudaMemcpyHostToDevice);    
    if (err != cudaSuccess){
        fprintf(stderr, "correctweight_cuda/cudaMemcpy dev_x returned error %s (code %d), line(%d)\n", cudaGetErrorString(err), err, __LINE__);
        freeobjects_cuda();
        return false;
    }

    int threadsPerBlock = 256;
    int blocksPerGrid =(sizex + threadsPerBlock - 1) / threadsPerBlock;

    vectorCorrect<<<blocksPerGrid, threadsPerBlock>>>(cuda_data::dev_roww, cuda_data::dev_x, sizex, sizey, d, row);

    cudaThreadSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to launch vectorCorrect kernel (error code %s)!\n", cudaGetErrorString(err));
        freeobjects_cuda();
        return false;
    }    

    //err=cudaMemcpy(w[row], &cuda_data::dev_w[row*sizey], size, cudaMemcpyDeviceToHost);
    err=cudaMemcpy(w[row], cuda_data::dev_roww, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess){
        fprintf(stderr, "correctweight_cuda/cudaMemcpy dev_roww returned error %s (code %d), line(%d)\n", cudaGetErrorString(err), err, __LINE__);
        freeobjects_cuda();
        return false;
    }

    return true;

}
