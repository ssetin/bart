#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>


const int BLOCK_SIZE=256;

/*!
    vectors for processing in gpu memory
*/
namespace device_data{
    double *w;
    double *x;
    double *y;

    __device__ double e;
    __device__ double e0;
    __device__ double s;
    __device__ int sizex;
    __device__ int sizey;
    __device__ double n;
}

/*!
    Free memory
*/
extern "C" void freedata_cuda(){
    cudaFree(device_data::x);
    cudaFree(device_data::w);
    cudaFree(device_data::y);
}


/*!
    Allocate device_data
*/
extern "C" bool allocatedata_cuda(const int sizex, const int sizey){
    cudaError_t err = cudaSuccess;

    err = cudaMalloc((void**)&device_data::w,  sizex * sizey * sizeof(double));
    if (err != cudaSuccess){
        fprintf(stderr, "allocateobjects_cuda/cudaMalloc w returned error %s (code %d), line(%d)\n", cudaGetErrorString(err), err, __LINE__);
        freedata_cuda();
        return false;
    }

    err = cudaMalloc((void**)&device_data::x, sizex * sizey * sizeof(double));
    if (err != cudaSuccess){
        fprintf(stderr, "allocateobjects_cuda/cudaMalloc x returned error %s (code %d), line(%d)\n", cudaGetErrorString(err), err, __LINE__);
        freedata_cuda();
        return false;
    }

    err = cudaMalloc((void**)&device_data::y, sizey * sizeof(double));
    if (err != cudaSuccess){
        fprintf(stderr, "allocateobjects_cuda/cudaMalloc y returned error %s (code %d), line(%d)\n", cudaGetErrorString(err), err, __LINE__);
        freedata_cuda();
        return false;
    }

    cudaMemcpyToSymbol((const void*)&device_data::sizex,(void*)&sizex, sizeof(int));
    cudaMemcpyToSymbol((const void*)&device_data::sizey,(void*)&sizey, sizeof(int));

    return true;
}


extern "C" bool setconstants_cuda(const double e,const double e0, const double s, const double n){   
    cudaMemcpyToSymbol((const void*)&device_data::e,(void*)&e, sizeof(double));
    cudaMemcpyToSymbol((const void*)&device_data::e0,(void*)&e0, sizeof(double));
    cudaMemcpyToSymbol((const void*)&device_data::s,(void*)&s, sizeof(double));
    cudaMemcpyToSymbol((const void*)&device_data::n,(void*)&n, sizeof(double));
    return true;
}

extern "C" bool setw_cuda(const int sizex,const int sizey, double *w){
    cudaError_t err = cudaSuccess;
    err=cudaMemcpy(device_data::w, w, sizex *sizey * sizeof(double) , cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        fprintf(stderr, "setw_cuda/cudaMemcpy w returned error %s (code %d), line(%d)\n", cudaGetErrorString(err), err, __LINE__);
        freedata_cuda();
        return false;
    }
    return true;
}

extern "C" bool getw_cuda(const int sizex,const int sizey, double *w){
    cudaError_t err = cudaSuccess;
    err=cudaMemcpy(w, device_data::w, sizex *sizey * sizeof(double) , cudaMemcpyDeviceToHost);
    if (err != cudaSuccess){
        fprintf(stderr, "getw_cuda/cudaMemcpy w returned error %s (code %d), line(%d)\n", cudaGetErrorString(err), err, __LINE__);
        freedata_cuda();
        return false;
    }
    return true;
}

extern "C" bool setx_cuda(const int sizex,const int sizey, double *x){
    cudaError_t err = cudaSuccess;
    err=cudaMemcpy(device_data::x, x, sizex * sizey * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        fprintf(stderr, "setx_cuda/cudaMemcpy x returned error %s (code %d), line(%d)\n", cudaGetErrorString(err), err, __LINE__);
        freedata_cuda();
        return false;
    }
    return true;
}

extern "C" bool sety_cuda(const int sizey, double *y){
    cudaError_t err = cudaSuccess;
    err=cudaMemcpy(device_data::y, y, sizey * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        fprintf(stderr, "sety_cuda/cudaMemcpy y returned error %s (code %d), line(%d)\n", cudaGetErrorString(err), err, __LINE__);
        freedata_cuda();
        return false;
    }
    return true;
}


/*!
    CUDA Kernel Device code
    for(int col=0;col<sizex;col++){
        w[row][col]=w[row][col] + d*x[col];
    }
    w[row*sizey+idx]+=d*x[idx];
*/
__global__ void correctweight_cuda(const int row, const int idx,const double d, double *x, double *w){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ double shared_w[BLOCK_SIZE];
    __shared__ double shared_x[BLOCK_SIZE];

    if(i<device_data::sizex){
        shared_x[threadIdx.x]=x[idx*device_data::sizex+i];
        shared_w[threadIdx.x]=w[row*device_data::sizey+i];

        __syncthreads();

        shared_w[threadIdx.x]+=d*shared_x[threadIdx.x];

        __syncthreads();

        w[row*device_data::sizey+i]=shared_w[threadIdx.x];
    }
}


__device__ void process_cuda(const int idx, double *x, double *y, double *w){
    for(int row=0;row<device_data::sizey;row++){
        double sum(0.0);
        for(int col=0;col<device_data::sizex;col++){
            sum+=x[device_data::sizex*idx+col]*w[row*device_data::sizey+col];
        }
        y[row]=1.0/(1.0+pow(M_E,-sum));
    }
}

__device__ void correctweight_cuda_2(const int row, const int idx, const double d, double *x, double *w){
    for(int col=0;col<device_data::sizex;col++){
        w[row*device_data::sizey+col]+=d*x[device_data::sizex*idx+col];
    }
}

/*!
    CUDA Kernel Device code
    for(row=0;row<sizey;row++){
        if(ind==row)
            T=1.0;
            else T=0.0;

        while((currente=0.5*((T-y[row])*(T-y[row])))>e){
            d=n * (T-y[row]) * y[row] * (1.0-y[row]);

            if(fabs(d)<=e0){
                break;
            }
            CorrectWeight(row,ind,d*n);
            process_cuda(ind);
        }
    }

*/
__global__ void correctmatrix_cuda(const int ind, double *x, double *y, double *w){
    double d(1.0), T(0.0);

    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;

    if(row<device_data::sizey){

        if(row==0)
            process_cuda(ind,x,y,w);

        if(ind==row)
            T=1.0;
        else
            T=0.0;


        while(0.5*((T-y[row])*(T-y[row])) > device_data::e){
            d = device_data::n * (T-y[row]) * y[row] * (1.0-y[row]);

            if(fabs(d)<=device_data::e0){
                break;
            }
            correctweight_cuda_2(row,ind,d*device_data::n,x,w);
            process_cuda(ind,x,y,w);
        }


    }


}


extern "C" bool teachsigma_cuda(const int stepscount, const int sizex, const int sizey){
    int step(0);
    int ind(0);

    srand(time(NULL));

    for(step=0;step<stepscount;step++){
          ind=rand()%sizey;

          dim3 threadsPerBlock(sizey,1);
          dim3 blocksPerGrid(1,1);

          correctmatrix_cuda<<<blocksPerGrid, threadsPerBlock>>>(ind,device_data::x,device_data::y,device_data::w);
          cudaThreadSynchronize();
    }

    return true;
}


