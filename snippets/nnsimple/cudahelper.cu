#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>


const int BLOCK_SIZE=256;

/*!
    vectors for processing in gpu memory
*/
namespace device_data{
    double *w=NULL;
    double *x=NULL;
    double *y=NULL;
}

/*!
    Free memory
*/
extern "C" void freedata_cuda(){
    cudaFree(device_data::x);
    cudaFree(device_data::w);
    cudaFree(device_data::y);
    device_data::x=NULL;
    device_data::y=NULL;
    device_data::w=NULL;
}


#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                        __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif


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

/*!
    CUDA Kernel Device code
    for(int col=0;col<sizex;col++){
        w[row][col]=w[row][col] + d*x[col];
    }
    w[row*sizey+idx]+=d*x[idx];
*/
__global__ void correctweight_cuda(const int row, const int idx, const int sizex, const int sizey,const double d, double *x, double *w){
    const int col = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ double shared_w[BLOCK_SIZE];
    __shared__ double shared_x[BLOCK_SIZE];

    if(col<sizex){
        shared_x[threadIdx.x]=x[idx*sizex+col];
        shared_w[threadIdx.x]=w[row*sizey+col];

        __syncthreads();

        shared_w[threadIdx.x]+=d*shared_x[threadIdx.x];

        __syncthreads();

        w[row*sizey+col]=shared_w[threadIdx.x];
    }
}

/*!
    for(int row=0;row<sizey;row++){
        double sum(0.0);
        for(int col=0;col<sizex;col++){
            sum+=x[sizex*idx+col]*w[row*sizey+col];
        }
        y[row]=AFunction(sum);
    }
}
*/
__global__ void process_cuda(const int idx, const int sizex, const int sizey, double *x, double *y, double *w){
    const int row = blockDim.y * blockIdx.y + threadIdx.y;

    if(row<sizey){
        double sum(0.0);
        for(int col=0;col<sizex;col++){
            sum+=x[sizex*idx+col]*w[row*sizey+col];
        }

        y[row]=1.0/(1.0+pow(M_E,-sum));
    }
}


/*!
    for(int row=0;row<sizey;row++){
        double sum(0.0);
        for(int col=0;col<sizex;col++){
            sum+=x[sizex*idx+col]*w[row*sizey+col];
        }
        y[row]=AFunction(sum);
    }
}
*/
__global__ void process_cuda_row(const int idx, const int row, const int sizex, const int sizey, double *x, double *y, double *w){
    const int col = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ double shared_w[BLOCK_SIZE];
    __shared__ double shared_x[BLOCK_SIZE];

    if(col<sizex){
        shared_x[threadIdx.x]=x[idx*sizex+col];
        shared_w[threadIdx.x]=w[row*sizey+col];
        __syncthreads();

        shared_w[threadIdx.x]=shared_x[threadIdx.x]*shared_w[threadIdx.x];
        __syncthreads();

        for(int s=1;s<blockDim.x;s<<=1){
            int  index = 2 * s * threadIdx.x;
            if ( index < blockDim.x )
                shared_w[index]+=shared_w[index + s];
            __syncthreads();
        }

        if(threadIdx.x==0)
            atomicAdd(&y[row], shared_w[0]);
    }
}


extern "C" bool teachsigma_cuda(const int stepscount, const int sizex, const int sizey, double *y,
        double n, double e, double e0){
    int step(0);
    double T(0.0), d(0.0);
    int ind(0);

    cudaError_t err = cudaSuccess;
    dim3 threadsPerBlock;
    dim3 blocksPerGrid;

    srand(time(NULL));

    for(step=0;step<stepscount;step++){
          ind=rand()%sizey;

          if(step%(stepscount/10)==0)
              printf("%d of %d\n", step, stepscount);

          threadsPerBlock.x=1;
          threadsPerBlock.y=BLOCK_SIZE;
          blocksPerGrid.x=1;
          blocksPerGrid.y=sizey / BLOCK_SIZE + 1;

          process_cuda<<<blocksPerGrid, threadsPerBlock>>>(ind, sizex, sizey, device_data::x,device_data::y,device_data::w);
          cudaThreadSynchronize();
          cudaMemcpy(y, device_data::y, sizey*sizeof(double), cudaMemcpyDeviceToHost);

          err=cudaGetLastError();
          if (err != cudaSuccess){
                fprintf(stderr, "process_cuda (error code %s)!\n", cudaGetErrorString(err));
                return false;
          }

          for(int row=0;row<sizey;row++){
                if(ind==row)
                    T=1.0;
                else T=0.0;

                while(0.5*((T-y[row])*(T-y[row]))>e){
                    d=n * (T-y[row]) * y[row] * (1.0-y[row]);
                    if(abs(d)<=e0){
                        break;
                    }
                    threadsPerBlock.x=BLOCK_SIZE;
                    threadsPerBlock.y=1;
                    blocksPerGrid.x=sizex / BLOCK_SIZE + 1;;
                    blocksPerGrid.y=1;
                    correctweight_cuda<<<blocksPerGrid, threadsPerBlock>>>(row,ind,sizex,sizey,d,device_data::x, device_data::w);
                    err=cudaGetLastError();
                    if (err != cudaSuccess){
                          fprintf(stderr, "correctweight_cuda (error code %s)!\n", cudaGetErrorString(err));
                          return false;
                    }
                    cudaThreadSynchronize();

                    threadsPerBlock.x=BLOCK_SIZE;
                    threadsPerBlock.y=1;
                    blocksPerGrid.x=sizex / BLOCK_SIZE + 1;;
                    blocksPerGrid.y=1;
                    y[row]=0.0;
                    process_cuda_row<<<blocksPerGrid, threadsPerBlock>>>(ind, row, sizex, sizey, device_data::x,device_data::y,device_data::w);
                    err=cudaGetLastError();
                    if (err != cudaSuccess){
                          fprintf(stderr, "process_cuda_row (error code %s)!\n", cudaGetErrorString(err));
                          return false;
                    }
                    cudaThreadSynchronize();
                    cudaMemcpy(&y[row],(const void*)&device_data::y[row], sizeof(double), cudaMemcpyDeviceToHost);
                    y[row]=1.0/(1.0+pow(M_E,-y[row]));
                }
          }


    }

    return true;
}


