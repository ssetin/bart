/*
 * Copyright 2016 Setin S.A.
*/

#include "nnsimple.h"
#include <ctime>

using namespace std;

NNSimple::NNSimple(Activate_Function nfunc, bool tryuse_cuda){
    n=0.5;
    s=0.6;
    e=0.0001;
    e0=1e-5;
    afunction=nfunc;
    sizex=0;
    sizey=0;
    y=NULL;
    w=NULL;
    use_cuda=tryuse_cuda;
}

bool NNSimple::CheckCuda(){
    if(use_cuda){
        int devID = 0;
        cudaError_t error;
        cudaDeviceProp deviceProp;
        error = cudaGetDevice(&devID);
        if (error != cudaSuccess){
            printf("cudaGetDevice returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
            use_cuda=false;
            return false;
        }
        error = cudaGetDeviceProperties(&deviceProp, devID);
        if (deviceProp.computeMode == cudaComputeModeProhibited){
            printf("Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
            use_cuda=false;
            return false;
        }
        if (error != cudaSuccess){
            printf("cudaGetDeviceProperties returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
            use_cuda=false;
            return false;
        }else{
            printf("Using GPU Device %d: \"%s\" with compute capability %d.%d\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
            printf("maxThreadsPerBlock=%d, sharedMemPerBlock=%zu Kb, totalGlobalMem=%zu Kb\n\n", deviceProp.maxThreadsPerBlock, deviceProp.sharedMemPerBlock/1024,deviceProp.totalGlobalMem/1024);
        }
        return true;
    }
    return false;
}

/*!
    Allocate and init matrix and vectors
    w[sizey][sizex], rows - sizey, cols - sizex
*/
void NNSimple::Init(){
    y=new double[sizey];
    w=new double[sizex*sizey];
    x=new double[sizex*sizey];
    srand(time(NULL));

   for(int i=0;i<sizex*sizey;i++)
            w[i]=0.0;
}

double NNSimple::GetW(int col, int row){
    if(w!=NULL && col<sizex && row<sizey)
        return w[sizey*row+col];
    else
        return 0;
}

void NNSimple::SetW(int col, int row, double value){
    if(w!=NULL && col<sizex && row<sizey)
        w[sizey*row+col]=value;
}

/*!
    Setting inaccuracy
    \param[in]  ne new inaccuracy
*/
void NNSimple::SetE(double ne){
    e=ne;
}

/*!
    Setting speed of teaching
    \param[in]  nn new speed of teaching
*/
void NNSimple::SetN(double nn){
    n=nn;
}

/*!
    Setting sensitivity (border of significance)
    \param[in]  ns new sensitivity
*/
void NNSimple::SetSensitivity(double ns){
    s=ns;
}

int NNSimple::MaxY(){
    int ind=0;
    double max(y[0]);
    for(int i=1;i<sizey;i++){
        if(y[i]>max){
            max=y[i];
            ind=i;
        }
    }
    return ind;
}

/*!
    Getting max value of y vector
*/
double NNSimple::MaxYVal(){
    double tmp(y[0]);
    for(int i=1;i<sizey;i++){
        if(y[i]>tmp){
            tmp=y[i];
        }
    }
    return tmp;
}

/*!
    Returns index of max value of y vector response
*/
int NNSimple::GetY(){
    double tmp(y[0]);
    int res(0);

    for(int i=1;i<sizey;i++){
        if(y[i]>tmp){
            tmp=y[i];
            res=i;
        }
    }
    return res;

}

void NNSimple::PrintY(int precision){
    cout.precision(precision);
    for(int i=0;i<sizey;i++){
        cout<<"y["<<i<<"] = "<<y[i]<<endl;
    }
}

void NNSimple::PrintW(int precision){
    cout.precision(precision);

    for(int row=0;row<sizey;row++){
        for(int col=0;col<sizex;col++){
            cout<<w[row*sizey+col]<<'\t';
        }
        cout<<endl;
    }
}

/*!
    Neuron activating function (threshold or sigma)
    \param[in] nsum - sum of neuron weights*input vector
*/
double NNSimple::AFunction(double nsum){
    if(afunction==AF_THRESH)
        return nsum>(sizey/2)?1:0;
    return 1.0/(1.0+pow(M_E,-nsum));
}

/*!
    Correct matrix weights for input vector
    \param[in] row - matrix row
    \param[in] idx - number of input vector
    \param[in] d   - koeff
*/
void NNSimple::CorrectWeight(const int row, const int idx, const double d){
    for(int col=0;col<sizex;col++){
        w[row*sizey+col]+=d*x[sizex*idx+col];
    }
}

void NNSimple::Clear(){
    if(y!=NULL)
        delete[] y;
    if(w!=NULL){
        delete[] w;
    }
    if(x!=NULL){
        delete[] x;
    }
    sizex=0;
    sizey=0;
}

NNSimple::~NNSimple(){
    Clear();
}

int NNSimple::Process(const int idx){
    if(x==NULL) return -1;

    for(int row=0;row<sizey;row++){
        double sum(0.0);
        for(int col=0;col<sizex;col++){
            sum+=x[sizex*idx+col]*w[row*sizey+col];
        }
        y[row]=AFunction(sum);
    }
    int res(MaxY());
    return y[res]>s?res:-1;
}


int NNSimple::Process(double *inputx){
    if(inputx==NULL) return -1;

    for(int row=0;row<sizey;row++){
        double sum(0.0);
        for(int col=0;col<sizex;col++){
            sum+=inputx[col]*w[row*sizey+col];
        }
        y[row]=AFunction(sum);
    }
    int res(MaxY());
    return y[res]>s?res:-1;
}

/*!
    Delta rule with threshold function
    δ = (T - Y)
    Δi = δxi
    w(step+1) = w(step) + Δi
    \param[in]  voc         input vectors
    \param[in]  stepscount  count of iterations
*/
void NNSimple::TeachThresh(int stepscount){
    int row(0), step(0);
    int steps(stepscount), ind(0);
    double d(0.0), T(0.0);
    srand(time(NULL));

    for(step=0;step<steps;step++){
          ind=rand()%sizey;
          Process(ind);
          for(row=0;row<sizey;row++){
              if(ind==row)
                  T=1.0;
              else T=0.0;
              d=T-y[row];
              CorrectWeight(row,ind,d);
          }
    }
}

/*!
    Delta rule with sigma function
    e = 0.5 (T-Y)(T-Y)
    δ = (T - Y)
    f' = Y (1-Y)    
    Δi = η δ f' xi    
    w(step+1) = w(step) + Δi
    \param[in]  voc         input vectors
    \param[in]  stepscount  count of iterations
*/
void NNSimple::TeachSigma(int stepscount){
    int row(0), step(0);
    int steps(stepscount), ind(0);
    double d(1.0), T(0.0),currente(0.0);
    srand(time(NULL));

    for(step=0;step<steps;step++){
          ind=rand()%sizey;
          if(step%(stepscount/10)==0)
              cout<<step<<" of "<<stepscount<<endl;
          Process(ind);          

          for(row=0;row<sizey;row++){
                if(ind==row)
                    T=1.0;
                else T=0.0;


                while((currente=0.5*((T-y[row])*(T-y[row])))>e){
                    d=n * (T-y[row]) * y[row] * (1.0-y[row]);

                    if(abs(d)<=e0){
                        break;
                    }
                    CorrectWeight(row,ind,d*n);
                    Process(ind);
                }
          }
    }
}

/*!
    Load weights matrix from file
    \param[in]  filename    file name
*/
void NNSimple::LoadWeights(const char *filename){
    ifstream fstr(filename);

    Clear();
    fstr>>sizex>>sizey;
    Init();

    for(int row=0;row<sizey;row++)
        for(int col=0;col<sizex;col++)
            fstr>>w[row*sizey+col];

    fstr.close();
}

/*!
    Save weights matrix to file
    \param[in]  filename    file name
*/
void NNSimple::SaveWeights(const char *filename){
    ofstream fstr(filename);
    fstr<<sizex<<" "<<sizey<<endl;

    for(int row=0;row<sizey;row++){
        for(int col=0;col<sizex;col++)
            fstr<<w[row*sizey+col]<<' ';
    }

    fstr.close();
}

/*!
    Load input vector from file and start teach process
    \param[in]  filename    file name
    \param[in]  stepscount  count of iterations
*/
void NNSimple::Teach(const char *filename, int stepscount){
    ifstream fstr(filename);

    if(!fstr.is_open()){
        cout<<"Error opening file "<<filename<<endl;
        return;
    }

    fstr>>sizex>>sizey;
    Init();

    for(int i=0;i<sizex*sizey;i++)
        fstr>>x[i];

    fstr.close();

    if(CheckCuda()){
        if(allocatedata_cuda(sizex, sizey)){

            setw_cuda(sizex, sizey, w);
            setx_cuda(sizex, sizey, x);

            if(teachsigma_cuda(stepscount, sizex, sizey, y, n, e, e0))
                getw_cuda(sizex, sizey, w);

            freedata_cuda();
        }
    } else {

        if(afunction==AF_THRESH)
            TeachThresh(stepscount);
        else
            TeachSigma(stepscount);

    }


}
