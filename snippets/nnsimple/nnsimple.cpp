/*
 * Copyright 2016 Setin S.A.
*/

#include "nnsimple.h"
#include <ctime>


using namespace std;

NNSimple::NNSimple(Activate_Function nfunc, bool tryuse_cuda){
    n=0.5;
    s=0.6;
    e=0.001;
    e0=1e-5;
    afunction=nfunc;
    sizex=0;
    sizey=0;
    y=NULL;
    w=NULL;
    use_cuda=tryuse_cuda;

    //Check CUDA
    if(use_cuda){
        int devID = 0;
        cudaError_t error;
        cudaDeviceProp deviceProp;
        error = cudaGetDevice(&devID);
        if (error != cudaSuccess){
            printf("cudaGetDevice returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
            use_cuda=false;
        }
        error = cudaGetDeviceProperties(&deviceProp, devID);
        if (deviceProp.computeMode == cudaComputeModeProhibited){
            printf("Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
            use_cuda=false;
        }
        if (error != cudaSuccess){
            printf("cudaGetDeviceProperties returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
            use_cuda=false;
        }else{
            printf("Using GPU Device %d: \"%s\" with compute capability %d.%d\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
            //printf("maxThreadsPerBlock=%d sharedMemPerBlock=%d totalGlobalMem=%d K\n\n", deviceProp.maxThreadsPerBlock, deviceProp.sharedMemPerBlock,deviceProp.totalGlobalMem/1024);
        }
    }

}

/*!
    Allocate and init matrix and vectors
    w[sizey][sizex], rows - sizey, cols - sizex
*/
void NNSimple::Init(){
    y=new double[sizey];
    w=new double*[sizey];
    srand(time(NULL));

    for(int row=0;row<sizey;row++){
        w[row]=new double[sizex];
    }
    for(int row=0;row<sizey;row++)
        for(int col=0;col<sizex;col++)
            w[row][col]=0.0;

    if(use_cuda){
        if(!allocateobjects_cuda(sizex,sizey,w))
            use_cuda=false;
    }
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
            cout<<w[row][col]<<'\t';
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


void NNSimple::CorrectWeight(int row, double d){
    if(!use_cuda){
        for(int col=0;col<sizex;col++){
            w[row][col]+=d*x[col];
        }
    }else{
        if(!correctweight_cuda(w,sizex,sizey,row,d)){
            use_cuda=false;
        }
    }
}

void NNSimple::Clear(){
    if(y!=NULL)
        delete[] y;
    if(w!=NULL){
        for(int row=0;row<sizey;row++){
            delete[] w[row];
        }
        delete[] w;
    }
    sizex=0;
    sizey=0;
    if(use_cuda){
        freeobjects_cuda();
    }
}

NNSimple::~NNSimple(){
    Clear();
}


int NNSimple::Process(double *inputx){
    if(inputx==NULL) return -1;
    x=inputx;
    if(use_cuda)
        setx_cuda(sizex, x);

    for(int row=0;row<sizey;row++){
        double sum(0.0);
        for(int col=0;col<sizex;col++){
            sum+=x[col]*w[row][col];
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
void NNSimple::TeachThresh(double **voc, int stepscount){
    int row(0), step(0);
    int steps(stepscount), ind(0);
    double d(0.0), T(0.0);
    srand(time(NULL));

    for(step=0;step<steps;step++){
          ind=rand()%sizey;
          Process(voc[ind]);
          for(row=0;row<sizey;row++){
              if(ind==row)
                  T=1.0;
              else T=0.0;
              d=T-y[row];
              CorrectWeight(row,d);
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
void NNSimple::TeachSigma(double **voc, int stepscount){
    int row(0), step(0);
    int steps(stepscount), ind(0);
    double d(1.0), T(0.0),currente(0.0);
    srand(time(NULL));

    for(step=0;step<steps;step++){
          ind=rand()%sizey;
          Process(voc[ind]);
          //cout<<"step="<<step<<endl;

          for(row=0;row<sizey;row++){
                if(ind==row)
                    T=1.0;
                else T=0.0;

                while((currente=0.5*((T-y[row])*(T-y[row])))>e){
                    d=n * (T-y[row]) * y[row] * (1.0-y[row]);

                    if(fabs(d)<=e0){
                        break;
                    }
                    CorrectWeight(row,d*n);
                    Process(voc[ind]);                    
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
            fstr>>w[row][col];

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
            fstr<<w[row][col]<<' ';
    }

    fstr.close();
}

/*!
    Load input vector from file and start teach process
    \param[in]  filename    file name
    \param[in]  stepscount  count of iterations
*/
void NNSimple::Teach(const char *filename, int stepscount){
    int row(0), col(0);
    ifstream fstr(filename);

    if(!fstr.is_open()){
        cout<<"Error opening file "<<filename<<endl;
        return;
    }

    double **voc=NULL;
    fstr>>sizex>>sizey;
    Init();

    voc=new double*[sizey];

    for(row=0;row<sizey;row++)
        voc[row]=new double[sizex];

    for(row=0;row<sizey;row++)
        for(col=0;col<sizex;col++)
            fstr>>voc[row][col];
    fstr.close();

    if(afunction==AF_THRESH)
        TeachThresh(voc,stepscount);
    else
        TeachSigma(voc,stepscount);

    for(row=0;row<sizey;row++){
        delete[] voc[row];
    }
    delete[] voc;
}
