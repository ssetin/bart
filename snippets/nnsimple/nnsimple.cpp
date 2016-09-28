/*
 * Copyright 2016 Setin S.A.
*/

#include <nnsimple.h>
#include <ctime>

using namespace std;

NNSimple::NNSimple(Activate_Function nfunc){
    n=0.5;
    s=0.5;
    e=0.001;
    afunction=nfunc;
    sizex=0;
    sizey=0;
    y=NULL;
    w=NULL;
}

void NNSimple::Init(){
    y=new double[sizey];
    w=new double*[sizex];
    srand(time(NULL));

    for(int i=0;i<sizex;i++){
        w[i]=new double[sizey];
    }
    for(int i=0;i<sizex;i++)
        for(int j=0;j<sizey;j++)
            w[i][j]=0;
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

double NNSimple::MaxYVal(){
    double tmp(y[0]);
    for(int i=1;i<sizey;i++){
        if(y[i]>tmp){
            tmp=y[i];
        }
    }
    return tmp;
}

void NNSimple::PrintY(int precision){
    cout.precision(precision);
    for(int i=0;i<sizey;i++){
        cout<<"y["<<i<<"] = "<<y[i]<<endl;
    }
}

/*!
    Neuron activating function (threshold or sigma)
    \param[in] nsum - sum of neuron weights*input vector
*/
double NNSimple::AFunction(double nsum){
    if(afunction==AF_THRESH)
        return nsum>(sizey/2)?1:0;
    return 1/(1+pow(M_E,-nsum));
}


void NNSimple::CorrectWeight(int j, double d){
    for(int i=0;i<sizex;i++){
        w[i][j]+=d*x[i];
    }
}

void NNSimple::Clear(){
    if(y!=NULL)
        delete[] y;
    if(w!=NULL)
    for(int i=0;i<sizex;i++){
        delete[] w[i];
    }
    delete[] w;
    sizex=0;
    sizey=0;
}

NNSimple::~NNSimple(){
    Clear();
}


int NNSimple::Process(double *inputx){
    if(inputx==NULL) return -1;
    x=inputx;
    for(int k=0;k<sizey;k++){
        double sum(0);
        for(int i=0;i<sizex;i++){
            sum+=x[i]*w[i][k];
        }
        y[k]=AFunction(sum);
    }
    int res(MaxY());
    return y[res]>s?res:-1;
}

/*!
    Delta rule with threshold function
    δ = (T - Y)
    Δi = δxi
    w(n+1) = w(n) + Δi
    \param[in]  voc         input vectors
    \param[in]  stepscount  count of iterations
*/
void NNSimple::TeachThresh(double **voc, int stepscount){
    int j(0), step(0);
    int steps(stepscount), ind(0);
    double d(0), T(0);
    srand(time(NULL));

    for(step=0;step<steps;step++){
          ind=rand()%sizey;
          Process(voc[ind]);
          for(j=0;j<sizey;j++){
              if(ind==j)
                  T=1;
              else T=0;
              d=T-y[j];
              CorrectWeight(j,d);
          }
    }
}

/*!
    Delta rule with sigma function
    e = 0.5 (T-Y)(T-Y)
    δ = (T - Y)
    f' = Y (1-Y)
    Δi = η δ f' xi
    w(n+1) = w(n) + Δi
    \param[in]  voc         input vectors
    \param[in]  stepscount  count of iterations
*/
void NNSimple::TeachSigma(double **voc, int stepscount){
    int j(0), step(0);
    int steps(stepscount), ind(0);
    double d(1), T(0);
    srand(time(NULL));

    for(step=0;step<steps;step++){
          ind=rand()%sizey;
          Process(voc[ind]);

          for(j=0;j<sizey;j++){
                if(ind==j)
                    T=1;
                else T=0;

                while(0.5*((T-y[j])*(T-y[j]))>e){
                    d=n*(T-y[j]) * y[j] * (1-y[j]);
                    CorrectWeight(j,d*n);
                    Process(voc[ind]);
                    //cout<<"step = "<<step<<" ind = "<<ind<<" d = "<<d<<" e = "<<0.5*((T-y[j])*(T-y[j]))<<endl;
                }
          }
    }
}

/*!
    Load input vector from file and start teach process
    \param[in]  filename    file name
    \param[in]  stepscount  count of iterations
*/
void NNSimple::Teach(const char *filename, int stepscount){
    int i(0), j(0);
    ifstream fstr(filename);

    if(!fstr.is_open()){
        cout<<"Error opening file "<<filename<<endl;
        return;
    }

    double **voc=NULL;
    fstr>>sizey>>sizex;
    Init();

    voc=new double*[sizey];
    for(i=0;i<sizey;i++)
        voc[i]=new double[sizex];

    for(i=0;i<sizey;i++)
        for(j=0;j<sizex;j++)
            fstr>>voc[i][j];
    fstr.close();

    if(afunction==AF_THRESH)
        TeachThresh(voc,stepscount);
    else
        TeachSigma(voc,stepscount);

    for(i=0;i<sizey;i++){
        delete[] voc[i];
    }
    delete[] voc;
}
