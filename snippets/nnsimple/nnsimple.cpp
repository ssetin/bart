#include <nnsimple.h>
#include <ctime>

using namespace std;

NNSimple::NNSimple(int xsize,int ysize, Correct_Rule nrule){
    sizex=xsize;
    sizey=ysize;
    n=0.1;
    rule=nrule;
    y=new double[sizey];
    w=new double*[sizex];

    for(int i=0;i<sizex;i++){
        w[i]=new double[sizey];
    }
    for(int i=0;i<sizex;i++)
        for(int j=0;j<sizey;j++)
            w[i][j]=0;
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

void NNSimple::PrintY(){
    cout.precision(17);
    for(int i=0;i<sizey;i++){
        cout<<"y["<<i<<"] = "<<y[i]<<endl;
    }
}

double NNSimple::AFunction(double nsum){
    if(rule==CR_DELTA)
        return nsum>5?1:0;
    return 1/(1+pow(M_E,-nsum));
}


/*
    Delta rule
    δ = (T - Y)
    Δi = ηδxi
    w(n+1) = w(n) + Δi
*/
void NNSimple::CorrectWeight(int j,double d){
    for(int i=0;i<sizex;i++){
        if(rule==CR_DELTA)
            w[i][j]+=d*x[i];
        else
            w[i][j]+=n*d*y[j];
    }
}


NNSimple::~NNSimple(){
    if(y!=NULL)
        delete[] y;
    for(int i=0;i<sizex;i++){
        delete[] w[i];
    }
    delete[] w;
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
    return MaxY();
}

void NNSimple::Teach(const char *filename, int stepscount){
    ifstream fstr(filename);
    int vcount(0), vlen(0), i(0), j(0), step(0);
    int steps(stepscount), ind(0);

    double **voc=NULL;
    fstr>>vcount>>vlen;
    voc=new double*[vcount];
    for(i=0;i<vcount;i++)
        voc[i]=new double[vlen];

    for(i=0;i<vcount;i++)
        for(j=0;j<vlen;j++)
            fstr>>voc[i][j];
    fstr.close();


    srand(time(NULL));
    /*
        Delta rule
        δ = (T - Y)
        Δi = ηδxi
        w(n+1) = w(n) + Δi
    */
    for(step=0;step<steps;step++){        
        ind=rand()%sizey;
        Process(voc[ind]);
        for(j=0;j<sizey;j++){
            double d(0), T(0);
            if(ind==j)
                T=1;
            //d=y[j]*(1-y[j])*(d-y[j]);
            d=T-y[j];
            CorrectWeight(j,d);
            //cout<<"step = "<<step<<" ind = "<<ind<<" delta = "<<delta<<endl;
        }
    }

    for(i=0;i<vcount;i++){
        delete[] voc[i];
    }
    delete[] voc;
}
