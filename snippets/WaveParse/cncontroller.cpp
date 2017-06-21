#include "cncontroller.h"
#include <sys/types.h>
#include <dirent.h>

#include<QDebug>

CNController::CNController(Activate_Function nfunce): NNSimple(nfunce){
    n=0.00001;
    s=0.5;
    e=0.1;
    sAlphabet=nullptr;
}


CNController::~CNController(){
    if(sAlphabet)
        delete[] sAlphabet;
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
void CNController::TeachSigma(CSoundInterval *voc, int stepscount){
    int j(0), step(0);
    int steps(stepscount), ind(0);
    double d(1), T(0);
    srand(time(NULL));

    qDebug()<<"Steps="<<stepscount;

    for(step=0;step<steps;step++){
          ind=rand()%sizey;
          if(step%50000==0)
              qDebug()<<step<<"...";

          Process(voc[ind].data);

          for(j=0;j<sizey;j++){                
                if(ind==j)
                    T=1.0;
                else T=0.0;

                while(0.5*((T-y[j])*(T-y[j]))>e){
                    d=n*(T-y[j]) * y[j] * (1.0-y[j]);
                    if(abs(d)<=0.0000000001){
                        //cout<<"fail"<<endl;
                        break;
                    }
                    //qDebug()<<"d="<<d<<" j="<<j<<" y[j]="<<y[j]<<" T="<<T;
                    CorrectWeight(j,d*n);
                    Process(voc[ind].data);                    
                }
          }
    }
}

void CNController::TeachAlphabet(string filename){
    int stepscount(500000);

    snd.LoadIntervalsFromFile(filename.c_str());

    sizey=snd.IntervalsCount();
    qDebug()<<"teach "<<filename.c_str()<<"...";

    if(sizey==0) return;
    sAlphabet=new string[sizey];

    for(int i=0;i<sizey;i++)
        sAlphabet[i]=snd.GetIntervals()[i].ch;

    sizex=snd.SamplesPerInterval();
    Init();

    TeachSigma(snd.GetIntervals(),stepscount);

    qDebug()<<"teach "<<filename.c_str()<<" done.";
}

void CNController::TeachAlphabets(const string path){
    DIR *dir = opendir(path.c_str());
    string fname;
    if(dir){
        struct dirent *ent;
        while((ent = readdir(dir)) != NULL){
            fname=ent->d_name;
            if(fname.find(".json")!=string::npos){
                TeachAlphabet(path+"/"+ent->d_name);
            }
        }
    }
    else{
        qDebug()<<"Error opening directory "<<path.c_str();
    }

    closedir(dir);

}

void CNController::LoadSound(const string filename){
    snd.LoadFromFile(filename.c_str());
    snd.Normalize();
}

/*!
    Load weights matrix and alphabet from file
    \param[in]  filename    file name
*/
void CNController::LoadWeights(const char *filename){
    ifstream fstr(filename);

    Clear();
    fstr>>sizey>>sizex;
    Init();

    sAlphabet=new string[sizey];
    for(int i=0;i<sizey;i++)
        fstr>>sAlphabet[i];

    for(int i=0;i<sizex;i++)
        for(int j=0;j<sizey;j++)
            fstr>>w[i][j];

    fstr.close();
}

/*!
    Save weights matrix and alphbet to file
    \param[in]  filename    file name
*/
void CNController::SaveWeights(const char *filename){
    ofstream fstr(filename);
    fstr<<sizey<<" "<<sizex<<endl;

    for(int i=0;i<sizey;i++)
        fstr<<sAlphabet[i]<<endl;

    for(int i=0;i<sizex;i++)
        for(int j=0;j<sizey;j++)
            fstr<<w[i][j]<<" ";

    fstr.close();
}

string CNController::GetAnswer(){
    int a(-1);
    string res("");
    a=GetY();
    qDebug()<<"a = "<<a;
    if(sAlphabet && a>=0)
        res=sAlphabet[a];
    return res;
}


void CNController::Recognize(){
    unsigned int i(0), size(0);
    string result("");

    if(snd.Size()==0)
        return;

    snd.FormIntervals(70,20);

    if(snd.IntervalsCount()==0)
        return;

    size=snd.SamplesPerInterval();
    if(size==0)
        return;

    double *data=new double[size];

    for(i=0;i<snd.IntervalsCount();i++){
        snd.GetFloatDataFromInterval(i,data);
        Process(data);
        //PrintY();
        result+=GetAnswer();
    }

    qDebug()<<"Answer = "<<result.c_str();

    delete[] data;

}

