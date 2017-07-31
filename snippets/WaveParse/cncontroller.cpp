#include "cncontroller.h"
#include <sys/types.h>
#include <dirent.h>
#include<chrono>

#include<QDebug>

CNController::CNController(Activate_Function nfunce, bool tryuse_cuda): NNSimple(nfunce, tryuse_cuda){
    n= 0.5;
    s= 0.5;
    e= 0.01;
    e0=1e-5;
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
    int row(0), step(0);
    int steps(stepscount), ind(0);
    double d(1.0), T(0.0);
    srand(time(NULL));


    for(step=0;step<steps;step++){
          ind=rand()%sizey;
          if(step%(stepscount/100)==0)
              qDebug()<<step<<" of "<<stepscount;

          Process(voc[ind].data);

          for(row=0;row<sizey;row++){
                if(ind==row)
                    T=1.0;
                else T=0.0;

                while(0.5*((T-y[row])*(T-y[row]))>e){
                    d=n*(T-y[row]) * y[row] * (1.0-y[row]);
                    if(abs(d)<=e0){
                        break;
                    }
                    //qDebug()<<"d="<<d<<" j="<<j<<" y[j]="<<y[j]<<" T="<<T;
                    CorrectWeight(row,d*n);
                    Process(voc[ind].data);                    
                }
          }
    }
}

/*!
    Process one alphabet
    \param[in]  filename file with chars and samples
*/
void CNController::TeachAlphabet(string filename){
    int stepscount(100);

    snd.LoadIntervalsFromFile(filename.c_str());

    sizey=snd.IntervalsCount();
    qDebug()<<"teach "<<filename.c_str()<<"...";

    if(sizey==0) return;
    sAlphabet=new string[sizey];

    for(int row=0;row<sizey;row++)
        sAlphabet[row]=snd.GetIntervals()[row].ch;

    sizex=snd.SamplesPerInterval();
    Init();

    TeachSigma(snd.GetIntervals(),stepscount);

    qDebug()<<"teach "<<filename.c_str()<<" done.";
}


/*!
    Processing few alphabets to generate general matrix of weights
    \param[in]  path path, where alphabets located
*/
void CNController::TeachAlphabets(const string path){

    chrono::time_point<chrono::high_resolution_clock> start, end;
    chrono::duration<double> elapsed;

    start = chrono::high_resolution_clock::now();

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

    end = chrono::high_resolution_clock::now();
    elapsed = end - start;
    elapsed=chrono::duration_cast<std::chrono::seconds>(elapsed);

    qDebug()<< "Done in " << elapsed.count()<< " seconds" << endl;
}

void CNController::LoadSound(const string filename){
    snd.LoadFromFile(filename.c_str());
    //snd.Normalize();
}

/*!
    Load weights matrix and alphabet from file
    \param[in]  filename    file name
*/
void CNController::LoadWeights(const char *filename){
    ifstream fstr(filename);

    Clear();
    fstr>>sizex>>sizey;
    Init();

    sAlphabet=new string[sizey];
    for(int row=0;row<sizey;row++)
        fstr>>sAlphabet[row];

    for(int row=0;row<sizey;row++)
        for(int col=0;col<sizex;col++)
            fstr>>w[row*sizey+col];

    fstr.close();
}

/*!
    Save weights matrix and alphbet to file
    \param[in]  filename    file name
*/
void CNController::SaveWeights(const char *filename){
    ofstream fstr(filename);

    fstr<<sizex<<" "<<sizey<<endl;

    for(int row=0;row<sizey;row++)
        fstr<<sAlphabet[row]<<endl;

    for(int row=0;row<sizey;row++){
        for(int col=0;col<sizex;col++)
            fstr<<w[row*sizey+col]<<' ';
    }

    fstr.close();
}

string CNController::GetAnswer(int idx){
    string res(" ");
    if(sAlphabet && idx>=0)
        res=sAlphabet[idx];
    return res;
}


string CNController::Recognize(){
    unsigned int i(0), size(0), res(-1);
    string result("");

    if(snd.Size()==0)
        return "";

    snd.FormIntervals(30,10);

    if(snd.IntervalsCount()==0)
        return "";

    size=snd.SamplesPerInterval();
    if(size==0)
        return "";

    double *data=new double[size];

    for(i=0;i<snd.IntervalsCount();i++){
        snd.GetFloatDataFromInterval(i,data);
        res=Process(data);
        result+=GetAnswer(res);
    }

    qDebug()<<"Answer = "<<result.c_str();

    delete[] data;

    return result;
}

