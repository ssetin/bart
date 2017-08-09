#include "cncontroller.h"
#include <sys/types.h>
#include <dirent.h>
#include <chrono>

#include<QDebug>

CNController::CNController(Activate_Function nfunce, bool tryuse_cuda): NNSimple(nfunce, tryuse_cuda){
    n=0.5;
    s=0.6;
    e=0.001;
    e0=1e-5;
    sAlphabet=nullptr;
}


CNController::~CNController(){
    if(sAlphabet)
        delete[] sAlphabet;
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

    for(unsigned int i=0;i<snd.IntervalsCount();i++){
        CSoundInterval *voc=snd.GetIntervals();
        for(int j=0;j<sizex;j++)
            x[i*snd.IntervalsCount()+j]=voc[i].data[j];
    }

    if(CheckCuda()){
        if(allocatedata_cuda(sizex, sizey)){

            setw_cuda(sizex, sizey, w);
            setx_cuda(sizex, sizey, x);

            if(teachsigma_cuda(stepscount, sizex, sizey, y, n, e, e0))
                getw_cuda(sizex, sizey, w);

            freedata_cuda();
        }
    } else {
        TeachSigma(stepscount);
    }

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

/*!
    Load sound data from wave file
    \param[in]  filename    file name
*/
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

/*!
    Get the real answer according to alphabet
    \param[in]  idx    index of element in alphabet
*/
string CNController::GetAnswer(int idx){
    string res(" ");
    if(sAlphabet && idx>=0)
        res=sAlphabet[idx];
    return res;
}

/*!
    Recognize signals from loaded file
*/
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

