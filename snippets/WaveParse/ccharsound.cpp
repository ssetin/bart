#include "ccharsound.h"
#include <stdio.h>
#include <math.h>
#include "../fjay/fjson.h"

#include<QDebug>

/*
    CWaveTimer
*/
CWaveTimer::CWaveTimer(){
    h=0;
    m=0;
    s=0;
    ms=0;
}

CWaveTimer::CWaveTimer(int msc){
    h=msc/1000/60/60;
    m=(msc-h*60*60*1000)/60/1000;
    s=(msc-h*60*60*1000-m*60*1000)/1000;
    ms=msc-h*60*60*1000-m*60*1000-s*1000;
}

CWaveTimer::~CWaveTimer(){
}

string CWaveTimer::ToStr(){
    string result;
    char buf[20];
    sprintf (buf, "%d:%d:%d.%d", h, m, s, ms);
    result=buf;
    return result;
}

/*
    CSoundInterval
*/
CSoundInterval::~CSoundInterval(){
    if(data)
        delete[] data;
}

/*
    CCharSound
*/

CCharSound::CCharSound()
{
    data=nullptr;
    intervals=nullptr;
    iSamplesCount=0;
    iIntervalsCount=0;
    iPeak=0;
    bSigned=false;
    ibytesPerSample=0;
    fDynamicRange=0;
}

CCharSound::~CCharSound()
{
    if(data)
        delete[] data;
    if(intervals)
        delete[] intervals;
}

WAVHEADER* CCharSound::Header(){
    return &header;
}

float CCharSound::Duration(){
    return fDuration;
}

unsigned int CCharSound::SamplesCount(){
    return iSamplesCount;
}

unsigned int CCharSound::IntervalsCount(){
    return iIntervalsCount;
}

unsigned int CCharSound::SamplesPerInterval(){
    return iSamplesPerInterval;
}

CSoundInterval* CCharSound::GetIntervals(){
    return intervals;
}

unsigned int CCharSound:: Size(){
    return header.subchunk2Size;
}

char* CCharSound::GetLastError(){
    return lasterror;
}

float CCharSound::SampleNoToSecond(unsigned int n){
    if(header.byteRate>0)
        return 1.0*header.blockAlign*n/header.byteRate;
    else return 0;
}

int CCharSound::Peak(){
    return iPeak;
}

bool CCharSound::IsSigned(){
    return bSigned;
}

short CCharSound::BytesPerSample(){
    return ibytesPerSample;
}

CWaveTimer CCharSound::SampleNoToTime(unsigned int n){
    return CWaveTimer((int)(SampleNoToSecond(n)*1000));
}

double CCharSound::VolumeToDBFS(double volume){
    return 20*log10(volume*1.22474487139);
}

double CCharSound::DBFSToVolume(double dbfs){
    return pow(10.0, dbfs * 0.05 )*0.81649658092;
}

int CCharSound::SampleToVal(CWaveSample &s){
    switch(header.bitsPerSample){
        case 8:
            return s.sample8;
        case 16:
            return s.sample16;
        case 24:
            return s.sample24;
        case 32:
            return s.sample32;
        default:
            return s.sample16;
    }
}

CWaveSample CCharSound::ValToSample(int value){
    CWaveSample s;

    switch(header.bitsPerSample){
        case 8:
            s.sample8=value;
            break;
        case 16:
            s.sample16=value;
            break;
        case 24:
            s.sample24=value;
            break;
        case 32:
            s.sample32=value;
            break;
        default:
            s.sample16=value;
            break;
    }
    return s;
}

int CCharSound::Data(unsigned int i){
    if(data==nullptr) return 0;
    CWaveSample amp;
    memcpy((void*)&amp.data[0],(void*)&data[i*ibytesPerSample*header.numChannels],ibytesPerSample);
    if(bSigned)
        return SampleToVal(amp);
    else
        return SampleToVal(amp)-(iPeak*0.5);
}

char* CCharSound::Data(){
    return data;
}

CSoundInterval* CCharSound::Interval(unsigned int i){
    if(i<iIntervalsCount)
        return &intervals[i];
    return nullptr;
}

void CCharSound::SetData(unsigned int pos, int value){
    if(data==nullptr) return;

    if(value>=iPeak*0.5)
        value=iPeak*0.5-1;
    else
    if(value<=-iPeak*0.5)
        value=-iPeak*0.5+1;

    CWaveSample amp;

    if(bSigned)
        amp=ValToSample(value);
    else
        amp=ValToSample(value+iPeak*0.5);

    memcpy((void*)&data[pos*ibytesPerSample*header.numChannels],(void*)&amp.data[0],ibytesPerSample);
}

void CCharSound::Normalize(short aligment, short rmswindow, short rmsoverlap, short silent){
    if(data==nullptr) return;
    unsigned int count(0), stepsamples(0), k(0), oversamples(0);
    unsigned int i(0);
    double dRMS(0), tmp(0);
    long double s(0);
    double dTargetMax=DBFSToVolume(fDynamicRange-aligment);
    double dSilent=DBFSToVolume(silent);
    stepsamples=rmswindow*(header.byteRate/ibytesPerSample/header.numChannels)/1000;
    oversamples=rmsoverlap*(header.byteRate/ibytesPerSample/header.numChannels)/1000;

    qDebug()<<"Normalazing. "<<"DynamicRange="<<fDynamicRange<<" dTargetMax="<<QString::number(dTargetMax,'f')
           <<" dSilent"<<dSilent<<" StepSamples="<<stepsamples<<" oversamples="<<oversamples;

    while(i<iSamplesCount){
        tmp=Data(i);
        if(abs(tmp)>dSilent){
            s+=tmp*tmp;
            count++;
        }

        if( k==stepsamples || i==iSamplesCount-1){
            if(count>0){
                dRMS=sqrt(s / (double)count);
                qDebug()<<"window: ["<<i-k<<", "<<i<<"]";
                for(unsigned int j=i-k;j<i;j++){
                    tmp=Data(j);
                    if(abs(tmp)>dSilent)
                        SetData(j,(dTargetMax/dRMS) * tmp);
                }
                count=0;
            }
            s=0;
            k=0;
            if(i>oversamples && i<iSamplesCount-oversamples)
                i-=oversamples;
        }
        i++;
        k++;
    }

    qDebug()<<"Done";
}

void CCharSound::GetFloatDataFromInterval(unsigned int i, double* array){
    if(array==nullptr) return;
    unsigned int k=intervals[i].begin;
    unsigned int j(0);
    for(unsigned int i=k;i<k+iSamplesPerInterval && i<Size();i++,j++){
        array[j]=Data(i);
    }
}

void CCharSound::FormIntervals(unsigned int msec, unsigned int overlap){
    if(data==nullptr) return;
    unsigned int samples=msec*(header.byteRate/ibytesPerSample/header.numChannels)/1000;
    unsigned int ovsamples=overlap*(header.byteRate/ibytesPerSample/header.numChannels)/1000;
    double dSilent=DBFSToVolume(SILENT);
    iSamplesPerInterval=samples+2*ovsamples;
    unsigned int k(0);

    if(this->intervals)
        delete[] intervals;
    iIntervalsCount=0;

    for(unsigned int i=0;i<iSamplesCount;i++){
        if(abs(Data(i))>dSilent){
            i+=samples;
            iIntervalsCount++;
        }
    }
    if(iIntervalsCount==0) return;

    intervals=new CSoundInterval[iIntervalsCount];
    for(unsigned int i=0;i<iSamplesCount;i++){
        if(abs(Data(i))>dSilent){
            if(i>ovsamples)
                intervals[k].begin=i-ovsamples;
            else
                intervals[k].begin=0;

            if(intervals[k].end+ovsamples<iSamplesCount)
                intervals[k].end=i+samples+ovsamples;
            else
                intervals[k].end=iSamplesCount-1;
            i+=samples;
            k++;
        }
    }

    qDebug()<<"Formed "<<iIntervalsCount<<" intervals. Samples count="<<samples;
}



bool CCharSound::SaveIntervalsToFile(const char* filename){
    fjObject obj;
    obj.Set("SampleRate", header.sampleRate);
    obj.Set("BytesPerSample", ibytesPerSample);
    obj.Set("PeakValue", iPeak);
    obj.Set("SamplesPerInterval",(int)iSamplesPerInterval);
    fjArray abc;

    for(unsigned int i=0;i<iIntervalsCount;i++){
        if(intervals[i].ch>""){
            fjObject tobj;
            tobj.Set("Char",make_shared<fjString>(intervals[i].ch));
            tobj.Set("Begin",make_shared<fjInt>(intervals[i].begin));
            fjArray tarr;
            for(unsigned int k=intervals[i].begin;k<=intervals[i].end;k++)
                tarr.Add((float)(Data(k)/(iPeak*0.5)));
                //tarr.Add(Data(k));
            tobj.Set("Samples", make_shared<fjArray>(tarr));
            abc.Add(make_shared<fjObjValue>(tobj));
        }
    }

    obj.Set("Alphabet", make_shared<fjArray>(abc));
    obj.SaveToFile(filename);
    return true;
}

bool CCharSound::LoadIntervalsFromFile(const char* filename){

    fjObject obj;
    obj.LoadFromFile(filename);

    if(!obj.exists("SampleRate") || !obj.exists("BytesPerSample") || !obj.exists("SamplesPerInterval") || !obj.exists("Alphabet") ){
        strcpy(lasterror,"Wrong file format! SampleRate, BytesPerSample, SamplesPerInterval and Alphabet fields needed");
        return false;
    }

    if(data!=nullptr){
        if(obj["SampleRate"]->asInt()!=header.sampleRate){
            strcpy(lasterror,"Another SampleRate detected!");
            return false;
        }
        if(obj["BytesPerSample"]->asInt()!=ibytesPerSample){
            strcpy(lasterror,"Another BytesPerSample detected!");
            return false;
        }
    }

    iSamplesPerInterval=obj["SamplesPerInterval"]->asInt();    

    if(intervals)
        delete[] intervals;

    fjArray *abcarr=(fjArray*)obj["Alphabet"].get();
    iIntervalsCount=abcarr->Size();
    intervals=new CSoundInterval[iIntervalsCount];

    qDebug()<<iIntervalsCount<<" intervals loaded";

    for(unsigned int k=0;k<abcarr->Size();k++){
        fjObjValue *tobj=(*abcarr)[k]->asfjObjValue().get();
        intervals[k].ch=(*tobj)["Char"]->asString();

        fjArray *subarr=(*tobj)["Samples"]->asfjArray().get();
        //qDebug()<<"ch: "<<intervals[k].ch.c_str()<<" begin: "<<(*tobj)["Begin"]->asInt();
        intervals[k].begin=(*tobj)["Begin"]->asInt();
        intervals[k].end=intervals[k].begin+iSamplesPerInterval;

        //load intervals data if sound data is empty
        if(data==nullptr){
            iPeak=obj["PeakValue"]->asInt();
            intervals[k].data=new double[subarr->Size()];
            for(unsigned int j=0;j<subarr->Size()-1;j++){
                intervals[k].data[j]=(*subarr)[j]->asFloat();
            }

        }
    }

    return true;
}

bool CCharSound::LoadFromFile(const char* filename){
    FILE *file;
    file = fopen(filename,"rb");
    if(file == nullptr) {
        strcpy(lasterror,"Can't load file");
        return false;
    }

    if(data)
        delete[] data;
    data=nullptr;
    iSamplesCount=0;
    iPeak=0;

    if(intervals)
        delete[] intervals;
    iIntervalsCount=0;
    intervals=nullptr;

    size_t aread(0);
    aread=fread((void*)&header, sizeof(WAVHEADER), 1, file);


    if(aread!=1){
        strcpy(lasterror,"Can't read wave header");
        fclose(file);
        return false;
    }

    if(memcmp(header.chunkId,"RIFF",4)!=0){
        strcpy(lasterror,"Wrong file format (RIFF)");
        fclose(file);
        return false;
    }
    if(memcmp(header.format,"WAVE",4)!=0){
        strcpy(lasterror,"Wrong file format (WAVE)");
        fclose(file);
        return false;
    }
    if(memcmp(header.subchunk1Id,"fmt ",4)!=0){
        strcpy(lasterror,"Wrong file format (fmt )");
        fclose(file);
        return false;
    }
    if(memcmp(header.subchunk2Id,"data",4)!=0){
        strcpy(lasterror,"Wrong file format (data section)");
        fclose(file);
        return false;
    }
    if(header.audioFormat!=1){
        strcpy(lasterror,"Wrong audio format. Support only PCM");
        fclose(file);
        return false;
    }

    ibytesPerSample=header.bitsPerSample / 8;

    iSamplesCount=header.subchunk2Size/ibytesPerSample;
    data=new char[header.subchunk2Size];
    aread=fread((void*)data, 1, header.subchunk2Size, file);
    if(aread!=header.subchunk2Size){
        strcpy(lasterror,"Can't read whole data section");
        fclose(file);
        return false;
    }
    fclose(file);

    CWaveSample amp;
    bSigned=false;
    if(header.bitsPerSample>8){
        for(unsigned int i=0;i<iSamplesCount;i++){
            amp.sample32=0;
            memcpy((void*)&amp.data[0],(void*)&data[i*ibytesPerSample*header.numChannels],ibytesPerSample);
            if(SampleToVal(amp)<0){
                bSigned=true;
                break;
            }
        }
    }

    iPeak=pow(2,header.bitsPerSample);

    fDuration=1.0 * header.subchunk2Size / ibytesPerSample / header.numChannels / header.sampleRate;
    fDynamicRange=VolumeToDBFS(iPeak*0.5);

    qDebug()<<"SubCh1Size="<<header.subchunk1Size<<" SizeofHeader="<<sizeof(WAVHEADER)<<" SamplesCount="<<iSamplesCount<<" subchunk2="<<header.subchunk2Size;
    qDebug()<<"peak: "<<iPeak<<" signed: "<<bSigned<<"Bits per sample:"<<header.bitsPerSample;

    return true;
}
