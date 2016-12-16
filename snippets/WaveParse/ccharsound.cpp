#include "ccharsound.h"
#include <stdio.h>
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
    iVolumeThresh=0.07;
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

int CCharSound:: Size(){
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

int CCharSound::BytesPerSample(){
    return ibytesPerSample;
}

CWaveTimer CCharSound::SampleNoToTime(unsigned int n){
    return CWaveTimer((int)(SampleNoToSecond(n)*1000));
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

int CCharSound::Data(unsigned int i){
    if(data==nullptr) return 0;
    CWaveSample amp;
    memcpy((void*)&amp.data[0],(void*)&data[i*ibytesPerSample*header.numChannels],ibytesPerSample);
    if(bSigned)
        return SampleToVal(amp);
    else
        return SampleToVal(amp)-(iPeak/2);
}

char* CCharSound::Data(){
    return data;
}

CSoundInterval* CCharSound::Interval(unsigned int i){
    if(i<iIntervalsCount)
        return &intervals[i];
    return nullptr;
}

void CCharSound::FormIntervals(unsigned int msec, unsigned int overlap){
    if(data==nullptr) return;
    unsigned int samples=msec*(header.byteRate/ibytesPerSample/header.numChannels)/1000;
    unsigned int ovsamples=overlap*(header.byteRate/ibytesPerSample/header.numChannels)/1000;
    iSamplesPerInterval=samples+2*ovsamples;
    unsigned int k(0);

    if(this->intervals)
        delete[] intervals;
    iIntervalsCount=0;

    for(unsigned int i=0;i<iSamplesCount;i++){
        if(abs(Data(i))>=iVolumeThresh*iPeak){
            i+=samples;
            iIntervalsCount++;
        }
    }
    if(iIntervalsCount==0) return;

    intervals=new CSoundInterval[iIntervalsCount];
    for(unsigned int i=0;i<iSamplesCount;i++){
        if(abs(Data(i))>=iVolumeThresh*iPeak){
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
    obj.Set("SamplesPerInterval",(int)iSamplesPerInterval);
    fjArray abc;

    for(unsigned int i=0;i<iIntervalsCount;i++){
        fjObject tobj;
        tobj.Set("Char",make_shared<fjString>(intervals[i].ch));
        tobj.Set("Begin",make_shared<fjInt>(intervals[i].begin));
        fjArray tarr;
        for(unsigned int k=intervals[i].begin;k<=intervals[i].end;k++)
            tarr.Add(Data(k));
        tobj.Set("Samples", make_shared<fjArray>(tarr));
        abc.Add(make_shared<fjObjValue>(tobj));
    }

    obj.Set("Alphabet", make_shared<fjArray>(abc));
    obj.SaveToFile(filename);
    return true;
}

bool CCharSound::LoadIntervalsFromFile(const char* filename){
    if(data==nullptr){
        strcpy(lasterror,"Can't load intervals data while audio data is empty!");
        return false;
    }

    fjObject obj;
    obj.LoadFromFile(filename);

    if(!obj.exists("SampleRate") || !obj.exists("BytesPerSample") || !obj.exists("SamplesPerInterval") || !obj.exists("Alphabet") ){
        strcpy(lasterror,"Wrong file format! SampleRate, BytesPerSample, SamplesPerInterval and Alphabet fields needed");
        return false;
    }
    if(obj["SampleRate"]->asInt()!=header.sampleRate){
        strcpy(lasterror,"Another SampleRate detected!");
        return false;
    }
    if(obj["BytesPerSample"]->asInt()!=ibytesPerSample){
        strcpy(lasterror,"Another BytesPerSample detected!");
        return false;
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

        //fjArray *subarr=(*tobj)["Samples"]->asfjArray().get();
        //qDebug()<<"ch: "<<intervals[k].ch.c_str()<<" begin: "<<(*tobj)["Begin"]->asInt();
        intervals[k].begin=(*tobj)["Begin"]->asInt();
        intervals[k].end=intervals[k].begin+iSamplesPerInterval;
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

    qDebug()<<"SubCh1Size="<<header.subchunk1Size<<" SizeofHeader="<<sizeof(WAVHEADER)<<" SamplesCount="<<iSamplesCount<<" subchunk2="<<header.subchunk2Size;
    qDebug()<<"sizeof(sample)"<<sizeof(CWaveSample)<<"peak: "<<iPeak<<" signed: "<<bSigned<<"Bits per sample:"<<header.bitsPerSample;

    return true;
}
