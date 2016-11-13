#include "ccharsound.h"
#include <stdio.h>
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
    data=NULL;
    iSamplesCount=0;
}

CCharSound::~CCharSound()
{
    if(data)
        delete data;
}

WAVHEADER* CCharSound::Header(){
    return &header;
}

float CCharSound::Duration(){
    return fDuration;
}

int CCharSound::SamplesCount(){
    return iSamplesCount;
}

int CCharSound:: Size(){
    return header.subchunk2Size;
}

char* CCharSound::GetLastError(){
    return lasterror;
}

float CCharSound::SampleNoToSecond(unsigned int n){
    if(header.byteRate>0)
        return 1.0*n/header.byteRate;
    else return 0;
}


CWaveTimer CCharSound::SampleNoToTime(unsigned int n){
    return CWaveTimer((int)(SampleNoToSecond(n)*1000));
}

int CCharSound::Data(unsigned int i){
    if(data==NULL) return 0;
    CWaveSample amp;
    memcpy((void*)&amp.data[0],(const void*)&data[i*(header.bitsPerSample/8)*header.numChannels],header.bitsPerSample/8);
    if(header.bitsPerSample==8)
        return amp.sample-128;
    else
        return amp.sample;
}

bool CCharSound::LoadFromFile(const char* filename){
    FILE *file;
    file = fopen(filename,"r");
    if(file == NULL) {
        strcpy(lasterror,"Can't load file");
        return false;
    }
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

    iSamplesCount=header.subchunk2Size/(header.bitsPerSample / 8);
    data=new char[header.subchunk2Size];
    fread((void*)data, 1, header.subchunk2Size, file);
    //samples=new CWaveSample[iSamplesCount];
    /*for(int i=0;i<header.subchunk2Size;i+=(header.bitsPerSample / 8)){
        aread=fread((void*)samples[i].data,(header.bitsPerSample / 8), 1 , file);
    }*/

    fclose(file);

    fDuration=1.0 * header.subchunk2Size / (header.bitsPerSample / 8) / header.numChannels / header.sampleRate;

    return true;
}
