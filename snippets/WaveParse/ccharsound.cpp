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
    iPeak=0;
    bSigned=false;
    ibytesPerSample=0;
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
        return 1.0*header.blockAlign*n/header.byteRate;
    else return 0;
}

int CCharSound::Peak(){
    return iPeak;
}

bool CCharSound::IsSigned(){
    return bSigned;
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
    if(data==NULL) return 0;
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

bool CCharSound::LoadFromFile(const char* filename){
    FILE *file;
    file = fopen(filename,"r");
    if(file == NULL) {
        strcpy(lasterror,"Can't load file");
        return false;
    }

    if(data)
        delete[] data;
    data=NULL;
    iSamplesCount=0;
    iPeak=0;

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
        for(int i=0;i<iSamplesCount;i++){
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
