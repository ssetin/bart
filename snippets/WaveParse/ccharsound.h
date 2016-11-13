#ifndef CCHARSOUND_H
#define CCHARSOUND_H

/*
    CCharSound

    Copyright 2016 Setin S.A.
*/

#include<string>

using namespace std;

/*
    Wave header struct
*/
struct WAVHEADER
{
    char chunkId[4];
    int chunkSize;
    char format[4];
    char subchunk1Id[4];
    int subchunk1Size;
    short audioFormat;
    short numChannels;
    int sampleRate;
    int byteRate;
    short blockAlign;
    short bitsPerSample;
    char subchunk2Id[4];
    int subchunk2Size;
};

/*
    Union for byte<->int value convertations
*/
union CWaveSample{
    char data[4];
    int  sample{0};
};

/*
    CWaveTimer class
*/
class CWaveTimer{
    int h;
    int m;
    int s;
    int ms;
public:
    CWaveTimer();
    CWaveTimer(int msec);
    ~CWaveTimer();
    string ToStr();
};


/*
    CCharSound class
*/
class CCharSound
{
    WAVHEADER header;
    float  fDuration;
    int    iSamplesCount;
    char   lasterror[64];
    char*  data;
public:
    CCharSound();
    ~CCharSound();
    bool LoadFromFile(const char* filename);
    char* GetLastError();
    int Data(unsigned int i);
    char* Data(){return data;}
    WAVHEADER* Header();
    float Duration();
    int SamplesCount();
    int Size();
    float SampleNoToSecond(unsigned int n);
    CWaveTimer SampleNoToTime(unsigned int n);
};

#endif // CCHARSOUND_H
