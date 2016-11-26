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
#pragma pack(push, 1)
union CWaveSample{
    char data[4];
    unsigned short sample8:8;
    short sample16:16;
    int   sample24:24;
    int   sample32:32;
};
#pragma pack(pop)

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
    bool   bSigned;
    int    iPeak;
    int    ibytesPerSample;
    char   lasterror[64];
    char*  data;
    int SampleToVal(CWaveSample &s);
public:
    CCharSound();
    ~CCharSound();
    bool LoadFromFile(const char* filename);
    char* GetLastError();
    int Data(unsigned int i);
    char* Data();
    WAVHEADER* Header();
    float Duration();
    int SamplesCount();
    int Size();
    int Peak();
    bool IsSigned();
    float SampleNoToSecond(unsigned int n);
    CWaveTimer SampleNoToTime(unsigned int n);
};

#endif // CCHARSOUND_H
