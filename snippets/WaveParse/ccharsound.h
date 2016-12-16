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
    unsigned int subchunk2Size;
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
    CSoundInterval
*/
struct CSoundInterval{
    unsigned int begin;
    unsigned int end;
    string ch;
    CSoundInterval():begin(-1),end(-1),ch(""){}
};


/*
    CCharSound class
*/
class CCharSound
{
    WAVHEADER header;
    float  fDuration;
    unsigned int iSamplesCount;
    bool   bSigned;
    int    iPeak;
    int    ibytesPerSample;
    char   lasterror[64];
    char*  data;
    CSoundInterval* intervals;
    unsigned int iIntervalsCount;
    unsigned int iSamplesPerInterval;
    float iVolumeThresh;
    int SampleToVal(CWaveSample &s);
public:
    CCharSound();
    ~CCharSound();
    bool LoadFromFile(const char* filename);
    char* GetLastError();
    int Data(unsigned int i);
    char* Data();
    CSoundInterval* Interval(unsigned int i);
    WAVHEADER* Header();
    float Duration();
    unsigned int SamplesCount();
    unsigned int IntervalsCount();
    int Size();    
    int Peak();
    bool IsSigned();
    int BytesPerSample();
    float SampleNoToSecond(unsigned int n);
    CWaveTimer SampleNoToTime(unsigned int n);    
    void FormIntervals(unsigned int msec, unsigned int overlap);
    bool SaveIntervalsToFile(const char* filename);
    bool LoadIntervalsFromFile(const char* filename);
};

#endif // CCHARSOUND_H
