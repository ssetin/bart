#ifndef CCHARSOUND_H
#define CCHARSOUND_H

/*
    CCharSound

    Copyright 2016 Setin S.A.
*/

#include<string>


/*
    Constants
    1. Maximum volume level for normalizing in -dBFS
    2. Window size in ms for RMS calculating
    2. Windows overlapping size in ms
    3. Silent in dB
*/
#define dBFS_ALIGMENT   16
#define RMS_WINDOW      80
#define RMS_OVERLAP     40
#define SILENT          32


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
    short  ibytesPerSample;
    char   lasterror[64];
    char*  data;
    CSoundInterval* intervals;
    unsigned int iIntervalsCount;
    unsigned int iSamplesPerInterval;
    float fDynamicRange;
    int SampleToVal(CWaveSample &s);
    CWaveSample ValToSample(int value);
    void SetData(unsigned int pos, int value);
    double VolumeToDBFS(double volume);
    double DBFSToVolume(double dbfs);
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
    short BytesPerSample();
    float SampleNoToSecond(unsigned int n);
    CWaveTimer SampleNoToTime(unsigned int n);    
    void FormIntervals(unsigned int msec, unsigned int overlap);
    bool SaveIntervalsToFile(const char* filename);
    bool LoadIntervalsFromFile(const char* filename);
    void Normalize(short aligment=dBFS_ALIGMENT, short rmswindow=RMS_WINDOW, short rmsoverlap=RMS_OVERLAP, short silent=SILENT);
};

#endif // CCHARSOUND_H
