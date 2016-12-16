#ifndef CAUDIOCONTROLLER_H
#define CAUDIOCONTROLLER_H

#include <QObject>
#include <QAudioOutput>
#include <QBuffer>
#include "ccharsound.h"
#include "qmywaveview.h"


class CAudioController: public QObject
{
    Q_OBJECT

    QAudioOutput *audio;
    QAudioFormat format;
    QByteArray *ba;
    QBuffer *dev;
    CCharSound *snd;
    QMyWaveView *view;
    QAudio::State state;
    int iNotifyDelay;
    void PrepareBuffer(unsigned int from=0,int to=-1);
public:
    CAudioController(QObject *parent=nullptr);
    CAudioController(CCharSound *newsnd, QMyWaveView *newview, int NotifyDelay=100);
    void Init(CCharSound *newsnd, QMyWaveView *newview, int NotifyDelay=100);
    void Clear();
    void Play();
    void Play(unsigned int from,int to=-1);
    void Stop();
    void Pause();
    virtual ~CAudioController();
signals:
public slots:
    void processaudio();    
    void audioStateChanged(QAudio::State newstate);
};

#endif // CAUDIOCONTROLLER_H
