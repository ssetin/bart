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
public:
    CAudioController(QObject *parent=NULL);
    CAudioController(CCharSound *newsnd, QMyWaveView *newview, int NotifyDelay=100);
    void Init(CCharSound *newsnd, QMyWaveView *newview, int NotifyDelay=100);
    void Clear();
    void Play();
    void Stop();
    void Pause();
    virtual ~CAudioController();
signals:
public slots:
    void processaudio();
    void audioStateChanged(QAudio::State newstate);
};

#endif // CAUDIOCONTROLLER_H
