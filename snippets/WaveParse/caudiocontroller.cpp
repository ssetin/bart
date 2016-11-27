#include "caudiocontroller.h"
#include <QDebug>


CAudioController::CAudioController(QObject *parent): QObject(parent)
{
    audio=NULL;
    ba=NULL;
    dev=NULL;
    view=NULL;
    snd=NULL;
    iNotifyDelay=500;
}

CAudioController::CAudioController(CCharSound *newsnd, QMyWaveView *newview, int NotifyDelay){
    audio=NULL;
    ba=NULL;
    dev=NULL;
    view=NULL;
    snd=NULL;
    Init(newsnd, newview, NotifyDelay);
}

void CAudioController::Init(CCharSound *newsnd, QMyWaveView *newview, int NotifyDelay){
    if(newsnd==NULL) return;
    snd=newsnd;
    view=newview;
    iNotifyDelay=NotifyDelay;

    Clear();

    format.setSampleRate(snd->Header()->sampleRate);
    format.setChannelCount(snd->Header()->numChannels);
    format.setSampleSize(snd->Header()->bitsPerSample);

    if(snd->IsSigned())
        format.setSampleType(QAudioFormat::SampleType::SignedInt);
    else
        format.setSampleType(QAudioFormat::SampleType::UnSignedInt);
    format.setCodec("audio/pcm");

    ba = new QByteArray(snd->Size(),0);
    ba->setRawData(snd->Data(),snd->Size());
    dev=new QBuffer();
    dev->setBuffer(ba);
    dev->open(QBuffer::ReadOnly);

    //ba->replace( 0,snd->Size() ,snd->Data());
    dev->seek(0);

    audio = new QAudioOutput(format);
    audio->setNotifyInterval(iNotifyDelay);
    connect(audio, SIGNAL(notify()), this, SLOT(processaudio()));
    connect(audio, SIGNAL(stateChanged(QAudio::State)), this, SLOT(audioStateChanged(QAudio::State)));


}

void CAudioController::Clear(){
    if(dev)
        delete dev;
    if(ba)
        delete ba;
    if(audio)
        delete audio;
}


void CAudioController::Play(){
    if(view)
        view->SetCursor(0);
    if(dev && audio){
        dev->seek(0);
        audio->start(dev);        
    }
    qDebug()<<"Play()";
}

void CAudioController::Stop(){
    if(audio)
        audio->stop();
    dev->seek(0);
    qDebug()<<"Stop()";
}

void CAudioController::Pause(){
    if(audio)
        audio->stop();
    qDebug()<<"Pause()";
}

CAudioController::~CAudioController(){
    Clear();
}

void CAudioController::audioStateChanged(QAudio::State newstate)
{
    qDebug() << "audioStateChanged from" << state << "to" << newstate;

    if (QAudio::IdleState == newstate && dev->pos() >= snd->Size() ) {
        Stop();
    } else {
        state=newstate;
    }
}



void CAudioController::processaudio(){
    if(view){        
        view->SetCursor(dev->pos()/(snd->Header()->bitsPerSample/8));
    }
}

