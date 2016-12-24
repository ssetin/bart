#include "caudiocontroller.h"
#include <QDebug>


CAudioController::CAudioController(QObject *parent): QObject(parent)
{
    audio=nullptr;
    ba=nullptr;
    dev=nullptr;
    view=nullptr;
    snd=nullptr;
    iNotifyDelay=400;
}

CAudioController::CAudioController(CCharSound *newsnd, QMyWaveView *newview, int NotifyDelay){
    audio=nullptr;
    ba=nullptr;
    dev=nullptr;
    view=nullptr;
    snd=nullptr;
    Init(newsnd, newview, NotifyDelay);
}

void CAudioController::PrepareBuffer(unsigned int from,int to){
    if(dev==nullptr || snd==nullptr) return;
    if(to==-1)
        to=snd->Size();
    else
        to*=snd->BytesPerSample();
    from*=snd->BytesPerSample();
    dev->close();
    dev->setData(&snd->Data()[from], to-from);
    dev->open(QBuffer::ReadOnly);
}

void CAudioController::Init(CCharSound *newsnd, QMyWaveView *newview, int NotifyDelay){
    if(newsnd==nullptr) return;
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
    PrepareBuffer();

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

void CAudioController::Play(unsigned int from,int to){
    if(view)
        view->SetCursor(from);
    if(dev){
        PrepareBuffer(from,to);
        dev->seek(0);
        // hmm....
        audio->setBufferSize(dev->size());
        if(audio)
            audio->start(dev);
    }    

    qDebug()<<"Play("<<from<<","<<to<<")";
}


void CAudioController::Play(){
    if(audio && dev){
        int p=dev->pos();
        PrepareBuffer();
        dev->seek(p);
        audio->start(dev);
    }
}


void CAudioController::Stop(){
    if(audio)
        audio->stop();    
    if(view)
        view->SetCursor(0);
    if(dev)
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

