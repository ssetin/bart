#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFileDialog>
#include <QDebug>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    audio=NULL;
    ba=NULL;
    dev=NULL;
    ui->setupUi(this);    
    ui->statusBar->addWidget(ui->lZoom);
    ui->statusBar->addWidget(ui->lPosition);

    ui->toolBar->addAction(QPixmap("icons/play_blue_button.png"), "Play", this, SLOT(on_play()));
    ui->graphicsView->SetZoomLabel(ui->lZoom);
    ui->graphicsView->SetPositionLabel(ui->lPosition);

}

MainWindow::~MainWindow()
{
    if(audio)
        delete audio;
    if(ba)
        delete ba;
    if(dev)
        delete dev;
    delete ui;
}

void MainWindow::on_actionOpen_triggered()
{
   QString fileName = QFileDialog::getOpenFileName(this, tr("Open File"), "",  tr("Wave (*.wav)"));

   if(fileName!=""){
       if(snd.LoadFromFile(fileName.toStdString().c_str())){
            ui->Log->append(fileName);
            ui->Log->append("Samplerate: "+QString::number(snd.Header()->sampleRate));
            ui->Log->append("Chanels: "+QString::number(snd.Header()->numChannels));
            ui->Log->append("SubChunk2Size: " +QString::number(snd.Header()->subchunk2Size));
            ui->Log->append("BitsPerSample: "+QString::number(snd.Header()->bitsPerSample));
            ui->Log->append("ByteRate: "+QString::number(snd.Header()->byteRate));
            ui->Log->append("Duration: "+QString::number(snd.Duration()));

            if(snd.Header()->numChannels>1){
                ui->Log->append("Warning! Use only 1 channel");
            }

            ui->graphicsView->AssignWave(&snd);

            format.setSampleRate(snd.Header()->sampleRate);
            format.setChannelCount(snd.Header()->numChannels);
            format.setSampleSize(snd.Header()->bitsPerSample);
            format.setCodec("audio/pcm");


       }else{
           ui->Log->append("Error opening file "+fileName+"\n"+snd.GetLastError());
       }
   }
}


void MainWindow::on_actionExit_triggered()
{
    exit(0);
}

void serialize(int* ar, int ar_size, QByteArray *result)
{
    QDataStream stream(result, QIODevice::WriteOnly);
    stream<<ar_size;

    for (int i=0; i<ar_size; i++)
        stream<<ar[i];
}


void MainWindow::processaudio(){
    ui->graphicsView->SetCursor(snd.Header()->byteRate/10);
    ui->graphicsView->Draw();
}

void MainWindow::on_play()
{
    ba = new QByteArray(snd.Size(),0);

    dev=new QBuffer();
    dev->setBuffer(ba);
    dev->open(QBuffer::ReadOnly);

    ba->replace(0,snd.Size() ,snd.Data());
    dev->seek(0);

    audio = new QAudioOutput(format, this);
    audio->setNotifyInterval(100);
    connect(audio, SIGNAL(notify()), this, SLOT(processaudio()));

    audio->start(dev);
}
