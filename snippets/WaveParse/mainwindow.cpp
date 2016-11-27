#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFileDialog>
#include <QDesktopWidget>
#include <QDebug>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);    
    ui->statusBar->addWidget(ui->lZoom);
    ui->statusBar->addWidget(ui->lPosition);

    ui->toolBar->addAction(QPixmap("icons/play_blue_button.png"), "Play", this, SLOT(on_play()));
    ui->toolBar->addAction(QPixmap("icons/pause_blue_button.png"), "Pause", this, SLOT(on_pause()));
    //ui->toolBar->addAction(QPixmap("icons/stop_blue_button.png"), "Stop", this, SLOT(on_stop()));

    ui->graphicsView->SetZoomLabel(ui->lZoom);
    ui->graphicsView->SetPositionLabel(ui->lPosition);

    move(qApp->desktop()->availableGeometry(this).center()-rect().center());



}

MainWindow::~MainWindow()
{
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
            audio.Init(&snd,ui->graphicsView);


       }else{
           ui->Log->append("Error opening file "+fileName+"\n"+snd.GetLastError());
           ui->graphicsView->Draw();
       }
   }
}


void MainWindow::on_actionExit_triggered()
{
    exit(0);
}

void MainWindow::on_play()
{
    audio.Play();
    //
}

void MainWindow::on_pause()
{
    audio.Pause();
    //
}
