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
    ui->statusBar->addWidget(ui->lSelection);

    ui->toolBar->addAction(QPixmap("icons/audio_wave2.png"), "Normalize", this, SLOT(on_normalize()));
    ui->toolBar->addSeparator();
    ui->toolBar->addAction(QPixmap("icons/StopPlay-Blue-Button.png"), "Play from begin", this, SLOT(on_stopplay()));
    ui->toolBar->addAction(QPixmap("icons/stop_button.png"), "Stop", this, SLOT(on_stop()));
    ui->toolBar->addAction(QPixmap("icons/play_blue_button.png"), "Play", this, SLOT(on_play()));
    ui->toolBar->addAction(QPixmap("icons/pause_blue_button.png"), "Pause", this, SLOT(on_pause()));
    ui->toolBar->addSeparator();
    ui->toolBar->addWidget(ui->eInterval);
    ui->toolBar->addWidget(new QLabel(" : "));
    ui->toolBar->addWidget(ui->eIntervalOverlap);
    ui->toolBar->addAction(QPixmap("icons/time-icon.png"), "Intervals", this, SLOT(on_intervals()));
    ui->toolBar->addSeparator();
    ui->toolBar->addSeparator();
    ui->toolBar->addWidget(ui->eChar);
    ui->toolBar->addAction(QPixmap("icons/iChat-Alt-icon.png"), "Play interval", this, SLOT(on_playinterval()));
    ui->toolBar->addSeparator();
    ui->eChar->setDisabled(true);

    ui->graphicsView->SetZoomLabel(ui->lZoom);
    ui->graphicsView->SetPositionLabel(ui->lPosition);
    ui->graphicsView->SetSelectionLabel(ui->lSelection);
    ui->graphicsView->SetSoundCharEdit(ui->eChar);

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

void MainWindow::on_playinterval(){
    int from(0),to(0);
    ui->graphicsView->GetSelectedInterval(from,to);
    audio.Play(from,to);
}

void MainWindow::on_intervals(){
    snd.FormIntervals(ui->eInterval->text().toInt(),ui->eIntervalOverlap->text().toInt());
    ui->graphicsView->Draw(true);
}

void MainWindow::on_stopplay()
{
    audio.Play(0);
}

void MainWindow::on_play()
{
    audio.Play();
}

void MainWindow::on_pause()
{
    audio.Pause();
}

void MainWindow::on_stop(){
    audio.Stop();
}

void MainWindow::on_normalize(){
    snd.Normalize();
    ui->graphicsView->Draw(true);
}


void MainWindow::on_actionSave_alphabet_triggered()
{
    QString filter="Alphabet (*.json)";
    QString fileName = QFileDialog::getSaveFileName( this, tr("Save File"), "", filter, &filter);

    if(fileName!=""){
        snd.SaveIntervalsToFile(fileName.toStdString().c_str());
    }
}

void MainWindow::on_eChar_textEdited(const QString &arg1)
{
    ui->graphicsView->WriteSoundCharToSelected(arg1);
}

void MainWindow::on_actionLoad_alphabet_triggered()
{
    QString filter="Alphabet (*.json)";
    QString fileName = QFileDialog::getOpenFileName( this, tr("Open File"), "", filter, &filter);

    if(fileName!=""){
        if(snd.LoadIntervalsFromFile(fileName.toStdString().c_str()))
            ui->graphicsView->Draw();
        else
           ui->Log->append(fileName+" - "+snd.GetLastError());
    }
}



