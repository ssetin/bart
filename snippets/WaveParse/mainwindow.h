#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "ccharsound.h"
#include <QAudioOutput>
#include <QBuffer>


namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT
public:
    CCharSound snd;
    QAudioOutput *audio;
    QAudioFormat format;
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
private slots:
    void on_actionOpen_triggered();
    void on_actionExit_triggered();
    void on_play();
    void processaudio();
public slots:

private:
    QByteArray *ba;
    QBuffer *dev;
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
