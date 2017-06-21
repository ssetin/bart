#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "ccharsound.h"
#include "caudiocontroller.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT
public:
    CCharSound snd;
    CAudioController audio;
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
private slots:
    void on_actionOpen_triggered();
    void on_actionExit_triggered();
    void on_play();
    void on_stop();
    void on_stopplay();
    void on_pause();
    void on_intervals();
    void on_playinterval();
    void on_normalize();
    void on_actionSave_alphabet_triggered();

    void on_eChar_textEdited(const QString &arg1);

    void on_actionLoad_alphabet_triggered();

    void on_actionTeach_network_triggered();

    void on_actionLoad_network_triggered();

public slots:

private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
