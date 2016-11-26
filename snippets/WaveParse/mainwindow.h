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
public slots:

private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
