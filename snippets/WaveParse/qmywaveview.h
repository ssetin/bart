#ifndef QMYWAVEVIEW_H
#define QMYWAVEVIEW_H

#include <QWidget>
#include <QGraphicsView>
#include <QGraphicsScene>
#include <QLabel>
#include <QGraphicsTextItem>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QGraphicsSceneWheelEvent>
#include "ccharsound.h"

class QMyWaveView: public QGraphicsView
{
    //Q_OBJECT
    QGraphicsScene *scene;
    QLabel *zoomlabel;
    QLabel *poslabel;
    bool eventFilter(QObject *obj, QEvent *event);
    void handleWheelOnGraphicsScene(QWheelEvent* scrollevent);
public:
    QMyWaveView(QWidget *parent=0);
    virtual ~QMyWaveView();
    void AssignWave(CCharSound *newsnd);
    void SetCursor(float newpos);
    void SetZoom(float newzoom);
    void SetZoomLabel(QLabel *l);
    void SetPositionLabel(QLabel *l);
    void Draw();
signals:
public slots:
    void mousePressEvent(QMouseEvent * e);
protected:
    float CurrentPosition;    
    float Zoom;
    int   Size;
    CCharSound *snd;
};

#endif // QMYWAVEVIEW_H
