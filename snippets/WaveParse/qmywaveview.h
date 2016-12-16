#ifndef QMYWAVEVIEW_H
#define QMYWAVEVIEW_H

#include <QWidget>
#include <QGraphicsView>
#include <QGraphicsScene>
#include <QLabel>
#include <QLineEdit>
#include <QGraphicsTextItem>
#include <QMouseEvent>
#include <QWheelEvent>
#include <vector>
#include <QGraphicsSceneWheelEvent>
#include "ccharsound.h"


/*
    QGraphicsIntervalItem
*/
class QGraphicsIntervalItem: public QGraphicsRectItem{
protected:
    virtual void Init();
public:
    QGraphicsIntervalItem(QGraphicsItem *parent=nullptr);
    QGraphicsIntervalItem(qreal x, qreal y, qreal width, qreal height, const QPen &pen);
    virtual ~QGraphicsIntervalItem();

    virtual void paint(QPainter * painter, const QStyleOptionGraphicsItem * option, QWidget * widget = 0);
signals:
public slots:

};



/*
    QMyWaveView
*/
class QMyWaveView: public QGraphicsView
{
    Q_OBJECT

    QGraphicsScene *scene;

    QGraphicsLineItem *lCursor;
    QGraphicsTextItem **tFrameNumbers;
    QGraphicsRectItem *rBar;
    QGraphicsLineItem **lAxises;
    QGraphicsLineItem **lVAxises;
    QGraphicsLineItem **lWaves;

    QGraphicsItem **rIntervals;

    QLabel *zoomlabel;
    QLabel *poslabel;
    QLabel *selectionlabel;
    QLineEdit *soundcharedit;
    bool eventFilter(QObject *obj, QEvent *event);
    void handleWheelOnGraphicsScene(QWheelEvent* scrollevent);
public:
    QMyWaveView(QWidget *parent=nullptr);
    virtual ~QMyWaveView();
    void AssignWave(CCharSound *newsnd);
    void SetCursor(float newpos);
    void IncCursor(float add);
    void SetZoom(float newzoom);

    void SetZoomLabel(QLabel *l);
    void SetPositionLabel(QLabel *l);
    void SetSelectionLabel(QLabel *l);
    void SetSoundCharEdit(QLineEdit *l);

    void Draw(bool redraw=true);
    void UnSelectItems();
    void WriteSoundCharToSelected(const QString &str);
    void GetSelectedInterval(int &begin, int &end);
signals:
public slots:
    void mousePressEvent(QMouseEvent * e);
protected:
    float fCurrentPosition;
    float fZoom;
    int   iSize;
    int   iSelectedInterval;
    CCharSound *snd;
    void resizeEvent(QResizeEvent *event);
};


#endif // QMYWAVEVIEW_H
