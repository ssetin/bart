#include "qmywaveview.h"
#include<QDebug>

QMyWaveView::QMyWaveView(QWidget *parent): QGraphicsView(parent){
    CurrentPosition=0;
    Zoom=1;
    Size=0;
    zoomlabel=NULL;
    poslabel=NULL;
    scene=NULL;
    snd=NULL;    
    scene = new QGraphicsScene;
    scene->setBackgroundBrush(QBrush(Qt::black));
    setScene(scene);
    Draw();
    installEventFilter(this);
}

QMyWaveView::~QMyWaveView()
{
    if(scene)
        delete scene;
}


bool QMyWaveView::eventFilter(QObject *obj, QEvent *event) {
  if (event->type() == QEvent::Wheel) {
     handleWheelOnGraphicsScene(static_cast<QWheelEvent*> (event));
     return true;
  }
  return false;
}

void QMyWaveView::handleWheelOnGraphicsScene(QWheelEvent* scrollevent)
{
    float prezoom(Zoom);
    Zoom+=0.001*(float)scrollevent->delta();
    if(Zoom*snd->SamplesCount()<width())
        Zoom=prezoom;
    if(Zoom>10)
        Zoom=10;
    Draw();

    QPointF remapped = mapToScene(scrollevent->pos());
    centerOn(remapped.x()/prezoom*Zoom,0);
    //centerOn(CurrentPosition,0);

    if(zoomlabel)
        zoomlabel->setText("Zoom: "+QString::number(Zoom,10,2));
}


void QMyWaveView::AssignWave(CCharSound *newsnd){
    if(newsnd==NULL){
        Size=0;
        return;
    }
    snd=newsnd;
    Size=snd->SamplesCount();
    Zoom=(float)width()/(float)Size;

    qDebug()<<" Size: "<<Size<<" Zoom: "<<QString::number(Zoom,10,2)<<" Width: "<<width();
    Draw();

    if(zoomlabel)
        zoomlabel->setText("Zoom: "+QString::number(Zoom,10,2));

    if(poslabel)
        poslabel->setText("Position: 00:00:00.000");


}

void QMyWaveView::SetZoomLabel(QLabel *l){
    zoomlabel=l;
}

void QMyWaveView::SetPositionLabel(QLabel *l){
    poslabel=l;
}

void QMyWaveView::SetCursor(float newpos){
    CurrentPosition=newpos;
    Draw();
}

void QMyWaveView::SetZoom(float newzoom){
    Zoom=newzoom;
    Draw();
}

void QMyWaveView::mousePressEvent(QMouseEvent * e)
{
    QPoint remapped = mapFromParent( e->pos() );
    if ( rect().contains( remapped ) )
    {
         QPointF mousePoint = mapToScene( remapped );
         CurrentPosition=1.0*mousePoint.rx()/Zoom;
         Draw();
    }
}

void QMyWaveView::showEvent(QShowEvent *event){
    //Draw();
}

void QMyWaveView::Draw(){
    scene->clear();

    QPen pen(Qt::green);
    QPen bpen(Qt::white);

    float px(0), py(0), y(0), scX(0), scY(0), peak(0);

    if(snd && snd->SamplesCount()>0){
        peak=256*(snd->Header()->bitsPerSample/8);
        //k=(float)snd->SamplesCount()/32.0;
        scY=0.9*height()/peak;
        scX=Zoom;

        for(float i=0;i<Size*Zoom;i+=scX){
            y=(float)snd->Data((int)(i/scX))*scY;
            //qDebug()<<snd->Data((int)i)<<" -> "<<y;
            scene->addLine(px,py,i,y,pen);
            px=i;
            py=y;
        }

    }

    if(snd && snd->SamplesCount() )
    if(snd->Header()->byteRate>0)
        qDebug()<<"Position: "<<CurrentPosition;

    //cursor
    if(Size>0){
        scene->addLine(CurrentPosition*Zoom,-height()/2+10,CurrentPosition*Zoom,height()/2-10, bpen);
        if(poslabel){
            poslabel->setText("Position: "+QString::fromStdString(snd->SampleNoToTime((int)CurrentPosition).ToStr() )) ;
        }
    }

    //axis
    if(Size>0)
        scene->addLine(0,0,Zoom*Size,0, bpen);

    //time bar
    if(Size>0){
        scene->addRect(0,-height()/2+10,Size*Zoom,20,bpen, QBrush(Qt::gray));
        for(int i=0;i<Zoom*Size;i+=200){
            scene->addLine(i,-height()/2+10,i,-height()/2+25,bpen);
            QGraphicsTextItem *atext=scene->addText(QString::number(i));
            atext->setPos(i,-height()/2+10);
        }
    }

    show();
}
