#include "qmywaveview.h"
#include <QDebug>



/*------------------------------*/

QMyWaveView::QMyWaveView(QWidget *parent): QGraphicsView(parent){
    CurrentPosition=0;
    Zoom=1;
    Size=0;
    zoomlabel=NULL;
    poslabel=NULL;
    scene=NULL;
    snd=NULL;    
    lCursor=NULL;
    tFrameNumbers=NULL;
    rBar=NULL;
    lAxises=NULL;
    lWaves=NULL;

    scene = new QGraphicsScene;
    scene->setBackgroundBrush(QBrush(Qt::black));
    setScene(scene);    
    installEventFilter(this);

    Draw();
}

QMyWaveView::~QMyWaveView()
{
    if(scene){
        delete scene;
    }
}


bool QMyWaveView::eventFilter(QObject *obj, QEvent *event) {
  if (obj==this)
  if (event->type() == QEvent::Wheel) {
     handleWheelOnGraphicsScene(static_cast<QWheelEvent*> (event));
     return true;
  }
  return false;
}

void QMyWaveView::handleWheelOnGraphicsScene(QWheelEvent* scrollevent)
{
    float prezoom(Zoom);
    Zoom+=0.0001*(float)scrollevent->angleDelta().y();
    if(Zoom<(float)width()/(float)Size)
        Zoom=(float)width()/(float)Size;
    if(Zoom>10)
        Zoom=10;

    Draw(true);

    QPointF remapped = mapToScene(scrollevent->pos());
    centerOn(remapped.x()*Zoom/prezoom,0);

    if(zoomlabel)
        zoomlabel->setText("Zoom: "+QString::number(Zoom,10,2));

    scrollevent->accept();
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
    if(poslabel){
        poslabel->setText("Position: "+QString::number((int)CurrentPosition)+" : "+QString::fromStdString(snd->SampleNoToTime((int)CurrentPosition).ToStr() )) ;
    }    
    lCursor->setLine(CurrentPosition*Zoom,-height()/2+10,CurrentPosition*Zoom,height()/2-10);
    Draw(false);
}

void QMyWaveView::IncCursor(float add){
    if(CurrentPosition+add>(float)snd->SamplesCount()){
        CurrentPosition=snd->SamplesCount();
        //return;
    }else
        CurrentPosition+=add;
    if(poslabel){
        poslabel->setText("Position: "+QString::number((int)CurrentPosition)+" : "+QString::fromStdString(snd->SampleNoToTime((int)CurrentPosition).ToStr() )) ;
    }
    lCursor->setLine(CurrentPosition*Zoom,-height()/2+10,CurrentPosition*Zoom,height()/2-10);
    Draw(false);
}

void QMyWaveView::SetZoom(float newzoom){
    Zoom=newzoom;
    Draw();
}

void QMyWaveView::mousePressEvent(QMouseEvent * e)
{
    if(snd==NULL) return;
    QPoint remapped = mapFromParent( e->pos() );
    if ( rect().contains( remapped ) )
    {
         QPointF mousePoint = mapToScene( remapped );
         SetCursor(1.0*mousePoint.rx()/Zoom);
    }
}

void QMyWaveView::Draw(bool redraw){
    const float dpy = 0.8;
    float px(0), py(0), y(0), x(0), scX(0), scY(0), k(0);
    QPen curpen(Qt::blue);

    if(redraw){
        QPen pen(Qt::green);
        QPen bpen(Qt::white);
        QPen axpen(Qt::white);
        axpen.setStyle(Qt::DashLine);

        scene->clear();
        int wavescount(0);

        if(snd && snd->SamplesCount()>0){
            k=((float)snd->SamplesCount()/width()/16);
            scY=dpy*height()/snd->Peak();
            scX=Zoom;

            wavescount=Size*Zoom/(scX*k)-1;
            lWaves=new QGraphicsLineItem*[wavescount];

            for(int i=0;i<wavescount;i++){
                x+=k;
                y=(float)snd->Data((int)x);
                lWaves[i]=scene->addLine(px*scX,py*scY,x*scX,y*scY,pen);
                px=x;
                py=y;
            }
        }


        if(Size>0){
           //axis
           lAxises=new QGraphicsLineItem*[5];
           lAxises[0]=scene->addLine(0,0,Zoom*Size,0, axpen);
           lAxises[1]=scene->addLine(0,-dpy*height()*0.25,Zoom*Size,-dpy*height()*0.25, axpen);
           lAxises[2]=scene->addLine(0, dpy*height()*0.25,Zoom*Size,dpy*height()*0.25, axpen);
           lAxises[3]=scene->addLine(0,-dpy*height()*0.5,Zoom*Size,-dpy*height()*0.5, axpen);
           lAxises[4]=scene->addLine(0, dpy*height()*0.5,Zoom*Size,dpy*height()*0.5, axpen);

           //time bar
           rBar=scene->addRect(0,-height()/2+10,Size*Zoom,20,bpen, QBrush(Qt::gray));

           int fcount=Zoom*Size/200+1;
           tFrameNumbers=new QGraphicsTextItem*[fcount];
           lVAxises=new QGraphicsLineItem*[fcount];
           for(int i=0;i<fcount;i++){
                lVAxises[i]=scene->addLine(i*200,-height()/2+10,i*200,height()/2-10,axpen);
                tFrameNumbers[i]=scene->addText(QString::number((int)(i*200/Zoom)));
                tFrameNumbers[i]->setPos(i*200,-height()/2+10);
            }

            //cursor
            lCursor=scene->addLine(CurrentPosition*Zoom,-height()/2+10,CurrentPosition*Zoom,height()/2-10, curpen);
       }

    }else{
        if(lCursor)
            lCursor->update();
    }

}
