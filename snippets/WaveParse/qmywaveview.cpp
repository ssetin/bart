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
    Zoom+=0.01*(float)scrollevent->delta();
    if(Zoom<(float)width()/(float)Size)
        Zoom=prezoom;
    if(Zoom>10)
        Zoom=10;

    Draw();


    QPointF remapped = mapToScene(scrollevent->pos());
    centerOn(remapped.x()*Zoom/prezoom,0);

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
    if(poslabel){
        poslabel->setText("Position: "+QString::number((int)CurrentPosition)+" : "+QString::fromStdString(snd->SampleNoToTime((int)CurrentPosition).ToStr() )) ;
    }
    Draw();
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
    Draw();
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


void QMyWaveView::Draw(){
    scene->clear();

    const float dpy = 0.8;

    QPen pen(Qt::green);
    QPen bpen(Qt::white);
    QPen curpen(Qt::blue);
    QPen axpen(Qt::white);
    axpen.setStyle(Qt::DashLine);

    float px(0), py(0), y(0), scX(0), scY(0), k(0);

    if(snd && snd->SamplesCount()>0){        
        k=((float)snd->SamplesCount()/width()/2);
        scY=dpy*height()/snd->Peak();
        scX=Zoom;

        for(float i=0;i<Size*Zoom;i+=scX*k){            
            y=snd->Data((int)(i/(scX)))*scY;
            //qDebug()<<(int)i<<". "<<snd->Data((int)i)<<" -> "<<y;
            scene->addLine(px,py,i,y,pen);
            px=i;
            py=y;
        }

    }    

    //axis
    if(Size>0){
        scene->addLine(0,0,Zoom*Size,0, axpen);
        scene->addLine(0,-dpy*height()*0.25,Zoom*Size,-dpy*height()*0.25, axpen);
        scene->addLine(0, dpy*height()*0.25,Zoom*Size,dpy*height()*0.25, axpen);
        scene->addLine(0,-dpy*height()*0.5,Zoom*Size,-dpy*height()*0.5, axpen);
        scene->addLine(0, dpy*height()*0.5,Zoom*Size,dpy*height()*0.5, axpen);
    }

    //time bar
    if(Size>0){
        scene->addRect(0,-height()/2+10,Size*Zoom,20,bpen, QBrush(Qt::gray));
        for(int i=0;i<Zoom*Size;i+=200){
            scene->addLine(i,-height()/2+10,i,height()/2-10,axpen);
            QGraphicsTextItem *atext=scene->addText(QString::number((int)(i/Zoom)));
            atext->setPos(i,-height()/2+10);
        }
    }

    //cursor
    if(Size>0){
        scene->addLine(CurrentPosition*Zoom,-height()/2+10,CurrentPosition*Zoom,height()/2-10, curpen);
    }

    show();
}
