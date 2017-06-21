#include "qmywaveview.h"
#include <QDebug>




/*
    QGraphicsIntervalItem
*/

QGraphicsIntervalItem::QGraphicsIntervalItem(QGraphicsItem *parent):QGraphicsRectItem(parent){
    Init();
}

QGraphicsIntervalItem::QGraphicsIntervalItem(qreal x, qreal y, qreal width, qreal height, const QPen &pen){
    Init();
    setRect(x,y,width,height);
    setPen(pen);
}


void QGraphicsIntervalItem::Init(){
    this->setCursor(Qt::PointingHandCursor);
    setOpacity(0.65);
    setZValue(1);
    setFlag(QGraphicsItem::ItemIsSelectable);
}

QGraphicsIntervalItem::~QGraphicsIntervalItem(){}

void QGraphicsIntervalItem::paint(QPainter * painter, const QStyleOptionGraphicsItem * option, QWidget * widget){
    if(this->isSelected()){
        setBrush(QBrush(pen().color()));
    }else{
        this->setBrush(QBrush());
    }
    QGraphicsRectItem::paint(painter,option,widget);
}



/*------------------------------*/
/*
    QMyWaveView
*/

QMyWaveView::QMyWaveView(QWidget *parent): QGraphicsView(parent){
    fCurrentPosition=0;
    fZoom=1;
    iSize=0;
    iSelectedInterval=-1;
    zoomlabel=nullptr;
    poslabel=nullptr;
    scene=nullptr;
    snd=nullptr;
    lCursor=nullptr;
    soundcharedit=nullptr;
    tFrameNumbers=nullptr;
    rBar=nullptr;
    lAxises=nullptr;
    lWaves=nullptr;
    rIntervals=nullptr;

    scene = new QGraphicsScene;
    scene->setBackgroundBrush(QBrush(Qt::black));
    setScene(scene);
    installEventFilter(this);

    Draw();
}

QMyWaveView::~QMyWaveView()
{
    //qDebug()<<"~QMyWaveView()";
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

void QMyWaveView::resizeEvent(QResizeEvent *event){
    QGraphicsView::resizeEvent(event);
}

void QMyWaveView::handleWheelOnGraphicsScene(QWheelEvent* scrollevent){
    float prezoom(fZoom);
    fZoom+=scrollevent->angleDelta().y()*0.0001;
    if(fZoom<fDefZoom || fZoom>2){
        fZoom=prezoom;
        return;
    }

    Draw(true);

    QPointF remapped = mapToScene(scrollevent->pos());
    centerOn(remapped.x()*fZoom/prezoom,0);

    if(zoomlabel)
        zoomlabel->setText("<b>Zoom:</b> "+QString::number(fZoom,10,3));

    scrollevent->accept();
}


void QMyWaveView::AssignWave(CCharSound *newsnd){
    if(newsnd==nullptr){
        iSize=0;
        return;
    }
    snd=newsnd;
    iSize=snd->SamplesCount();
    fZoom=(float)width()/(float)iSize;
    fDefZoom=fZoom;

    qDebug()<<" Size: "<<iSize<<" Zoom: "<<QString::number(fZoom,10,3)<<" Width: "<<width();

    Draw();

    if(zoomlabel)
        zoomlabel->setText("<b>Zoom:</b> "+QString::number(fZoom,10,3));

    if(poslabel)
        poslabel->setText("<b>Position:</b> 00:00:00.000");


}

void QMyWaveView::SetZoomLabel(QLabel *l){
    zoomlabel=l;
}

void QMyWaveView::SetPositionLabel(QLabel *l){
    poslabel=l;
}

void QMyWaveView::SetSelectionLabel(QLabel *l){
    selectionlabel=l;
}

void QMyWaveView::SetSoundCharEdit(QLineEdit *l){
    soundcharedit=l;
}

void QMyWaveView::SetCursor(float newpos){
    fCurrentPosition=newpos;
    if(poslabel){
        poslabel->setText("<b>Position:</b> "+QString::number((int)fCurrentPosition)+" : "+QString::fromStdString(snd->SampleNoToTime((int)fCurrentPosition).ToStr() )) ;
    }
    lCursor->setLine(fCurrentPosition*fZoom,-height()/2+10,fCurrentPosition*fZoom,height()/2-10);
    Draw(false);
}

void QMyWaveView::IncCursor(float add){
    if(fCurrentPosition+add>(float)snd->SamplesCount()){
        fCurrentPosition=snd->SamplesCount();
        //return;
    }else
        fCurrentPosition+=add;
    if(poslabel){
        poslabel->setText("<b>Position:</b> "+QString::number((int)fCurrentPosition)+" : "+QString::fromStdString(snd->SampleNoToTime((int)fCurrentPosition).ToStr() )) ;
    }
    lCursor->setLine(fCurrentPosition*fZoom,-height()/2+10,fCurrentPosition*fZoom,height()/2-10);
    Draw(false);
}

void QMyWaveView::SetZoom(float newzoom){
    fZoom=newzoom;
    Draw();
}

void QMyWaveView::mousePressEvent(QMouseEvent * e)
{
    if(snd==nullptr) return;

    QPointF mousePoint = mapToScene(e->pos());
    SetCursor(1.0*mousePoint.rx()/fZoom);

    //find interval
    QGraphicsItem *item=scene->itemAt(mousePoint, QGraphicsView::transform());

    if(item!=0){
       QGraphicsRectItem *rectItm;
       if((rectItm = dynamic_cast<QGraphicsIntervalItem*>(item)) ){
          UnSelectItems();
          rectItm->setSelected(true);
          iSelectedInterval=rectItm->data(0).toInt();
          CSoundInterval* csi=snd->Interval(iSelectedInterval);

          if(csi!=nullptr){
            if(soundcharedit){
                soundcharedit->setEnabled(true);
                soundcharedit->setText(QString::fromStdString(csi->ch));
            }
            if(selectionlabel){
                selectionlabel->setText("<b>Selection</b> "+QString::number(iSelectedInterval)+": "+QString::number(csi->begin)+":"+QString::number(csi->end));
            }
          }
       }else{
           soundcharedit->setEnabled(false);
           soundcharedit->setText("");
           iSelectedInterval=-1;
           selectionlabel->setText("");
       }
    }

}

void QMyWaveView::UnSelectItems(){
   for(QGraphicsItem *i:scene->selectedItems()){
       i->setSelected(false);
   }
}

void QMyWaveView::GetSelectedInterval(int &begin, int &end){
    for(QGraphicsItem *i:scene->selectedItems()){
        begin=snd->Interval(i->data(0).toInt())->begin;
        end=snd->Interval(i->data(0).toInt())->end;
    }
}

void QMyWaveView::WriteSoundCharToSelected(const QString &str){
   for(QGraphicsItem *i:scene->selectedItems()){
       snd->Interval(i->data(0).toInt())->ch=str.toStdString();
   }
}

/*
    Draw waves, grid, cursor
*/
void QMyWaveView::Draw(bool redraw){
    const float dpy = 0.8;
    float px(0), py(0), y(0), x(0), scX(0), scY(0), k(0);
    QPen curpen(Qt::blue);

    if(redraw){
        QPen pen(Qt::green);
        QPen bpen(Qt::white);
        QPen axpen(Qt::white);
        QPen ixpen(Qt::red);
        axpen.setStyle(Qt::DashLine);
        ixpen.setStyle(Qt::DashLine);
        pen.setWidth(2);

        scene->clear();
        int wavescount(0);

        if(snd && snd->SamplesCount()>0){
            k=32;//((float)snd->SamplesCount()/width()/4);
            scY=dpy*height()/snd->Peak();
            scX=fZoom;

            wavescount=snd->SamplesCount()/k;//iSize*fZoom/(scX*k)-1;
            lWaves=new QGraphicsLineItem*[wavescount];

            for(int i=0;i<wavescount;i++){
                x+=k;
                y=(float)snd->Data((int)x);
                lWaves[i]=scene->addLine(px*scX,py*scY,x*scX,y*scY,pen);
                px=x;
                py=y;
            }
        }


        if(iSize>0){
           //axis
           lAxises=new QGraphicsLineItem*[5];
           lAxises[0]=scene->addLine(0,0,fZoom*iSize,0, axpen);
           lAxises[1]=scene->addLine(0,-dpy*height()*0.25,fZoom*iSize,-dpy*height()*0.25, axpen);
           lAxises[2]=scene->addLine(0, dpy*height()*0.25,fZoom*iSize,dpy*height()*0.25, axpen);
           lAxises[3]=scene->addLine(0,-dpy*height()*0.5,fZoom*iSize,-dpy*height()*0.5, axpen);
           lAxises[4]=scene->addLine(0, dpy*height()*0.5,fZoom*iSize,dpy*height()*0.5, axpen);

           //time bar
           rBar=scene->addRect(0,-height()/2+10,iSize*fZoom,20,bpen, QBrush(Qt::gray));

           int fcount=fZoom*iSize/200+1;
           tFrameNumbers=new QGraphicsTextItem*[fcount];
           lVAxises=new QGraphicsLineItem*[fcount];
           for(int i=0;i<fcount;i++){
                lVAxises[i]=scene->addLine(i*200,-height()/2+10,i*200,height()/2-10,axpen);
                tFrameNumbers[i]=scene->addText(QString::number((int)(i*200/fZoom)));
                tFrameNumbers[i]->setPos(i*200,-height()/2+10);
            }

            //intervals
           if(snd->IntervalsCount()>0){
                rIntervals=new QGraphicsItem*[snd->IntervalsCount()];
                for(unsigned int i=0;i<snd->IntervalsCount();i++){
                    rIntervals[i]=new QGraphicsIntervalItem(snd->Interval(i)->begin*fZoom, -height()/2+32,
                                                            snd->Interval(i)->end*fZoom-snd->Interval(i)->begin*fZoom,height()-40,ixpen);
                    scene->addItem(rIntervals[i]);
                    rIntervals[i]->setData(0,i);
                    if(iSelectedInterval==i)
                        rIntervals[i]->setSelected(true);
                }

           }

            //cursor
            lCursor=scene->addLine(fCurrentPosition*fZoom,-height()/2+10,fCurrentPosition*fZoom,height()/2-10, curpen);

            //fix scene size
            QRectF srect=sceneRect();
            srect.setWidth(rBar->rect().width());
            setSceneRect(srect);
       }

    }else{
        if(lCursor)
            lCursor->update();
    }

}
