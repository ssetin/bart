/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 5.5.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QTextBrowser>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>
#include "qmywaveview.h"

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QAction *actionOpen;
    QAction *actionExit;
    QAction *actionSave_alphabet;
    QAction *actionLoad_alphabet;
    QWidget *centralWidget;
    QVBoxLayout *verticalLayout;
    QTabWidget *tabWidget;
    QWidget *tab;
    QGridLayout *gridLayout;
    QMyWaveView *graphicsView;
    QWidget *tab_2;
    QGridLayout *gridLayout_2;
    QTextBrowser *Log;
    QLabel *lZoom;
    QLabel *lPosition;
    QLabel *lSelection;
    QLineEdit *eChar;
    QLineEdit *eInterval;
    QLineEdit *eIntervalOverlap;
    QMenuBar *menuBar;
    QMenu *menuFile;
    QMenu *menuEdit;
    QStatusBar *statusBar;
    QToolBar *toolBar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(1101, 629);
        MainWindow->setMinimumSize(QSize(973, 585));
        QFont font;
        font.setFamily(QStringLiteral("Tahoma"));
        font.setPointSize(12);
        MainWindow->setFont(font);
        QIcon icon;
        icon.addFile(QStringLiteral("icons/application wave.png"), QSize(), QIcon::Normal, QIcon::Off);
        MainWindow->setWindowIcon(icon);
        MainWindow->setDocumentMode(false);
        MainWindow->setTabShape(QTabWidget::Rounded);
        actionOpen = new QAction(MainWindow);
        actionOpen->setObjectName(QStringLiteral("actionOpen"));
        actionExit = new QAction(MainWindow);
        actionExit->setObjectName(QStringLiteral("actionExit"));
        actionSave_alphabet = new QAction(MainWindow);
        actionSave_alphabet->setObjectName(QStringLiteral("actionSave_alphabet"));
        actionLoad_alphabet = new QAction(MainWindow);
        actionLoad_alphabet->setObjectName(QStringLiteral("actionLoad_alphabet"));
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        verticalLayout = new QVBoxLayout(centralWidget);
        verticalLayout->setSpacing(6);
        verticalLayout->setContentsMargins(11, 11, 11, 11);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        verticalLayout->setContentsMargins(2, 2, 2, 2);
        tabWidget = new QTabWidget(centralWidget);
        tabWidget->setObjectName(QStringLiteral("tabWidget"));
        tabWidget->setMinimumSize(QSize(931, 0));
        tabWidget->setAutoFillBackground(true);
        tabWidget->setTabPosition(QTabWidget::South);
        tabWidget->setTabShape(QTabWidget::Rounded);
        tabWidget->setUsesScrollButtons(true);
        tabWidget->setDocumentMode(true);
        tabWidget->setTabsClosable(false);
        tabWidget->setMovable(false);
        tabWidget->setTabBarAutoHide(false);
        tab = new QWidget();
        tab->setObjectName(QStringLiteral("tab"));
        gridLayout = new QGridLayout(tab);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        gridLayout->setContentsMargins(0, 0, 0, 0);
        graphicsView = new QMyWaveView(tab);
        graphicsView->setObjectName(QStringLiteral("graphicsView"));
        QSizePolicy sizePolicy(QSizePolicy::Ignored, QSizePolicy::Expanding);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(graphicsView->sizePolicy().hasHeightForWidth());
        graphicsView->setSizePolicy(sizePolicy);
        graphicsView->viewport()->setProperty("cursor", QVariant(QCursor(Qt::IBeamCursor)));
        graphicsView->setFrameShape(QFrame::Box);
        graphicsView->setFrameShadow(QFrame::Plain);
        graphicsView->setLineWidth(1);
        graphicsView->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
        graphicsView->setSizeAdjustPolicy(QAbstractScrollArea::AdjustToContents);
        QBrush brush(QColor(0, 0, 0, 0));
        brush.setStyle(Qt::NoBrush);
        graphicsView->setBackgroundBrush(brush);
        graphicsView->setInteractive(true);
        graphicsView->setRenderHints(QPainter::Antialiasing|QPainter::TextAntialiasing);
        graphicsView->setCacheMode(QGraphicsView::CacheNone);
        graphicsView->setTransformationAnchor(QGraphicsView::NoAnchor);
        graphicsView->setResizeAnchor(QGraphicsView::NoAnchor);
        graphicsView->setViewportUpdateMode(QGraphicsView::SmartViewportUpdate);
        graphicsView->setRubberBandSelectionMode(Qt::ContainsItemShape);
        graphicsView->setOptimizationFlags(QGraphicsView::DontClipPainter);

        gridLayout->addWidget(graphicsView, 0, 0, 1, 1);

        tabWidget->addTab(tab, QString());
        tab_2 = new QWidget();
        tab_2->setObjectName(QStringLiteral("tab_2"));
        gridLayout_2 = new QGridLayout(tab_2);
        gridLayout_2->setSpacing(6);
        gridLayout_2->setContentsMargins(11, 11, 11, 11);
        gridLayout_2->setObjectName(QStringLiteral("gridLayout_2"));
        gridLayout_2->setSizeConstraint(QLayout::SetMaximumSize);
        gridLayout_2->setContentsMargins(0, 0, 0, 0);
        Log = new QTextBrowser(tab_2);
        Log->setObjectName(QStringLiteral("Log"));

        gridLayout_2->addWidget(Log, 0, 0, 1, 1);

        tabWidget->addTab(tab_2, QString());

        verticalLayout->addWidget(tabWidget);

        lZoom = new QLabel(centralWidget);
        lZoom->setObjectName(QStringLiteral("lZoom"));
        QFont font1;
        font1.setPointSize(10);
        font1.setBold(false);
        font1.setWeight(50);
        lZoom->setFont(font1);
        lZoom->setTextFormat(Qt::RichText);
        lZoom->setScaledContents(true);
        lZoom->setMargin(0);

        verticalLayout->addWidget(lZoom);

        lPosition = new QLabel(centralWidget);
        lPosition->setObjectName(QStringLiteral("lPosition"));
        lPosition->setEnabled(true);
        lPosition->setFont(font1);
        lPosition->setTextFormat(Qt::RichText);
        lPosition->setScaledContents(true);

        verticalLayout->addWidget(lPosition);

        lSelection = new QLabel(centralWidget);
        lSelection->setObjectName(QStringLiteral("lSelection"));
        QFont font2;
        font2.setPointSize(10);
        lSelection->setFont(font2);
        lSelection->setTextFormat(Qt::RichText);

        verticalLayout->addWidget(lSelection);

        eChar = new QLineEdit(centralWidget);
        eChar->setObjectName(QStringLiteral("eChar"));
        eChar->setMaximumSize(QSize(60, 50));
        eChar->setLayoutDirection(Qt::LeftToRight);

        verticalLayout->addWidget(eChar);

        eInterval = new QLineEdit(centralWidget);
        eInterval->setObjectName(QStringLiteral("eInterval"));
        eInterval->setMaximumSize(QSize(60, 50));
        eInterval->setLayoutDirection(Qt::LeftToRight);
        eInterval->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        verticalLayout->addWidget(eInterval);

        eIntervalOverlap = new QLineEdit(centralWidget);
        eIntervalOverlap->setObjectName(QStringLiteral("eIntervalOverlap"));
        eIntervalOverlap->setMaximumSize(QSize(60, 50));
        eIntervalOverlap->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        verticalLayout->addWidget(eIntervalOverlap);

        MainWindow->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(MainWindow);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 1101, 21));
        menuBar->setDefaultUp(false);
        menuBar->setNativeMenuBar(true);
        menuFile = new QMenu(menuBar);
        menuFile->setObjectName(QStringLiteral("menuFile"));
        menuEdit = new QMenu(menuBar);
        menuEdit->setObjectName(QStringLiteral("menuEdit"));
        MainWindow->setMenuBar(menuBar);
        statusBar = new QStatusBar(MainWindow);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        MainWindow->setStatusBar(statusBar);
        toolBar = new QToolBar(MainWindow);
        toolBar->setObjectName(QStringLiteral("toolBar"));
        MainWindow->addToolBar(Qt::TopToolBarArea, toolBar);

        menuBar->addAction(menuFile->menuAction());
        menuBar->addAction(menuEdit->menuAction());
        menuFile->addAction(actionOpen);
        menuFile->addSeparator();
        menuFile->addAction(actionLoad_alphabet);
        menuFile->addAction(actionSave_alphabet);
        menuFile->addSeparator();
        menuFile->addAction(actionExit);

        retranslateUi(MainWindow);

        tabWidget->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "Wave parser", 0));
#ifndef QT_NO_ACCESSIBILITY
        MainWindow->setAccessibleName(QString());
#endif // QT_NO_ACCESSIBILITY
        actionOpen->setText(QApplication::translate("MainWindow", "Open wave", 0));
        actionExit->setText(QApplication::translate("MainWindow", "Exit", 0));
        actionSave_alphabet->setText(QApplication::translate("MainWindow", "Save alphabet", 0));
        actionLoad_alphabet->setText(QApplication::translate("MainWindow", "Load alphabet", 0));
        tabWidget->setTabText(tabWidget->indexOf(tab), QApplication::translate("MainWindow", "Wave", 0));
        tabWidget->setTabText(tabWidget->indexOf(tab_2), QApplication::translate("MainWindow", "Log", 0));
        lZoom->setText(QApplication::translate("MainWindow", "<b>Zoom</b>: 1", 0));
        lPosition->setText(QApplication::translate("MainWindow", "<b>Position:</b>", 0));
        lSelection->setText(QString());
#ifndef QT_NO_TOOLTIP
        eChar->setToolTip(QApplication::translate("MainWindow", "Sound", 0));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_TOOLTIP
        eInterval->setToolTip(QApplication::translate("MainWindow", "Interval, msec", 0));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_STATUSTIP
        eInterval->setStatusTip(QString());
#endif // QT_NO_STATUSTIP
#ifndef QT_NO_WHATSTHIS
        eInterval->setWhatsThis(QString());
#endif // QT_NO_WHATSTHIS
        eInterval->setText(QApplication::translate("MainWindow", "70", 0));
#ifndef QT_NO_TOOLTIP
        eIntervalOverlap->setToolTip(QApplication::translate("MainWindow", "Overlap, msec", 0));
#endif // QT_NO_TOOLTIP
        eIntervalOverlap->setText(QApplication::translate("MainWindow", "20", 0));
        menuFile->setTitle(QApplication::translate("MainWindow", "File", 0));
        menuEdit->setTitle(QApplication::translate("MainWindow", "Edit", 0));
        toolBar->setWindowTitle(QApplication::translate("MainWindow", "toolBar", 0));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
