#-------------------------------------------------
#
# Project created by QtCreator 2016-09-30T16:24:57
#
#-------------------------------------------------

QT       += core gui multimedia

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = WavParse
TEMPLATE = app

CONFIG += c++11

SOURCES += main.cpp\
        mainwindow.cpp \
    ccharsound.cpp \
    qmywaveview.cpp

HEADERS  += mainwindow.h \
    ccharsound.h \
    qmywaveview.h

FORMS    += mainwindow.ui

