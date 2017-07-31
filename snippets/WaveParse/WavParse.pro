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
    qmywaveview.cpp \
    ../fjay/fjson.cpp \
    caudiocontroller.cpp \
    cncontroller.cpp \
    ../nnsimple/nnsimple.cpp

CUDA_SOURCES += \
    ../nnsimple/cudahelper.cu

HEADERS  += mainwindow.h \
    ccharsound.h \
    qmywaveview.h \
    caudiocontroller.h \
    ../fjay/fjson.h \
    ui_mainwindow.h \
    cncontroller.h \
    ../nnsimple/nnsimple.h

FORMS    += mainwindow.ui


# CUDA settings <-- may change depending on your system
CUDA_SDK = "/usr/local/cuda-8.0/samples"    # Path to cuda SDK install
CUDA_DIR = "/usr/local/cuda-8.0"            # Path to cuda toolkit install
SYSTEM_NAME = 64                            # Depending on your system either 'Win32', 'x64', or 'Win64'
SYSTEM_TYPE = 64                            # '32' or '64', depending on your system
CUDA_ARCH=sm_21                             # Type of CUDA architecture, for example 'compute_10', 'compute_30', 'sm_30'
NVCC_OPTIONS = --use_fast_math


# include paths
INCLUDEPATH += $$CUDA_DIR/include \
               $$CUDA_SDK/common/inc/ \
               $$CUDA_SDK/../shared/inc/

# library directories
QMAKE_LIBDIR += $$CUDA_DIR/lib/$$SYSTEM_NAME \
                $$CUDA_DIR/lib$$SYSTEM_NAME
# Add the necessary libraries
LIBS += -lcuda -lcudart

# The following makes sure all path names (which often include spaces) are put between quotation marks
CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')


# Configuration of the Cuda compiler
CONFIG(debug, debug|release) {
    # Debug mode
    cuda_d.input = CUDA_SOURCES
    cuda_d.output = ${OBJECTS_DIR}/${QMAKE_FILE_BASE}_cuda.o
    cuda_d.commands = $$CUDA_DIR/bin/nvcc -D_DEBUG $$NVCC_OPTIONS $$CUDA_INC $$LIBS -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda_d.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda_d
}
else {
    # Release mode
    cuda.input = CUDA_SOURCES
    cuda.output = ${OBJECTS_DIR}/${QMAKE_FILE_BASE}_cuda.o
    cuda.commands = $$CUDA_DIR/bin/nvcc $$NVCC_OPTIONS $$CUDA_INC $$LIBS -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda
}
