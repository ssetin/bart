/*
 * Copyright 2016 Setin S.A.
 *
 * TODO:
 * - supporting CUDA, optimize data structures and gpu memory usage
 * - few layers
*/

#include <iostream>
#include <math.h>
#include <fstream>

#include <cuda_runtime.h>
#include <helper_cuda.h>

#define USE_CUDA true
#ifdef _WIN32
    const double M_E=2.718281828459;
#endif

enum Activate_Function {AF_THRESH, AF_SIGMA};

extern "C" bool allocateobjects_cuda(const int sizex, const int sizey, double *w);
extern "C" bool setx_cuda(const int sizex, double* x);
extern "C" bool freeobjects_cuda();
extern "C" bool correctweight_cuda(double *w,const int sizex, const int sizey,const int row,const double d);

/*!
    Simple artificial neural network
    sizex - size of input signal vector
    sizey - size of output signal vector
    w - matrix(1 dim array) of weights (rows=sizey cols=sizex)
    x - input signal vector
    y - output signal vector
    n - speed of teaching
    e - inaccuracy
    e0 - "zero"
    s - sensitivity (border of significance)
    afunction - activate function
*/
class NNSimple{
protected:
    double *w;
    double *x;
    double *y;
    double e;
    double e0;
    double s;
    int sizex, sizey;
    double n;
    bool use_cuda;
    Activate_Function afunction;
    virtual void CorrectWeight(int row,double d);
    virtual double AFunction(double nsum);
    virtual int    MaxY();
    virtual double MaxYVal();
    virtual void TeachThresh(double **voc, int stepscount);
    virtual void TeachSigma(double **voc, int stepscount);
    virtual void Init();
    virtual void Clear();    
public:
    NNSimple(Activate_Function nfunce=AF_THRESH, bool tryuse_cuda=USE_CUDA);
    virtual ~NNSimple();
    virtual int Process(double *inputx);

    virtual void PrintY(int precision=4);
    virtual void PrintW(int precision=4);
    virtual int  GetY();    
    virtual void SetE(double ne);
    virtual void SetN(double nn);

    virtual double GetW(int col, int row);
    virtual void SetW(int col, int row, double value);

    virtual void SetSensitivity(double ns);
    virtual void Teach(const char *filename, int stepscount);
    virtual void LoadWeights(const char *filename);
    virtual void SaveWeights(const char *filename);
};
