/*
 * Copyright 2016 Setin S.A.
 *
 * TODO:
 * - supporting CUDA, optimize data structures and gpu memory usage
 * - few layers
*/

#include <iostream>
#include <cmath>
#include <fstream>

#include <cuda_runtime.h>
#include <helper_cuda.h>

#define USE_CUDA true
#ifdef _WIN32
    const double M_E=2.718281828459;
#endif

enum Activate_Function {AF_THRESH, AF_SIGMA};

extern "C" bool allocatedata_cuda(const int sizex, const int sizey);
extern "C" bool setw_cuda(const int sizex,const int sizey, double *w);
extern "C" bool getw_cuda(const int sizex,const int sizey, double *w);
extern "C" bool setx_cuda(const int sizex,const int sizey, double *x);
extern "C" bool teachsigma_cuda(const int stepscount, const int sizex, const int sizey, double *y,double n, double e, double e0);
extern "C" void freedata_cuda();


/*!
    Simple artificial neural network
    sizex - size of input signal vector
    sizey - size of output signal vector
    w - matrix(1 dim array) of weights (rows=sizey cols=sizex)
    x - array of input signal vectors
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
    virtual void CorrectWeight(const int row, const int idx, const double d);
    virtual double AFunction(double nsum);
    virtual int    MaxY();
    virtual double MaxYVal();
    virtual void TeachThresh(int stepscount);
    virtual void TeachSigma(int stepscount);
    virtual void Init();
    virtual void Clear();
    virtual bool CheckCuda();
public:
    NNSimple(Activate_Function nfunce=AF_THRESH, bool tryuse_cuda=USE_CUDA);
    virtual ~NNSimple();
    virtual int Process(double *inputx);
    virtual int Process(const int idx);

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
