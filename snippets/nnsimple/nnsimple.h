/*
 * Copyright 2016 Setin S.A.
*/

#include <iostream>
#include <math.h>
#include <fstream>

enum Activate_Function {AF_THRESH, AF_SIGMA};
//const double M_E=2.718281828459;

/*!
    Simple artificial neural network
    sizex - size of input signal vector
    sizey - size of output signal vector
    w - matrix of weights (sizex*sizey)
    x - input signal vector
    y - output signal vector
    n - speed of teaching
    e - inaccuracy
    s - border of significance
    afunction - activate function
*/
class NNSimple{
protected:
    double **w;
    double *x;
    double *y;
    double e;
    double s;
    int sizex, sizey;
    double n;
    Activate_Function afunction;
    virtual void CorrectWeight(int j,double d);
    virtual double AFunction(double nsum);
    virtual int    MaxY();
    virtual double MaxYVal();
    virtual void TeachThresh(double **voc, int stepscount);
    virtual void TeachSigma(double **voc, int stepscount);
    virtual void Init();
    virtual void Clear();
public:
    NNSimple(Activate_Function nfunce=AF_THRESH);
    virtual ~NNSimple();
    virtual int Process(double *inputx);
    virtual void PrintY(int precision=10);
    virtual int  GetY();
    virtual void SetE(double ne);
    virtual void SetN(double nn);
    virtual void Teach(const char *filename, int stepscount);
    virtual void LoadWeights(const char *filename);
    virtual void SaveWeights(const char *filename);
};
