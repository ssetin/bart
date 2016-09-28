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
    double **w;
    double *x;
    double *y;
    double e;
    double s;
    int sizex, sizey;
    double n;
    Activate_Function afunction;
    void CorrectWeight(int j,double d);
    double AFunction(double nsum);
    int    MaxY();
    double MaxYVal();
    void TeachThresh(double **voc, int stepscount);
    void TeachSigma(double **voc, int stepscount);
    void Init();
    void Clear();
public:
    NNSimple(Activate_Function nfunce=AF_THRESH);
    ~NNSimple();
    int Process(double *inputx);
    void PrintY(int precision=10);
    void SetE(double ne);
    void SetN(double nn);
    void Teach(const char *filename, int stepscount);
};
