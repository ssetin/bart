#include <iostream>
#include <math.h>
#include <fstream>

enum Correct_Rule {CR_DELTA, CR_SIGMA};

/*!
    Simple artificial neural network
    sizex - size of input signal vector
    sizey - size of output signal vector
    w - matrix of weights (sizex*sizey)
    x - input signal vector
    y - output signal vector
    n - speed of teaching
    rule - how we do correction of weights and what activate function we use
*/
class NNSimple{
    double **w;
    double *x;
    double *y;
    int    sizex, sizey;
    int    n;
    Correct_Rule rule;
    void CorrectWeight(int j,double d);
    double AFunction(double nsum);
    int    MaxY();
    double MaxYVal();
public:
    NNSimple(int xsize,int ysize, Correct_Rule nrule=CR_DELTA);
    ~NNSimple();
    int Process(double *inputx);
    void PrintY();
    void Teach(const char *filename, int stepscount);
};
