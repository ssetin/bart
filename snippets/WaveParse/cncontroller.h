#ifndef CNCONTROLLER_H
#define CNCONTROLLER_H

#include "ccharsound.h"
#include "../nnsimple/nnsimple.h"


class CNController:public NNSimple{
protected:
    CCharSound snd;
    string *sAlphabet;
    virtual void TeachAlphabet(string filename);
public:
    CNController(Activate_Function nfunce=AF_SIGMA);
    virtual ~CNController();
    virtual void LoadWeights(const char *filename);
    virtual void SaveWeights(const char *filename);
    virtual void TeachAlphabets(const string path);
    virtual void TeachSigma(CSoundInterval *voc, int stepscount);
    virtual void LoadSound(const string filename);
    virtual void Recognize();
    virtual string GetAnswer();
};

#endif // CNCONTROLLER_H
