/*
 * Copyright 2017 Setin S.A.
 *
 * TODO:
 * - processing few alphabets
 * - result correction
*/

#ifndef CNCONTROLLER_H
#define CNCONTROLLER_H

#include "ccharsound.h"
#include "../nnsimple/nnsimple.h"


class CNController:public NNSimple{
protected:
    CCharSound snd;
    string *sAlphabet;    
    virtual string GetAnswer(int idx);
    virtual void TeachSigma(CSoundInterval *voc, int stepscount);
public:
    CNController()=delete;
    CNController(Activate_Function nfunce=AF_SIGMA,bool tryuse_cuda=true);
    virtual ~CNController();
    virtual void LoadWeights(const char *filename);
    virtual void SaveWeights(const char *filename);
    virtual void TeachAlphabet(string filename);
    virtual void TeachAlphabets(const string path);    
    virtual void LoadSound(const string filename);
    virtual string Recognize();
};

#endif // CNCONTROLLER_H
