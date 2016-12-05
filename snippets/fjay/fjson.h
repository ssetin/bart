/*
 * Copyright 2016 Setin S.A.
 * fantasies...
*/

#ifndef FJSON_H
#define FJSON_H

#include<string>
#include<vector>
#include<memory>
#include<iostream>
#include<fstream>

using namespace std;

class fjObject;
class fjArray;
class fjInt;
class fjFloat;
class fjString;

enum eStringIs{SI_STRING, SI_INT, SI_FLOAT, SI_ARRAY, SI_OBJECT};

eStringIs DetectType(string &value);
string trimstr(const string& str);

/*
 * fjValue - base fj abstract type
*/
class fjValue{
protected:
public:
    fjValue(){}

    virtual string asString(bool quotes=false)const=0;
    virtual shared_ptr<fjArray> asfjArray()const=0;
    virtual float asFloat()const=0;
    virtual shared_ptr<fjObject>asfjObject()const=0;
    virtual int asInt()const=0;

    virtual bool IsArray()const=0;
    virtual bool IsObject()const=0;
    virtual size_t Size()const=0;

    virtual void Add(shared_ptr<fjValue>){}
    virtual void Add(const string&){}
    virtual void Add(const int&){}
    virtual void Add(const float&){}
    virtual void Delete(unsigned int){}

    friend ostream& operator<<(ostream& s, const fjValue *t);

    virtual ~fjValue(){}
};

/*
 * fjPair - pair of name and value
*/
struct fjPair{
    string name;
    shared_ptr<fjValue> value;
    fjPair()=delete;
    fjPair(string pname, shared_ptr<fjValue> nvalue){
        name=pname;
        value=nvalue;
    }  
    ~fjPair(){        
    }
};

/*
 * fjString - string fj type
*/
class fjString: public fjValue {
    string value;
public:
    fjString();
    fjString(const fjString &val);
    fjString(fjString &&val);
    fjString(const char *val);
    fjString(const string &val);
    fjString& operator=(const fjString &a);
    fjString& operator=(fjString &&a);

    virtual string asString(bool quotes=false) const;
    virtual shared_ptr<fjArray> asfjArray() const;
    virtual float asFloat()const;
    virtual shared_ptr<fjObject> asfjObject() const;
    virtual int asInt() const;

    virtual bool IsArray()const;
    virtual bool IsObject()const;
    virtual size_t Size()const;

    friend ostream& operator<<(ostream& s, const fjString &t);
    virtual ~fjString();
};

/*
 * fjInt - int fj type
*/
class fjInt: public fjValue {
    int value;
public:
    fjInt();
    fjInt(const fjInt &val);
    fjInt(fjInt &&val);
    fjInt(const int &val);
    fjInt& operator=(const fjInt &a);
    fjInt& operator=(fjInt &&a);

    virtual string asString(bool quotes=false) const;
    virtual shared_ptr<fjArray> asfjArray() const;
    virtual float asFloat() const;
    virtual shared_ptr<fjObject> asfjObject() const;
    virtual int asInt() const;

    virtual bool IsArray()const;
    virtual bool IsObject()const;
    virtual size_t Size()const;

    friend ostream& operator<<(ostream& s, const fjInt &t);
    virtual ~fjInt();
};

/*
 * fjFloat - float fj type
*/
class fjFloat: public fjValue {
    float value;
public:
    fjFloat();
    fjFloat(const fjFloat &val);
    fjFloat(fjFloat &&val);
    fjFloat(const float &val);
    fjFloat& operator=(const fjFloat &a);
    fjFloat& operator=(fjFloat &&a);

    virtual string asString(bool quotes=false)const;
    virtual shared_ptr<fjArray> asfjArray()const;
    virtual float asFloat() const;
    virtual shared_ptr<fjObject> asfjObject()const;
    virtual int asInt()const;

    virtual bool IsArray()const;
    virtual bool IsObject()const;
    virtual size_t Size()const;

    friend ostream& operator<<(ostream& s, const fjFloat &t);
    virtual ~fjFloat();
};


/*
    fjObjValue - value of fjObject type
*/
class fjObjValue: public fjValue {
    shared_ptr<fjObject> value;
public:
    fjObjValue();
    fjObjValue(const fjObjValue &val);
    fjObjValue(fjObjValue &&val);
    fjObjValue(const fjObject &val);
    fjObjValue& operator=(const fjObjValue &a);
    fjObjValue& operator=(fjObjValue &&a);

    virtual string asString(bool quotes=false)const;
    virtual shared_ptr<fjArray> asfjArray()const;
    virtual float asFloat() const;
    virtual shared_ptr<fjObject> asfjObject()const;
    virtual int asInt()const;

    virtual bool IsArray()const;
    virtual bool IsObject()const;
    virtual size_t Size()const;

    friend ostream& operator<<(ostream& s, const fjObjValue &t);
    virtual ~fjObjValue();
};


/*
 * fjArray - array of fj values
*/
class fjArray: public fjValue {
    vector<shared_ptr<fjValue>> value;
    void AddValue(string &value);
public:
    fjArray();
    fjArray(const fjArray &val);
    fjArray(fjArray &&val);
    fjArray(const string &jsonstring);
    fjArray& operator=(const fjArray &a);
    fjArray& operator=(fjArray &&a);

    virtual void Set(const string &jstr);
    virtual void Add(shared_ptr<fjValue> val);
    virtual void Add(const int &val);
    virtual void Add(const float &val);
    virtual void Add(const string &val);
    virtual void Delete(unsigned int ind);
    virtual void Clear();
    shared_ptr<fjValue>& operator[](unsigned int i);

    virtual string asString(bool quotes=false)const;
    virtual shared_ptr<fjArray> asfjArray()const;
    virtual float asFloat()const;
    virtual shared_ptr<fjObject> asfjObject()const;
    virtual int asInt()const;

    virtual bool IsArray()const;
    virtual bool IsObject()const;
    virtual size_t Size()const;

    friend ostream& operator<<(ostream& s, const fjArray &t);

    virtual ~fjArray();
};

/*
 * fjObject - fj object
*/
class fjObject{
protected:    
    void Parse(const string jstr);
    void AddValue(const string &name, string &value);
public:
    vector<fjPair> pair;
    fjObject();
    fjObject(const fjObject &obj);
    fjObject(fjObject &&obj);
    fjObject(const string &jsonstring);
    fjObject& operator=(const fjObject &obj);
    fjObject& operator=(fjObject &&obj);

    void Init(const string &jsonstring);
    void Clear();

    shared_ptr<fjValue>& operator[](string name);
    virtual bool exists(string name);
    virtual void Set(string name, shared_ptr<fjValue> value);
    virtual void Set(string name, const int &value);
    virtual void Set(string name, const string &value);
    virtual void Set(string name, const float &value);
    size_t Size();

    friend ostream& operator<<(ostream& s, const fjObject &t);
    void SaveToFile(const string filename);
    void LoadFromFile(const string filename);

    virtual  ~fjObject();
};



#endif // FJSON_H






