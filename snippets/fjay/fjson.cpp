/*
 * Copyright 2016 Setin S.A.
 * fantasies...
*/

#include"fjson.h"

using namespace std;

/*
    helpers
*/
string trimstr(const string& str)
{
    size_t first = str.find_first_not_of(" \n");
    if (string::npos == first)
    {
        return str;
    }
    size_t last = str.find_last_not_of(" \n");
    return str.substr(first, (last - first + 1));
}

eStringIs DetectType(string &value){
    value=trimstr(value);
    unsigned short digits(0), points(0);

    if(value[0]=='[' && value[value.size()-1]==']')
        return SI_ARRAY;
    if(value[0]=='{' && value[value.size()-1]=='}')
        return SI_OBJECT;
    if(value[0]=='"' && value[value.size()-1]=='"'){
        value=value.substr(1,value.size()-2);
        return SI_STRING;
    }

    for(unsigned int i=0;i<value.size();i++){
        if(value[i]=='.')
            points++;
        if(value[i]>='0' && value[i]<='9')
            digits++;
    }

    if(points+digits==value.size()){
        if(points==1 && digits>0)
            return SI_FLOAT;
        if(points==0 && digits>0)
            return SI_INT;
    }

    return SI_STRING;
}

size_t FindMineBrace(const char closebr, const size_t pos, const string &str){
    short subcount(0);
    char openbr('[');
    if(closebr=='}')
        openbr='{';
    for(unsigned int i=pos;i<str.length();i++){
        if(subcount==0 && str[i]==closebr)
            return i;
        if(str[i]==openbr)
            subcount++;
        else if(str[i]==closebr)
            subcount--;
    }
    return string::npos;
}




/*
    fjString
*/
fjString::fjString(): value(""){}
fjString::fjString(const fjString &val): fjValue(), value(val.value){}

fjString::fjString(fjString &&val): fjValue(){
    value=move(val.value);
}

fjString::fjString(const char *val){ value=val;}
fjString::fjString(const string &val):value(val){}


fjString& fjString::operator=(const fjString &a){
    if(this==&a)
        return *this;
    value=a.value;
    return *this;
}

fjString& fjString::operator=(fjString &&a){
    if(this==&a)
        return *this;
    value=move(a.value);
    return *this;
}

string fjString::asString(bool quotes) const{
    if(quotes)
        return "\""+value+"\"";
    return value;
}

shared_ptr<fjArray> fjString::asfjArray() const{
    return nullptr;
}

float fjString::asFloat() const{
    return stof(value);
}

shared_ptr<fjObject> fjString::asfjObject() const{
    return nullptr;
}

int fjString::asInt() const{
    return stoi(value);
}

fjString::~fjString(){
    cout<<"~fjString("<<value<<")"<<endl;
}

bool fjString::IsArray()const{return false;}
bool fjString::IsObject()const{return false;}
size_t fjString::Size()const{return 1;}

/*
    fjInt
*/
fjInt::fjInt(): value(0){}
fjInt::fjInt(const fjInt &val): fjValue(), value(val.value){}

fjInt::fjInt(fjInt &&val): fjValue(){
    value=move(val.value);
}

fjInt::fjInt(const int &val): value(val){}

fjInt& fjInt::operator=(const fjInt &a){
    if(this==&a)
        return *this;
    value=a.value;
    return *this;
}

fjInt& fjInt::operator=(fjInt &&a){
    if(this==&a)
        return *this;
    value=move(a.value);
    return *this;

}

string fjInt::asString(bool) const{
    return std::to_string(value);
}

shared_ptr<fjArray> fjInt::asfjArray() const{
    return nullptr;
}

float fjInt::asFloat() const{
    return (float)value;
}

shared_ptr<fjObject> fjInt::asfjObject() const{
    return nullptr;
}

int fjInt::asInt() const{
    return value;
}

fjInt::~fjInt(){
    cout<<"~fjInt("<<value<<")"<<endl;
}

bool fjInt::IsArray()const{return false;}
bool fjInt::IsObject()const{return false;}
size_t fjInt::Size()const{return 1;}

/*
    fjFloat
*/
fjFloat::fjFloat(): value(0.0){}
fjFloat::fjFloat(const fjFloat &val): fjValue(), value(val.value){}

fjFloat::fjFloat(fjFloat &&val): fjValue(){
    value=move(val.value);
}

fjFloat::fjFloat(const float &val): value(val){}

fjFloat& fjFloat::operator=(const fjFloat &a){
    if(this==&a)
        return *this;
    value=a.value;
    return *this;
}

fjFloat& fjFloat::operator=(fjFloat &&a){
    if(this==&a)
        return *this;
    value=move(a.value);
    return *this;
}


string fjFloat::asString(bool) const{
    return std::to_string(value);
}

shared_ptr<fjArray> fjFloat::asfjArray() const{
    return nullptr;
}
float fjFloat::asFloat() const {
    return value;
}

shared_ptr<fjObject> fjFloat::asfjObject() const{
    return nullptr;
}

int fjFloat::asInt() const{
    return (int)value;
}
fjFloat::~fjFloat(){
    cout<<"~fjString("<<value<<")"<<endl;
}

bool fjFloat::IsArray()const{return false;}
bool fjFloat::IsObject()const{return false;}
size_t fjFloat::Size()const{return 1;}

/*
    fjObjValue
*/
fjObjValue::fjObjValue(): fjValue(), value(nullptr){}
fjObjValue::fjObjValue(const fjObjValue &val): fjValue(){
        value=val.value;
}

fjObjValue::fjObjValue(const fjObject &val){
    value=make_shared<fjObject>(val);
}

fjObjValue& fjObjValue::operator=(const fjObjValue &a){
    if(this==&a)
        return *this;
    value=a.value;
    return *this;
}

fjObjValue& fjObjValue::operator=(fjObjValue &&a){
    if(this==&a)
        return *this;
    value=move(a.value);
    return *this;
}

string fjObjValue::asString(bool)const{
    string str;
    str="{";
    for(unsigned int i=0;i<value.get()->pair.size();i++){
        str+=value.get()->pair.at(i).name+": "+(*value.get()->pair.at(i).value).asString(true);
        if(i<value.get()->pair.size()-1)
            str+=", ";
    }
    str+="}";
    return str;
}

shared_ptr<fjArray> fjObjValue::asfjArray()const{
    return nullptr;
}

float fjObjValue::asFloat()const{
    return 0;
}
shared_ptr<fjObject> fjObjValue::asfjObject()const{
    return nullptr;
}

int fjObjValue::asInt()const{
    return 0;
}


fjObjValue::~fjObjValue(){
    cout<<"~fjObjValue()"<<endl;
}

bool fjObjValue::IsArray()const{return false;}
bool fjObjValue::IsObject()const{return true;}
size_t fjObjValue::Size()const{return 1;}

/*
    fjArray
*/
fjArray::fjArray(){}

fjArray::fjArray(const fjArray &val): fjValue(){
    value=val.value;
}

fjArray::fjArray(fjArray &&val): fjValue(){
    value=move(val.value);
}

fjArray::fjArray(const string &jsonstring){
    Set(jsonstring);
}

fjArray& fjArray::operator=(const fjArray &a){
    if(this==&a)
        return *this;
    value=a.value;
    return *this;
}

fjArray& fjArray::operator=(fjArray &&a){
    if(this==&a)
        return *this;
    value=move(a.value);
    return *this;
}

void fjArray::AddValue(string &value){
    eStringIs tp=DetectType(value);
    cout<<"AddValue[]("<<value<<")"<<endl;
    switch(tp){
        case SI_STRING:            
            Add(value);
            break;
        case SI_INT:
            Add(atoi(value.c_str()));
            break;
        case SI_FLOAT:
            Add((float)atof(value.c_str()));
            break;
        case SI_ARRAY:
            Add(make_shared<fjArray>(value));
            break;
        case SI_OBJECT:
            Add(make_shared<fjObjValue>(value));
            break;
        default:
            break;
    }
}

void fjArray::Set(const string &jstr){
    size_t elemb(0),eleme(0), objb(0), obje(0), arrayb(0), arraye(0);
    string valstr;

    elemb=jstr.find('[')+1;

    while(eleme<jstr.size()){
            eleme=jstr.find(',',elemb);
            arrayb=jstr.find('[',elemb);
            objb=jstr.find('{',elemb);

            if(arrayb!=string::npos && arrayb<eleme && (objb>arrayb || objb==string::npos)){
                arraye=FindMineBrace(']',arrayb+1,jstr);
                if(arraye!=string::npos){
                    valstr=jstr.substr(arrayb,arraye-arrayb+1);
                    AddValue(valstr);
                    elemb=jstr.find_first_not_of(" \n",arraye+1)+1;
                }
            }else
            if(objb!=string::npos && objb<eleme && (objb<arrayb || arrayb==string::npos)){
                obje=FindMineBrace('}',objb+1,jstr);
                if(obje!=string::npos){
                    valstr=jstr.substr(objb,obje-objb+1);
                    AddValue(valstr);
                    elemb=jstr.find_first_not_of(" \n",obje+1)+1;
                }
            }else
            if(eleme!=string::npos){
                valstr=jstr.substr(elemb,eleme-elemb);
                AddValue(valstr);
                elemb=eleme+1;
            }else{
                eleme=FindMineBrace(']',elemb,jstr);

                if(eleme!=string::npos){
                    valstr=jstr.substr(elemb,eleme-elemb);
                    AddValue(valstr);
                    elemb=eleme+1;
                }
            }

    }
}

void fjArray::Add(shared_ptr<fjValue> val){
    if(val==nullptr) return;
    value.push_back(val);
}

void fjArray::Add(const int &val){
    value.push_back(make_shared<fjInt>(val));
}

void fjArray::Add(const float &val){
    value.push_back(make_shared<fjFloat>(val));
}

void fjArray::Add(const string &val){
    value.push_back(make_shared<fjString>(val));
}

void fjArray::Delete(unsigned int ind){
    if(ind>=value.size())
        return;
    value.at(ind).reset();
    value.erase(value.begin() + ind - 1);
}

void fjArray::Clear(){
    value.clear();
}


shared_ptr<fjValue>& fjArray::operator[](unsigned int i){
    return value.at(i);
}

string fjArray::asString(bool quotes) const{
    string res("");
    for(unsigned int i=0;i<value.size();i++){
        res+=value.at(i)->asString(quotes);
        if(i<value.size()-1)
            res+=", ";
    }
    return "["+res+"]";
}

shared_ptr<fjArray> fjArray::asfjArray() const{
    return make_shared<fjArray>(*this);
}

float fjArray::asFloat() const{
    return 0;
}

shared_ptr<fjObject> fjArray::asfjObject() const{
    return nullptr;
}

int fjArray::asInt() const{
    return 0;
}

fjArray::~fjArray(){
    Clear();
    cout<<"~fjArray("<<asString(true)<<")"<<endl;    
}

bool fjArray::IsArray()const{return true;}
bool fjArray::IsObject()const{return false;}
size_t fjArray::Size()const{return value.size();}

/*
    fjObject
*/
fjObject::fjObject(){
}

fjObject::fjObject(const fjObject &obj){
    pair=obj.pair;
}

fjObject::fjObject(fjObject &&obj){
    pair=move(obj.pair);
}

fjObject& fjObject::operator=(const fjObject &obj){
    if(this==&obj)
        return *this;
    Clear();
    pair=obj.pair;
    return *this;
}

fjObject& fjObject::operator=(fjObject &&obj){
    if(this==&obj)
        return *this;
    Clear();
    pair=move(obj.pair);
    return *this;
}

fjObject::fjObject(const string &jsonstring){
    Init(jsonstring);
}

void fjObject::Init(const string &jsonstring){
    Clear();
    Parse(jsonstring);
}


void fjObject::AddValue(const string &name, string &value){
    eStringIs tp=DetectType(value);
    cout<<"AddValue{}(\""<<name<<"\",\""<<value<<"\")"<<endl;
    switch(tp){
        case SI_STRING:
            Set(name, value);
            break;
        case SI_INT:
            Set(name, atoi(value.c_str()));
            break;
        case SI_FLOAT:
            Set(name,(float)atof(value.c_str()));
            break;
        case SI_ARRAY:
            Set(name, make_shared<fjArray>(value));
            break;
        case SI_OBJECT:
            Set(name, make_shared<fjObjValue>(value));
            break;
        default:
            break;
    }
}

/*
 * {field1:
 *    {
 *       name: \"subobj\",
 *       value: 100,
 *       subarray: [1,2,{name: \"hop-hay\", value: 999},4,5.67],
 *       fld: \"filename\"
 *     },
 * zo: 1 ,
 * field2: \"string field\",
 * file3array: [ 11.3, 1, \"new\" ]}"
 *
*/
void fjObject::Parse(const string jstr){
    size_t elemb(0),eleme(0), objb(0), obje(0), arrayb(0), arraye(0);
    string name, valstr;

    elemb=jstr.find('{')+1;

    while(eleme<jstr.size()){
        eleme=jstr.find(':',elemb);
        if(eleme!=string::npos){
            name=trimstr(jstr.substr(elemb,eleme-elemb));
            elemb=eleme+1;

            eleme=jstr.find(',',elemb);
            arrayb=jstr.find('[',elemb);
            objb=jstr.find('{',elemb);

            if(arrayb!=string::npos && arrayb<eleme && (objb>arrayb || objb==string::npos)){
                arraye=FindMineBrace(']',arrayb+1,jstr);
                if(arraye!=string::npos){
                    valstr=jstr.substr(arrayb,arraye-arrayb+1);
                    AddValue(name,valstr);
                    elemb=jstr.find_first_not_of(" \n",arraye+1)+1;
                }
            }else
            if(objb!=string::npos && objb<eleme && (objb<arrayb || arrayb==string::npos)){
                obje=FindMineBrace('}',objb+1,jstr);
                if(obje!=string::npos){
                    valstr=jstr.substr(objb,obje-objb+1);
                    AddValue(name,valstr);
                    elemb=jstr.find_first_not_of(" \n",obje+1)+1;
                }
            }else
            if(eleme!=string::npos){
                valstr=jstr.substr(elemb,eleme-elemb);
                AddValue(name,valstr);
                elemb=eleme+1;
            }else{
                eleme=FindMineBrace('}',elemb,jstr);

                if(eleme!=string::npos){
                    valstr=jstr.substr(elemb,eleme-elemb);
                    AddValue(name,valstr);
                    elemb=eleme+1;
                }
            }

        }

    }
}


shared_ptr<fjValue>& fjObject::operator[](string name){
    for(unsigned int i=0;i<pair.size();i++){
        if(pair.at(i).name==name){
            return pair.at(i).value;
            break;
        }
    }
    pair.emplace_back(fjPair(name,nullptr));
    return pair.back().value;
}

bool fjObject::exists(string name){
    if(name>""){
        for(unsigned int i=0;i<pair.size();i++){
            if(pair.at(i).name==name && pair.at(i).value!=nullptr){
                return true;
            }
        }
    }
    return false;
}

void fjObject::Set(string name, shared_ptr<fjValue> value){
    if(name>"" && value!=nullptr){
        for(unsigned int i=0;i<pair.size();i++){
            if(pair.at(i).name==name){
                pair.at(i).value=value;
                return;
            }
        }
        pair.emplace_back(fjPair(name,value));
    }
}

void fjObject::Set(string name, const int &value){
    if(name>""){
        for(unsigned int i=0;i<pair.size();i++){
            if(pair.at(i).name==name){
                pair.at(i).value=make_shared<fjInt>(value);
                return;
            }
        }
        pair.emplace_back(fjPair(name,make_shared<fjInt>(value)));
    }
}

void fjObject::Set(string name, const float &value){
    if(name>""){
        for(unsigned int i=0;i<pair.size();i++){
            if(pair.at(i).name==name){
                pair.at(i).value=make_shared<fjFloat>(value);
                return;
            }
        }
        pair.emplace_back(fjPair(name,make_shared<fjFloat>(value)));
    }
}

void fjObject::Set(string name,const string &value){
    if(name>""){
        for(unsigned int i=0;i<pair.size();i++){
            if(pair.at(i).name==name){
                pair.at(i).value=make_shared<fjString>(value);
                return;
            }
        }
        pair.emplace_back(fjPair(name,make_shared<fjString>(value)));
    }
}

void fjObject::Clear(){
    pair.clear();
}

size_t fjObject::Size(){return pair.size();}

void fjObject::LoadFromFile(const string filename){
    ifstream fstr(filename);
    string t, jstr;
    while(fstr>>t)
        jstr+=t;
    fstr.close();
    Init(jstr);
}

void fjObject::SaveToFile(const string filename){
    ofstream fstr(filename);
    fstr<<this;
    fstr.close();
}


fjObject::~fjObject(){
    Clear();
}

/*
    ostream <<
*/

ostream& operator<<(ostream& s, const fjValue *t){
    s<<t->asString(true);
    return s;
}

ostream& operator<<(ostream& s, const fjString &t)
{
    s<<t.value;
    return s;
}

ostream& operator<<(ostream& s, const fjInt &t)
{
    s<<t.value;
    return s;
}

ostream& operator<<(ostream& s, const fjFloat &t)
{
    s<<t.value;
    return s;
}

ostream& operator<<(ostream& s, const fjArray &t)
{
    s<<t.asString(true);
    return s;
}

ostream& operator<<(ostream& s, const fjObjValue &t)
{
    s<<t.value;
    return s;
}

ostream& operator<<(ostream& s, const fjObject &t)
{
    s<<"{";
    for(unsigned int i=0;i<t.pair.size();i++){
        if(t.pair.at(i).value!=nullptr){
            s<<t.pair.at(i).name<<": "<<(*t.pair.at(i).value).asString(true);
            if(i<t.pair.size()-1)
                s<<", ";
        }
    }
    s<<"}";
    return s;
}




