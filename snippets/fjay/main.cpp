#include <iostream>
#include <fjson.h>

using namespace std;

int main(int argc, char *argv[])
{
    fjString js="jstring";
    fjString js2=js;
    cout<<"js="<<js<<" js2="<<js2<<endl;

    fjInt ji=55;
    cout<<ji.asString(true)<<" "<<ji<<endl;

    fjArray fa;
    float s=1.5;
    fa.Add(s);
    fa.Add(2.345667f);
    fa.Add(2);
    fa.Add("char*");
    fa.Add(90);
    fa.Add("hoho");    

    fa[2]=make_shared<fjString>(js);
    cout<<"fa[2]="<<fa[2]<<endl;

    fa.Add(make_shared<fjInt>(34));

    cout<<"fa="<<fa<<endl<<endl;
    cout<<"js="<<js<<endl;
    fa.Set("[1,2,\"string\" ]");
    cout<<"new fa="<<fa<<endl<<endl;


    //**************************************************
    cout<<"json test"<<endl;
    fjObject json;

    json.Set("MyArray",make_shared<fjArray>(fa));

    json.SaveToFile("temp.json");

    cout << "Size: "<<json.Size() << endl;
    json.Set("me","hoha");
    json.Set("me","songslist");
    json.Set("me number",make_shared<fjInt>(84));

    json["he"]=json["me"];

    json.Set("me","old mememe");

    cout << "Size: "<<json.Size() << endl;

    if(json.exists("me"))
        cout << "json[me]="<<json["me"]->asString()<< endl;
    if(json.exists("me number"))
        cout << "json[me number]="<<json["me number"]<< endl;
    if(json.exists("he"))
        cout << "json[he]="<<json["he"]<<endl;

    fjObject obj;
    obj.Set("insideObject", 3.1415982f);
    obj.Set("mememe", "Junior");

    fa.Add(make_shared<fjObjValue>(obj));

    json.Set("MyArray", make_shared<fjArray>(fa));

    cout<<endl<<endl<<"JSON: "<<json<<endl<<endl;

    cout<<"JSON from string"<<endl;
    json.Init("{field1: {name: \"subobj\", value: 100 , subarray: [1, 2 , {name: \"hop-hay\", value: 999} ,4,5.67]\n, fld: \"filename\" } , zo: 1 , field2: \"string field\", file3array: [ 11.3, 1, \"new\"] }");

    json["kukumber"]=make_shared<fjString>("first kukumber");
    json["kukumber"]=make_shared<fjString>("magical");

    cout<<"JSON: "<<json<<endl<<endl;

    cout<<"json[field1][name]="<<(*json["field1"]->asfjObject())["name"]<<endl;
    (*json["field1"]->asfjObject())["name"]=make_shared<fjString>("new subobj");
    cout<<"json[field1][name]="<<(*json["field1"]->asfjObject())["name"]<<endl;

    cout<<"size="<<json.Size()<<endl;

    cin.get();

    return 0;
}
