#include <iostream>
#include <nnsimple.h>

using namespace std;

int main()
{
    NNSimple nn(15,10);
    cout<<"Teaching..."<<endl;
    nn.Teach("teach.txt",10000);
    cout<<"Done"<<endl;

    double a1[15]={1,1,1,1,0,1,1,0,1,1,0,1,1,1,1};
    double a2[15]={0,0,1,0,0,1,0,0,1,0,0,1,0,0,1};
    double a3[15]={1,1,1,0,0,1,1,1,1,1,0,0,1,1,1};
    double a4[15]={1,1,1,0,0,1,1,1,1,0,0,1,1,1,1};

    double a5[15]={1,0,1,1,0,1,1,1,1,0,0,1,0,0,1};
    double b5[15]={1,0,1,1,0,1,1,0,1,0,0,1,0,0,0};

    double a6[15]={1,1,1,1,0,0,1,1,1,0,0,1,1,1,1};
    double a7[15]={1,1,1,1,0,0,1,1,1,1,0,1,1,1,1};
    double a8[15]={1,1,1,0,0,1,0,0,1,0,0,1,0,0,1};
    double a9[15]={1,1,1,1,0,1,1,1,1,1,0,1,1,1,1};
    double a10[15]={1,1,1,1,0,1,1,1,1,0,0,1,1,1,1};

    cout<<"clear  4 -> "<<nn.Process(a4)+1<<endl;
    nn.PrintY();
    cout<<"noize  5 -> "<<nn.Process(b5)+1<<endl;
    nn.PrintY();

    cout << "End!" << endl;
    return 0;
}
