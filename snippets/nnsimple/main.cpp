#include <iostream>
#include <nnsimple.h>

using namespace std;

int main()
{
    NNSimple nn(AF_SIGMA);
    cout<<"Teaching..."<<endl;
    nn.Teach("teach.txt",500);
    cout<<"Done"<<endl;

    /*
        input signal - digits (0..9) in 3x5 field
    */
    double a0[15]={1,1,1,1,0,1,1,0,1,1,0,1,1,1,1};
    double a1[15]={0,0,1,0,0,1,0,0,1,0,0,1,0,0,1};
    double a2[15]={1,1,1,0,0,1,1,1,1,1,0,0,1,1,1};
    double a3[15]={1,1,1,0,0,1,1,1,1,0,0,1,1,1,1};

    double a4[15]={1,0,1,1,0,1,1,1,1,0,0,1,0,0,1};
    double b4[15]={1,0.8,1,1,0,1,1,1,1,0,0,1,0,0,1};

    double a5[15]={1,1,1,1,0,0,1,1,1,0,0,1,1,1,1};
    double a6[15]={1,1,1,1,0,0,1,1,1,1,0,1,1,1,1};
    double a7[15]={1,1,1,0,0,1,0,0,1,0,0,1,0,0,1};
    double a8[15]={1,1,1,1,0,1,1,1,1,1,0,1,1,1,1};
    double a9[15]={1,1,1,1,0,1,1,1,1,0,0,1,1,1,1};

    cout<<"clear 2 -> "<<nn.Process(a2)<<endl;
    nn.PrintY();
    cout<<"noise 4 -> "<<nn.Process(b4)<<endl;
    nn.PrintY();

    //nn.PrintW(2);
    //cout<<"w[0,1]="<<nn.w[0][1]<<endl;
    //cout<<"w[1,1]="<<nn.w[1][1]<<endl;

    cout << "End!" << endl;
    return 0;
}
