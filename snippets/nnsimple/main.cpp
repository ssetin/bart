#include <iostream>
#include <nnsimple.h>
#include <chrono>

using namespace std;

int main()
{
    NNSimple nn(AF_SIGMA,true);
    cout<<"Teaching..."<<endl;


    chrono::time_point<chrono::high_resolution_clock> start, end;
    chrono::duration<double> elapsed;

    start = chrono::high_resolution_clock::now();

    nn.Teach("teach.txt",10000);

    end = chrono::high_resolution_clock::now();
    elapsed = end - start;

    cout<< "Done in " << elapsed.count()<< " seconds" << endl;


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

    cout << "End!" << endl;
    exit(0);
    //return 0;
}
