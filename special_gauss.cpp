#include <iostream>
#include <fstream>
#include <iomanip>
#include <time.h>
#include <sys/time.h>
#include <cmath>
#include <string>
#include <string.h>
using namespace std;

//#define D1
#define DEBUG
//#define ARM
#define X86

#ifdef X86
#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE2
#include <pmmintrin.h> //SSE3
#include <tmmintrin.h> //SSSE3
#include <smmintrin.h> //SSE4.1
#include <nmmintrin.h> //SSSE4.2
#include <immintrin.h> //AVX、AVX2
#endif

#ifdef ARM
#include<arm_neon.h>
#endif

#define path "./1_130_22_8/"

#define col 130
#define row 22
#define row2 8

#define width 32

uint32_t m[col][(col/width)/16*16+16]={0};
uint32_t n[row2][(col/width)/16*16+16]={0};

#ifndef ALIGN
uint32_t m1[col][(col/width)/16*16+16]={0};
uint32_t n1[row2][(col/width)/16*16+16]={0};
#else
mat_t ele_tmp[COL][(COL / mat_L + 1) / 16 * 16 + 16] __attribute__((aligned(64))) = {0};
mat_t row_tmp[ROW][(COL / mat_L + 1) / 16 * 16 + 16] __attribute__((aligned(64))) = {0};
#endif


void reset()
{
    string pad=(string)path+"1.txt";
    ifstream ifs(pad);
    for(int i=0;i<row;i++)
    {
        int token_f;
        int token;
        string line;
        getline(ifs,line);
        //将读入的一行进行对应的处理
        istringstream lines(line);
        lines>>token_f;
        m[token_f][token_f/width]+=(uint32_t)1<<(token_f%width);
        while(lines>>token)
        {
            m[token_f][token/width]+=(uint32_t)1<<(token%width);
        }
    }
    ifs.close();

    string pad2=(string)path+"2.txt";
    ifstream ifs2(pad2);
    for(int i=0;i<row2;i++)
    {
        int token;
        string line;
        getline(ifs2,line);
        //将读入的一行进行对应的处理
        istringstream lines(line);
        while(lines>>token)
        {
            
            n[i][token/width]+=(uint32_t)1<<(token%width);
        }
    }
    ifs2.close();
    

}

void test(void (*func)(), int times)
{
    double time=0;
    timespec start, end;
    for(int i=0;i<times;i++)
    {
        
        asign();
        clock_gettime(CLOCK_REALTIME, &start);
        func();
        clock_gettime(CLOCK_REALTIME, &end);
        time += end.tv_sec - start.tv_sec;
        time += double(end.tv_nsec - start.tv_nsec) / 1000000000;
        memset(m1,0,sizeof(m1));
    }
    cout<<"测试次数: "<<times<<" "<<"测试用时: "<<time<<endl;
}
void asign()
{
    memcpy(m1,m,sizeof(m));
    memcpy(n1,n,sizeof(n));
}


void clear()
{
    memset(m1,0,sizeof(m));
    memset(n1,0,sizeof(n));

}



//普通的特殊高斯算法
void gb()
{
    for(int i=0;i<row2;i++)
    {
        for(int j=col-1;j>=0;j--)
        {
            bool t1=((uint32_t) 1<<(j%width))&n1[i][j/width];

            if(t1)
            {
                bool k1=((uint32_t) 1<<(j%width))&m1[j][j/width];
                if(k1)
                {
                    for(int u=0;u<col/width+1;u++)
                    {
                        n1[i][u]^=m1[j][u];
                    }
                }
                else
                {
                    memcpy(m1[j],n1[i],sizeof(n1[0]));
                    break;
                }

            }
        }
    }
#ifdef DEBUG
for(int i=0;i<row2;i++)
{
    for(int j=col/width;j>=0;j--)
    {
        uint32_t temp=n1[i][j];
        //cout<<temp<<endl;
        int count=0;
        uint32_t b=1;
        b<<=width-1; 
        while(temp!=0)
        {
            if((temp&b)!=0)
            {
                cout<<(j+1)*width-count-1<<" ";
            }
            temp<<=1;
            count++;
        }
    }
    cout<<endl;
}

#endif
}

#ifdef ARM

void gb_neon()
{
    for(int i=0;i<row2;i++)
    {
        for(int j=col-1;j>=0;j--)
        {
            bool t1=((uint32_t) 1<<(j%width))&n1[i][j/width];

            if(t1)
            {
                bool k1=((uint32_t) 1<<(j%width))&m1[j][j/width];
                if(k1)
                {
                    int l1=col/width+1;
                    int l2=l1/4;
                    int l3=l2*4;
                    for(int u=0;u<l2;u+=4)
                    {
                        uint32x4_t t1=vld1q_u32(&n1[i][u]);
                        uint32x4_t t2=vld1q_u32(&m1[j][u]);
                        vst1q_u32(&n1[i][u],veorq_u32(t1,t2));
                    }
                    for(int u=l3;u<l1;u++)
                    {
                        n1[i][u]^=m1[j][u];
                    }
                }
                else
                {
                    memcpy(m1[j],n1[i],sizeof(n1[0]));
                    break;
                }

            }
        }
    }

#ifdef DEBUG
for(int i=0;i<row2;i++)
{
    for(int j=col/width;j>=0;j--)
    {
        uint32_t temp=n1[i][j];
        //cout<<temp<<endl;
        int count=0;
        uint32_t b=1;
        b<<=width-1; 
        while(temp!=0)
        {
            if((temp&b)!=0)
            {
                cout<<(j+1)*width-count-1<<" ";
            }
            temp<<=1;
            count++;
        }
    }
    cout<<endl;
}

#endif
}

#endif



#ifdef X86

void gb_avx()
{
    for(int i=0;i<row2;i++)
    {
        for(int j=col-1;j>=0;j--)
        {
            bool t1=((uint32_t) 1<<(j%width))&n1[i][j/width];

            if(t1)
            {
                bool k1=((uint32_t) 1<<(j%width))&m1[j][j/width];
                if(k1)
                {
                    int l1=(col/width)+1;
                    int l2=l1/8;
                    int l3=l2*8;
                    for(int u=0;u<l2;u+=8)
                    {
                        #ifdef ALIGH
                        __m256i t1=_mm256_load_si256((__m256i *)&n1[i][u]);
                         __m256i t2=_mm256_load_si256((__m256i *)&m1[i][u]);
                         __m256i t3 =_mm256_xor_si256(t1, t2);
                         _mm256_store_si256((__m256i *)&n1[i][u],t3);
                        #else
                         __m256i t1=_mm256_loadu_si256((__m256i *)&n1[i][u]);
                         __m256i t2=_mm256_loadu_si256((__m256i *)&m1[i][u]);
                         __m256i t3 =_mm256_xor_si256(t1, t2);
                         _mm256_storeu_si256((__m256i *)&n1[i][u],t3);
                        
                        #endif
                    }
                    for(int u=l3;u<l1;u++)
                    {
                        n1[i][u]^=m1[j][u];
                    }
                }
                else
                {
                    memcpy(m1[j],n1[i],sizeof(n1[0]));
                    break;
                }

            }
        }
    }

#ifdef DEBUG
for(int i=0;i<row2;i++)
{
    for(int j=col/width;j>=0;j--)
    {
        uint32_t temp=n1[i][j];
        //cout<<temp<<endl;
        int count=0;
        uint32_t b=1;
        b<<=width-1; 
        while(temp!=0)
        {
            if((temp&b)!=0)
            {
                cout<<(j+1)*width-count-1<<" ";
            }
            temp<<=1;
            count++;
        }
    }
    cout<<endl;
}

#endif
}

#endif



int main()
{
    reset();
    asign();
    gb_avx();


    


}









