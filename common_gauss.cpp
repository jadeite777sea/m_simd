#include <iostream>
#include <fstream>
#include <iomanip>
#include <time.h>
#include <sys/time.h>
#include <cmath>
#include<string.h>


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

using namespace std;
#define ALIGH

//#define ARM

#ifdef ARM
#include<arm_neon.h>
#endif

#ifdef ARM
__extension__ extern __inline float32x4_t
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vdivq_f32 (float32x4_t __a, float32x4_t __b)
{
  return __a / __b;
}
#endif 


#define N 16

#define DEBUG

float m [N][N]={0};

#ifdef ALIGH
float m1[N][N]__attribute__((aligned(64)))={0};
#else
float m1[N][N]={0};
#endif



void reset()
{
    for(int i=0;i<N;i++)
    {
        m[i][i]=1.0;
        for(int j=i+1;j<N;j++)
        {
            m[i][j]=rand()/2000;
            
        }

    }

    for(int k=0;k<N;k++)
        for(int i=k+1;i<N;i++)
            for(int j=0;j<N;j++)
                m[i][j]+=m[k][j];

}

//用于创建一个与原数组相同的二维数组
void asign()
{
    for(int i=0;i<N;i++)
    {
        for(int j=0;j<N;j++)
        {
            m1[i][j]=m[i][j];
        }
    }
}


//普通高斯消去算法
void cg()
{

#ifdef DEBUG
    cout<<"原数组"<<endl;
    for(int i=0;i<N;i++)
    {
        for(int j=0;j<N;j++)
        {
            cout<<m[i][j]<<" ";
        }
        cout<<endl;
    }

#endif

    

    for(int k=0;k<N;k++)
    {
        for(int j=k+1;j<N;j++)
        {
            m1[k][j]=m1[k][j]/m1[k][k]; //可向量化
        }
        m1[k][k]=1.0;
        for(int i=k+1;i<N;i++)
        {
            for (int j=k+1;j<N;j++)
            {
                m1[i][j]-=m1[i][k]*m1[k][j]; // 可向量化
            }
            m1[i][k]=0;
        }
    }

#ifdef DEBUG
    cout<<"消去后数组" <<endl;
    for(int i=0;i<N;i++)
    {
        for(int j=0;j<N;j++)
        {
            cout<<m1[i][j]<<" ";
        }
        cout<<endl;
    }
#endif

}

#ifdef ARM
void neon()
{
#ifdef DEBUG
    cout<<"原数组"<<endl;
    for(int i=0;i<N;i++)
    {
        for(int j=0;j<N;j++)
        {
            cout<<m[i][j]<<" ";
        }
        cout<<endl;
    }

#endif

for(int k=0;k<N;k++)
    {
        //对齐处理
        int o=(N-k-1)%4+k;
        for(int j=k+1;j<=o;j++)
        {
            m1[k][j]=m1[k][j]/m1[k][k];
        }
        for(int j=o+1;j<N;j+=4)
        {
             //可向量化
            float32x4_t t1=vld1q_f32(&m1[k][j]);
            float32x4_t t2;
            vst1q_f32(&m1[k][k],t2);
            float32x4_t t3=vdivq_f32(t1,t2);
            vst1q_f32(&m1[k][j],t3);

        }
        m1[k][k]=1.0;
        for(int i=k+1;i<N;i++)
        {
            int q=(N-k-1)%4+k;
            for(int j=k+1;j<=q;j++)
            {
                m1[i][j]-=m1[i][k]*m1[k][j];
            }
            for (int j=q+1;j<N;j+=4)
            {

                float32x4_t t1=vld1q_f32(&m1[i][j]);
                float32x4_t t2=vld1q_f32(&m1[k][j]);
                float32x4_t t3;
                vst1q_f32(&m1[i][k],t3);
                float32x4_t t4=vmulq_f32(t1,t2);
                float32x4_t t5=vsubq_f32(t3,t4);
                vst1q_f32(&m1[i][j],t5);

            }
            m1[i][k]=0;
        }
    }

    

    

#ifdef DEBUG
    cout<<"消去后数组" <<endl;
    for(int i=0;i<N;i++)
    {
        for(int j=0;j<N;j++)
        {
            cout<<m1[i][j]<<" ";
        }
        cout<<endl;
    }
#endif

}
#endif

#ifdef X86
//未要求对齐
void sse()
{
#ifdef DEBUG
    cout<<"原数组"<<endl;
    for(int i=0;i<N;i++)
    {
        for(int j=0;j<N;j++)
        {
            cout<<m[i][j]<<" ";
        }
        cout<<endl;
    }
#endif

for(int k=0;k<N;k++)
    {
        //对齐处理
        int o=(N-k-1)%4+k;
        for(int j=k+1;j<=o;j++)
        {
            m1[k][j]=m1[k][j]/m1[k][k];
        }
        for(int j=o+1;j<N;j+=4)
        {
            #ifndef ALIGH
             //可向量化
            __m128 t1=_mm_loadu_ps(&m1[k][j]);
            __m128 t2=_mm_set1_ps(m1[k][k]);
            __m128 t3=_mm_div_ps(t1,t2);
            _mm_storeu_ps(&m1[k][j],t3);
            #else
            __m128 t1=_mm_load_ps(&m1[k][j]);
            __m128 t2=_mm_set1_ps(m1[k][k]);
            __m128 t3=_mm_div_ps(t1,t2);
            _mm_store_ps(&m1[k][j],t3);

            #endif

        }
        m1[k][k]=1.0;
        for(int i=k+1;i<N;i++)
        {
            int q=(N-k-1)%4+k;
            for(int j=k+1;j<=q;j++)
            {
                m1[i][j]-=m1[i][k]*m1[k][j];
            }
            for (int j=q+1;j<N;j+=4)
            {
                #ifndef ALIGH

                __m128 t1=_mm_loadu_ps(&m1[i][j]);
                __m128 t2=_mm_loadu_ps(&m1[k][j]);
                __m128 t3=_mm_set1_ps(m1[i][k]);
                __m128 t4=_mm_mul_ps(t1,t2);
                __m128 t5=_mm_sub_ps(t3,t4);
                _mm_storeu_ps(&m[i][j],t5);
                #else
                __m128 t1=_mm_load_ps(&m1[i][j]);
                __m128 t2=_mm_load_ps(&m1[k][j]);
                __m128 t3=_mm_set1_ps(m1[i][k]);
                __m128 t4=_mm_mul_ps(t1,t2);
                __m128 t5=_mm_sub_ps(t3,t4);
                _mm_store_ps(&m[i][j],t5);
                #endif

            }
            m1[i][k]=0;
        }
    }

    

    

#ifdef DEBUG
    cout<<"消去后数组" <<endl;
    for(int i=0;i<N;i++)
    {
        for(int j=0;j<N;j++)
        {
            cout<<m1[i][j]<<" ";
        }
        cout<<endl;
    }
#endif


}



void avx()
{
#ifdef DEBUG
    cout<<"原数组"<<endl;
    for(int i=0;i<N;i++)
    {
        for(int j=0;j<N;j++)
        {
            cout<<m[i][j]<<" ";
        }
        cout<<endl;
    }
#endif

for(int k=0;k<N;k++)
    {
        //对齐处理
        int o=(N-k-1)%8+k;
        for(int j=k+1;j<=o;j++)
        {
            m1[k][j]=m1[k][j]/m1[k][k];
        }
        for(int j=o+1;j<N;j+=8)
        {
            #ifndef ALIGH
             //可向量化
            __m256 t1=_mm256_loadu_ps(&m1[k][j]);
            __m256 t2=_mm256_set1_ps(m1[k][k]);
            __m256 t3=_mm256_div_ps(t1,t2);
            _mm256_storeu_ps(&m1[k][j],t3);
            #else
            __m256 t1=_mm256_load_ps(&m1[k][j]);
            __m256 t2=_mm256_set1_ps(m1[k][k]);
            __m256 t3=_mm256_div_ps(t1,t2);
            _mm256_store_ps(&m1[k][j],t3);
            #endif

        }
        m1[k][k]=1.0;
        for(int i=k+1;i<N;i++)
        {
            int q=(N-k-1)%8+k;
            for(int j=k+1;j<=q;j++)
            {
                m1[i][j]-=m1[i][k]*m1[k][j];
            }
            for (int j=q+1;j<N;j+=8)
            {

                #ifndef ALIGH
                __m256 t1=_mm256_loadu_ps(&m1[i][j]);
                __m256 t2=_mm256_loadu_ps(&m1[k][j]);
                __m256 t3=_mm256_set1_ps(m1[i][k]);
                __m256 t4=_mm256_mul_ps(t1,t2);
                __m256 t5=_mm256_sub_ps(t3,t4);
                _mm256_storeu_ps(&m[i][j],t5);
                #else
                __m256 t1=_mm256_load_ps(&m1[i][j]);
                __m256 t2=_mm256_load_ps(&m1[k][j]);
                __m256 t3=_mm256_set1_ps(m1[i][k]);
                __m256 t4=_mm256_mul_ps(t1,t2);
                __m256 t5=_mm256_sub_ps(t3,t4);
                _mm256_store_ps(&m[i][j],t5);
                #endif

            }
            m1[i][k]=0;
        }
    }

    

#ifdef DEBUG
    cout<<"消去后数组" <<endl;
    for(int i=0;i<N;i++)
    {
        for(int j=0;j<N;j++)
        {
            cout<<m1[i][j]<<" ";
        }
        cout<<endl;
    }
#endif


}




void avx512()
{
#ifdef DEBUG
    cout<<"原数组"<<endl;
    for(int i=0;i<N;i++)
    {
        for(int j=0;j<N;j++)
        {
            cout<<m[i][j]<<" ";
        }
        cout<<endl;
    }
#endif

for(int k=0;k<N;k++)
    {
        //对齐处理
        int o=(N-k-1)%16+k;
        for(int j=k+1;j<=o;j++)
        {
            m1[k][j]=m1[k][j]/m1[k][k];
        }
        for(int j=o+1;j<N;j+=16)
        {
            #ifndef ALIGH
             //可向量化
            __m512 t1=_mm512_loadu_ps(&m1[k][j]);
            __m512 t2=_mm512_set1_ps(m1[k][k]);
            __m512 t3=_mm512_div_ps(t1,t2);
            _mm512_storeu_ps(&m1[k][j],t3);
            #else
             //可向量化
            __m512 t1=_mm512_load_ps(&m1[k][j]);
            __m512 t2=_mm512_set1_ps(m1[k][k]);
            __m512 t3=_mm512_div_ps(t1,t2);
            _mm512_store_ps(&m1[k][j],t3);
            #endif

        }
        m1[k][k]=1.0;
        for(int i=k+1;i<N;i++)
        {
            int q=(N-k-1)%16+k;
            for(int j=k+1;j<=q;j++)
            {
                m1[i][j]-=m1[i][k]*m1[k][j];
            }
            for (int j=q+1;j<N;j+=16)
            {

                #ifndef ALIGH
                __m512 t1=_mm512_loadu_ps(&m1[i][j]);
                __m512 t2=_mm512_loadu_ps(&m1[k][j]);
                __m512 t3=_mm512_set1_ps(m1[i][k]);
                __m512 t4=_mm512_mul_ps(t1,t2);
                __m512 t5=_mm512_sub_ps(t3,t4);
                _mm512_storeu_ps(&m[i][j],t5);
                #else
                __m512 t1=_mm512_load_ps(&m1[i][j]);
                __m512 t2=_mm512_load_ps(&m1[k][j]);
                __m512 t3=_mm512_set1_ps(m1[i][k]);
                __m512 t4=_mm512_mul_ps(t1,t2);
                __m512 t5=_mm512_sub_ps(t3,t4);
                _mm512_store_ps(&m[i][j],t5);
                #endif

            }
            m1[i][k]=0;
        }
    }

    

#ifdef DEBUG
    cout<<"消去后数组" <<endl;
    for(int i=0;i<N;i++)
    {
        for(int j=0;j<N;j++)
        {
            cout<<m1[i][j]<<" ";
        }
        cout<<endl;
    }
#endif


}







#endif


void test(void (*func)(),int times )
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
    cout<<"测试规模: "<<N<<" "<<"测试次数: "<<times<<" "<<"测试用时: "<<time<<endl;
}



int main()
{
    

   

}
