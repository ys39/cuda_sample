/*
1024×1024行列の乗算プログラム
*/

#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <sys/time.h>

const int N = 1024;

double cpuSecond(){   
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

// 関数の定義
void matrix_product_host(float *A_host, float *B_host, float *C_host)
{
	for(int i=0; i<N; i++)
	{
		for(int j=0; j<N; j++)
		{
			for(int k=0; k<N; k++)
			{
				A_host[i*N+j] += B_host[i*N+k] * C_host[k*N+j];
			}
		}
	}
}

// カーネル関数の定義
__global__ void matrix_product_device(float *A_dev, float *B_dev, float *C_dev)
{

	unsigned int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	float tmp = 0.0;
	for(int i=0; i<N; i++)
	{
		tmp += B_dev[index_y * N + i] * C_dev[i * N + index_x];
	}
	A_device[index_y * N + index_x] = tmp;
}

int main(int argc, char **argv){

// JetsonTK1の情報を出力
    printf ("%s Starting...\n",argv[0]);
   	cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    printf("Using Device %s\n",deviceProp.name);

    double Start, Elaps;

// CPU（ホスト）側メモリの宣言
    float *A_host,*B_host,*C_host;

// GPU（デバイス）側メモリの宣言
    float *A_dev, *B_dev, *C_dev;

// dim3型で使用するブロック数とスレッド数を決定
	dim3 blocks(64,64,1);
    dim3 threads(16,16,1);

// CPU（ホスト）側メモリの確保
    A_host = (float*)malloc(N * N * sizeof(float));
    B_host = (float*)malloc(N * N * sizeof(float));
    C_host = (float*)malloc(N * N * sizeof(float));

// CPU（ホスト）側メモリへの割り当て
    for(int j=0; j<N; j++){
        for(int i=0; i<N; i++){
			B_host[j*N+i] = j*N+i;
			C_host[j*N+i] = j*N+i;
        }
    }

// GPU（デバイス）側メモリの確保
    cudaMalloc((void**)&A_dev, N * N * sizeof(float));
    cudaMalloc((void**)&B_dev, N * N * sizeof(float));
    cudaMalloc((void**)&C_dev, N * N * sizeof(float));

// 時間計測開始
    Start = cpuSecond();

/* GPU（デバイス）側
*****************************************/

// データ転送（CPU（ホスト）→  GPU（デバイス））
	cudaMemcpy(A_dev, A_host,N * N * sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(B_dev, B_host,N * N * sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(C_dev, C_host,N * N * sizeof(float),cudaMemcpyHostToDevice);

// カーネル関数の起動、カーネル関数の実行
    matrix_product_device<<<blocks, threads>>>(A_dev,B_dev,C_dev);

// カーネル関数が終了するまでCPUの処理を待機させる（同期処理）
	cudaDeviceSynchronize();

// データ転送（GPU（デバイス）→  CPU（ホスト））
    cudaMemcpy(A_host, A_dev, N * N * sizeof(float),cudaMemcpyDeviceToHost);

/* CPU（ホスト）側
*****************************************/

// 関数の起動、関数の実行
/*
	matrix_product_host(A_host, B_host, C_host); 
*/

// 時間計測終了
    all_Elaps = cpuSecond()-Start;
    printf("matrix_product_device<<<(%d,%d,%d),(%d,%d,%d)>>> all : Time elapsed %fmsec\n",
	blocks.x,blocks.y,blocks.z,threads.x,threads.y,threads.z,all_Elaps*1000);
	printf("A_host[1048575] = %f\n",A_host[1048575]);

//CPU（ホスト）で確保したメモリの開放
    free(A_host);
    free(B_host);
    free(C_host);

//GPU（デバイス）で確保したメモリの開放
    cudaFree(A_device);
    cudaFree(B_device);
    cudaFree(C_device);

    return 0;
}
