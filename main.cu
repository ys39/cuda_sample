/*
1024×1024行列の加算プログラム
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

__global__ void matrix_vector_device(float *A_dev, float *B_dev, float *C_dev)
{
	/* <<<1,1>>> ブロック数 1 スレッド数1
	***************************************************************/

		for(int j=0; j<N; j++){
			for(int i=0; i<N; i++){
				A_dev[j*N+i] = B_dev[j*N+i] + C_dev[j*N+i];
			}
		}


	/* <<<1,2>>> ブロック数 1 スレッド数2
	***************************************************************/
    /*
		int N_start;
		N_start = threadIdx.x * 512;
		for(int j = N_start; j<N_start+512; j++)
		{
			for(int i=0; i<N; i++)
			{
				A_dev[j*N+i] = B_dev[j*N+i] + C_dev[j*N+i];
			}
		}
    */

	/* <<<1,1024>>> ブロック数 1 スレッド数1024
	***************************************************************/
    /*
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		for(int i=0; i<N; i++)
		{
			A_dev[index*N+i] = B_dev[index*N+i] + C_dev[index*N+i];
		}
    */

	/* <<<4,256>>> ブロック数 4 スレッド数256
	***************************************************************/
    /*
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		for(int i=0; i<N; i++)
		{
			A_dev[index*N+i] = B_dev[index*N+i] + C_dev[index*N+i];
		}
    */
}

int main(int argc, char **argv){

// JetsonTK1の情報を出力
    printf ("%s Starting...\n",argv[0]);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    printf("Using Device %s\n",deviceProp.name);

    double all_Start, all_Elaps;

// CPU（ホスト）側メモリの宣言
    float *A_host,*B_host,*C_host;

// GPU（デバイス）側メモリの宣言
    float *A_dev, *B_dev, *C_dev;

// dim3型で使用するブロック数とスレッド数を決定
    dim3 blocks(1,1,1);
    dim3 threads(1,1,1);

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
    all_Start = cpuSecond();

// データ転送（CPU（ホスト）→  GPU（デバイス））
    cudaMemcpy(A_dev, A_host, N * N * sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(B_dev, B_host, N * N * sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(C_dev, C_host, N * N * sizeof(float),cudaMemcpyHostToDevice);

// カーネル関数の起動、カーネル関数の実行
    matrix_vector_device<<<blocks, threads>>>(A_dev,B_dev,C_dev);


// データ転送（GPU（デバイス）→  CPU（ホスト））
    cudaMemcpy(A_host, A_dev, N * N * sizeof(float),cudaMemcpyDeviceToHost);

// 時間計測終了
    all_Elaps = cpuSecond()-all_Start;
	printf("matrix_vector_device<<<(%d,%d,%d),(%d,%d,%d)>>> all : Time elapsed %fmsec\n",
			blocks.x,blocks.y,blocks.z,threads.x,threads.y,threads.z,all_Elaps*1000);

    printf("A_host[1048575] = %f\n",A_host[1048575]);

// CPU（ホスト）で確保したメモリの開放
    free(A_host);
    free(B_host);
    free(C_host);

// GPU（デバイス）で確保したメモリの開放
    cudaFree(A_dev);
    cudaFree(B_dev);
    cudaFree(C_dev);

    return 0;
}
