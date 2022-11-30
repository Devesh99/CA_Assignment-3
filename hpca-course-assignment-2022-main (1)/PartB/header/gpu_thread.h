#include <ctime>
#include <chrono>
#include <stdio.h>
#define TIME_NOW std::chrono::high_resolution_clock::now()
#define TIME_DIFF(gran, start, end) std::chrono::duration_cast<gran>(end - start).count()

#define nthread 32


//----------------------------------------------------Best Attempt----------------------------------------------------------------//

__global__ void matrixMul(int *a, int *b, int *out, int N) {

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;


  int sum = 0;
  for (int k = 0; k < N; k++) {
 
    sum += a[2*row * N + k] * b[k * N + 2*col];
    sum += a[2*row*N + N + k] * b[k * N + 2*col];
    sum += a[2*row * N + k] * b[k * N + 2*col+1];
    sum += a[2*row*N + N + k] * b[k * N + 2*col+1];
    
    //printf("row = %d and col = %d and k = %d and [2*row * N + k] = %d and k * N + 2*col = %d\n", row, col, k, 2*row * N + k, k * N + 2*col);

  }
  out[row * (N>>1) + col] = sum;
  
}

void gpuThread(int N, int *matA, int *matB, int *output)
{
	auto begin = TIME_NOW;

  int *da, *db, *dout;
  cudaMalloc(&da, N * N * sizeof(int));
  cudaMalloc(&db,N * N * sizeof(int));
  cudaMalloc(&dout, (N * N * sizeof(int))>>2);

  cudaMemcpy(da, matA, N * N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(db, matB, N * N * sizeof(int), cudaMemcpyHostToDevice);

  int nblock = (N>>1) / nthread;

  dim3 threads(nthread, nthread);
  dim3 blocks(nblock, nblock);

  matrixMul<<<blocks, threads>>>(da, db, dout, N);


  cudaMemcpy(output, dout, (N * N * sizeof(int))>>2, cudaMemcpyDeviceToHost);
  
  
  auto end = TIME_NOW;
  cout << "CUDA execution time: " << 
  (double)TIME_DIFF(std::chrono::microseconds, begin, end) / 1000.0 << " ms\n"; 

  cudaFree(da);
  cudaFree(db);
  cudaFree(dout);

}







//-------------------------------------------------------------ATTEMPT -2-------------------------------------------------------//
/*
int N;


__global__ void matrixMul(int *a,int *b, int *output) {
	int smem_size = N;
	
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

 
  __shared__ int sa[smem_size];
  __shared__ int sb[smem_size];
  __shared__ int sc[smem_size];
  __shared__ int sd[smem_size];

  int sum = 0;

  for (int i = 0; i < N; i += blockDim.x) {

    sa[threadIdx.y * blockDim.x + threadIdx.x] = a[2*row * N + i + threadIdx.x];
    sb[threadIdx.y * blockDim.x + threadIdx.x] = b[i * N + threadIdx.y * N + 2*col];
    
    sc[threadIdx.y * blockDim.x + threadIdx.x] = a[2*row * N + N + i + threadIdx.x];
    sd[threadIdx.y * blockDim.x + threadIdx.x] = b[i * N + threadIdx.y * N + 2*col + 1];

    // sync all threads
    __syncthreads();

    for (int j = 0; j < blockDim.x; j++) {
      sum += sa[threadIdx.y * blockDim.x + j] * sb[j * blockDim.x + threadIdx.x];
      sum += sc[threadIdx.y * blockDim.x + j] * sd[j * blockDim.x + threadIdx.x];
      sum += sc[threadIdx.y * blockDim.x + j] * sb[j * blockDim.x + threadIdx.x];
      sum += sa[threadIdx.y * blockDim.x + j] * sd[j * blockDim.x + threadIdx.x];
      
    }
    __syncthreads();
  }

  output[row * Nby2 + col] = tmp;
}


void gpuThread(int Np, int *matA, int *matB, int *output)
{
	auto begin = TIME_NOW;
	N=Np;
  int *da, *db, *dout;
  cudaMalloc(&da, N*N*sizeof(int));
  cudaMalloc(&db, N*N*sizeof(int));
  cudaMalloc(&dout, (N*N*sizeof(int))>>2);

  cudaMemcpy(da, matA, , cudaMemcpyHostToDevice);
  cudaMemcpy(db, matB, bytes, cudaMemcpyHostToDevice);

  int nblock = (N>>1) / nthread;

  dim3 threads(nthread, nthread);
  dim3 blocks(nblock, nblock);
 
  matrixMul<<<blocks, threads>>>(da, db, dout, N);

  cudaMemcpy(output, dout, N*N*sizeof(int)>>2, cudaMemcpyDeviceToHost);
  
  cudaFree(da);
  cudaFree(db);
  cudaFree(dout);

  auto end = TIME_NOW;
  cout << "CUDA execution time: " << 
  (double)TIME_DIFF(std::chrono::microseconds, begin, end) / 1000.0 << " ms\n"; 
  
}*/








//-------------------------------------------------------ATTEMPT -1-------------------------------------------------------//
/*
__global__ void matrixMul(int *a, int *b, int *c, int N) {
  
  int Nby2=N>>1;
  

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
 
  for (int k = 0; k < N; k++) {
    c[row * Nby2 + col] += a[row * N+k] * b[col* N + k];
  }
  
}
// Create other necessary functions here

// Fill in this function
void gpuThread1(int N, int *matA, int *matB, int *output)
{
	auto begin = TIME_NOW;

  
  int Nby2 = N >> 1;
	int i,j;
  
  int *matBT = new int[N*N];
  int *B = new int[N*(N>>1)];
  int *A = new int[N*(N>>1)];
  
	//Taking Transpose of Matrix B
	for(i=0;i<N;++i){
	  	for(j=0;j<N;++j){
	  		matBT[j*N+i]=matB[i*N+j];
	  	}
	  }
  
  
  //Summing Adjacent Rows for Matrix A
  int index,indexN;
  for(i=0;i<Nby2;i++){
  	index=2*i*N;
  	indexN=index+N;
	  for(j=0;j<N;j++){
  			A[i*N+j]=matA[index+j]+matA[indexN+j];
  			
  		}	
  }
  
  //Summing Adjacent Rows for Matrix BT
  for(i=0;i<Nby2;i++){
  	index=2*i*N;
  	indexN=index+N;
	  for(j=0;j<N;j++){
  			B[i*N+j]=matBT[index+j]+matBT[indexN+j];
  			
  		}	
  }

  int *da, *db, *dout;
  cudaMalloc(&da, (N * N * sizeof(int))>>1);
  cudaMalloc(&db, (N * N * sizeof(int))>>1);
  cudaMalloc(&dout, (N * N * sizeof(int))>>2);

  cudaMemcpy(da, matA, (N * N * sizeof(int))>>1, cudaMemcpyHostToDevice);
  cudaMemcpy(db, matB, (N * N * sizeof(int))>>1, cudaMemcpyHostToDevice);

  int nblock = (N>>1) / nthread;

  dim3 threads(nthread, nthread);
  dim3 blocks(nblock, nblock);

  matrixMul1<<<blocks, threads>>>(da, db, dout, N);


  cudaMemcpy(output, dout, (N * N * sizeof(int))>>2, cudaMemcpyDeviceToHost);
  
  cudaFree(da);
  cudaFree(db);
  cudaFree(dout);
  
  auto end = TIME_NOW;
  cout << "CUDA execution time: " << 
  (double)TIME_DIFF(std::chrono::microseconds, begin, end) / 1000.0 << " ms\n"; 

}

*/
