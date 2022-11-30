#include <immintrin.h>

//----------------------------------- ATTEMPT - 7 Summing up rows and transposed columns and using AVX2 ----------------------//
// Best attempt at reducing execution time yet


// Optimize this function

void singleThread(int N, int *matA, int *matB, int *output)
{
  assert( N>=4 and N == ( N &~ (N-1)));
  
  int Nby2=N>>1;
  
  int *A=new int[Nby2*N];
  int *B=new int[Nby2*N];
  int *Bt=new int[N*N];
  
  int i,j,k,index,indexN;
  
  __m256i a0,b0,c0,m;
  
  // Taking transpose of Matrix B, to serialize matrix multiplication
  for(i=0;i<N;++i){
  	for(j=0;j<N;++j){
  		Bt[j*N+i]=matB[i*N+j];
  	}
  }
  // Adding adjacent Rows of Matrix A, 8 8-bit integers at a time using 256 bit avx registers
  for(i=0;i<Nby2;i++){
  	index=2*i*N;
  	indexN=index+N;
	  for(j=0;j<N;j+=8){
	  		a0 = _mm256_loadu_si256((__m256i *)&matA[index+j]);
	  		b0 = _mm256_loadu_si256((__m256i *)&matA[indexN+j]);
	  		m = _mm256_add_epi32(a0, b0);
  			_mm256_storeu_si256((__m256i *)&A[i*N+j], m);
  		}	
  }
  
  
  // Adding adjacent Rows of Matrix B, 8 8-bit integers at a time using 256 bit avx registers
    for(i=0;i<Nby2;i++){
  	index=2*i*N;
  	indexN=index+N;
	  for(j=0;j<N;j+=8){
  			a0 = _mm256_loadu_si256((__m256i *)&Bt[index+j]);
	  		b0 = _mm256_loadu_si256((__m256i *)&Bt[indexN+j]);
	  		m = _mm256_add_epi32(a0, b0);
  			_mm256_storeu_si256((__m256i *)&B[i*N+j], m);
  		}	
  }
  
  	int val,l;
  	int *sum = new int[8];
  	
  	for(i=0;i<Nby2;i++){
  		for(j=0;j<Nby2;j++){
  			m = _mm256_setzero_si256();
  			for(k=0;k<N;k+=8){

	  			a0 = _mm256_loadu_si256((__m256i *)&A[i*N+k]);
		  		b0 = _mm256_loadu_si256((__m256i *)&B[j*N+k]);
		  		c0 = _mm256_mullo_epi32(a0, b0);
	  			m = _mm256_add_epi32(m, c0);
  			}
  			val=0;
  			_mm256_storeu_si256((__m256i *)sum, m);
  			
  			for(l=0;l<8;++l){
  				val+=sum[l];
  			}
  			output[i*Nby2+j]=val;
  		}
  	}
 }
  
  
  
//----------------------------------------------- ATTEMPT - 6 Summing up rows and columns ------------------------------------------//
//Reduced Execution time by doing (N*N*N)/4 multiplications compared to (N*N*N). Yay
  
  /*
  int Nby2=N>>1;
  int *A=new int[Nby2*N];
  int *B=new int[N*Nby2];
  int i,j,k,index,indexN;
  
  for(i=0;i<Nby2;i++){
  	index=2*i*N;
  	indexN=index+N;
	  for(j=0;j<N;j++){
  			A[i*N+j]=matA[index+j]+matA[indexN+j];
  			
  		}	
  }
  for(i=0;i<N;++i){
  	index=i*N;
  	indexN=i*Nby2;
  	for(j=0;j<Nby2;++j){	
  		B[indexN+j]=matB[index+2*j]+matB[index+2*j+1];
  	}
  }	
  for(k=0;k<N;++k){
  	for(i=0;i<Nby2;++i){
  		for(j=0;j<Nby2;++j){
  			output[i*Nby2+j]+=A[i*N+k]*B[k*Nby2+j];
  			}
  		}
  	}
  */
  
//-------------------------------------------------- ATTEMPT - 5 --tiling on ikj --------------------------------------------/
// One last try with tiling but for different loop order. Didnt work. sigh
// Tried for block sizes 2,4,8,16,32,64
  /*
  int i,j,k;
  int sum, indexC, Nby2=N>>1;
  int rowA, iter, colB, rowAby2, rowAN, rowA1N;
  
  int b=32;
  for(i=0;i<N; i=i+b){
		for(j=0; j<N; j=j+b){ 
			for (k=0; k<N; k=k+b){
			  for(rowA = i; rowA < i+b; rowA +=2){
			  	rowAby2 = (rowA>>1)*Nby2;
			  	rowAN=rowA*N;
			  	rowA1N=rowAN+N;
			  	for(iter = k; iter < k+b; ++iter){
					for(colB = j; colB < j+b; colB += 2){
						sum = 0;
						indexC = rowAby2 + (colB>>1);
						sum += matA[rowAN + iter] * matB[iter * N + colB];
						sum += matA[rowA1N + iter] * matB[iter * N + colB];
						sum += matA[rowAN + iter] * matB[iter * N + (colB+1)];
						sum += matA[rowA1N + iter] * matB[iter * N + (colB+1)];
						output[indexC] += sum;
						}
				}
			  }
			}
		}
	}
  */
  
  
 
  
  
//--------------------------------------------------- ATTEMPT - 4 -- tiling on kij -------------------------------------------//
// Worse than loop reordering. Means Reduced MatMul dont cache well with tiles. hmmmm...
  
/*
	int i,j,k;
	
	int sum, indexC, Nby2=N>>1;
    int rowA, iter, colB, rowAby2, rowAN, rowA1N,iterN;
	int b = 16;
	for(i=0;i<N; i=i+b){
		for(j=0; j<N; j=j+b){ 
			for (k=0; k<N; k=k+b){
			
				for(iter=k; iter < k+b;++iter){
				  	iterN=iter*N;
				  	
				  	for(rowA=i; rowA < i+b; rowA+=2){
				  		rowAby2 = (rowA>>1)*Nby2;
				  		rowAN=rowA*N+iter;
				  		rowA1N=(rowA+1)*N+iter;
				  		
				  		for(colB=j;colB<j+b;colB+=2){
				  			sum = 0;
							indexC = rowAby2 + (colB>>1);
							sum += matA[rowAN] * matB[iterN + colB];
							sum += matA[rowA1N] * matB[iterN + colB];
							sum += matA[rowAN] * matB[iterN + (colB+1)];
							sum += matA[rowA1N] * matB[iterN + (colB+1)];
							output[indexC] += sum;
				  		
				  		}
				  	}
				  }	
				
			}
		}
	}
  */
  
  
//-------------------------------------------- ATTEMPT - 3 -- loop reordering k,i,j ---------------------------------------------//
// Better Execution time than i,k,j. Progresss
  
  
  /*
  int sum, indexC, Nby2=N>>1;
  int rowA, iter, colB, rowAby2, rowAN, rowA1N,iterN;
  
  
  for(iter=0; iter < N;++iter){
  	iterN=iter*N;
  	for(rowA=0; rowA < N; rowA+=2){
  		rowAby2 = (rowA>>1)*Nby2;
  		rowAN=rowA*N+iter;
  		rowA1N=(rowA+1)*N+iter;
  		for(colB=0;colB<N;colB+=2){
  			sum = 0;
    		indexC = rowAby2 + (colB>>1);
		    sum += matA[rowAN] * matB[iterN + colB];
		    sum += matA[rowA1N] * matB[iterN + colB];
		    sum += matA[rowAN] * matB[iterN + (colB+1)];
		    sum += matA[rowA1N] * matB[iterN + (colB+1)];
		    output[indexC] += sum;
  		
  		}
  	}
  }
  */
  
  
  
//------------------------------------------ ATTEMPT - 2 -- loop reordering i,k,j ------------------------------------------//
// Basic loop reordering. Obvv better than Vanilla
  /*
  int sum, indexC, Nby2=N>>1;
  int rowA, iter, colB, rowAby2, rowAN, rowA1N;
  for(rowA = 0; rowA < N; rowA +=2){
  	rowAby2 = (rowA>>1)*Nby2;
  	rowAN=rowA*N;
  	rowA1N=rowAN+N;
  	for(iter = 0; iter < N; ++iter){
    	for(colB = 0; colB < N; colB += 2){
    		sum = 0;
    		indexC = rowAby2 + (colB>>1);
		    sum += matA[rowAN + iter] * matB[iter * N + colB];
		    sum += matA[rowA1N + iter] * matB[iter * N + colB];
		    sum += matA[rowAN + iter] * matB[iter * N + (colB+1)];
		    sum += matA[rowA1N + iter] * matB[iter * N + (colB+1)];
		    output[indexC] += sum;
		    }
	}
  }
  */
  
  
//-------------------------------------------- ATTEMPT - 1 -- normal MatMul -----------------------------------------------//
// Vanilla Matrix Mul. The slowest one can find
  /*
	int i, j, k,val;
	for(i=0; i<N; i++){
			val=(i>>1)*(N>>1);
			for(k=0; k<N; k++){
				for(j=0; j<N; j++){ 
					output[val+(j>>1)] += matA[i*N+k] * matB[k*N+j]; 
				} 
			}
		}
	*/

