#include <pthread.h>
#include <immintrin.h>

// Create other necessary functions here


//ATTEMPT 2 - Multi-Threading with Vectorization
// Combining AVX with pthreads. Cuz every core has a vector processor 

int tile_size,t_count;
int Np,Nby2,*outputp,*A,*Bt,*B;

void* parallel(void* arg) {
	
	// Matrix size is tile_size*tile_size
	int N=tile_size;
	
	// ii and jj represent the starting row and column index for matrix multiplication
	int ii = ((int*)arg)[0], jj = ((int*)arg)[1];
	
  	int i,j,k,index,indexN;
  
  	__m256i a0,b0,c0,m;
  
  	int val,l;
  	int temp[8];
  	
  	for(i=ii;i<ii+N;i++){
  		for(j=jj;j<jj+N;j++){
  			m = _mm256_setzero_si256();
  			for(k=0;k<Np;k+=8){
	  			a0 = _mm256_loadu_si256((__m256i *)&A[i*Np+k]);
		  		b0 = _mm256_loadu_si256((__m256i *)&B[j*Np+k]);
		  		c0 = _mm256_mullo_epi32(a0, b0);
	  			m = _mm256_add_epi32(m, c0);
  			}
  			val=0;
  			_mm256_storeu_si256((__m256i *)&temp, m);
  			
  			for(l=0;l<8;++l){
  				val+=temp[l];
  			}
  			outputp[i*Nby2+j]=val;
  		}
  	}
  	
  	free(arg);
    return 0;
    
}

// Fill in this function
void multiThread(int N, int *matA, int *matB, int *output)
{
	Np=N;
	outputp=output;
  	
  	Nby2=N>>1;
  	
  	A=new int[Nby2*N];
	Bt=new int[N*N];
	B=new int[Nby2*N];
  	
  	int i,j,k,index,indexN;
  
  	__m256i a0,b0,c0,m;
  
  for(i=0;i<N;++i){
  	for(j=0;j<N;++j){
  		Bt[j*N+i]=matB[i*N+j];
  	}
  }
  
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
  	
	t_count = 4;
	tile_size = Nby2/t_count;
	// Total 16 threads taken as my device has 8 cores, each with dual thread capacity.
	pthread_t threads[t_count][t_count];
	
    for(i=0; i<t_count; i++){
        for(j=0; j<t_count; j++){
            int *s;
            s = (int*) malloc(sizeof(int)*2);
            
            s[0] = i*tile_size;
            s[1] = j*tile_size;
            pthread_create(&threads[i][j], NULL, &parallel, s);
        }
    }
    
    for(i=0; i<t_count; i++){
        for(j=0; j<t_count; j++){
            pthread_join(threads[i][j], NULL);
        }
    }
}





//---------------------------------------------------- ATTEMPT 1 - Basic Multi-Threading -------------------------------------------//
// Using pthreads to divide the work between cores. 
/*

#include <pthread.h>
#include <immintrin.h>

// Create other necessary functions here

int tile_size,t_count;
int Np,*matAp, *matBp, *outputp;

void* parallel(void* arg) {

	int i = ((int*)arg)[0], j = ((int*)arg)[1];
    
    int sum, indexC, Nby2 = Np>>1;
    int rowA, iter, colB, rowAby2, rowAN, rowA1N;
    
	for(rowA = 0; rowA < tile_size; rowA +=2){
	  	rowAby2 = ((i+rowA)>>1)*Nby2;
	  	rowAN=(i+rowA)*Np;
	  	rowA1N=rowAN+Np;
	  	for(iter = 0; iter < Np; ++iter){
			for(colB = 0; colB < tile_size; colB += 2){
				sum = 0;
				indexC = rowAby2 + ((colB+j)>>1);
				sum += matAp[rowAN + iter] * matBp[iter * Np + colB+j];
				sum += matAp[rowA1N + iter] * matBp[iter * Np + colB+j];
				sum += matAp[rowAN + iter] * matBp[iter * Np + (colB+1+j)];
				sum += matAp[rowA1N + iter] * matBp[iter * Np + (colB+1+j)];
				outputp[indexC] += sum;
			}
		}
	}
    
    free(arg);
    
    return 0;
    
}

// Fill in this function
void multiThread(int N, int *matA, int *matB, int *output)
{
	Np=N;
	matAp=matA;
	matBp=matB;
	outputp=output;
  
	int i,j,k;
	t_count = 4;
	tile_size = N/t_count;
	
	pthread_t threads[t_count][t_count];
	
    for(i=0; i<t_count; i++){
        for(j=0; j<t_count; j++){
            int *s;
            s = (int*) malloc(sizeof(int)*2);
            
            s[0] = i*tile_size;
            s[1] = j*tile_size;

            pthread_create(&threads[i][j], NULL, &parallel, s);
        }
    }
    
    for(i=0; i<t_count; i++){
        for(j=0; j<t_count; j++){
            pthread_join(threads[i][j], NULL);
        }
    } 

}

*/

