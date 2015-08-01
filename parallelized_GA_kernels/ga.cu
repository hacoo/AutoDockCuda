__global__ void evolve_subpopulation(int iterations) {
	const int POP_SIZE = 2 * blockDim.x; //not sure if this is allowed to be const but it should be (also must be power of 2)
	const int GENOME_SIZE = /*tba*/; //idk here like 12 or something
	__shared__ int permutation[POP_SIZE]; //store permutation
	__shared__ float population [POP_SIZE * GENOME_SIZE]; //assume this has been populated


	for (int gen = 0; gen < num_generations; ++gen) {

		//shuffle population

		//parallelized fisher-yates
		const int id = threadIdx.x + blockIdx.x * blockDim.x;
		permutation[2*id] = 2*id;
		permutation[2*id+1] = 2*id+1;
		__syncthreads();

		unsigned int shift = 1;
		unsigned int pos = id * 2;
		int temp;
		while (shift <= blockDim.x) {
			if (curand(&curand_state) & 1) {
				temp = permuation[pos];
				permutation[pos] = permutation[pos+shift];
				permutation[pos+shift] = temp;
			}
			shift = shift << 1;
			pos = (pos & ~shift) | ((pos & shift) >> 1); //not sure what this does i found it on SO and it looks legit, probly something with modulos
			__syncthreads();
		}

		//shuffle the genes
		int my_val1[GENOME_SIZE];
		int my_val2[GENOME_SIZE]; 
		
		cudaMemcpy(population[permutation[2*id]], population[2*id], 
				sizeof(float)*GENOME_SIZE, cudaDeviceToDevice)
		cudaMemcpy(population[permutation[2*id+1]], population[2*id+1], 
				sizeof(float)*GENOME_SIZE, cudaDeviceToDevice)
		__syncthreads();
				
		//endshuffle

		//make babies
		//code here
		//stop making babies
		
		

		//selection
		//code here...
		//end selection

		//repeat
		__syncthreads();
	}
	
}
