#include "App.h"
#include "SHA256.cuh"
#include <cublas_v2.h>
#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#define cudaCheck(err) (cudaErrorCheck(err, __FILE__, __LINE__))


void cudaErrorCheck(cudaError_t error, const char *file, int line)
{
    if (error != cudaSuccess)
    {
        printf("[CUDA ERROR] at file %s:%d:\n%s: %s\n", file, line,
               cudaGetErrorName(error), cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

CHUNK * GPU_Chunk_Init(/*CHUNK * chunk,*/uint64_t size,  char * data ){
	CHUNK *chunk;
    cudaCheck(cudaMallocManaged(&chunk, sizeof(CHUNK)));	//j = (JOB *)malloc(sizeof(JOB));
	//cudaCheck(cudaMallocManaged(&(chunk->data), size));
	//chunk->data = (unsigned char*)data;
	chunk->size = size;
    for (int i = 0; i < 32; i++)
	{
		chunk->digest[i] = 0xff;
	}
	return chunk;
}

__global__ void sha256_gpu(CHUNK ** chunk, int n, unsigned char * file_data) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < n){
		//uint32_t message_update[64]={0};
		//uint32_t message_padding[64]={0};
		SHA256_CTX ctx;
		sha256_init(&ctx);
		sha256_update(&ctx, chunk[i]->size,chunk[i]->start,file_data);
		sha256_padding(&ctx, chunk[i]->digest);
	}
}

void RUN_SHA256_GPU(CHUNK ** chunk, int chunk_num, unsigned char * file_data){

    int BlockSize = 4;
	int numBlocks = (chunk_num + BlockSize - 1) / BlockSize;
	sha256_gpu <<< numBlocks, BlockSize >>> (chunk, chunk_num,file_data);
	cudaCheck(cudaDeviceSynchronize()); // wait for kernel to finish
}

int main(int argc, char** argv) {
    if (argc != 2) {
	std::cout << "usage: ./gpu <path>" << "\n";
	exit(0);
    }
    stopwatch cpu_sha_timer;
    stopwatch cpu_cdc_timer;
	stopwatch cpu_dedup_timer;
    stopwatch gpu_sha_timer;
    std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    char* buf = 0;//(char*)malloc(size);
	cudaCheck(cudaMallocManaged(&buf, (size+1)*sizeof(char)));
    file.read(buf, size);

	// FILE * f = 0;
	// f = fopen(argv[1], "rb");
	// unsigned char * gpu_buffer = 0;
	// cudaCheck(cudaMallocManaged(&gpu_buffer, (size+1)*sizeof(char)));
	// fread(gpu_buffer, size, 1, f);
	// fclose(f);
	
	int *boundary=(int *)malloc(sizeof(int)*(size));
	uint8_t chunk_num;
	char  *chunk=(char *)malloc(sizeof(char)*1000);

	CHUNK ** GPU_Chunk;
    cpu_cdc_timer.start();
    chunk_num=cdc_base(buf,size,boundary);
    cpu_cdc_timer.stop();
    cudaCheck(cudaMallocManaged(&GPU_Chunk, chunk_num * sizeof(CHUNK *)));
	int cpu_dedup_result[chunk_num]={0};
	int gpu_dedup_result[chunk_num]={0};
	// for(int i=0;i<chunk_num;i++){
	// 	std::string str_chunk(buf+boundary[i], boundary[i+1]-boundary[i]);
    //     std::cout << "Chunk" << i<<": " << str_chunk << std::endl;
	// 	std::cout<<"\n";
	// }

    for(int i=0;i<chunk_num;i++)
    {
       	memcpy((chunk),buf+boundary[i],boundary[i+1]-boundary[i]); 
        //int chunk_size=boundary[i+1]-boundary[i];
        cpu_sha_timer.start();
        auto hash = sha_256(chunk, boundary[i+1]-boundary[i]);
        cpu_sha_timer.stop();

        for (int j = 0; j < 32; j++) 
            printf("%02x", static_cast<uint8_t>(hash[j]));
        std::cout<<"\n";

        GPU_Chunk[i]=GPU_Chunk_Init(boundary[i+1]-boundary[i],chunk);
		GPU_Chunk[i]->start=boundary[i];

   	    //char* HashArray = reinterpret_cast<char*>(hash.get());
   	    unsigned char* hashPtr = (unsigned char *)malloc(sizeof(unsigned char)*32);
   	    hashPtr=reinterpret_cast<unsigned char*>(hash.get());

	    cpu_dedup_timer.start();
	    cpu_dedup_result[i]=match_map(hashPtr);
	    cpu_dedup_timer.stop();

	    if(cpu_dedup_result[i]==-1)
		    std::cout<< "chunk"<<i<<" "<<"is distinct!"<<std::endl;
	    else
		    std::cout<< "chunk"<<i<<" "<<"is deduplicated with chunk"<<cpu_dedup_result[i]<<std::endl;
    	std::cout<<"\n";
    }
	//GPU SHA256
	cudaCheck(cudaMemcpyToSymbol(dev_k, host_k, sizeof(host_k), 0, cudaMemcpyHostToDevice));
	unsigned char * dev_buf=nullptr; //char * host_buf=nullptr;
	cudaCheck(cudaMalloc(&dev_buf, size));
	//cudaCheck(cudaMallocManaged(&dev_buf, size*sizeof(unsigned char)));
	cudaCheck(cudaMemcpy(dev_buf, buf, size, cudaMemcpyHostToDevice));

    gpu_sha_timer.start();
    RUN_SHA256_GPU(GPU_Chunk,chunk_num,dev_buf);
    gpu_sha_timer.stop();

	cudaCheck(cudaGetLastError());      // check for errors from kernel run
	//cudaCheck(cudaMemcpy(buf, dev_buf, size, cudaMemcpyDeviceToHost));
	//cudaDeviceReset();
	//printf("%s", buf);
	int match_chunk=0;
	for	(int m = 0; m < chunk_num; m++){
		gpu_dedup_result[m]=match_map_gpu(GPU_Chunk[m]->digest);
		if(gpu_dedup_result[m]!=cpu_dedup_result[m]){
			std::cout<< "deduplication result of chunk"<<m<<" "<<"cannot match!"<<std::endl;
		}
		else
			match_chunk++;

		for (int n = 0; n < 32; n++){
	 		printf("%02x", GPU_Chunk[m]->digest[n]);
			//printf("%s \n", hash_transform(GPU_Chunk[m]->digest));
		}
		std::cout<<"\n";

		if(gpu_dedup_result[m]==-1)
		    std::cout<< "gpu:chunk"<<m<<" "<<"is distinct!"<<std::endl;
	    else
		    std::cout<< "gpu: chunk"<<m<<" "<<"is deduplicated with chunk"<<gpu_dedup_result[m]<<std::endl;
    	std::cout<<"\n";
	}

	if(match_chunk==chunk_num)
		std::cout << "GPU-Computed Deduplication Result of " << argv[1] << " can be verified by CPU Version!\n\n";
	else
		std::cout << "GPU-Computed Deduplication Result of " << argv[1] << " is wrong!!\n\n";

    std::cout << "--------------- cdc_cpu Throughputs ---------------" << std::endl;
	float output_latency_cdc_cpu = cpu_cdc_timer.latency() / 1000.0;
	float output_throughput_cdc_cpu = (size / 1000000.0) / output_latency_cdc_cpu; // Mb/s
	std::cout << "Output Throughput of CPU_CDC: " << output_throughput_cdc_cpu << " Mb/s."
			<< " (Latency: " << output_latency_cdc_cpu << "s)." << std::endl;

    std::cout << "--------------- sha256_cpu Throughputs ---------------" << std::endl;
	float output_latency_cpu = cpu_sha_timer.latency() / 1000.0;
	float output_throughput_cpu = (size / 1000000.0) / output_latency_cpu; // Mb/s
	std::cout << "Output Throughput of CPU_SHA256: " << output_throughput_cpu << " Mb/s."
			<< " (Latency: " << output_latency_cpu << "s)." << std::endl;

	std::cout << "--------------- sha256_gpu Throughputs ---------------" << std::endl;
	float output_latency_gpu = gpu_sha_timer.latency() / 1000.0;
	float output_throughput_gpu = (chunk_num*256 / 1000000.0) / output_latency_gpu; // Mb/s
	std::cout << "Output Throughput of GPU_SHA256: " << output_throughput_gpu << " Mb/s."
			<< " (Latency: " << output_latency_gpu << "s)." << std::endl;

	std::cout << "--------------- deduplication_cpu Throughputs ---------------" << std::endl;
	float output_latency_dedup_cpu = cpu_dedup_timer.latency() / 1000.0;
	float output_throughput_dedup_cpu = (size / 1000000.0) / output_latency_dedup_cpu; // Mb/s
	std::cout << "Output Throughput of CPU_Deduplication: " << output_throughput_dedup_cpu << " Mb/s."
			<< " (Latency: " << output_latency_dedup_cpu << "s)." << std::endl;
}