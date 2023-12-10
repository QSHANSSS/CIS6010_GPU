#include "App.h"
#include "SHA256.cuh"
#include "new_sha256.cuh"
#include <cublas_v2.h>
#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#define cudaCheck(err) (cudaErrorCheck(err, __FILE__, __LINE__))
#define NO_STREAM
#define SET_CHUNK_PER_STREAM 7
#define DIGEST_SIZE 32
stopwatch gpu_sha_timer;
cudaEvent_t beg, stop;

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

__global__ void sha256_stream_kernel(unsigned char hash[], int n, unsigned char * file_data,int* boundary) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n){
		//uint32_t message_update[64]={0};
		//uint32_t message_padding[64]={0};
		SHA256_CTX ctx;
		//sha256_init(&ctx);
		sha256_update(&ctx, boundary[i+1]-boundary[i],boundary[i]-boundary[0],file_data);
		sha256_padding(&ctx, hash+32*i);
	} 
}

void RUN_SHA256_GPU_STREAM(int boundary[], int chunk_num, char data[],uint32_t file_len, unsigned char digest[]){
	const int CHUNK_PER_STREAM=(chunk_num>SET_CHUNK_PER_STREAM)? SET_CHUNK_PER_STREAM:chunk_num;
	const int nStreams = (chunk_num+CHUNK_PER_STREAM-1)/CHUNK_PER_STREAM;
	cudaEventCreate(&beg);
    cudaEventCreate(&stop);
	cudaStream_t streams[nStreams];
	cudaStream_t streams2[nStreams];
	for (int i = 0; i < nStreams; i++) {
  		cudaStreamCreate(&streams[i]);
		cudaStreamCreate(&streams2[i]);
	}
	int offset=0;
	unsigned char * dev_data=nullptr; 
	//unsigned char *digest=nullptr;
	unsigned char *dev_digest=nullptr;
	//cudaMallocHost(&digest, 32);
	cudaMalloc(&dev_digest, DIGEST_SIZE*CHUNK_PER_STREAM);
	int currentChunk_num=CHUNK_PER_STREAM;
	
	for(int i=0; i<nStreams; i++)
	{
		//printf("current_chunk:%d\n",currentChunk_num);
		cudaMalloc(&dev_data, boundary[i*CHUNK_PER_STREAM+currentChunk_num]-boundary[i*CHUNK_PER_STREAM]);
		cudaEventRecord(beg);
		gpu_sha_timer.start();
		cudaMemcpyAsync(dev_data,
                    data+offset,
                    boundary[i*CHUNK_PER_STREAM+currentChunk_num]-boundary[i*CHUNK_PER_STREAM],
                    cudaMemcpyHostToDevice,
                    streams[i] );

		int BlockSize = 256;
		int numBlocks = (chunk_num + BlockSize - 1) / BlockSize;
		
		sha256_stream_kernel <<< numBlocks, BlockSize >>> (dev_digest, 
								currentChunk_num,
								dev_data,
								boundary+i*CHUNK_PER_STREAM);
		
		cudaMemcpyAsync( digest+i*CHUNK_PER_STREAM*DIGEST_SIZE,
                    dev_digest,
                    DIGEST_SIZE*currentChunk_num,
                    cudaMemcpyDeviceToHost,
                    streams2[i] );
		
		gpu_sha_timer.stop();
		offset +=boundary[i*CHUNK_PER_STREAM+currentChunk_num]-boundary[i*CHUNK_PER_STREAM];
		currentChunk_num = ((i+1) != (nStreams - 1)) ?  CHUNK_PER_STREAM 
							:(chunk_num % CHUNK_PER_STREAM)? chunk_num % CHUNK_PER_STREAM
							: CHUNK_PER_STREAM;
	} 
	cudaDeviceSynchronize();	
	cudaEventRecord(stop);
	cudaEventSynchronize(beg);
	cudaEventSynchronize(stop);

}

__global__ void sha256_gpu(CHUNK ** chunk, int n, unsigned char * file_data) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < n){
		//uint32_t message_update[64]={0};
		//uint32_t message_padding[64]={0};
		SHA256_CTX ctx;
		//sha256_init(&ctx);
		sha256_update(&ctx, chunk[i]->size,chunk[i]->start,file_data);
		sha256_padding(&ctx, chunk[i]->digest);
	}
}

void RUN_SHA256_GPU(CHUNK ** chunk, int chunk_num, unsigned char * file_data){

    int BlockSize = 8;
	int numBlocks = (chunk_num + BlockSize - 1) / BlockSize;
	sha256_gpu <<< numBlocks, BlockSize >>> (chunk, chunk_num,file_data);
	cudaDeviceSynchronize(); // wait for kernel to finish
}

int main(int argc, char** argv) {
    if (argc != 2) {
	std::cout << "usage: ./gpu <path>" << "\n";
	exit(0);
    }
    stopwatch cpu_sha_timer;
    stopwatch cpu_cdc_timer;
	stopwatch cpu_dedup_timer;
    stopwatch gpu_sha_nostream_timer;
    std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    char* buf = 0;//(char*)malloc(size);
	cudaCheck(cudaMallocManaged(&buf, (size+1)*sizeof(char)));
    file.read(buf, size);

	cudaCheck(cudaMemcpyToSymbol(new_dev_k, new_host_k, sizeof(new_host_k), 0, cudaMemcpyHostToDevice));
	//int *boundary=(int *)malloc(sizeof(int)*(size));
	int *boundary=0;
	cudaCheck(cudaMallocManaged(&boundary, (1000)*sizeof(int)));
	char  *chunk=(char *)malloc(sizeof(char)*10000);
	uint8_t chunk_num;
	CHUNK ** GPU_Chunk;

    cpu_cdc_timer.start();
    chunk_num=cdc_base(buf,size,boundary);
    cpu_cdc_timer.stop();

    cudaCheck(cudaMallocManaged(&GPU_Chunk, chunk_num * sizeof(CHUNK *)));
	int cpu_dedup_result[chunk_num]={0};
	int gpu_dedup_result[chunk_num]={0};
	int gpu_dedup_result_2[chunk_num]={0};
	unsigned char *host_digest;
	// cudaMallocHost((void**)&host_digest, chunk_num * sizeof(unsigned char *));
	// for (int i = 0; i < chunk_num; i++) 
        cudaMallocHost(&host_digest, chunk_num*DIGEST_SIZE * sizeof(unsigned char));
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

        GPU_Chunk[i]=GPU_Chunk_Init(boundary[i+1]-boundary[i],chunk);
		GPU_Chunk[i]->start=boundary[i];

		for (int j = 0; j < 32; j++){
            printf("%02x", static_cast<uint8_t>(hash[j]));
			//printf("GPU:%02x", static_cast<uint8_t>(host_ctx->hash[j]));
		}
        std::cout<<"\n";

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
	cudaCheck(cudaMemcpy(dev_buf, buf, size, cudaMemcpyHostToDevice));

	RUN_SHA256_GPU_STREAM(boundary,chunk_num,buf,size,host_digest);
	cudaCheck(cudaGetLastError());      // check for errors from kernel run
#ifdef NO_STREAM
	gpu_sha_nostream_timer.start();
    RUN_SHA256_GPU(GPU_Chunk,chunk_num,dev_buf);
    gpu_sha_nostream_timer.stop();
#endif
	
	//cudaCheck(cudaMemcpy(buf, dev_buf, size, cudaMemcpyDeviceToHost));
	//printf("%s", buf);
	int match_chunk=0;
	for	(int m = 0; m < chunk_num; m++){
#ifdef NO_STREAM
		//gpu_dedup_result_2[m]=match_map_gpu(GPU_Chunk[m]->digest);
#endif
		gpu_dedup_result_2[m]=match_map_gpu(host_digest+m*DIGEST_SIZE);
		if(gpu_dedup_result_2[m]!=cpu_dedup_result[m]){
			std::cout<< "deduplication result of chunk"<<m<<" "<<"cannot match!"<<std::endl;
		}
		else
			match_chunk++;

		for (int n = 0; n < 32; n++){
			#ifdef NO_STREAM
	 		//printf("%02x", GPU_Chunk[m]->digest[n]);
			#endif
			printf("%02x", host_digest[m*DIGEST_SIZE+n]);
		}
		std::cout<<"\n";

		if(gpu_dedup_result_2[m]==-1)
		    std::cout<< "gpu:chunk"<<m<<" "<<"is distinct!"<<std::endl;
	    else
		    std::cout<< "gpu: chunk"<<m<<" "<<"is deduplicated with chunk"<<gpu_dedup_result_2[m]<<std::endl;
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
	float output_latency_gpu = gpu_sha_timer.latency() /(1000.0);
	float output_throughput_gpu = (size / 1000000.0) / output_latency_gpu; // Mb/s
	std::cout << "Output Throughput of GPU_SHA256: " << output_throughput_gpu << " Mb/s."
			<< " (Latency: " << output_latency_gpu << "s)." << std::endl;
#ifdef NO_STREAM	
	std::cout << "--------------- nostream_sha256_gpu Throughputs ---------------" << std::endl;
	float output_latency_gpu2 = gpu_sha_nostream_timer.latency() / 1000.0;
	float output_throughput_gpu2 = (size / 1000000.0) / output_latency_gpu2; // Mb/s
	std::cout << "Output Throughput of NO_STREAM_GPU_SHA256: " << output_throughput_gpu2 << " Mb/s."
			<< " (Latency: " << output_latency_gpu2 << "s)." << std::endl;
#endif
	std::cout << "--------------- deduplication_cpu Throughputs ---------------" << std::endl;
	float output_latency_dedup_cpu = cpu_dedup_timer.latency() / 1000.0;
	float output_throughput_dedup_cpu = (size / 1000000.0) / output_latency_dedup_cpu; // Mb/s
	std::cout << "Output Throughput of CPU_Deduplication: " << output_throughput_dedup_cpu << " Mb/s."
			<< " (Latency: " << output_latency_dedup_cpu << "s)." << std::endl;
		float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, beg, stop);
    elapsed_time /= 1000.; // Convert to seconds
	double flops = (double)2 * size;
	// printf(
    //     "Average elapsed time: (%7.6f) s, performance: (%7.2f) GFLOPS. size: (%ld).\n",
    //     elapsed_time ,
    //     (flops * 1e-9) / elapsed_time,
    //     size);

}