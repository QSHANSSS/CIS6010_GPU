#include "App.h"
#include "SHA256.cuh"
#define MAX_CHUNK_SIZE 256
#define cudaCheck(err) (cudaErrorCheck(err, __FILE__, __LINE__))

char * hash_transform(unsigned char * buff) {
	char * string = (char *)malloc(70);
	int k, i;
	for (i = 0, k = 0; i < 32; i++, k+= 2)
	{
		sprintf(string + k, "%.2x", buff[i]);
		//printf("%02x", buff[i]);
	}
	string[64] = 0;
	return string;
}

void cudaErrorCheck(cudaError_t error, const char *file, int line)
{
    if (error != cudaSuccess)
    {
        printf("[CUDA ERROR] at file %s:%d:\n%s: %s\n", file, line,
               cudaGetErrorName(error), cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

CHUNK * GPU_Chunk_Init(/*CHUNK * chunk,*/uint64_t size, unsigned char * data ){
	CHUNK *chunk;
    cudaCheck(cudaMallocManaged(&chunk, sizeof(CHUNK)));	//j = (JOB *)malloc(sizeof(JOB));
	cudaCheck(cudaMallocManaged(&(chunk->data), 10000));
	chunk->data = data;
	chunk->size = size;
    for (int i = 0; i < 64; i++)
	{
		chunk->digest[i] = 0xff;
	}
	return chunk;
}

__global__ void sha256_gpu(CHUNK ** chunk, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t message[64];
	
	if (i < n){
		SHA256_CTX ctx;
		sha256_init(&ctx);
		sha256_update(&ctx, chunk[i]->data, chunk[i]->size,message);
		sha256_padding(&ctx, chunk[i]->digest,message);
	}
}

void RUN_SHA256_GPU(CHUNK ** chunk, int chunk_num){

    cudaCheck(cudaMemcpyToSymbol(dev_k, host_k, sizeof(host_k), 0, cudaMemcpyHostToDevice));
    int BlockSize = 4;
	int numBlocks = (chunk_num + BlockSize - 1) / BlockSize;
	sha256_gpu <<< numBlocks, BlockSize >>> (chunk, chunk_num);
    cudaCheck(cudaDeviceSynchronize()); // wait for kernel to finish
    cudaCheck(cudaGetLastError());      // check for errors from kernel run
}



int main(int argc, char** argv) {
    if (argc != 2) {
	std::cout << "usage: ./sha <path>" << "\n";
	exit(0);
    }

    stopwatch cpu_sha_timer;
    stopwatch cpu_cdc_timer;
	stopwatch cpu_dedup_timer;
    stopwatch gpu_sha_timer;
    std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    char* buf = (char*)malloc(size);
    file.read(buf, size);
	
	int *boundary=(int *)malloc(sizeof(int)*(size));
	uint8_t chunk_num;
	char  *chunk=(char *)malloc(sizeof(char)*1000);

	CHUNK ** GPU_Chunk;
    cpu_cdc_timer.start();
    chunk_num=cdc_base(buf,size,boundary);
    cpu_cdc_timer.stop();
    cudaCheck(cudaMallocManaged(&GPU_Chunk, chunk_num * sizeof(CHUNK *)));

    for(int i=0;i<chunk_num;i++)
    {
       	memcpy((chunk),buf+boundary[i],boundary[i+1]-boundary[i]); 
        //int chunk_size=boundary[i+1]-boundary[i];
        
        cpu_sha_timer.start();
        auto hash = sha_256(chunk, boundary[i+1]-boundary[i]);
        cpu_sha_timer.stop();

        //CHUNK * GPU_Chunk;

        for (int j = 0; j < 32; j++) 
            printf("%02x", static_cast<uint8_t>(hash[j]));
        std::cout<<"\n";

        GPU_Chunk[i]=GPU_Chunk_Init(/*GPU_Chunk[i],*/boundary[i+1]-boundary[i],(unsigned char *)chunk);

   	    //char* HashArray = reinterpret_cast<char*>(hash.get());
   	    unsigned char* hashPtr = (unsigned char *)malloc(sizeof(unsigned char)*32);
   	    hashPtr=reinterpret_cast<unsigned char*>(hash.get());

	    cpu_dedup_timer.start();
	    int dedup_result=match_map(hashPtr);
	    cpu_dedup_timer.stop();
	    //for (int i = 0; i < 32; ++i) {
        //	printf("%c",hashPtr[i]);
    	//}
    	//std::cout<<"\n";
	    if(dedup_result==-1)
		    std::cout<< "chunk"<<i<<" "<<"is distinct!"<<std::endl;
	    else
		    std::cout<< "chunk"<<i<<" "<<"is deduplicated with chunk"<<dedup_result<<std::endl;
    	std::cout<<"\n";
    }
    gpu_sha_timer.start();
    RUN_SHA256_GPU(GPU_Chunk,chunk_num);
    gpu_sha_timer.stop();

	// for	(int m = 0; m < chunk_num; m++)
	// 	for (int n = 0; n < 32; n++)
	// 		printf("%02x", GPU_Chunk[m]->digest[n]);


    std::cout << "  " << argv[1] << "\n";

    std::cout << "--------------- cdc_cpu Throughputs ---------------" << std::endl;
	float output_latency_cdc_cpu = cpu_cdc_timer.latency() / 1000.0;
	float output_throughput_cdc_cpu = (size / 1000000.0) / output_latency_cdc_cpu; // Mb/s
	std::cout << "Output Throughput of CPU_CDC: " << output_throughput_cdc_cpu << " Mb/s."
			<< " (Latency: " << output_latency_cdc_cpu << "s)." << std::endl;

    std::cout << "--------------- sha256_cpu Throughputs ---------------" << std::endl;
	float output_latency_cpu = cpu_sha_timer.latency() / 1000.0;
	float output_throughput_cpu = (256 / 1000000.0) / output_latency_cpu; // Mb/s
	std::cout << "Output Throughput of CPU_SHA256: " << output_throughput_cpu << " Mb/s."
			<< " (Latency: " << output_latency_cpu << "s)." << std::endl;
	
	std::cout << "--------------- sha256_gpu Throughputs ---------------" << std::endl;
	float output_latency_gpu = gpu_sha_timer.latency() / 1000.0;
	float output_throughput_gpu = (256 / 1000000.0) / output_latency_gpu; // Mb/s
	std::cout << "Output Throughput of GPU_SHA256: " << output_throughput_gpu << " Mb/s."
			<< " (Latency: " << output_latency_gpu << "s)." << std::endl;

	std::cout << "--------------- deduplication_cpu Throughputs ---------------" << std::endl;
	float output_latency_dedup_cpu = cpu_dedup_timer.latency() / 1000.0;
	float output_throughput_dedup_cpu = (size / 1000000.0) / output_latency_dedup_cpu; // Mb/s
	std::cout << "Output Throughput of CPU_Deduplication: " << output_throughput_dedup_cpu << " Mb/s."
			<< " (Latency: " << output_latency_dedup_cpu << "s)." << std::endl;
}