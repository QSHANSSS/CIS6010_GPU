#include "App.h"
#include "SHA256.cuh"
#define MAX_CHUNK_SIZE 256

void SHA256_GPU(JOB ** jobs, int chunk_size){

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
	char  *chunk=(char *)malloc(sizeof(char)*size);

    cpu_cdc_timer.start();
    chunk_num=cdc_base(buf,size,boundary);
    cpu_cdc_timer.stop();
    
    for(int i=0;i<chunk_num;i++)
    {
       	memcpy((chunk),buf+boundary[i],boundary[i+1]-boundary[i]); 
        cpu_sha_timer.start();
        auto hash = sha_256(chunk, size);
        cpu_sha_timer.stop();

        gpu_sha_timer.start();
        auto hash = sha_256(chunk, size);
        gpu_sha_timer.stop();
        
        for (int i = 0; i < 32; i++) 
            printf("%02x", static_cast<uint8_t>(hash[i]));
        std::cout<<"\n";
        
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

	std::cout << "--------------- deduplication_cpu Throughputs ---------------" << std::endl;
	float output_latency_dedup_cpu = cpu_dedup_timer.latency() / 1000.0;
	float output_throughput_dedup_cpu = (size / 1000000.0) / output_latency_dedup_cpu; // Mb/s
	std::cout << "Output Throughput of CPU_Deduplication: " << output_throughput_dedup_cpu << " Mb/s."
			<< " (Latency: " << output_latency_dedup_cpu << "s)." << std::endl;
}