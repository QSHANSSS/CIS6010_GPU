#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <bits/stdc++.h>
#include <vector>
#include <iostream>
#include <string.h>
#include "stopwatch.h"

//#include "../SHA256.h"

#define WIN_SIZE 16  //window length
#define PRIME 3   //
#define MODULUS 256
#define TARGET 0
using namespace std;
uint64_t prime[16]={3,9,27,81,243,729,2187,6561,19683,59049,177147,531441,1594323,4782969,14348907,43046721};


uint64_t hash_func(const char *input, unsigned int pos)
{
	// put your hash function implementation here
	uint64_t temp[WIN_SIZE];
	uint64_t hash=0;
	uint64_t sum=0;
	for(int j=0;j<WIN_SIZE;j=j+2){
		temp[j]=int(input[pos+WIN_SIZE-1-j])*prime[j];
		temp[j+1]=int(input[pos+WIN_SIZE-1-j-1]*prime[j+1]);
		sum=sum+temp[j]+temp[j+1];
	}
	/*for(int i=0;i<WIN_SIZE;i++)//reuse window
	{
		//unsigned char temp=temp[j];
		//for (int k=0;k<(i+1);k++)
	 	//{
	 		//temp[i]=temp[i]*prime[i];
		//}
		 hash+=int(temp[i])*(pow(PRIME, i+1));//hash+=input[pos+15-i]*3^(i+1)
		 hash = hash + temp[i];
	}*/

	return sum;
}

uint8_t cdc_new(const char *buff,int buff_size,int *chunk_bound)
{  
   uint8_t chunk=0;
   int last_boundary=0;
   static int i;	
   chunk_bound[0]=0;
   unsigned char temp1[4];
   unsigned char temp2[4];
   temp1[0]=buff[16];temp1[1]=buff[17];temp1[2]=buff[18];temp1[3]=buff[19];//temp1[4]=buff[20];
    temp2[0]=buff[32];temp2[1]=buff[33];temp2[2]=buff[34];temp2[3]=buff[35];//temp2[4]=buff[36];
   uint64_t a=hash_func(buff,WIN_SIZE);
   for(i=0;i<buff_size-WIN_SIZE;i=i+4)
   {
	 if(i>WIN_SIZE ){
		temp1[3]=buff[i+2];temp2[3]=buff[i+WIN_SIZE+2];
		a=a*PRIME-int(temp1[0])*129140163+int(temp2[0])*PRIME;
		a=a*PRIME-int(temp1[1])*129140163+int(temp2[1])*PRIME;
		a=a*PRIME-int(temp1[2])*129140163+int(temp2[2])*PRIME;
		a=a*PRIME-int(temp1[3])*129140163+int(temp2[3])*PRIME;
		//a=a*PRIME-int(temp1[4])*129140163+int(temp2[4])*PRIME;
		temp1[0]=temp1[1];temp1[1]=temp1[2];temp1[2]=temp1[3];//temp1[3]=temp1[4];
		temp2[0]=temp2[1];temp2[1]=temp2[2];temp2[2]=temp2[3];//temp2[3]=temp2[4];
		//temp[i%16]=buff[i-1+WIN_SIZE];
	 }
           if ((a%MODULUS)==TARGET || i+4-last_boundary>=8191 ) {
		   //total_size+=i-last_boundary;
	        	printf("\n************chunk%d: size=%d***********\n",chunk,i-last_boundary);
          		chunk++;
		    	chunk_bound[chunk]=i+3;
	        	last_boundary=i+3;
	 	   }
	   
   }
    chunk_bound[++chunk]=buff_size;
    return chunk;
   //forloop.stop();
   /*std::cout << "--------------- hash Throughputs ---------------" << std::endl;
	float output_latency_cdc_para = hash.latency() / 1000.0;
	float output_throughput_cdc_para = (14247 * 8 / 1000000.0) / output_latency_cdc_para; // Mb/s
	std::cout << "Output Throughput of hash function: " << output_throughput_cdc_para << " Mb/s."
			<< " (Latency: " << output_latency_cdc_para << "s)." << std::endl;
	std::cout << "--------------- for loop Throughputs ---------------" << std::endl;
	float output_latency_for = forloop.latency() / 1000.0;
	float output_throughput_for = (14247 * 8 / 1000000.0) / output_latency_for; // Mb/s
	std::cout << "Output Throughput of hash function: " << output_throughput_for << " Mb/s."
			<< " (Latency: " << output_latency_for << "s)." << std::endl;*/
  //printf("\n************chunk%d: size=%d***********\n",chunk,buff_size+WIN_SIZE-last_boundary);

}


