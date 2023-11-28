#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <unordered_map>
#include <bits/stdc++.h>
#include "stopwatch.h"
using namespace std;
int match_map(/*unordered_map<string,int> table,*/unsigned char *sha)
{
     //unsigned int index=0;
     //string str=sha;
     std::string str(reinterpret_cast<char*>(sha));
     static unordered_map<string,int> table;
     //unordered_map<string,int> table;
     static int count=0;
     /*if(index==0){
        table[str]=count++;
        return -1;
     }
     index++;*/
     //for (int i = 0; i < count; i++) {
        if( table.find(str) != table.end() )
            return table[str];
        else{
            table[str]=count++;
            return -1;
        }
     //}
    
}

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
int match_map_gpu(/*unordered_map<string,int> table,*/unsigned char *sha)
{
     //unsigned int index=0;
     string str=hash_transform(sha);
     //std::string str(reinterpret_cast<char*>(sha));
     static unordered_map<string,int> table;
     //unordered_map<string,int> table;
     static int count=0;
     /*if(index==0){
        table[str]=count++;
        return -1;
     }
     index++;*/
     //for (int i = 0; i < count; i++) {
        if( table.find(str) != table.end() )
            return table[str];
        else{
            table[str]=count++;
            return -1;
        }
     //}
    
}
