#include <vector>
#include <stdint.h>
#include <stdlib.h>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cstdio>
#include <cassert>
#include <array>
#include <bit>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <ios>
#include <string>
#include <vector>
#include "stopwatch.h"
using namespace std;
using Hash = std::unique_ptr<uint8_t[]>;
uint8_t cdc_new(char *buff,int buff_size,int *chunk_bound);
uint8_t cdc(const char *buff,int buff_size,int *chunk_bound);
int cdc_base(char *buff, unsigned int buff_size,int *chunk_bound);
Hash sha_256(char* bytes, uint64_t len);
int match_map(/*unordered_map<string,int> table,*/unsigned char *sha);
int match_map_gpu(/*unordered_map<string,int> table,*/unsigned char *sha);
