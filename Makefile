CXX = g++
CXXFLAGS = -std=c++17 -O3
TARGET = cpu
SRC = SHA256.cpp

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $<

GPU_SOURCE_FILE=sha.cu

# optimized binary
gpu: $(GPU_SOURCE_FILE)
	nvcc -std=c++17 --generate-code=arch=compute_75,code=[compute_75,sm_75] $^ -lcublas -o $@

clean:
	rm -f $(TARGET)