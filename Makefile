CXX = g++
CXXFLAGS = -std=c++17 -O3
TARGET = cpu
LDFLAGS = -lpthread -pthread
SRCS = $(wildcard ./*.cpp module/*.cpp)#SHA256.cpp ./module/cdc.cpp
OBJS = $(SRCS:.cpp=.o)
GPU_SOURCE_FILE=App.cu cdc_cpu.cpp deduplication.cpp SHA256.cpp

#%.o: %.cpp
#	$(CXX) $(CXXFLAGS) -c $< -o $@ 
$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

gpu: $(GPU_SOURCE_FILE)
	nvcc -std=c++17 --generate-code=arch=compute_75,code=[compute_75,sm_75] $^ -lcublas -o $@

clean:
	rm -f $(TARGET) $(OBJS)