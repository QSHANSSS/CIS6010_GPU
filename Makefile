CXX = g++
CXXFLAGS = -std=c++17 -O3
TARGET = cpu
LDFLAGS = -lpthread -pthread
SRCS = $(wildcard ./*.cpp module/*.cpp)#SHA256.cpp ./module/cdc.cpp
OBJS = $(SRCS:.cpp=.o)

#%.o: %.cpp
#	$(CXX) $(CXXFLAGS) -c $< -o $@ 
$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET) $(OBJS)