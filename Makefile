CXX = g++
CXXFLAGS = -std=c++20 -O3
TARGET = cpu
SRC = SHA256.cpp

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)
