NVCC = nvcc
CXXFLAGS = -std=c++17
OPENCV_FLAGS = $(shell pkg-config --cflags --libs opencv4)

TARGET = cuda_image_processor
SOURCE = src/main.cu

.PHONY: all build run clean

all: build

build:
	$(NVCC) $(CXXFLAGS) $(SOURCE) -o $(TARGET) $(OPENCV_FLAGS)

run: build
	./$(TARGET) 100

clean:
	rm -f $(TARGET)
	rm -rf output_images