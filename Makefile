# Thin wrapper around CMake. Prefer: make cpu  (CI / no GPU)
BUILD_DIR ?= build
CMAKE ?= cmake
CXX ?= g++

.PHONY: all cpu cuda test clean help

all: cpu

help:
	@echo "make cpu    - host-only build (no nvcc required)"
	@echo "make cuda   - enable CUDA if nvcc is on PATH"
	@echo "make test   - run CTest on the current build/"
	@echo "make clean  - remove build/"

cpu:
	$(CMAKE) -S . -B $(BUILD_DIR) -DCKLAB_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_CXX_COMPILER=$(CXX)
	$(CMAKE) --build $(BUILD_DIR) -j

cuda:
	$(CMAKE) -S . -B $(BUILD_DIR) -DCKLAB_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_CXX_COMPILER=$(CXX)
	$(CMAKE) --build $(BUILD_DIR) -j

test:
	ctest --test-dir $(BUILD_DIR) --output-on-failure

clean:
	rm -rf $(BUILD_DIR)
