SYCL_CXX ?= $(ONEAPI_ROOT)/compiler/latest/linux/bin/dpcpp
MKL_ROOT ?= $(ONEAPI_ROOT)/mkl/latest
SYCL_CXX_FLAGS = -std=c++17 -O2 -g
COMPLEX_FLAGS = -device-math-lib=fp32,fp64
#MKL_FLAGS = -DMKL_ILP64 -I$(MKL_ROOT)/include -L$(MKL_ROOT)/lib/intel64 -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core
MKL_FLAGS = -I$(MKL_ROOT)/include -L$(MKL_ROOT)/lib/intel64 -lmkl_sycl -lmkl_intel_lp64 -lmkl_sequential -lmkl_core

# See
# https://github.com/jinge90/llvm/blob/sycl/sycl/doc/extensions/C-CXX-StandardLibrary/DeviceLibExtensions.rst
# and
# https://github.com/jinge90/llvm/blob/sycl/sycl/test/devicelib/std_complex_math_fp64_test.cpp
#DPCPP_LIBDIR ?= /opt/intel/inteloneapi/compiler/latest/linux/lib
#DPCPP_OBJS = libsycl-complex-fp64.o libsycl-cmath-fp64.o
#LIBS = $(addprefix $(DPCPP_LIBDIR)/,$(DPCPP_OBJS))

BUILD_DIR=build-intelone

SYCL_SOURCES = $(wildcard *.cxx)
EXES = $(addprefix $(BUILD_DIR)/,$(basename $(SYCL_SOURCES)))

.PHONY: all
all: $(EXES)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/% : %.cxx | $(BUILD_DIR)
	@echo "Compiling "$<
	$(SYCL_CXX) $(SYCL_CXX_FLAGS) -o $@ $< $(LIBS)

$(BUILD_DIR)/complex : complex.cxx | $(BUILD_DIR)
	@echo "Compiling "$<
	$(SYCL_CXX) $(SYCL_CXX_FLAGS) $(COMPLEX_FLAGS) -o $@ $< $(LIBS)

$(BUILD_DIR)/batched_zgetrs : batched_zgetrs.cxx | $(BUILD_DIR)
	@echo "Compiling "$<
	$(SYCL_CXX) $(SYCL_CXX_FLAGS) $(COMPLEX_FLAGS) $(MKL_FLAGS) -o $@ $< $(LIBS)

$(BUILD_DIR)/sparse_solve_npvt : sparse_solve.cxx | $(BUILD_DIR)
	@echo "Compiling "$<
	$(SYCL_CXX) $(SYCL_CXX_FLAGS) $(COMPLEX_FLAGS) $(MKL_FLAGS) -DNOPIVOT -DOLD_NAME -o $@ $< $(LIBS)

$(BUILD_DIR)/sparse_solve : sparse_solve.cxx | $(BUILD_DIR)
	@echo "Compiling "$<
	$(SYCL_CXX) $(SYCL_CXX_FLAGS) $(COMPLEX_FLAGS) $(MKL_FLAGS) -DOLD_NAME -o $@ $< $(LIBS)

$(BUILD_DIR)/batched_fft : batched_fft.cxx | $(BUILD_DIR)
	@echo "Compiling "$<
	$(SYCL_CXX) $(SYCL_CXX_FLAGS) $(COMPLEX_FLAGS) $(MKL_FLAGS) -o $@ $< $(LIBS)

.PHONY: clean
clean:
	rm -f $(BUILD_DIR)/*
