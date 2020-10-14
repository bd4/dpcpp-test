SYCL_CXX ?= $(ONEAPI_ROOT)/compiler/latest/linux/bin/dpcpp
SYCL_CXX_FLAGS = -std=c++17
COMPLEX_FLAGS = -device-math-lib=fp32,fp64

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

.PHONY: clean
clean:
	rm -f $(BUILD_DIR)/*
