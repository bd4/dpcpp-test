CPP = dpcpp
CPP_FLAGS = -std=c++14

# See
# https://github.com/jinge90/llvm/blob/sycl/sycl/doc/extensions/C-CXX-StandardLibrary/DeviceLibExtensions.rst
# and
# https://github.com/jinge90/llvm/blob/sycl/sycl/test/devicelib/std_complex_math_fp64_test.cpp
ONEAPI_LIBDIR=/opt/intel/inteloneapi/compiler/latest/linux/lib
ONEAPI_OBJS = libsycl-complex-fp64.o libsycl-cmath-fp64.o
LIBS = $(addprefix $(ONEAPI_LIBDIR)/,$(ONEAPI_OBJS))

BUILD_DIR=build-intelone

SYCL_SOURCES = $(wildcard *.cxx)
EXES = $(addprefix $(BUILD_DIR)/,$(basename $(SYCL_SOURCES)))

.PHONY: all
all: $(EXES)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/% : %.cxx | $(BUILD_DIR)
	@echo "Compiling "$<
	$(CPP) $(CPP_FLAGS) -o $@ $< $(LIBS)

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)/*
