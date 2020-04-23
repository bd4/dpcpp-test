CPP = dpcpp
CPP_FLAGS = -std=c++17

BUILD_DIR=build-intelone

SYCL_SOURCES = $(wildcard *.cxx)
EXES = $(addprefix $(BUILD_DIR)/,$(basename $(SYCL_SOURCES)))

.PHONY: all
all: $(EXES)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/% : %.cxx | $(BUILD_DIR)
	@echo "Compiling "$<
	$(CPP) $(CPP_FLAGS) -o $@ $<

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)/*
