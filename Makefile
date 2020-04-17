.PHONY: all
all: listdev gpuinfo usmmem usmvec simplebuffer


listdev: listdev.cxx
	dpcpp -std=c++17 -o $@ $<

gpuinfo: gpuinfo.cxx
	dpcpp -std=c++17 -o $@ $<

usmmem: usmmem.cxx
	dpcpp -std=c++17 -o $@ $<

usmvec: usmvec.cxx
	dpcpp -std=c++17 -o $@ $<

simplebuffer: simplebuffer.cxx
	dpcpp -std=c++17 -o $@ $<

.PHONY: clean
clean:
	rm -rf gpuinfo listdev usmmem usmvec simplebuffer
