#include <iostream>

#include <CL/sycl.hpp>

using namespace cl::sycl;

int main() {
    queue myQueue( gpu_selector{} );
    std::cout << "Selected device: " <<
        myQueue.get_device().get_info<info::device::name>() << std::endl;
    std::cout << " -> Device vendor: " <<
        myQueue.get_device().get_info<info::device::vendor>() << std::endl;

    return 0;
}
