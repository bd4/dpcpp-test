#include <iostream>
#include <chrono>
#include <thread>

#ifdef COMPUTECPP_USM
#include <SYCL/experimental/usm_wrapper.h>
#include <CL/sycl.hpp>
#include <SYCL/experimental.hpp>

using namespace cl::sycl;
using namespace cl::sycl::experimental;
#else
#include <CL/sycl.hpp>
using namespace cl::sycl;
namespace sycl = cl::sycl;
#endif

using namespace std::chrono_literals;

int main(int argc, char **argv) {
    const int N = 32;
    int i;

    std::string arg_device_type;
    if (argc < 2) {
        arg_device_type = "gpu";
    } else {
        arg_device_type = argv[1];
    }

    queue q;
    if (arg_device_type == "cpu") {
        q = queue( cpu_selector() );
    } else if (arg_device_type == "gpu") {
        q = queue( gpu_selector() );
    } else if (arg_device_type == "host") {
        q = queue( host_selector() );
    } else {
        std::cout << "Usage: " << argv[0] << " cpu|gpu|host" << std::endl;
        return 1;
    }

    auto dev = q.get_device();
    std::string type;
    if (dev.is_cpu()) {
        type = "CPU  ";
    } else if (dev.is_gpu()) {
        type = "GPU  ";
    } else if (dev.is_host()) {
        type = "HOST ";
    } else {
        type = "OTHER";
    }
    std::cout << "[" << type << "] "
              << dev.get_info<info::device::name>()
              << " {" << dev.get_info<info::device::vendor>() << "}"
              << std::endl;

    // test sycl usm C based api
    int *h_a_ptr = static_cast<int *>(malloc(N*sizeof(int)));

#ifdef COMPUTECPP_USM
    auto d_a_ptr = malloc_device<int>(N, q);
    auto d_a_ptr_capture = usm_wrapper<int>{d_a_ptr};
#else
    auto d_a_ptr = malloc_device<int>(N, q);
    if (d_a_ptr == nullptr) {
        std::cout << "Error: unable to allocat device memory" << std::endl;
        return 1;
    }
    auto d_a_ptr_capture = d_a_ptr;
#endif

    auto e0 = q.submit([&](handler & cgh) {
        cgh.parallel_for<class FillVector>(range<1>(N), [=](id<1> idx) {
            int i = idx[0];
            d_a_ptr_capture[i] = i*i;
        });
    });
    e0.wait();

    // TODO: fixes race on gpu, cpu and host work as expected
    //std::this_thread::sleep_for(5s);

    q.memcpy(h_a_ptr, d_a_ptr, N*sizeof(int));
    q.wait();

    for (i=0; i<N; i++) {
        std::cout << i << ": " << h_a_ptr[i] << std::endl;
    }

#ifdef COMPUTECPP_USM
    sycl::experimental::free(d_a_ptr, q);
#else
    sycl::free(d_a_ptr, q);
#endif
    free(h_a_ptr);

    return 0;
}
