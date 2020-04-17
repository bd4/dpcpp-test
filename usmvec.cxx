#include <iostream>
#include <chrono>
#include <thread>

#include <CL/sycl.hpp>
#include <CL/sycl/usm.hpp>

using namespace std::chrono_literals;

using namespace cl::sycl;

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

    // test sycl usm allocator based api using STL
    using device_int_alloc = usm_allocator<int, usm::alloc::device>;
    using shared_int_alloc = usm_allocator<int, usm::alloc::shared>;
    using host_int_alloc = usm_allocator<int, usm::alloc::host>;
    using device_int_vector = std::vector<int,  device_int_alloc>;
    using shared_int_vector = std::vector<int,  shared_int_alloc>;
    using host_int_vector = std::vector<int,  host_int_alloc>;

    device_int_vector d_b{N, device_int_alloc(q)};
    // TODO: why doesn't this work?
    //host_int_vector h_b{N, host_int_alloc(q)};
    std::vector<int> h_b(N);

    int *d_b_data = d_b.data();

    auto e0 = q.submit([&](handler & cgh) {
        cgh.parallel_for<class DeviceVectorInit>(range<1>(N), [=](id<1> idx) {
            d_b_data[idx] = idx*idx;
        });
    });
    e0.wait();

    // TODO: why is this needed, e0.wait is not actually waiting?
    //std::this_thread::sleep_for(5s);

    q.memcpy(h_b.data(), d_b_data, N*sizeof(int));
    q.wait();

    for (i=0; i<N; i++) {
        std::cout << i << ": " << h_b[i] << std::endl;
    }

    return 0;
}
