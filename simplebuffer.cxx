#include <iostream>
#include <chrono>
#include <thread>

#include <CL/sycl.hpp>

using namespace std::chrono_literals;

using namespace cl::sycl;

int main(int argc, char **argv) {
    constexpr int N = 16;
    int h_a[N];
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

    {
        // scope for device vector, to force synchronize at the end and allow
        // host access without explicit host accessor
        buffer<int, 1> d_a_buf(h_a, range<1>(N));

        q.submit([&](handler & cgh) {
            auto d_a_write = d_a_buf.get_access<access::mode::discard_write>(cgh);

            cgh.parallel_for<class FillVector>(range<1>(N), [=](id<1> idx) {
                d_a_write[idx[0]] = idx[0]*idx[0];
            });
        });
    }
    q.wait();

    for (i=0; i<N; i++) {
        std::cout << i << ": " << h_a[i] << std::endl;
    }

    return 0;
}
