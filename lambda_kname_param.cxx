#include <iostream>
#include <chrono>
#include <thread>

#include <CL/sycl.hpp>

using namespace std::chrono_literals;

using namespace cl::sycl;

template <typename F>
class KernelOp;

template <typename F>
void index_op_kernel(queue& q, int n, int *h_a, F&& op) {
    {
        // scope for device vector, to force synchronize at the end and allow
        // host access without explicit host accessor
        buffer<int, 1> d_a_buf(h_a, range<1>(n));

        q.submit([&](handler & cgh) {
            auto d_a_write = d_a_buf.get_access<access::mode::discard_write>(cgh);

            using kname = KernelOp<decltype(op)>;
            cgh.parallel_for<kname>(range<1>(n), [=](id<1> idx) {
                d_a_write[idx[0]] = op(idx[0]);
            });
        });
    }
    q.wait();

    for (int i=0; i<n; i++) {
        std::cout << i << ": " << h_a[i] << std::endl;
    }
}

int square(int i) {
    return i*i;
}

int main(int argc, char **argv) {
    constexpr int N = 16;
    int h_a[N];

    std::string arg_device_type;
    if (argc < 2) {
        arg_device_type = "default";
    } else {
        arg_device_type = argv[1];
    }

    queue q;
    if (arg_device_type == "cpu") {
        q = queue( cpu_selector{} );
    } else if (arg_device_type == "gpu") {
        q = queue( gpu_selector{} );
    } else if (arg_device_type == "default") {
        q = queue( default_selector{} );
    } else {
        std::cout << "Usage: " << argv[0] << " cpu|gpu|default" << std::endl;
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

    std::cout << "==== square" << std::endl;
    index_op_kernel(q, N, h_a, [=](int i) { return i*i; });

    std::cout << "==== 10 * i" << std::endl;
    int b = 10;
    index_op_kernel(q, N, h_a, [=](int i) { return b*i; });
    /*
    {
        // scope for device vector, to force synchronize at the end and allow
        // host access without explicit host accessor
        buffer<int, 1> d_a_buf(h_a, range<1>(N));

        q.submit([&](handler & cgh) {
            auto d_a_write = d_a_buf.get_access<access::mode::discard_write>(cgh);

            using kname = KernelOp<decltype(square)>;
            cgh.parallel_for<>(range<1>(N), [=](id<1> idx) {
                d_a_write[idx[0]] = square(idx[0]);
            });
        });
    }
    q.wait();

    for (i=0; i<N; i++) {
        std::cout << i << ": " << h_a[i] << std::endl;
    }
    */

    return 0;
}
