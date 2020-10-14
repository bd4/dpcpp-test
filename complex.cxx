#include <CL/sycl.hpp>
#include <complex>
#include <iostream>

#define OP *
#define OP_STR " * "

using namespace cl::sycl;

namespace kernels {
class array_op_kernel {};
}  // namespace kernels

int main(int argc, char **argv) {
    constexpr int N = 16;
    using dcomplex = std::complex<double>;
    dcomplex h_a[N];
    dcomplex h_b[N];
    dcomplex h_c[N];
    int i;

    for (i = 0; i < N; i++) {
        h_a[i] = dcomplex(i, -i);
        h_b[i] = dcomplex(i,  i);
    }

    std::string arg_device_type;
    if (argc < 2) {
        arg_device_type = "gpu";
    } else {
        arg_device_type = argv[1];
    }

    queue q;
    if (arg_device_type == "cpu") {
        q = queue(cpu_selector());
    } else if (arg_device_type == "gpu") {
        q = queue(gpu_selector());
    } else if (arg_device_type == "host") {
        q = queue(host_selector());
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
    std::cout << "[" << type << "] " << dev.get_info<info::device::name>()
              << " {" << dev.get_info<info::device::vendor>() << "}"
              << std::endl;

    {
        // scope for device vector, to force synchronize at the end and allow
        // host access without explicit host accessor
        buffer<dcomplex, 1> d_a_buf(h_a, range<1>(N));
        buffer<dcomplex, 1> d_b_buf(h_b, range<1>(N));
        buffer<dcomplex, 1> d_c_buf(h_c, range<1>(N));

        q.submit([&](handler &cgh) {
            auto d_a_read = d_a_buf.get_access<access::mode::read>(cgh);
            auto d_b_read = d_b_buf.get_access<access::mode::read>(cgh);
            auto d_c_write =
                d_c_buf.get_access<access::mode::discard_write>(cgh);

            cgh.parallel_for<kernels::array_op_kernel>(
                range<1>(N), [=](id<1> idx) {
                    int i = idx[0];
                    d_c_write[i] = d_a_read[i] OP d_b_read[i];
                });
        });
    }
    q.wait();

    for (i = 0; i < N; i++) {
        std::cout << i << ": " << h_a[i] << OP_STR << h_b[i]
                  << " = " << h_c[i] << std::endl;
    }

    return 0;
}
