#include <iostream>

#include <CL/sycl.hpp>
#include <CL/sycl/usm.hpp>

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

    /*
    shared_int_array s_b{N, shared_int_alloc(q)};
    int * s_b_data = s_b.data();

    for (i=0; i<N; i++) {
        s_b[i] = i;
    }
    for (i=0; i<N; i++) {
        std::cout << i << ": " << s_b_data[i] << endl;
    }
    */

    // Q: vector doesn't actually alloc here?
    device_int_vector d_b{N, device_int_alloc(q)};
    host_int_vector h_b{N, host_int_alloc(q)};

    std::cout << "submit d_b init" << std::endl;
    auto event2 = q.submit([&](handler & cgh) {
        // Q: buf wrapper doesn't seem to be necessary?
        //auto buf_b = buffer(d_b.data());
        //auto acc_b = buf_b.get_access<access::write>(cgh);
        
        // Q: forces vector alloc in device context?
        auto d_b_data = d_b.data();
        cgh.parallel_for<class DeviceVectorInit>(range<1>(N), [=](id<1> idx) {
            //acc_b[idx] = idx*idx;
            //d_b[idx] = idx*idx;
            d_b_data[idx] = idx*idx;
        });
    });
    event2.wait();

    //std::copy(d_b, h_b);
    q.memcpy(h_b.data(), d_b.data(), N*sizeof(int));
    q.wait();

    for (i=0; i<N; i++) {
        std::cout << i << ": " << h_b[i] << std::endl;
    }

    return 0;
}
