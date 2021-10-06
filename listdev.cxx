#include <iostream>

#include <CL/sycl.hpp>

using namespace cl::sycl;

int main() {
    auto platformlist = platform::get_platforms();

    std::string type;
    for(auto p : platformlist) {
        decltype(p.get_devices(info::device_type::all)) devicelist;
        devicelist = p.get_devices(info::device_type::all);

        std::cout << p.get_info<info::platform::name>()
                  << " {" << p.get_info<info::platform::vendor>() << "}"
                  << std::endl;
        for(const auto& dev : devicelist) {
            if (dev.is_cpu()) {
                type = "CPU  ";
            } else if (dev.is_gpu()) {
                type = "GPU  ";
            } else if (dev.is_host()) {
                type = "HOST ";
            } else {
                type = "OTHER";
            }

            std::cout << "  [" << type << "] "
                      << dev.get_info<info::device::name>()
                      << " {" << dev.get_info<info::device::vendor>() << "}"
                      << " (" << dev.get_info<info::device::vendor_id>() << ")"
                      << std::endl;
        }
    }

    return 0;
}
