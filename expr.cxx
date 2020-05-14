/*
 * =====================================================================================
 *
 *       Filename:  expr.cxx
 *
 *    Description: Use expression mini-language in SYCL kernels. 
 *
 *        Version:  1.0
 *        Created:  05/12/2020 02:28:57 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#include <iostream>

#include <CL/sycl.hpp>

#include "expr.hpp"

using namespace cl;

int main(int argc, char **argv) {
    constexpr std::size_t N = 16;
    std::vector<double> h_x(N);
    std::vector<double> h_y(N);
    std::vector<double> h_z(N);

    for (int i=0; i<N; i++) {
        h_x[i] = static_cast<double>(i);
    }

    auto q = sycl::queue{};
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
              << dev.get_info<sycl::info::device::name>()
              << " {" << dev.get_info<sycl::info::device::vendor>() << "}"
              << std::endl;

    auto k_fun = binaryclosure<plus<double>, double>(plus<double>{}, 7, 11);

    using dconst = constfn<double, double>;
    using dlinear = linearfn<double, double>;
    auto k_expr_const = mkexpr(plus<double>{}, dconst{2.0}, dconst{7.5});
    auto k_expr_linear = mkexpr(plus<double>{}, dlinear{2.0, 1.0},
                                dlinear{0.0, 7.5});
    auto k_expr_complex = mkexpr(plus<double>{}, std::move(k_expr_const),
                                 std::move(k_expr_linear));

    // 4x^2
    auto k_expr_mult = mkexpr(times<double>{}, dlinear{2.0, 0.0},
                              dlinear{2.0, 0.0});
    {
        sycl::buffer<double, 1> buf_x{h_x.data(), sycl::range<1>{N}};
        sycl::buffer<double, 1> buf_y{h_y.data(), sycl::range<1>{N}};
        sycl::buffer<double, 1> buf_z{h_z.data(), sycl::range<1>{N}};

        auto e = q.submit([&](sycl::handler &cgh) {
          auto acc_x = buf_x.get_access<sycl::access::mode::write>(cgh);
          auto acc_y = buf_y.get_access<sycl::access::mode::write>(cgh);
          auto acc_z = buf_z.get_access<sycl::access::mode::write>(cgh);
          cgh.parallel_for<class ExprTest>(sycl::range<1>(N),
          [=](sycl::item<1> item) {
             int i = item.get_linear_id();
             acc_x[i] = k_expr_const(i);
             acc_y[i] = k_expr_linear(i);
             acc_z[i] = k_expr_complex(i);
          });
        });
    }

    for (int i=0; i<N; i++) {
        std::cout << h_x[i] << " + " << h_y[i] << " = " << h_z[i] << std::endl;
    }
}
