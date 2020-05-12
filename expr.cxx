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


template <typename T>
struct plus {
  auto operator()(T left, T right) -> T {
    return left + right;
  }
};

template <typename T>
struct minus {
  auto operator()(T left, T right) -> T {
    return left + right;
  }
};

template <typename T>
struct times {
  auto operator()(T left, T right) -> T {
    return left * right;
  }
};


template <typename Result, typename Arg>
class constfn {
    public:
      using result_type = Result;
      using arg_type = Arg;

      constfn(result_type v) : v_(v) {}

      Result operator() (arg_type arg) {
          return v_;
      }

    private:
      result_type v_;
};


template <typename F, typename T>
class binaryclosure {
  public:
    binaryclosure(F&& f, T&& arg0, T&& arg1)
    : f_(std::forward<F>(f)),
      arg0_(std::forward<T>(arg0)),
      arg1_(std::forward<T>(arg1))
    {}

    T operator() () {
      return f_(arg0_, arg1_);
    }

  private:
    F f_;
    T arg0_;
    T arg1_;
};


template <typename F, typename E, typename T, typename Arg>
class binaryexpr {
  public:
    binaryexpr(F&& f, E&& e0, E&& e1)
    : f_(std::forward<F>(f)),
      e0_(std::forward<E>(e0)),
      e1_(std::forward<E>(e1))
    {}

    T operator() (Arg arg) {
      return f_(e0_(arg), e1_(arg));
    }

  private:
    F f_;
    E e0_;
    E e1_;
};


int main(int argc, char **argv) {
    constexpr std::size_t N = 16;
    std::vector<double> h_x(N);

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

    using dfn = constfn<double, size_t>;
    using plusexpr = binaryexpr<plus<double>, dfn, double, size_t>;
    auto k_expr = plusexpr(plus<double>{}, dfn{2.0}, dfn{7.5});

    {
        sycl::buffer<double, 1> buf{h_x.data(), sycl::range<1>{N}};
        auto e = q.submit([&](sycl::handler &cgh) {
          auto acc = buf.get_access<sycl::access::mode::write>(cgh);
          cgh.parallel_for(sycl::range<1>(N),
          [=](sycl::item<1> item) mutable {
             int i = item.get_linear_id();
             acc[i] = k_expr(i);
          });
        });
    }

    for (int i=0; i<N; i++) {
        std::cout << h_x[i] << std::endl;
    }
}
