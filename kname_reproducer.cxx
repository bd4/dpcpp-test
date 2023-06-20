#include <iostream>
#include <type_traits>

#include <sycl/sycl.hpp>

#define _SYCL_CPLX_NAMESPACE gt::sycl_cplx
#include "sycl_ext_complex.hpp"

// replicate how gtensor handles complex types, with gt::complex<T> alias
namespace gt
{

template <typename T>
using complex = gt::sycl_cplx::complex<T>;

using gt::sycl_cplx::pow;

template <typename T1, typename T2>
struct KName;

} // namespace gt


template <typename T>
struct plus {
  auto operator()(T left, T right) const -> T {
    return left + right;
  }
};


template <typename DerivedExpression>
class expression {
protected:
  expression() = default;

public:
  using derived_type = DerivedExpression;

  const derived_type& derived() const&;
  derived_type& derived() &;
  derived_type derived() &&;
};


template <typename DerivedExpression>
inline auto expression<DerivedExpression>::derived() const&
-> const derived_type& {
    return static_cast<const derived_type&>(*this);
}


template <typename DerivedExpression>
inline auto expression<DerivedExpression>::derived() & -> derived_type& {
    return static_cast<derived_type&>(*this);
}


template <typename DerivedExpression>
inline auto expression<DerivedExpression>::derived() && -> derived_type {
    return static_cast<derived_type&&>(*this);
}


template <typename F, typename E1, typename E2>
class binaryexpr;

template <typename F, typename E1, typename E2>
class binaryexpr : public expression<binaryexpr<F, E1, E2>> {
  public:
    using self_type = binaryexpr<F, E1, E2>;
    using base_type = expression<self_type>;
    using function_type = F;
    using expression_type_1 = E1;
    using expression_type_2 = E2;

    binaryexpr(F&& f, E1&& e1, E2&& e2)
    : f_(std::forward<F>(f)),
      e1_(std::forward<E1>(e1)),
      e2_(std::forward<E2>(e2))
    {}

    template <typename Arg>
    auto operator() (Arg arg) const;

  private:
    F f_;
    E1 e1_;
    E2 e2_;
};


template <typename F, typename E1, typename E2>
template <typename Arg>
auto binaryexpr<F, E1, E2>::operator()(Arg arg) const {
    return f_(e1_(arg), e2_(arg));
}


template <typename F, typename E1, typename E2>
auto mkexpr(F&& f, E1&& e1, E2&& e2)
{
  return binaryexpr<F, E1, E2>(std::forward<F>(f),
                               std::forward<E1>(e1),
                               std::forward<E2>(e2));
}

template <typename T>
class container : public expression<container<T>> {
  public:
    using value_type = std::decay_t<T>;
    using pointer = std::add_pointer_t<value_type>;
    using reference = std::add_lvalue_reference_t<value_type>;
    using self_type = container<T>;
    using base_type = expression<self_type>;

    container(std::size_t count, pointer data) : count_(count), data_(data)
    {}

    reference operator[] (std::size_t i) const { return data_[i]; };
    reference operator() (std::size_t i) const { return data_[i]; };

    std::size_t size() const { return count_; }

  private:
    pointer data_;
    std::size_t count_;
};

template <typename E1, typename E2>
class Assign1;

template <typename E1, typename E2>
void assign(E1 &lhs, const E2& rhs, sycl::queue &q)
{
  std::size_t size = lhs.size();
  auto e = q.submit([&](sycl::handler& cgh) {
    using ltype = decltype(lhs);
    using rtype = decltype(rhs);
    using kname = Assign1<ltype, rtype>;
    cgh.parallel_for<kname>(sycl::range<1>(size), [=](sycl::item<1> item) {
      auto i = item.get_id();
      lhs(i) = rhs(i);
    });
  });
}

template <typename T>
void test() {
    constexpr std::size_t N = 16;

    auto q = sycl::queue{};
    auto dev = q.get_device();
    std::string type;
    if (dev.is_cpu()) {
        type = "CPU  ";
    } else if (dev.is_gpu()) {
        type = "GPU  ";
    } else {
        type = "OTHER";
    }
    std::cout << "[" << type << "] "
              << dev.get_info<sycl::info::device::name>()
              << " {" << dev.get_info<sycl::info::device::vendor>() << "}"
              << std::endl;

    auto Adata = sycl::malloc_shared<T>(N, q);
    auto A = container<T>(N, Adata);
    auto Bdata = sycl::malloc_shared<T>(N, q);
    auto B = container<T>(N, Bdata);
    auto Cdata = sycl::malloc_shared<T>(N, q);
    auto C = container<T>(N, Cdata);

    for (int i = 0; i < N; i++) {
      B[i] = T{1.0 * i};
      C[i] = T{1.0 * i};
    }

    auto k_expr_containers = mkexpr(plus<T>{}, std::move(B), std::move(C));

    assign(C, k_expr_containers, q);
}

int main(int argc, char **argv) {
  test<gt::complex<double>>();
}
