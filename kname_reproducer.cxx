#include <iostream>
#include <type_traits>

#include <sycl/sycl.hpp>

// replicate how gtensor handles complex types, with gt::complex<T> alias
namespace sycl_cplx {

template <class _Tp, class _Enable = void> class complex;

template <class _Tp>
class complex<
    _Tp, typename std::enable_if<std::is_floating_point<_Tp>::value>::type> {
public:
  typedef _Tp value_type;

  constexpr complex(value_type __re = value_type(),
                    value_type __im = value_type())
      : __re_(__re), __im_(__im) {}

private:
  value_type __re_;
  value_type __im_;
};

} // namespace sycl_cplx

template <typename DerivedExpression> class expression {
protected:
  expression() = default;

public:
  using derived_type = DerivedExpression;

  const derived_type &derived() const &;
  derived_type &derived() &;
  derived_type derived() &&;
};

template <typename DerivedExpression>
inline auto
expression<DerivedExpression>::derived() const & -> const derived_type & {
  return static_cast<const derived_type &>(*this);
}

template <typename DerivedExpression>
inline auto expression<DerivedExpression>::derived() & -> derived_type & {
  return static_cast<derived_type &>(*this);
}

template <typename DerivedExpression>
inline auto expression<DerivedExpression>::derived() && -> derived_type {
  return static_cast<derived_type &&>(*this);
}

template <typename T> class container : public expression<container<T>> {
public:
  using value_type = std::decay_t<T>;
  using pointer = std::add_pointer_t<value_type>;
  using reference = std::add_lvalue_reference_t<value_type>;
  using self_type = container<T>;
  using base_type = expression<self_type>;

  container(std::size_t count, pointer data) : count_(count), data_(data) {}

  reference operator[](std::size_t i) const { return data_[i]; };
  reference operator()(std::size_t i) const { return data_[i]; };

  std::size_t size() const { return count_; }

private:
  pointer data_;
  std::size_t count_;
};

template <typename E1, typename E2> class Assign1;

// Note: passing expressions by value works
// void assign(E1 lhs, const E2 rhs, sycl::queue &q)

template <typename E1, typename E2>
void assign(E1 &lhs, const E2 &rhs, sycl::queue &q) {
  std::size_t size = lhs.size();
  auto e = q.submit([&](sycl::handler &cgh) {
    using ltype = decltype(lhs);
    using rtype = decltype(rhs);
    using kname = Assign1<ltype, rtype>;
    cgh.parallel_for<kname>(sycl::range<1>(size), [=](sycl::item<1> item) {
      auto i = item.get_id();
      lhs(i) = rhs(i);
    });
  });
}

template <typename T> void test() {
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
  std::cout << "[" << type << "] " << dev.get_info<sycl::info::device::name>()
            << " {" << dev.get_info<sycl::info::device::vendor>() << "}"
            << std::endl;

  auto Adata = sycl::malloc_shared<T>(N, q);
  auto A = container<T>(N, Adata);
  auto Bdata = sycl::malloc_shared<T>(N, q);
  auto B = container<T>(N, Bdata);

  for (int i = 0; i < N; i++) {
    B[i] = T{1.0 * i};
  }

  assign(A, B, q);

  /*
  std::size_t size = A.size();
  auto e = q.submit([&](sycl::handler& cgh) {
    using ltype = decltype(A);
    using rtype = decltype(B);
    using kname = Assign1<ltype, rtype>;
    cgh.parallel_for<kname>(sycl::range<1>(size), [=](sycl::item<1> item) {
      auto i = item.get_id();
      A(i) = B(i);
    });
  });
  */
}

int main(int argc, char **argv) { test<sycl_cplx::complex<double>>(); }
