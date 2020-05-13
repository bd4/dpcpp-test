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

      result_type operator() (arg_type arg) {
          return v_;
      }

    private:
      result_type v_;
};


template <typename Result, typename Arg>
class linearfn {
    public:
      using result_type = Result;
      using arg_type = Arg;

      linearfn(result_type m, result_type b) : m_(m), b_(b) {}

      result_type operator() (arg_type arg) {
          return m_ * arg + b_;
      }

    private:
      result_type m_;
      result_type b_;
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


template <typename F, typename E>
class binaryexpr {
  public:
    binaryexpr(F&& f, E&& e0, E&& e1)
    : f_(std::forward<F>(f)),
      e0_(std::forward<E>(e0)),
      e1_(std::forward<E>(e1))
    {}

    template <typename Arg>
    auto operator() (Arg arg);

  private:
    F f_;
    E e0_;
    E e1_;
};


template <typename F, typename E>
template <typename Arg>
auto binaryexpr<F, E>::operator()(Arg arg) {
    return f_(e0_(arg), e1_(arg));
}


template <typename F, typename E>
auto mkexpr(F&& f, E&& e1, E&& e2)
{
  return binaryexpr<F, E>(std::forward<F>(f),
                          std::forward<E>(e1),
                          std::forward<E>(e2));
}
