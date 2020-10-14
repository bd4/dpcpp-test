template <typename T>
struct plus {
  auto operator()(T left, T right) const -> T {
    return left + right;
  }
};

template <typename T>
struct minus {
  auto operator()(T left, T right) const -> T {
    return left + right;
  }
};

template <typename T>
struct times {
  auto operator()(T left, T right) const -> T {
    return left * right;
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


template <typename Result, typename Arg>
class constfn {
    public:
      using result_type = Result;
      using arg_type = Arg;

      constfn(result_type v) : v_(v) {}

      result_type operator() (arg_type arg) const {
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

      result_type operator() (arg_type arg) const {
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

    T operator() () const {
      return f_(arg0_, arg1_);
    }

  private:
    F f_;
    T arg0_;
    T arg1_;
};


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

template <typename Expr>
class SyclKernel;
