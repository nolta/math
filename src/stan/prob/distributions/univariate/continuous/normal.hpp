#ifndef STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__NORMAL_HPP
#define STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__NORMAL_HPP

#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/utility/enable_if.hpp>

#include <stan/agrad/partials_vari.hpp>
#include <stan/math.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/traits.hpp>

namespace stan {

  namespace prob {

    /**
     * The log of the normal density for the specified scalar(s) given
     * the specified mean(s) and deviation(s). y, mu, or sigma can
     * each be either a scalar or a std::vector. Any vector inputs
     * must be the same length.
     *
     * <p>The result log probability is defined to be the sum of the
     * log probabilities for each observation/mean/deviation triple.
     * @param y (Sequence of) scalar(s).
     * @param mu (Sequence of) location parameter(s)
     * for the normal distribution.
     * @param sigma (Sequence of) scale parameters for the normal
     * distribution.
     * @return The log of the product of the densities.
     * @throw std::domain_error if the scale is not positive.
     * @tparam T_y Underlying type of scalar in sequence.
     * @tparam T_loc Type of location parameter.
     */
    template <bool propto, 
              typename T_y, typename T_loc, typename T_scale>
    typename boost::enable_if_c<is_var_or_arithmetic<T_y,T_loc,T_scale>::value,
                                typename return_type<T_y,T_loc,T_scale>::type>::type
    normal_log(const T_y& y, const T_loc& mu, const T_scale& sigma) {
      static const char* function = "stan::prob::normal_log(%1%)";

      using std::log;
      using stan::is_constant_struct;
      using stan::math::check_positive;
      using stan::math::check_finite;
      using stan::math::check_not_nan;
      using stan::math::check_consistent_sizes;
      using stan::math::value_of;
      using stan::prob::include_summand;

      // check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(mu) 
            && stan::length(sigma)))
        return 0.0;

      // set up return value accumulator
      double logp(0.0);

      // validate args (here done over var, which should be OK)
      check_not_nan(function, y, "Random variable", &logp);
      check_finite(function, mu, "Location parameter", &logp);
      check_positive(function, sigma, "Scale parameter", &logp);
      check_consistent_sizes(function,
                             y,mu,sigma,
                             "Random variable","Location parameter",
                             "Scale parameter",
                             &logp);
      // check if no variables are involved and prop-to
      if (!include_summand<propto,T_y,T_loc,T_scale>::value)
        return 0.0;
      
      // set up template expressions wrapping scalars into vector views
      agrad::OperandsAndPartials<T_y, T_loc, T_scale> operands_and_partials(y, mu, sigma);

      VectorView<const T_y> y_vec(y);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> sigma_vec(sigma);
      size_t N = max_size(y, mu, sigma);

      DoubleVectorView<true,is_vector<T_scale>::value> inv_sigma(length(sigma));
      DoubleVectorView<include_summand<propto,T_scale>::value,is_vector<T_scale>::value> log_sigma(length(sigma));
      for (size_t i = 0; i < length(sigma); i++) {
        inv_sigma[i] = 1.0 / value_of(sigma_vec[i]);
        if (include_summand<propto,T_scale>::value)
          log_sigma[i] = log(value_of(sigma_vec[i]));
      }

      for (size_t n = 0; n < N; n++) {
        // pull out values of arguments
        const double y_dbl = value_of(y_vec[n]);
        const double mu_dbl = value_of(mu_vec[n]);
      
        // reusable subexpression values
        const double y_minus_mu_over_sigma 
          = (y_dbl - mu_dbl) * inv_sigma[n];
        const double y_minus_mu_over_sigma_squared 
          = y_minus_mu_over_sigma * y_minus_mu_over_sigma;

        static double NEGATIVE_HALF = - 0.5;

        // log probability
        if (include_summand<propto>::value)
          logp += NEG_LOG_SQRT_TWO_PI;
        if (include_summand<propto,T_scale>::value)
          logp -= log_sigma[n];
        if (include_summand<propto,T_y,T_loc,T_scale>::value)
          logp += NEGATIVE_HALF * y_minus_mu_over_sigma_squared;

        // gradients
        double scaled_diff = inv_sigma[n] * y_minus_mu_over_sigma;
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] -= scaled_diff;
        if (!is_constant_struct<T_loc>::value)
          operands_and_partials.d_x2[n] += scaled_diff;
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n] 
            += -inv_sigma[n] + inv_sigma[n] * y_minus_mu_over_sigma_squared;
      }
      return operands_and_partials.to_var(logp);
    }

    template <typename T_y, typename T_loc, typename T_scale>
    inline
    typename return_type<T_y,T_loc,T_scale>::type
    normal_log(const T_y& y, const T_loc& mu, const T_scale& sigma) {
      return normal_log<false>(y,mu,sigma);
    }
    
    /**
     * The log of the normal density for the specified scalar(s) given
     * the specified mean(s) and deviation(s). y, mu, or sigma can
     * each be either a scalar or a std::vector. Any vector inputs
     * must be the same length.
     *
     * <p>The result log probability is defined to be the sum of the
     * log probabilities for each observation/mean/deviation triple.
     * @param y (Sequence of) scalar(s).
     * @param mu (Sequence of) location parameter(s)
     * for the normal distribution.
     * @param sigma (Sequence of) scale parameters for the normal
     * distribution.
     * @return The log of the product of the densities.
     * @throw std::domain_error if the scale is not positive.
     * @tparam T_y Underlying type of scalar in sequence.
     * @tparam T_loc Type of location parameter.
     */
    template <bool propto, 
              typename T_y, typename T_s, typename T_n, typename T_loc, typename T_scale>
    typename boost::enable_if_c<is_var_or_arithmetic<T_y,T_s,T_loc,T_scale>::value,
                                typename return_type<T_y,T_s,T_loc,T_scale>::type>::type
    normal_ss_log(const T_y& y_bar, const T_s& s_squared, const T_n& n_obs, const T_loc& mu, const T_scale& sigma) {
      static const char* function = "stan::prob::normal_log(%1%)";

      using std::log;
      using stan::is_constant_struct;
      using stan::math::check_positive;
      using stan::math::check_finite;
      using stan::math::check_not_nan;
      using stan::math::check_consistent_sizes;
      using stan::math::value_of;
      using stan::prob::include_summand;

      // check if any vectors are zero length
      if (!(stan::length(y_bar) 
            && stan::length(s_squared) 
            && stan::length(n_obs) 
            && stan::length(mu) 
            && stan::length(sigma)))
        return 0.0;

      // set up return value accumulator
      double logp(0.0);

      // validate args (here done over var, which should be OK)
      check_not_nan(function, y_bar, "Location parameter sufficient statistic", &logp);
      check_not_nan(function, s_squared, "Scale parameter sufficient statistic", &logp);
      check_not_nan(function, n_obs, "Number of observations", &logp);
      check_finite(function, n_obs, "Number of observations", &logp);
      check_positive(function, n_obs, "Number of observations", &logp);
      check_finite(function, mu, "Location parameter", &logp);
      check_positive(function, sigma, "Scale parameter", &logp);
      check_consistent_sizes(function,
                             y_bar, s_squared, n_obs, mu, sigma,
                             "Location parameter sufficient statistic",
                             "Scale parameter sufficient statistic",
                             "Number of observations",
                             "Location parameter",
                             "Scale parameter",
                             &logp);
      // check if no variables are involved and prop-to
      if (!include_summand<propto,T_y,T_s,T_loc,T_scale>::value)
        return 0.0;
      
      // set up template expressions wrapping scalars into vector views
      agrad::OperandsAndPartials<T_y, T_s, T_loc, T_scale> operands_and_partials(y_bar, s_squared, mu, sigma);

      VectorView<const T_y> y_bar_vec(y_bar);
      VectorView<const T_s> s_squared_vec(s_squared);
      VectorView<const T_n> n_obs_vec(n_obs);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> sigma_vec(sigma);
      size_t N = max_size(y_bar, s_squared, n_obs, mu, sigma);
      
      for (size_t i = 0; i < N; i++) {
        const double y_bar_dbl = value_of(y_bar_vec[i]);
        const double s_squared_dbl = value_of(s_squared_vec[i]);
        const double n_obs_dbl = n_obs_vec[i];
        const double mu_dbl = value_of(mu_vec[i]);
        const double sigma_dbl = value_of(sigma_vec[i]);
        const double sigma_squared = pow(sigma_dbl,2);
        
        if (include_summand<propto>::value)
          logp += NEG_LOG_SQRT_TWO_PI * n_obs_dbl;
        
        if (include_summand<propto, T_scale>::value)
          logp -= n_obs_dbl*log(sigma_dbl);
        
           
        const double cons_expr = 
          (s_squared_dbl
           + n_obs_dbl * pow(y_bar_dbl - mu_dbl, 2));
          
        logp -= cons_expr / (2 * sigma_squared);
        

        // gradients
        if (!is_constant_struct<T_y>::value || !is_constant_struct<T_loc>::value) {
          const double common_derivative = n_obs_dbl * (mu_dbl - y_bar_dbl) / sigma_squared;
          if (!is_constant_struct<T_y>::value)
            operands_and_partials.d_x1[i] += common_derivative;
          if (!is_constant_struct<T_loc>::value)
            operands_and_partials.d_x3[i] -= common_derivative;
        }
        if (!is_constant_struct<T_s>::value)
          operands_and_partials.d_x2[i] -= 
            1 / (2 * sigma_squared);
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x4[i] 
            += cons_expr / pow(sigma_dbl, 3) - n_obs_dbl / sigma_dbl;
      }
      return operands_and_partials.to_var(logp);
    }

    template <typename T_y, typename T_s, typename T_n, typename T_loc, typename T_scale>
    inline
    typename return_type<T_y,T_s,T_loc,T_scale>::type
    normal_ss_log(const T_y& y_bar, const T_s& s_squared, const T_n& n_obs, const T_loc& mu, const T_scale& sigma) {
      return normal_ss_log<false>(y_bar,s_squared,n_obs,mu,sigma);    
    }
       /**
     * Calculates the normal cumulative distribution function for the given
     * variate, location, and scale.
     * 
     * \f$\Phi(x) = \frac{1}{\sqrt{2 \pi}} \int_{-\inf}^x e^{-t^2/2} dt\f$.
     * 
     * Errors are configured by policy.  All variables must be finite
     * and the scale must be strictly greater than zero.
     * 
     * @param y A scalar variate.
     * @param mu The location of the normal distribution.
     * @param sigma The scale of the normal distriubtion
     * @return The unit normal cdf evaluated at the specified arguments.
     * @tparam T_y Type of y.
     * @tparam T_loc Type of mean parameter.
     * @tparam T_scale Type of standard deviation paramater.
     * @tparam Policy Error-handling policy.
     */
    template <typename T_y, typename T_loc, typename T_scale>
    typename return_type<T_y,T_loc,T_scale>::type
    normal_cdf(const T_y& y, const T_loc& mu, const T_scale& sigma) {
      static const char* function = "stan::prob::normal_cdf(%1%)";

      using stan::math::check_positive;
      using stan::math::check_finite;
      using stan::math::check_not_nan;
      using stan::math::value_of;
      using stan::math::check_consistent_sizes;
      using stan::math::INV_SQRT_2;

      double cdf(1.0);

      // check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(mu) 
            && stan::length(sigma)))
        return cdf;

      check_not_nan(function, y, "Random variable", &cdf);
      check_finite(function, mu, "Location parameter", &cdf);
      check_not_nan(function, sigma, "Scale parameter", &cdf);
      check_positive(function, sigma, "Scale parameter", &cdf);
      check_consistent_sizes(function,
                             y,mu,sigma,
                             "Random variable","Location parameter",
                             "Scale parameter",
                             &cdf);

     agrad::OperandsAndPartials<T_y, T_loc, T_scale> 
       operands_and_partials(y, mu, sigma);

      VectorView<const T_y> y_vec(y);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> sigma_vec(sigma);
      size_t N = max_size(y, mu, sigma);
      const double SQRT_TWO_OVER_PI = std::sqrt(2.0 / stan::math::pi());

      for (size_t n = 0; n < N; n++) {
        const double y_dbl = value_of(y_vec[n]);
        const double mu_dbl = value_of(mu_vec[n]);
        const double sigma_dbl = value_of(sigma_vec[n]);
        const double scaled_diff = (y_dbl - mu_dbl) / (sigma_dbl * SQRT_2);
        double cdf_;
        if (scaled_diff < -37.5 * INV_SQRT_2)
          cdf_ = 0.0;
        else if (scaled_diff < -5.0 * INV_SQRT_2)
          cdf_ = 0.5 * erfc(-scaled_diff);
        else if (scaled_diff > 8.25 * INV_SQRT_2)
          cdf_ = 1;
        else
          cdf_ = 0.5 * (1.0 + erf(scaled_diff));

        // cdf
        cdf *= cdf_;

        // gradients
        const double rep_deriv = SQRT_TWO_OVER_PI * 0.5 
          * exp(-scaled_diff * scaled_diff) / cdf_ / sigma_dbl;
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] += rep_deriv;
        if (!is_constant_struct<T_loc>::value)
          operands_and_partials.d_x2[n] -= rep_deriv;
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n] -= rep_deriv * scaled_diff * SQRT_2;
      }

      if (!is_constant_struct<T_y>::value)
        for (size_t n = 0; n < stan::length(y); ++n) 
          operands_and_partials.d_x1[n] *= cdf;
      if (!is_constant_struct<T_loc>::value)
        for (size_t n = 0; n < stan::length(mu); ++n) 
          operands_and_partials.d_x2[n] *= cdf;
      if (!is_constant_struct<T_scale>::value)
        for (size_t n = 0; n < stan::length(sigma); ++n) 
          operands_and_partials.d_x3[n] *= cdf;

      return operands_and_partials.to_var(cdf);
    }

    template <typename T_y, typename T_loc, typename T_scale>
    typename return_type<T_y,T_loc,T_scale>::type
    normal_cdf_log(const T_y& y, const T_loc& mu, const T_scale& sigma) {
      static const char* function = "stan::prob::normal_cdf_log(%1%)";

      using stan::math::check_positive;
      using stan::math::check_finite;
      using stan::math::check_not_nan;
      using stan::math::check_consistent_sizes;
      using stan::math::value_of;
      using stan::math::INV_SQRT_2;

      double cdf_log(0.0);
      // check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(mu) 
            && stan::length(sigma)))
        return cdf_log;

      check_not_nan(function, y, "Random variable", &cdf_log);
      check_finite(function, mu, "Location parameter", &cdf_log);
      check_not_nan(function, sigma, "Scale parameter", &cdf_log);
      check_positive(function, sigma, "Scale parameter", &cdf_log);
      check_consistent_sizes(function,
                             y,mu,sigma,
                             "Random variable","Location parameter",
                             "Scale parameter", &cdf_log);

      agrad::OperandsAndPartials<T_y, T_loc, T_scale> 
        operands_and_partials(y, mu, sigma);

      VectorView<const T_y> y_vec(y);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> sigma_vec(sigma);
      size_t N = max_size(y, mu, sigma);
      double log_half = std::log(0.5);  
    
      const double SQRT_TWO_OVER_PI = std::sqrt(2.0 / stan::math::pi());
      for (size_t n = 0; n < N; n++) {
        const double y_dbl = value_of(y_vec[n]);
        const double mu_dbl = value_of(mu_vec[n]);
        const double sigma_dbl = value_of(sigma_vec[n]);

        const double scaled_diff = (y_dbl - mu_dbl) / (sigma_dbl * SQRT_2);
        
        double one_p_erf;
        if (scaled_diff < -37.5 * INV_SQRT_2)
          one_p_erf = 0.0;
        else if (scaled_diff < -5.0 * INV_SQRT_2)
          one_p_erf =  erfc(-scaled_diff);
        else if (scaled_diff > 8.25 * INV_SQRT_2)
          one_p_erf = 2.0;
        else
          one_p_erf = 1.0 + erf(scaled_diff);

        // log cdf
        cdf_log += log_half + log(one_p_erf);

        // gradients
        const double rep_deriv = SQRT_TWO_OVER_PI 
          * exp(-scaled_diff * scaled_diff) / one_p_erf;
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] += rep_deriv / sigma_dbl;
        if (!is_constant_struct<T_loc>::value)
          operands_and_partials.d_x2[n] -= rep_deriv / sigma_dbl;
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n] -= rep_deriv * scaled_diff 
            * stan::math::SQRT_2 / sigma_dbl;
      }
      return operands_and_partials.to_var(cdf_log);
    }

    template <typename T_y, typename T_loc, typename T_scale>
    typename return_type<T_y,T_loc,T_scale>::type
    normal_ccdf_log(const T_y& y, const T_loc& mu, const T_scale& sigma) {
      static const char* function = "stan::prob::normal_ccdf_log(%1%)";

      using stan::math::check_positive;
      using stan::math::check_finite;
      using stan::math::check_not_nan;
      using stan::math::check_consistent_sizes;
      using stan::math::value_of;
      using stan::math::INV_SQRT_2;

      double ccdf_log(0.0);
      // check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(mu) 
            && stan::length(sigma)))
        return ccdf_log;

      check_not_nan(function, y, "Random variable", &ccdf_log);
      check_finite(function, mu, "Location parameter", &ccdf_log);
      check_not_nan(function, sigma, "Scale parameter", &ccdf_log);
      check_positive(function, sigma, "Scale parameter", &ccdf_log);
      check_consistent_sizes(function,
                             y,mu,sigma,
                             "Random variable","Location parameter",
                             "Scale parameter", &ccdf_log);

      agrad::OperandsAndPartials<T_y, T_loc, T_scale>
        operands_and_partials(y, mu, sigma);

      VectorView<const T_y> y_vec(y);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> sigma_vec(sigma);
      size_t N = max_size(y, mu, sigma);
      double log_half = std::log(0.5);  
    
      const double SQRT_TWO_OVER_PI = std::sqrt(2.0 / stan::math::pi());
      for (size_t n = 0; n < N; n++) {
        const double y_dbl = value_of(y_vec[n]);
        const double mu_dbl = value_of(mu_vec[n]);
        const double sigma_dbl = value_of(sigma_vec[n]);

        const double scaled_diff = (y_dbl - mu_dbl) / (sigma_dbl * SQRT_2);
        
        double one_m_erf;
        if (scaled_diff < -37.5 * INV_SQRT_2)
          one_m_erf = 2.0;
        else if (scaled_diff < -5.0 * INV_SQRT_2)
          one_m_erf =  2.0 - erfc(-scaled_diff);
        else if (scaled_diff > 8.25 * INV_SQRT_2)
          one_m_erf = 0.0;
        else
          one_m_erf = 1.0 - erf(scaled_diff);

        // log ccdf
        ccdf_log += log_half + log(one_m_erf);

        // gradients
        const double rep_deriv = SQRT_TWO_OVER_PI 
          * exp(-scaled_diff * scaled_diff) / one_m_erf;
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] -= rep_deriv / sigma_dbl;
        if (!is_constant_struct<T_loc>::value)
          operands_and_partials.d_x2[n] += rep_deriv / sigma_dbl;
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n] += rep_deriv * scaled_diff 
            * stan::math::SQRT_2 / sigma_dbl;
      }
      return operands_and_partials.to_var(ccdf_log);
    }

    template <class RNG>
    inline double
    normal_rng(const double mu,
               const double sigma,
               RNG& rng) {
      using boost::variate_generator;
      using boost::normal_distribution;
      using stan::math::check_positive;
      using stan::math::check_finite;
      using stan::math::check_not_nan;

      static const char* function = "stan::prob::normal_rng(%1%)";

      check_finite(function, mu, "Location parameter", (double*)0);
      check_not_nan(function, mu, "Location parameter", (double*)0);
      check_positive(function, sigma, "Scale parameter", (double*)0);
      check_not_nan(function, sigma, "Scale parameter", (double*)0);

      variate_generator<RNG&, normal_distribution<> >
        norm_rng(rng, normal_distribution<>(mu, sigma));
      return norm_rng();
    }
  }
}
#endif
