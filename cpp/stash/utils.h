#include "ll.hpp"

#define TIME_COUNTER(tag)                                             \
  LOG(INFO) << ">>>>>> | enter " #tag;                                \
  ll::TimeCounter __tc([](uint64_t ms) {                              \
    LOG(INFO) << "<<<<<< | exit " #tag << " " << ms << " ms passed."; \
  })


template <typename T>
std::vector<T> PercentileOf(std::vector<T> con,
                            const std::vector<double> &ratios) {
  CHECK(!con.empty() && std::all_of(ratios.begin(), ratios.end(),
                                    [](double r) { return r > 0. && r < 1.; }));
  return ll::mapf(
      [&con](double r) {
        const std::size_t n = static_cast<std::size_t>(con.size() * r);
        std::nth_element(con.begin(), con.begin() + n, con.end());
        return con[n];
      },
      ratios);
}

template <typename T>
T MedianOf(std::vector<T> vals) {
  return PercentileOf(vals, std::vector<double>{0.5}).front();
}

template <typename T>
std::pair<T, T> CalcMuSigma(const std::vector<T> &lens) {
  CHECK(!lens.empty());
  T mu = ll::sum(lens) / lens.size();
  T d2 = ll::sum_by([mu](double l) { return (l - mu) * (l - mu); }, lens);
  T sigma = std::sqrt(d2 / lens.size());

  return std::make_pair(mu, sigma);
}

std::string GetTimeStr() {
  using std::chrono::system_clock;
  auto now = system_clock::now();
  auto now_c = system_clock::to_time_t(now);

  std::ostringstream ss;
  ss << std::put_time(localtime(&now_c), "%Y%m%d_%H%M%m");

  return ss.str();
}

