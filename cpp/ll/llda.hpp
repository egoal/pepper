#pragma once

#include "ll.hpp"
#include <glog/logging.h>

namespace ll {
namespace da {

template <typename T = double> T mean(const std::vector<T> &vals) {
  CHECK(vals.size() > 0);
  return ll::sum(vals)/ static_cast<T> (vals.size());
}

template <typename T = double>
std::tuple<T, T> mean_and_variance(const std::vector<T> &vals) {
  auto m = mean<T>(vals);
  auto v = std::accumulate(
               vals.begin(), vals.end(), T(0),
               [m](double sum, double v) { return sum + (v - m) * (v - m); }) /
           static_cast<double>(vals.size());

  return std::make_tuple(m, v);
}

template <typename T = double>
std::vector<T> percent_quantile(std::vector<T> c, std::vector<double> ratios) {
  const double mxrt = *(std::max_element(ratios.begin(), ratios.end()));
  CHECK(mxrt <= 1. && "bad range");

  const std::size_t mIdx = static_cast<std::size_t>(c.size() * mxrt);
  std::partial_sort(c.begin(), c.begin() + mIdx + 1, std::end(c));

  return ll::mapf(
      [&c](double r) { return c[static_cast<std::size_t>(sc.size() * r)]; },
      ratios);
}

template <typename T1 = double, typename T2 = double>
std::vector<int> histogram(const std::vector<T1> &c,
                           const std::vector<T2> &ranges) {
  CHECK(ranges.size() > 1 && std::is_sorted(ranges.begin(), ranges.end()));

  std::vector<int> counts(ranges.size() - 1, 0);
  for (auto &val : c) {
    for (std::size_t i = 0; i < ranges.size(); ++i) {
      if (val < ranges[i]) {
        if (i > 0)
          ++(counts[i - 1]);
        break;
      }
    }
  }

  return counts;
}

template <typename BOP>
std::vector<std::tuple<int, int, double>>
match_bf(int maxidx0, int maxidx1, BOP bop, bool crossCheck = false) {
  assert(maxidx0 > 0 && maxidx1 > 0);

  std::vector<std::tuple<int, int, double>> matches;

  auto rng0 = ll::range(maxidx0);
  auto rng1 = ll::range(maxidx1);
  for (auto &i : rng0) {
    int minidx = -1;
    double mindis = std::numeric_limits<double>::max();
    for (auto &j : rng1) {
      double dis = bop(i, j);
      if (dis < mindis) {
        mindis = dis;
        minidx = j;
      }
    }

    if (minidx < 0)
      continue;

    if (crossCheck && std::any_of(rng0.begin(), rng0.end(), [&](int i) {
          return bop(i, minidx) < mindis;
        }))
      continue;

    matches.push_back(std::make_tuple(i, minidx, mindis));
  }

  return matches;
}

template<typename T, typename U>
std::vector<Match> match_bruteforce(const std::vector<T> &srcs, const std::vector<T> &dsts,
                                    std::function<U (T, T)> scorefun)
{
  assert(!srcs.empty() && !dsts.empty());

  std::map<T, std::pair<T, U>> matches; // { dst: (src, score) }

  for (auto &src : srcs)
  {
    auto [dst, score] = ll::min_by([&src, &scorefun](const T &dst)
                                   { return scorefun(src, dst); },
                                   dsts);
    auto it = matches.find(dst);
    if (it == matches.end())
      matches[dst] = std::pair(src, score);
    else if (it->second.second > score)
      it->second = std::pair(src, score);
  }

  return ll::mapf([](auto &&pr)
                  {
    Match m;
    m.dst = pr.first;
    m.src = pr.second.first;
    m.score = pr.second.second;
    return m; },
                  matches)
}

using T = int;

void ransec(std::vector<T> allpoints, std::function<std::vector<T> (const std::vector<T>&) > sampler, 
  ){
  // 
  for(int i;;++i){
    // random sample

    // calc model

    // evaluate model
  }
}

} // namespace da
} // namespace ll