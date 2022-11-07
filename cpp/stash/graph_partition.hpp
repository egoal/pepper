#pragma once

#include "ll.hpp"

#include "Eigen/Core"
#include "Eigen/Sparse"

namespace water {

template <typename T> class GraphPartition {
public:
  GraphPartition(int max_iterations = 10) : maxiters(max_iterations) {}

  void Clear() {
    vertices_.clear();
    edges_.clear();
  }

  ///
  ///@brief undirected
  ///
  void Link(T t1, T t2) { edges_.emplace_back(TryAdd(t1), TryAdd(t2)); }

  std::vector<std::vector<T>> ComputeSubgraphs() const {
    if (edges_.empty())
      return {};

    AdjacentMatrix A = MakeAdjacentMatrix();
    // LOG_INFO << ll::unsafe_format(
    //     "adjacent matrix built from %d edges between %d vertices.",
    //     edges_.size(), vertices_.size());

    {
      AdjacentMatrix lastA = A;
      for (int i = 0; i < maxiters; ++i) {
        A = lastA * lastA;
        if (A.nonZeros() == lastA.nonZeros()) {
          LOG_INFO << "connection found on iteration: " << i;
          break;
        }
        lastA = A;
      }
    }

    std::vector<std::vector<T>> subgraphs;

    std::unordered_set<int> walked;
    for (int i = 0; i < A.outerSize(); ++i) {
      if (A.col(i).nonZeros() < 2)
        continue;

      if (walked.count(i))
        continue;

      std::vector<T> subgraph;
      for (auto it = AdjacentMatrix::InnerIterator(A, i); it; ++it) {
        subgraph.emplace_back(vertices_[it.row()]);
        walked.insert(it.row());
      }

      subgraphs.emplace_back(std::move(subgraph));
    }

    LOG_INFO << ll::unsafe_format("%d subgraphs found from %d vertices.",
                                  subgraphs.size(), vertices_.size());

    return subgraphs;
  }

private:
  using AdjacentMatrix = Eigen::SparseMatrix<int, Eigen::ColMajor>;
  const int maxiters;

  std::vector<T> vertices_;
  std::vector<std::pair<std::size_t, std::size_t>> edges_;

  std::size_t TryAdd(T t) {
    auto it = std::find(vertices_.begin(), vertices_.end(), t);
    if (it == vertices_.end()) {
      vertices_.emplace_back(t);
      return vertices_.size() - 1;
    } else
      return std::distance(vertices_.begin(), it);
  }

  AdjacentMatrix MakeAdjacentMatrix() const {
    std::vector<Eigen::Triplet<int>> triples;
    triples.reserve(vertices_.size() + edges_.size() * 2);
    for (std::size_t i = 0; i < vertices_.size(); ++i)
      triples.emplace_back(i, i, 1);
    for (auto &pr : edges_) {
      triples.emplace_back(pr.first, pr.second, 1);
      triples.emplace_back(pr.second, pr.first, 1);
    }

    AdjacentMatrix A(vertices_.size(), vertices_.size());
    A.setFromTriplets(triples.begin(), triples.end());

    return A;
  }
};

} // namespace water
