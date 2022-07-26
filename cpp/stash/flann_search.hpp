#pragma once

#include "ll.hpp"
#include <Eigen/Core>
#include <flann/flann.h>

namespace water {

// todo:
template <typename Point, typename Points = std::vector<Point>>
class FlannIndex {
  using T = typename Point::value_type;
  static constexpr int DIM = Point::RowsAtCompileTime;
  using Index = flann::Index<L2_Simple<T>>;

public:
  struct QueryRe {
    std::vector<std::vector<int>> indices;
    std::vector<std::vector<T>> dists;
  };

  void Build(const Points &points, int leafsize) {
    static_assert(Point::ColsAtCompileTime == 1, "use column vector");

    // LOG_IF(WARN, !index_) << "rebuild flann index.";
    if (index_)
      LOG_WARN << "rebuild flann index.";
    index_ = std::make_shared<Index>(AsMatrix(points),
                                     flann::KDTreeSingleIndexParams(leafsize));
    index_->buildIndex();
  }

  QueryRe QueryRadius(const Points &points, T r) const {
    QueryRe qr;
    qr.indices.reserve(points.size());
    qr.dists.reserve(points.size());
    index_->radiusSearch(AsMatrix(points), qr.indices, qr.dists, r * r,
                         sparam_);
    return qr;
  }

  QueryRe QueryNearestN(const Points &points, std::size_t n) const {
    QueryRe qr;
    qr.indices.reserve(points.size());
    qr.dists.reserve(points.size());
    index_->knnSearch(AsMatrix(points), qr.indices, qr.dists, n, sparam_);
    return qr;
  }

  Point GetPoint(int i) const { return Eigen::Map<Point>(index_->getPoint(i)); }

  const Index &GetIndex() const { return *index_; }

private:
  std::shared_ptr<Index> index_{nullptr};
  flann::SearchParams sparam_{-1, 1e-6f};

  inline flann::Matrix<T> AsMatrix(const Points &points) const {
    return flann::Matrix<T>(const_cast<T *>(&points[0][0]), points.size(), DIM);
  }
};

template <typename EigenVec>
auto BuildFlannIndex(const EigenVec &points, int leafsize = 15) {
  CHECK(!points.empty());
  FlannIndex<typename EigenVec::value_type, EigenVec> fi;
  fi.Build(points, leafsize);
  return fi;
}

} // namespace water