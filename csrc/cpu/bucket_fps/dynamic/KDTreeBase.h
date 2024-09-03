//
// Created by 韩萌 on 2022/6/14.
// Refactored by AyajiLin on 2024/09/03.
//

#pragma once

#include "KDNode.h"
#include "Point.h"
#include <algorithm>
#include <array>
#include <numeric>

namespace quickfps::dynamic {

template <typename T, typename S> class KDTreeBase {
  public:
    using _Point = Point<T, S>;
    using _Points = _Point *;
    using NodePtr = KDNode<T, S> *;
    using _Interval = Interval<T>;

    size_t pointSize;
    _Points sample_points;
    NodePtr root_;
    _Points points_;

  public:
    KDTreeBase(_Points data, size_t pointSize, _Points samplePoints);

    ~KDTreeBase();

    void buildKDtree();

    NodePtr get_root() const { return this->root_; };

    void init(const _Point &ref);

    size_t dim() const { return points_[0].dim(); }

    virtual _Point max_point() = 0;

    virtual void sample(size_t sample_num) = 0;

  protected:
    void deleteNode(NodePtr node_p);
    virtual void addNode(NodePtr p) = 0;
    virtual bool leftNode(size_t high, size_t count) const = 0;
    virtual void update_distance(const _Point &ref_point) = 0;

    NodePtr divideTree(ssize_t left, ssize_t right,
                       const std::vector<_Interval> &bboxs, size_t curr_high);

    size_t planeSplit(ssize_t left, ssize_t right, size_t split_dim,
                      T split_val);

    T qSelectMedian(size_t dim, size_t left, size_t right);
    static size_t findSplitDim(const std::vector<_Interval> &bboxs, size_t dim);
    inline std::vector<_Interval> computeBoundingBox(size_t left, size_t right);
};

template <typename T, typename S>
KDTreeBase<T, S>::KDTreeBase(_Points data, size_t pointSize,
                             _Points samplePoints)
    : pointSize(pointSize), sample_points(samplePoints), root_(nullptr),
      points_(data) {}

template <typename T, typename S> KDTreeBase<T, S>::~KDTreeBase() {
    if (root_ != nullptr)
        deleteNode(root_);
}

template <typename T, typename S>
void KDTreeBase<T, S>::deleteNode(NodePtr node_p) {
    if (node_p->left)
        deleteNode(node_p->left);
    if (node_p->right)
        deleteNode(node_p->right);
    delete node_p;
}

template <typename T, typename S> void KDTreeBase<T, S>::buildKDtree() {
    size_t left = 0;
    size_t right = pointSize;
    std::vector<_Interval> bboxs = this->computeBoundingBox(left, right);
    this->root_ = divideTree(left, right, bboxs, 0);
}

template <typename T, typename S>
typename KDTreeBase<T, S>::NodePtr
KDTreeBase<T, S>::divideTree(ssize_t left, ssize_t right,
                             const std::vector<_Interval> &bboxs,
                             size_t curr_high) {
    NodePtr node = new KDNode<T, S>(bboxs);

    ssize_t count = right - left;
    if (this->leftNode(curr_high, count)) {
        node->pointLeft = left;
        node->pointRight = right;
        node->points = this->points_;
        this->addNode(node);
        return node;
    } else {
        size_t split_dim = this->findSplitDim(bboxs, dim());
        T split_val = this->qSelectMedian(split_dim, left, right);

        size_t split_delta = planeSplit(left, right, split_dim, split_val);

        std::vector<_Interval> bbox_cur =
            this->computeBoundingBox(left, left + split_delta);
        node->left =
            this->divideTree(left, left + split_delta, bbox_cur, curr_high + 1);
        bbox_cur = this->computeBoundingBox(left + split_delta, right);
        node->right = this->divideTree(left + split_delta, right, bbox_cur,
                                       curr_high + 1);
        return node;
    }
}

template <typename T, typename S>
size_t KDTreeBase<T, S>::planeSplit(ssize_t left, ssize_t right,
                                    size_t split_dim, T split_val) {
    ssize_t start = left;
    ssize_t end = right - 1;

    for (;;) {
        while (start <= end && points_[start].pos[split_dim] < split_val)
            ++start;
        while (start <= end && points_[end].pos[split_dim] >= split_val)
            --end;

        if (start > end)
            break;
        std::swap(points_[start], points_[end]);
        ++start;
        --end;
    }

    ssize_t lim1 = start - left;
    if (start == left)
        lim1 = 1;
    if (start == right)
        lim1 = (right - left - 1);

    return static_cast<ssize_t>(lim1);
}

template <typename T, typename S>
T KDTreeBase<T, S>::qSelectMedian(size_t dim, size_t left, size_t right) {
    T sum = std::accumulate(this->points_ + left, this->points_ + right, 0.0,
                            [dim](const T &acc, const _Point &point) {
                                return acc + point.pos[dim];
                            });
    return sum / (right - left);
}

template <typename T, typename S>
size_t KDTreeBase<T, S>::findSplitDim(const std::vector<_Interval> &bboxs,
                                      size_t dim) {
    T min_, max_;
    T span = 0;
    size_t best_dim = 0;

    for (size_t cur_dim = 0; cur_dim < dim; cur_dim++) {
        min_ = bboxs[cur_dim].low;
        max_ = bboxs[cur_dim].high;
        T cur_span = (max_ - min_);

        if (cur_span > span) {
            best_dim = cur_dim;
            span = cur_span;
        }
    }

    return best_dim;
}

template <typename T, typename S>
inline std::vector<Interval<T>>
KDTreeBase<T, S>::computeBoundingBox(size_t left, size_t right) {
    std::vector<T> min_vals(this->dim(), std::numeric_limits<T>::max());
    std::vector<T> max_vals(this->dim(), std::numeric_limits<T>::lowest());

    for (size_t i = left; i < right; ++i) {
        const _Point &pos = points_[i];

        for (size_t cur_dim = 0; cur_dim < this->dim(); cur_dim++) {
            T val = pos[cur_dim];
            min_vals[cur_dim] = std::min(min_vals[cur_dim], val);
            max_vals[cur_dim] = std::max(max_vals[cur_dim], val);
        }
    }

    std::vector<_Interval> bboxs(dim());

    for (size_t cur_dim = 0; cur_dim < dim(); cur_dim++) {
        bboxs[cur_dim].low = min_vals[cur_dim];
        bboxs[cur_dim].high = max_vals[cur_dim];
    }

    return bboxs;
}

template <typename T, typename S>
void KDTreeBase<T, S>::init(const _Point &ref) {
    this->sample_points[0] = ref;
    this->root_->init(ref);
}

} // namespace quickfps::dynamic
