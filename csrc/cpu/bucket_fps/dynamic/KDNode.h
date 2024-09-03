//
// Created by 韩萌 on 2022/6/14.
// Refactored by AyajiLin on 2024/09/03.
//

#pragma once
#include <array>
#include <limits>
#include <queue>
#include <vector>

#include "Interval.h"
#include "Point.h"

namespace quickfps::dynamic {

template <typename T, typename S> class KDNode {
  public:
    using _Point = Point<T, S>;
    using _Points = _Point *;
    _Points points;
    size_t pointLeft, pointRight;
    size_t idx;

    std::vector<Interval<T>> bboxs;
    std::vector<_Point> waitpoints;
    std::vector<_Point> delaypoints;
    _Point max_point;
    KDNode *left;
    KDNode *right;

    KDNode();

    KDNode(const KDNode &a);

    KDNode(const std::vector<Interval<T>> &bboxs);

    void init(const _Point &ref);

    size_t dim() const { return max_point.dim(); };

    void updateMaxPoint(const _Point &lpoint, const _Point &rpoint) {
        if (lpoint.dis > rpoint.dis)
            this->max_point = lpoint;
        else
            this->max_point = rpoint;
    }

    S bound_distance(const _Point &ref_point) const;

    void send_delay_point(const _Point &point) {
        this->waitpoints.push_back(point);
    }

    void update_distance();

    void reset();

    size_t size() const;
};

template <typename T, typename S>
KDNode<T, S>::KDNode()
    : points(nullptr), pointLeft(0), pointRight(0), left(nullptr),
      right(nullptr), max_point(0) {}

template <typename T, typename S>
KDNode<T, S>::KDNode(const std::vector<Interval<T>> &other_bboxs)
    : points(nullptr), pointLeft(0), pointRight(0), left(nullptr),
      right(nullptr), bboxs(other_bboxs), max_point(other_bboxs.size()) {}

template <typename T, typename S>
KDNode<T, S>::KDNode(const KDNode &a)
    : points(a.points), pointLeft(a.pointLeft), pointRight(a.pointRight),
      left(a.left), right(a.right), idx(a.idx), bboxs(a.bboxs),
      waitpoints(a.waitpoints), delaypoints(a.delaypoints),
      max_point(a.max_point) {}

template <typename T, typename S> void KDNode<T, S>::init(const _Point &ref) {
    waitpoints.clear();
    delaypoints.clear();
    if (this->left && this->right) {
        this->left->init(ref);
        this->right->init(ref);
        updateMaxPoint(this->left->max_point, this->right->max_point);
    } else {
        S dis;
        S maxdis = std::numeric_limits<S>::lowest();
        for (size_t i = pointLeft; i < pointRight; i++) {
            dis = points[i].updatedistance(ref);
            if (dis > maxdis) {
                maxdis = dis;
                max_point = points[i];
            }
        }
    }
}

template <typename T, typename S>
S KDNode<T, S>::bound_distance(const _Point &ref_point) const {
    S bound_dis(0);
    S dim_distance;
    for (size_t cur_dim = 0; cur_dim < dim(); cur_dim++) {
        dim_distance = 0;
        if (ref_point[cur_dim] > this->bboxs[cur_dim].high)
            dim_distance = ref_point[cur_dim] - this->bboxs[cur_dim].high;
        else if (ref_point[cur_dim] < this->bboxs[cur_dim].low)
            dim_distance = this->bboxs[cur_dim].low - ref_point[cur_dim];
        bound_dis += powi(dim_distance, 2);
    }
    return bound_dis;
}

template <typename T, typename S> void KDNode<T, S>::update_distance() {
    for (const auto &ref_point : this->waitpoints) {
        S lastmax_distance = this->max_point.dis;
        S cur_distance = this->max_point.distance(ref_point);
        // cur_distance >
        // lastmax_distance意味着当前Node的max_point不会进行更新
        if (cur_distance > lastmax_distance) {
            S boundary_distance = bound_distance(ref_point);
            if (boundary_distance < lastmax_distance)
                this->delaypoints.push_back(ref_point);
        } else {
            if (this->right && this->left) {
                if (!delaypoints.empty()) {
                    for (const auto &delay_point : delaypoints) {
                        this->left->send_delay_point(delay_point);
                        this->right->send_delay_point(delay_point);
                    }
                    delaypoints.clear();
                }
                this->left->send_delay_point(ref_point);
                this->left->update_distance();

                this->right->send_delay_point(ref_point);
                this->right->update_distance();

                updateMaxPoint(this->left->max_point, this->right->max_point);
            } else {
                S dis;
                S maxdis;
                this->delaypoints.push_back(ref_point);
                for (const auto &delay_point : delaypoints) {
                    maxdis = std::numeric_limits<S>::lowest();
                    for (size_t i = pointLeft; i < pointRight; i++) {
                        dis = points[i].updatedistance(delay_point);
                        if (dis > maxdis) {
                            maxdis = dis;
                            max_point = points[i];
                        }
                    }
                }
                this->delaypoints.clear();
            }
        }
    }
    this->waitpoints.clear();
}

template <typename T, typename S> void KDNode<T, S>::reset() {
    for (size_t i = pointLeft; i < pointRight; i++) {
        points[i].reset();
    }
    this->waitpoints.clear();
    this->delaypoints.clear();
    this->max_point.reset();
    if (this->left && this->right) {
        this->left->reset();
        this->right->reset();
    }
}

template <typename T, typename S> size_t KDNode<T, S>::size() const {
    if (this->left && this->right)
        return this->left->size() + this->right->size();
    return (pointRight - pointLeft);
}

} // namespace quickfps::dynamic
