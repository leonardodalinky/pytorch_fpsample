//
// Created by hanm on 22-6-15.
// Refactored by AyajiLin on 2024/09/03.
//

#pragma once

#include <limits>
#include <vector>

#include "KDTreeBase.h"

namespace quickfps::dynamic {

template <typename T, typename S = T>
class KDLineTree : public KDTreeBase<T, S> {
  public:
    using typename KDTreeBase<T, S>::_Point;
    using typename KDTreeBase<T, S>::_Points;
    using typename KDTreeBase<T, S>::NodePtr;

    KDLineTree(_Points data, size_t pointSize, size_t treeHigh,
               _Points samplePoints);
    ~KDLineTree();

    std::vector<NodePtr> KDNode_list;

    size_t high_;

    _Point max_point() override;

    void update_distance(const _Point &ref_point) override;

    void sample(size_t sample_num) override;

    bool leftNode(size_t high, size_t count) const override {
        return high == this->high_ || count == 1;
    };

    void addNode(NodePtr p) override;
};

template <typename T, typename S>
KDLineTree<T, S>::KDLineTree(_Points data, size_t pointSize, size_t treeHigh,
                             _Points samplePoints)
    : KDTreeBase<T, S>(data, pointSize, samplePoints), high_(treeHigh) {
    KDNode_list.clear();
}

template <typename T, typename S> KDLineTree<T, S>::~KDLineTree() {
    KDNode_list.clear();
}

template <typename T, typename S>
typename KDLineTree<T, S>::_Point KDLineTree<T, S>::max_point() {
    _Point tmpPoint(this->dim());
    S max_distance = std::numeric_limits<S>::lowest();
    for (const auto &bucket : KDNode_list) {
        if (bucket->max_point.dis > max_distance) {
            max_distance = bucket->max_point.dis;
            tmpPoint = bucket->max_point;
        }
    }
    return tmpPoint;
}

template <typename T, typename S>
void KDLineTree<T, S>::update_distance(const _Point &ref_point) {
    for (const auto &bucket : KDNode_list) {
        bucket->send_delay_point(ref_point);
        bucket->update_distance();
    }
}

template <typename T, typename S>
void KDLineTree<T, S>::sample(size_t sample_num) {
    for (size_t i = 1; i < sample_num; i++) {
        _Point ref_point = this->max_point();
        this->sample_points[i] = ref_point;
        this->update_distance(ref_point);
    }
}

template <typename T, typename S> void KDLineTree<T, S>::addNode(NodePtr p) {
    size_t nodeIdx = KDNode_list.size();
    p->idx = nodeIdx;
    KDNode_list.push_back(p);
}

} // namespace quickfps::dynamic
