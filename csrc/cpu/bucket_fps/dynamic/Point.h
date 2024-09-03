//
// Created by 韩萌 on 2022/6/14.
// Refactored by AyajiLin on 2024/09/03.
//

#pragma once

#include "utils.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace quickfps::dynamic {

template <typename T, typename S = T> class Point {
  public:
    std::vector<T> pos; // x, y, z, ...
    S dis;
    size_t id;

    Point(size_t dim);
    Point(const std::vector<T> &pos, size_t id);
    Point(const std::vector<T> &pos, size_t id, S dis);
    Point(const Point &obj);
    ~Point() {};

    bool operator<(const Point &aii) const;

    constexpr T operator[](size_t i) const { return pos.at(i); }

    Point &operator=(const Point &obj) {
        this->pos = obj.pos;
        this->dis = obj.dis;
        this->id = obj.id;
        return *this;
    }

    constexpr size_t dim() const { return pos.size(); }

    constexpr S distance(const Point &b) {
        S ret = 0;
        for (size_t i = 0; i < pos.size(); i++) {
            S temp = pos[i] - b.pos[i];
            ret += temp * temp;
        }
        return ret;
    }

    void reset();

    S updatedistance(const Point &ref);

    S updateDistanceAndCount(const Point &ref, size_t &count);
};

template <typename T, typename S>
Point<T, S>::Point(size_t dim)
    : pos(dim, 0), dis(std::numeric_limits<S>::max()), id(0) {}

template <typename T, typename S>
Point<T, S>::Point(const std::vector<T> &pos, size_t id)
    : pos(pos), dis(std::numeric_limits<S>::max()), id(id) {}

template <typename T, typename S>
Point<T, S>::Point(const std::vector<T> &pos, size_t id, S dis)
    : pos(pos), dis(dis), id(id) {}

template <typename T, typename S>
Point<T, S>::Point(const Point &obj) : pos(obj.pos), dis(obj.dis), id(obj.id) {}

template <typename T, typename S>
bool Point<T, S>::operator<(const Point &aii) const {
    return dis < aii.dis;
}

template <typename T, typename S>
S Point<T, S>::updatedistance(const Point &ref) {
    this->dis = std::min(this->dis, this->distance(ref));
    return this->dis;
}

template <typename T, typename S>
S Point<T, S>::updateDistanceAndCount(const Point &ref, size_t &count) {
    S tempDistance = this->distance(ref);
    if (tempDistance < this->dis) {
        this->dis = tempDistance;
        count++;
    }
    return this->dis;
}

template <typename T, typename S> void Point<T, S>::reset() {
    this->dis = std::numeric_limits<S>::max();
}

} // namespace quickfps::dynamic
