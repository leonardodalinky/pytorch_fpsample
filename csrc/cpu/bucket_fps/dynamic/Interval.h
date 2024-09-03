//
// Created by 韩萌 on 2022/6/14.
// Refactored by AyajiLin on 2024/09/03.
//

#pragma once

namespace quickfps::dynamic {
template <typename S> class Interval {
  public:
    S low, high;
    Interval() : low(0), high(0) {};
    Interval(S low, S high) : low(low), high(high) {};
    Interval(const Interval &o) : low(o.low), high(o.high) {};
};
} // namespace quickfps::dynamic
