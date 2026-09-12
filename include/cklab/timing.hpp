#pragma once

#include <chrono>

namespace cklab {

class Stopwatch {
public:
    void start() { t0_ = Clock::now(); }
    double elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(Clock::now() - t0_).count();
    }

private:
    using Clock = std::chrono::steady_clock;
    Clock::time_point t0_{};
};

// Warm up once, then time `reps` calls. Returns mean milliseconds.
template <typename Fn>
double time_ms(Fn&& fn, int reps) {
    fn();
    Stopwatch sw;
    sw.start();
    for (int i = 0; i < reps; ++i) {
        fn();
    }
    return sw.elapsed_ms() / static_cast<double>(reps);
}

inline double gflops(double flops, double ms) {
    if (ms <= 0.0) {
        return 0.0;
    }
    return (flops / 1.0e9) / (ms / 1.0e3);
}

inline double gbps(double bytes, double ms) {
    if (ms <= 0.0) {
        return 0.0;
    }
    return (bytes / 1.0e9) / (ms / 1.0e3);
}

} // namespace cklab
