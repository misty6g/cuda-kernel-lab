#pragma once

#include <string>

namespace cklab {

struct CompareResult {
    bool ok = false;
    int mismatches = 0;
    int first_index = -1;
    float max_abs = 0.0f;
    float max_rel = 0.0f;
};

// Elementwise |a-b| <= atol + rtol * |ref| (NumPy allclose convention).
CompareResult allclose(int n, const float* got, const float* ref, float atol, float rtol);

std::string format_compare(const CompareResult& r);

} // namespace cklab
