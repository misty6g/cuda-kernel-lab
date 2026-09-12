#include "cklab/verify.hpp"

#include <cmath>
#include <sstream>

namespace cklab {

CompareResult allclose(int n, const float* got, const float* ref, float atol, float rtol) {
    CompareResult r;
    r.ok = true;
    for (int i = 0; i < n; ++i) {
        const float a = got[i];
        const float b = ref[i];
        const float diff = std::fabs(a - b);
        const float rel = diff / std::max(std::fabs(b), 1.0e-12f);
        r.max_abs = std::max(r.max_abs, diff);
        r.max_rel = std::max(r.max_rel, rel);
        if (diff > atol + rtol * std::fabs(b)) {
            r.ok = false;
            ++r.mismatches;
            if (r.first_index < 0) {
                r.first_index = i;
            }
        }
    }
    return r;
}

std::string format_compare(const CompareResult& r) {
    std::ostringstream os;
    if (r.ok) {
        os << "PASS (max_abs=" << r.max_abs << ", max_rel=" << r.max_rel << ")";
    } else {
        os << "FAIL mismatches=" << r.mismatches << " first=" << r.first_index
           << " max_abs=" << r.max_abs << " max_rel=" << r.max_rel;
    }
    return os.str();
}

} // namespace cklab
