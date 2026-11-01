// Synthetic checks for the pre-bake QEM target and the xatlas chart gutter.
// No model weights. Build:
//   cmake --build build --target trellis-test-bake-pack && ./build/trellis-test-bake-pack
//
// Prints faces in/out and the tightest chart-to-chart gap at two atlas sizes.
#include "uv_bake.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

int fail(const std::string& msg) {
    std::fprintf(stderr, "FAIL: %s\n", msg.c_str());
    return 1;
}

void add_grid(std::vector<float>& v, std::vector<int32_t>& f, int n) {
    v.clear();
    f.clear();
    v.reserve((size_t)(n + 1) * (n + 1) * 3);
    for (int y = 0; y <= n; ++y) {
        for (int x = 0; x <= n; ++x) {
            const float u = (float)x / (float)n;
            const float w = (float)y / (float)n;
            v.push_back((u - 0.5f) * 0.8f);
            v.push_back((w - 0.5f) * 0.8f);
            v.push_back(0.04f * std::sin(u * 6.2831853f) * std::sin(w * 6.2831853f));
        }
    }
    auto id = [n](int x, int y) { return y * (n + 1) + x; };
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            const int a = id(x, y), b = id(x + 1, y), c = id(x + 1, y + 1), d = id(x, y + 1);
            f.insert(f.end(), {a, b, c, a, c, d});
        }
    }
}

struct Box { float x0, y0, x1, y1; int quad; };

float aabb_gap(const Box& a, const Box& b) {
    const bool ox = a.x1 >= b.x0 && b.x1 >= a.x0;
    const bool oy = a.y1 >= b.y0 && b.y1 >= a.y0;
    const float gx = a.x1 < b.x0 ? b.x0 - a.x1 : (b.x1 < a.x0 ? a.x0 - b.x1 : 0.f);
    const float gy = a.y1 < b.y0 ? b.y0 - a.y1 : (b.y1 < a.y0 ? a.y0 - b.y1 : 0.f);
    if (ox && oy) {
        const float px = std::min(a.x1, b.x1) - std::max(a.x0, b.x0);
        const float py = std::min(a.y1, b.y1) - std::max(a.y0, b.y0);
        return -std::min(px, py);
    }
    if (ox) return gy;
    if (oy) return gx;
    return std::sqrt(gx * gx + gy * gy);
}

struct PackStats {
    float min_gap = 0;
    int faces = 0;
    int faces_with_center = 0;
    int texels = 0;
    float uv_min = 0, uv_max = 0;
};

PackStats measure_pack(const trellis::BakedMesh& bm, int nq, int T) {
    PackStats s;
    s.faces = (int)bm.faces.size() / 3;
    const int V = (int)bm.verts.size() / 3;
    std::vector<int> vq((size_t)V, -1);
    for (int i = 0; i < V; ++i) {
        const float x = bm.verts[3 * i], y = bm.verts[3 * i + 1];
        // Quads were laid out on a 6x6 grid; recover the cell from position.
        const int gx = (int)std::floor((x + 0.36f) / 0.12f);
        const int gy = (int)std::floor((y + 0.36f) / 0.12f);
        if (gx >= 0 && gy >= 0 && gx < nq && gy < nq) vq[i] = gy * nq + gx;
    }
    std::vector<Box> boxes((size_t)nq * nq);
    std::vector<uint8_t> seen(boxes.size(), 0);
    for (int i = 0; i < (int)boxes.size(); ++i) {
        boxes[i] = {1e9f, 1e9f, -1e9f, -1e9f, i};
    }
    s.uv_min = 1e9f;
    s.uv_max = -1e9f;
    for (int i = 0; i < V; ++i) {
        const float u = bm.uv[2 * i], v = bm.uv[2 * i + 1];
        s.uv_min = std::min(s.uv_min, std::min(u, v));
        s.uv_max = std::max(s.uv_max, std::max(u, v));
        const int q = vq[i];
        if (q < 0) continue;
        seen[q] = 1;
        Box& b = boxes[q];
        b.x0 = std::min(b.x0, u * T);
        b.y0 = std::min(b.y0, v * T);
        b.x1 = std::max(b.x1, u * T);
        b.y1 = std::max(b.y1, v * T);
    }
    s.min_gap = 1e9f;
    for (int i = 0; i < (int)boxes.size(); ++i) {
        if (!seen[i]) continue;
        for (int j = i + 1; j < (int)boxes.size(); ++j) {
            if (!seen[j]) continue;
            s.min_gap = std::min(s.min_gap, aabb_gap(boxes[i], boxes[j]));
        }
    }
    std::vector<uint8_t> hit((size_t)T * T, 0);
    for (int f = 0; f < s.faces; ++f) {
        const int i0 = bm.faces[3 * f], i1 = bm.faces[3 * f + 1], i2 = bm.faces[3 * f + 2];
        const float px[3] = {bm.uv[2 * i0] * T, bm.uv[2 * i1] * T, bm.uv[2 * i2] * T};
        const float py[3] = {bm.uv[2 * i0 + 1] * T, bm.uv[2 * i1 + 1] * T, bm.uv[2 * i2 + 1] * T};
        int x0 = (int)std::floor(std::min({px[0], px[1], px[2]}));
        int x1 = (int)std::ceil(std::max({px[0], px[1], px[2]}));
        int y0 = (int)std::floor(std::min({py[0], py[1], py[2]}));
        int y1 = (int)std::ceil(std::max({py[0], py[1], py[2]}));
        x0 = std::max(0, x0); y0 = std::max(0, y0);
        x1 = std::min(T - 1, x1); y1 = std::min(T - 1, y1);
        const float d = (py[1] - py[2]) * (px[0] - px[2]) + (px[2] - px[1]) * (py[0] - py[2]);
        if (std::fabs(d) < 1e-8f) continue;
        bool covered = false;
        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                const float fx = x + 0.5f, fy = y + 0.5f;
                const float w0 = ((py[1] - py[2]) * (fx - px[2]) + (px[2] - px[1]) * (fy - py[2])) / d;
                const float w1 = ((py[2] - py[0]) * (fx - px[2]) + (px[0] - px[2]) * (fy - py[2])) / d;
                const float w2 = 1.f - w0 - w1;
                if (w0 < -1e-4f || w1 < -1e-4f || w2 < -1e-4f) continue;
                covered = true;
                const size_t t = (size_t)y * T + x;
                if (!hit[t]) { hit[t] = 1; ++s.texels; }
            }
        }
        if (covered) ++s.faces_with_center;
    }
    return s;
}

}  // namespace

int main() {
    int rc = 0;
    {
        std::vector<float> v;
        std::vector<int32_t> f;
        add_grid(v, f, 48);
        const int Fin = (int)f.size() / 3;
        std::vector<float> ov;
        std::vector<int32_t> of;
        const int target = 600;
        trellis::decimate_qem(v, (int)v.size() / 3, f, Fin, target, ov, of);
        const int Fout = (int)of.size() / 3;
        std::printf("decimate: faces %d -> %d (target %d), verts %d -> %d\n",
                    Fin, Fout, target, (int)v.size() / 3, (int)ov.size() / 3);
        if (Fout <= 0 || Fout > target) {
            rc |= fail("QEM did not land in (0, target]");
        }
        std::vector<float> ov2;
        std::vector<int32_t> of2;
        trellis::decimate_qem(v, (int)v.size() / 3, f, Fin, 1000000, ov2, of2);
        const int Fnoop = (int)of2.size() / 3;
        std::printf("decimate no-op: faces %d -> %d (target 1000000, above the mesh)\n", Fin, Fnoop);
        if (Fnoop != Fin) rc |= fail("target above the mesh must keep every face");
    }

    const int n = 6;
    std::vector<float> v;
    std::vector<int32_t> f;
    for (int gy = 0; gy < n; ++gy) {
        for (int gx = 0; gx < n; ++gx) {
            const float x0 = -0.36f + gx * 0.12f;
            const float y0 = -0.36f + gy * 0.12f;
            const int b = (int)v.size() / 3;
            const float z = 0.f;
            const float p[4][3] = {
                {x0, y0, z}, {x0 + 0.08f, y0, z}, {x0 + 0.08f, y0 + 0.08f, z}, {x0, y0 + 0.08f, z}};
            for (int k = 0; k < 4; ++k) {
                v.push_back(p[k][0]); v.push_back(p[k][1]); v.push_back(p[k][2]);
            }
            f.insert(f.end(), {b, b + 1, b + 2, b, b + 2, b + 3});
        }
    }
    const std::vector<float> no_pbr;
    PackStats prev;
    bool have_prev = false;
    for (int T : {1024, 2048}) {
        trellis::BakedMesh bm = trellis::uv_bake(v, (int)v.size() / 3, f, (int)f.size() / 3, no_pbr, T, nullptr);
        if (!bm.ok()) return fail("uv_bake failed at atlas " + std::to_string(T));
        PackStats s = measure_pack(bm, n, T);
        const float tpf = s.faces > 0 ? (float)s.texels / (float)s.faces : 0.f;
        std::printf("pack T=%d: charts faces=%d  min_gap=%.2f px  uv=[%.4f, %.4f]  "
                    "texels=%d (%.2f / face)  faces_with_texel_center=%d/%d\n",
                    T, s.faces, s.min_gap, s.uv_min, s.uv_max, s.texels, tpf,
                    s.faces_with_center, s.faces);
        if (s.uv_min < -0.02f || s.uv_max > 1.02f)
            rc |= fail("UVs fell outside the atlas at " + std::to_string(T));
        if (s.min_gap < 4.f)
            rc |= fail("chart gutter under 4 px at atlas " + std::to_string(T));
        if (have_prev) {
            const float ratio = s.min_gap / prev.min_gap;
            std::printf("pack scale: gutter %.2f px @%d / %.2f px @%d = %.3f (expect ~2)\n",
                        s.min_gap, T, prev.min_gap, T / 2, ratio);
            if (ratio < 1.6f || ratio > 2.4f)
                rc |= fail("gutter did not scale with atlas size");
        }
        prev = s;
        have_prev = true;
    }
    // Denser single mesh: many faces into a small atlas, then the same mesh after
    // QEM. Reports the texel-center coverage the Pixal speck comes from.
    {
        std::vector<float> v;
        std::vector<int32_t> f;
        add_grid(v, f, 64);
        const int Fin = (int)f.size() / 3;
        const int T = 128;
        auto coverage = [&](const char* tag, const std::vector<float>& vv, const std::vector<int32_t>& ff) {
            trellis::BakedMesh bm = trellis::uv_bake(vv, (int)vv.size() / 3, ff, (int)ff.size() / 3,
                                                     no_pbr, T, nullptr);
            if (!bm.ok()) { rc |= fail(std::string("coverage bake failed: ") + tag); return; }
            const int Fo = (int)bm.faces.size() / 3;
            std::vector<uint8_t> hit((size_t)T * T, 0);
            int covered = 0, texels = 0;
            for (int fi = 0; fi < Fo; ++fi) {
                const int i0 = bm.faces[3 * fi], i1 = bm.faces[3 * fi + 1], i2 = bm.faces[3 * fi + 2];
                const float px[3] = {bm.uv[2 * i0] * T, bm.uv[2 * i1] * T, bm.uv[2 * i2] * T};
                const float py[3] = {bm.uv[2 * i0 + 1] * T, bm.uv[2 * i1 + 1] * T, bm.uv[2 * i2 + 1] * T};
                int x0 = (int)std::floor(std::min({px[0], px[1], px[2]}));
                int x1 = (int)std::ceil(std::max({px[0], px[1], px[2]}));
                int y0 = (int)std::floor(std::min({py[0], py[1], py[2]}));
                int y1 = (int)std::ceil(std::max({py[0], py[1], py[2]}));
                x0 = std::max(0, x0); y0 = std::max(0, y0);
                x1 = std::min(T - 1, x1); y1 = std::min(T - 1, y1);
                const float d = (py[1] - py[2]) * (px[0] - px[2]) + (px[2] - px[1]) * (py[0] - py[2]);
                if (std::fabs(d) < 1e-8f) continue;
                bool any = false;
                for (int y = y0; y <= y1; ++y) for (int x = x0; x <= x1; ++x) {
                    const float fx = x + 0.5f, fy = y + 0.5f;
                    const float w0 = ((py[1] - py[2]) * (fx - px[2]) + (px[2] - px[1]) * (fy - py[2])) / d;
                    const float w1 = ((py[2] - py[0]) * (fx - px[2]) + (px[0] - px[2]) * (fy - py[2])) / d;
                    const float w2 = 1.f - w0 - w1;
                    if (w0 < -1e-4f || w1 < -1e-4f || w2 < -1e-4f) continue;
                    any = true;
                    const size_t t = (size_t)y * T + (size_t)x;
                    if (!hit[t]) { hit[t] = 1; ++texels; }
                }
                if (any) ++covered;
            }
            const float tpf = Fo > 0 ? (float)texels / (float)Fo : 0.f;
            std::printf("coverage %s T=%d: faces=%d texels=%d (%.2f / face) with_center=%d (%.1f%%)\n",
                        tag, T, Fo, texels, tpf, covered, Fo ? 100.0 * covered / Fo : 0.0);
        };
        coverage("dense", v, f);
        std::vector<float> ov;
        std::vector<int32_t> of;
        const int target = 800;
        trellis::decimate_qem(v, (int)v.size() / 3, f, Fin, target, ov, of);
        std::printf("coverage decimate: faces %d -> %d (target %d)\n", Fin, (int)of.size() / 3, target);
        coverage("qem", ov, of);
    }

    if (rc == 0) std::printf("ok\n");
    return rc;
}
