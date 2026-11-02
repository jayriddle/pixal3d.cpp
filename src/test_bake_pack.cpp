// Synthetic checks for the pre-bake QEM target and the xatlas chart gutter.
// No model weights. Build:
//   cmake --build build --target trellis-test-bake-pack && ./build/trellis-test-bake-pack
//
// Prints faces in/out and the tightest chart-to-chart gap at two atlas sizes.
#include "uv_bake.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>
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

    // Padding policy: full gutter at the measured 213k/1024 bake, none at the
    // historical 1M/1024 pack (padding 0). A fixed pad of 4 there inflated the
    // packer and the downscale threw texels away.
    {
        const uint32_t full = trellis::xatlas_pack_padding(1024, 1);
        const uint32_t at_213k = trellis::xatlas_pack_padding(1024, 212818);
        const uint32_t at_250k = trellis::xatlas_pack_padding(1024, 250000);
        const uint32_t at_1m = trellis::xatlas_pack_padding(1024, 1000000);
        const uint32_t at_820k = trellis::xatlas_pack_padding(1024, 820534);
        const uint32_t at_300k = trellis::xatlas_pack_padding(2048, 300000);
        std::printf("padding: full@1024=%u  213k=%u  250k=%u  820k=%u  1M=%u  trellis 300k@2048=%u\n",
                    full, at_213k, at_250k, at_820k, at_1m, at_300k);
        if (at_213k != full || at_250k != full)
            rc |= fail("250k/1024 should keep the full gutter (the measured bake)");
        if (at_1m != 0 || at_820k != 0)
            rc |= fail("1M and the undecimated 820k mesh should pack with padding 0");
        if (at_300k != trellis::xatlas_pack_padding(2048, 1))
            rc |= fail("TRELLIS 300k/2048 should keep a full gutter");
    }

    // Chart-heavy mesh with fewer atlas texels than faces. Padding must be 0
    // (the historical pack) and UVs must still land in the bake atlas.
    {
        const int nq = 24;          // 1152 faces
        const int T = 32;           // 1024 texels → < 1 texel/face
        std::vector<float> hv;
        std::vector<int32_t> hf;
        for (int gy = 0; gy < nq; ++gy) {
            for (int gx = 0; gx < nq; ++gx) {
                const float x0 = -0.45f + gx * (0.9f / nq);
                const float y0 = -0.45f + gy * (0.9f / nq);
                const float s = 0.55f * (0.9f / nq);
                const int b = (int)hv.size() / 3;
                hv.insert(hv.end(), {x0, y0, 0.f, x0 + s, y0, 0.f, x0 + s, y0 + s, 0.f, x0, y0 + s, 0.f});
                hf.insert(hf.end(), {b, b + 1, b + 2, b, b + 2, b + 3});
            }
        }
        const int faces = (int)hf.size() / 3;
        const uint32_t expect = trellis::xatlas_pack_padding(T, faces);
        std::printf("dense charts: faces=%d atlas=%d expected padding=%u (%.2f atlas texels/face)\n",
                    faces, T, expect, (double)T * T / faces);
        if (expect != 0) rc |= fail("chart-heavy mesh should pack with padding 0");
        const std::vector<float> no;
        trellis::BakedMesh bm = trellis::uv_bake(hv, (int)hv.size() / 3, hf, faces, no, T, nullptr);
        if (!bm.ok()) rc |= fail("dense-chart bake failed");
        else {
            float umin = 1e9f, umax = -1e9f;
            for (float u : bm.uv) { umin = std::min(umin, u); umax = std::max(umax, u); }
            std::printf("dense charts: uv=[%.4f, %.4f] faces_out=%zu\n", umin, umax, bm.faces.size() / 3);
            if (umin < -0.02f || umax > 1.02f)
                rc |= fail("dense-chart UVs fell outside the atlas");
        }
    }

    // Pre-decimation: 2% always; under 3% only when farther than 3 remesh
    // voxels from the body. An attached 2.5% part stays, a detached 2.5%
    // shell goes, a far 4% foot stays. Post-QEM 3% still drops a flat slab.
    {
        const float voxel = 0.01f;
        std::vector<float> v;
        std::vector<int32_t> f;
        add_grid(v, f, 56);                       // smooth body, 6272 faces, x in [-0.4, 0.4]
        const int body_faces = (int)f.size() / 3;
        auto add_fan = [&](float cx, float cy, float cz, int n, float radius, float spike) {
            const int c = (int)v.size() / 3;
            v.insert(v.end(), {cx, cy, cz});
            for (int i = 0; i < n; ++i) {
                const float ang = 6.2831853f * (float)i / (float)n;
                v.push_back(cx + radius * std::cos(ang));
                v.push_back(cy + radius * std::sin(ang));
                v.push_back(cz + ((i % 2) ? spike : -0.25f * spike));
            }
            for (int i = 0; i < n; ++i) {
                f.push_back(c);
                f.push_back(c + 1 + i);
                f.push_back(c + 1 + (i + 1) % n);
            }
        };
        // Attached ear: center is 0.5 voxel off the body face at x=0.4. 160/6272 = 2.55%.
        add_fan(0.4f + 0.5f * voxel, 0.f, 0.f, 160, 0.03f, 0.12f);
        // Detached shell, same face count, many voxels away.
        add_fan(1.2f, 0.f, 0.f, 160, 0.05f, 0.18f);
        // Dust under 2%.
        const int dust_faces = 40;
        add_fan(2.4f, 0.f, 0.f, dust_faces, 0.02f, 0.05f);
        // Foot at ~4% of the body, far away. Must survive the proximity cut.
        const int foot_faces = 250;
        add_fan(-1.2f, 0.f, 0.f, foot_faces, 0.05f, 0.18f);
        const int floater_faces = 160;
        // Flat slab, ~25% of the body, at y=-1. QEM reduces it to a few triangles.
        {
            const int n = 28;
            const int base = (int)v.size() / 3;
            for (int y = 0; y <= n; ++y)
                for (int x = 0; x <= n; ++x) {
                    v.push_back(-0.2f + 0.4f * (float)x / (float)n);
                    v.push_back(-1.2f + 0.02f * (float)y / (float)n);
                    v.push_back(0.f);
                }
            auto id = [n, base](int x, int y) { return base + y * (n + 1) + x; };
            for (int y = 0; y < n; ++y)
                for (int x = 0; x < n; ++x) {
                    const int a = id(x, y), b = id(x + 1, y), c = id(x + 1, y + 1), d = id(x, y + 1);
                    f.insert(f.end(), {a, b, c, a, c, d});
                }
        }
        // Second smooth body, large enough to survive both cuts.
        {
            std::vector<float> v2;
            std::vector<int32_t> f2;
            add_grid(v2, f2, 36);
            const int base = (int)v.size() / 3;
            for (size_t i = 0; i < v2.size(); i += 3) {
                v.push_back(v2[i]);
                v.push_back(v2[i + 1] + 1.5f);
                v.push_back(v2[i + 2]);
            }
            for (int idx : f2) f.push_back(base + idx);
        }
        auto comp_sizes = [](const std::vector<float>& vv, const std::vector<int32_t>& ff) {
            const int V = (int)vv.size() / 3;
            std::vector<int> par((size_t)V);
            for (int i = 0; i < V; ++i) par[i] = i;
            auto find = [&](int x) { while (par[x] != x) { par[x] = par[par[x]]; x = par[x]; } return x; };
            for (size_t i = 0; i < ff.size(); i += 3) {
                int r0 = find(ff[i]), r1 = find(ff[i + 1]), r2 = find(ff[i + 2]);
                if (r0 != r1) par[r1] = r0;
                r0 = find(ff[i]);
                if (r0 != r2) par[r2] = r0;
            }
            std::unordered_map<int, int> fc;
            for (size_t i = 0; i < ff.size(); i += 3) fc[find(ff[i])]++;
            std::vector<int> sizes;
            for (auto& kv : fc) sizes.push_back(kv.second);
            std::sort(sizes.begin(), sizes.end(), std::greater<int>());
            return sizes;
        };
        auto bbox_max_x = [](const std::vector<float>& vv) {
            float m = -1e9f;
            for (size_t i = 0; i < vv.size(); i += 3) m = std::max(m, vv[i]);
            return m;
        };
        const auto before = comp_sizes(v, f);
        std::printf("floater before: components");
        for (int s : before) std::printf(" %d", s);
        std::printf(" (body %d, 2.5%% parts %d = %.2f%%, dust %d = %.2f%%, foot %d = %.2f%%)\n",
                    body_faces, floater_faces, 100.0 * floater_faces / body_faces,
                    dust_faces, 100.0 * dust_faces / body_faces,
                    foot_faces, 100.0 * foot_faces / body_faces);

        {
            std::vector<float> a = v;
            std::vector<int32_t> b = f;
            const int dropped = trellis::drop_pre_decimation_components(a, b, voxel);
            const float xmax_pre = bbox_max_x(a);
            float xmin_pre = 1e9f;
            for (size_t i = 0; i < a.size(); i += 3) xmin_pre = std::min(xmin_pre, a[i]);
            const auto pre = comp_sizes(a, b);
            std::printf("floater after pre: dropped=%d xmax=%.3f xmin=%.3f components", dropped, xmax_pre, xmin_pre);
            for (int s : pre) std::printf(" %d", s);
            std::printf("\n");
            if (dropped < 2)
                rc |= fail("expected the detached shell and the dust to drop");
            if (!(xmax_pre > 0.402f && xmax_pre < 0.5f))
                rc |= fail("attached 2.5% part should stay and the detached shell should go");
            if (!(xmin_pre < -1.0f))
                rc |= fail("detached foot at ~4% of the body was dropped");
            std::vector<float> ov;
            std::vector<int32_t> of;
            trellis::decimate_qem(a, (int)a.size() / 3, b, (int)b.size() / 3, 900, ov, of);
            trellis::drop_small_components(ov, of, 0.03f);
            const auto after = comp_sizes(ov, of);
            const float xmax = bbox_max_x(ov);
            float xmin = 1e9f, ymin = 1e9f;
            for (size_t i = 0; i < ov.size(); i += 3) {
                xmin = std::min(xmin, ov[i]);
                ymin = std::min(ymin, ov[i + 1]);
            }
            std::printf("floater after post 3%%:");
            for (int s : after) std::printf(" %d", s);
            std::printf("  bbox x=[%.3f, %.3f] ymin=%.3f\n", xmin, xmax, ymin);
            if (ymin < -0.8f)
                rc |= fail("flat slab survived the post-QEM face-count cut");
            if (!(xmax > 0.40f && xmax < 0.5f))
                rc |= fail("attached part did not survive QEM");
            if (!(xmin < -1.0f))
                rc |= fail("foot did not survive QEM");
            if (after.size() != 4)
                rc |= fail("expected body, secondary, attached part, and foot");
            if (after.empty() || after[0] < 200)
                rc |= fail("main body collapsed below the QEM target's useful range");
        }
    }

    // Pre-pass keeps an against-body 2.5% part and drops a far shell. The
    // post-QEM pass is the plain 3% face cut, which then removes that part.
    // Those pieces are inward skins ~2 voxels inside the wall, not ears.
    {
        const float voxel = 0.01f;
        std::vector<float> v;
        std::vector<int32_t> f;
        add_grid(v, f, 40);
        const int body = (int)f.size() / 3;
        auto add_fan = [&](float cx, int n) {
            const int c = (int)v.size() / 3;
            v.insert(v.end(), {cx, 0.f, 0.f});
            for (int i = 0; i < n; ++i) {
                const float ang = 6.2831853f * (float)i / (float)n;
                v.push_back(cx);
                v.push_back(0.02f * std::sin(ang));
                v.push_back(((i % 2) ? 0.08f : -0.02f));
            }
            for (int i = 0; i < n; ++i) {
                f.push_back(c);
                f.push_back(c + 1 + i);
                f.push_back(c + 1 + (i + 1) % n);
            }
        };
        const int npart = 80;   // 80/3200 = 2.5%
        add_fan(0.4f + 0.5f * voxel, npart);
        add_fan(1.5f, npart);
        std::printf("post-pass consistency: body=%d part=%d (%.2f%%)\n", body, npart, 100.0 * npart / body);
        trellis::drop_pre_decimation_components(v, f, voxel);
        const int mid = (int)f.size() / 3;
        float xmax = -1e9f, xmin = 1e9f;
        for (size_t i = 0; i < v.size(); i += 3) {
            xmax = std::max(xmax, v[i]);
            xmin = std::min(xmin, v[i]);
        }
        if (!(xmax > 0.40f && xmax < 0.6f))
            rc |= fail("consistency: attached 2.5% part was dropped on the pre-pass");
        if (xmax > 1.0f)
            rc |= fail("consistency: detached 2.5% shell survived the pre-pass");
        trellis::drop_small_components(v, f, 0.03f);
        const int after = (int)f.size() / 3;
        float xmax2 = -1e9f;
        for (size_t i = 0; i < v.size(); i += 3) xmax2 = std::max(xmax2, v[i]);
        if (after != mid - npart)
            rc |= fail("consistency: post-QEM 3% cut should remove the 2.5% inner piece");
        if (xmax2 > 0.401f)
            rc |= fail("consistency: post-QEM 3% cut left the against-body part");
        if (!(xmin > -0.5f))
            rc |= fail("consistency: unexpected geometry");
        std::printf("post-pass consistency: faces %d -> %d -> %d xmax %.3f -> %.3f\n",
                    body + 2 * npart, mid, after, xmax, xmax2);
    }

    // Large-mesh timing. Dust under 2% must not enter the gap search.
    // Also times the old per-vertex ring walk on a slice of the far shell so
    // the speedup is measured, not just the new absolute time.
    {
        std::vector<float> v;
        std::vector<int32_t> f;
        add_grid(v, f, 420);                 // 352800 body faces
        const int body = (int)f.size() / 3;
        const float voxel = 0.01f;
        const double cell = 3.0 * voxel;
        auto add_fan = [&](float cx, float cy, int n) {
            const int c = (int)v.size() / 3;
            v.insert(v.end(), {cx, cy, 0.f});
            for (int i = 0; i < n; ++i) {
                const float ang = 6.2831853f * (float)i / (float)n;
                v.push_back(cx);
                v.push_back(cy + 0.01f * std::sin(ang));
                v.push_back((i % 2) ? 0.05f : -0.01f);
            }
            for (int i = 0; i < n; ++i) {
                f.push_back(c);
                f.push_back(c + 1 + i);
                f.push_back(c + 1 + (i + 1) % n);
            }
        };
        const int band = std::max(2, (int)(0.025f * body));
        const int shell_base = (int)v.size() / 3;
        add_fan(1.6f, 0.f, band);
        const int shell_verts = (int)v.size() / 3 - shell_base;
        add_fan(0.4f + 0.5f * voxel, 0.f, band);
        for (int i = 0; i < 2000; ++i) add_fan(3.f + 0.01f * (i % 50), 3.f + 0.01f * (i / 50), 1);

        // Old search: expand coarse rings per shell vertex until a body sample is found.
        std::unordered_map<uint64_t, int> rep;
        auto pack = [](int x, int y, int z) {
            return (uint64_t)(uint32_t)(x + 0x100000) << 42 |
                   (uint64_t)(uint32_t)(y + 0x100000) << 21 |
                   (uint32_t)(z + 0x100000);
        };
        const double coarse = 8.0 * voxel;
        for (int i = 0; i < shell_base; ++i) {
            const float* p = v.data() + 3 * i;
            const int x = (int)std::floor(p[0] / coarse);
            const int y = (int)std::floor(p[1] / coarse);
            const int z = (int)std::floor(p[2] / coarse);
            rep.emplace(pack(x, y, z), i);
        }
        const int sample = std::min(shell_verts, 400);
        const auto t_old0 = std::chrono::steady_clock::now();
        for (int s = 0; s < sample; ++s) {
            const float* p = v.data() + 3 * (shell_base + 1 + s);
            const int cx = (int)std::floor(p[0] / coarse);
            const int cy = (int)std::floor(p[1] / coarse);
            const int cz = (int)std::floor(p[2] / coarse);
            for (int R = 0; R < 24; ++R) {
                bool hit = false;
                for (int dz = -R; dz <= R && !hit; ++dz)
                    for (int dy = -R; dy <= R && !hit; ++dy)
                        for (int dx = -R; dx <= R; ++dx) {
                            if (R && std::max(std::abs(dx), std::max(std::abs(dy), std::abs(dz))) != R) continue;
                            if (rep.find(pack(cx + dx, cy + dy, cz + dz)) != rep.end()) { hit = true; break; }
                        }
                if (hit) break;
            }
        }
        const double old_sample = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_old0).count();
        const double old_shell = old_sample * (double)shell_verts / (double)sample;
        const auto t0 = std::chrono::steady_clock::now();
        trellis::drop_pre_decimation_components(v, f, voxel);
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("bench: body=%d shell_verts=%d dust=2000  new_filter %.3fs  "
                    "old_ring %.3fs/%d verts (%.1fs extrapolated to this shell)\n",
                    body, shell_verts, dt, old_sample, sample, old_shell);
        (void)cell;
        if (dt > 15.0)
            rc |= fail("component filter slower than 15s on the synthetic large mesh");
        float xmax = -1e9f;
        for (size_t i = 0; i < v.size(); i += 3) xmax = std::max(xmax, v[i]);
        if (!(xmax > 0.40f && xmax < 0.6f))
            rc |= fail("bench: attached part or detached shell decision changed");
    }

    if (rc == 0) std::printf("ok\n");
    return rc;
}
