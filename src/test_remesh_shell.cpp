// Outer-shell check for narrow-band dual contouring. No model weights.
//
//   cmake --build build --target trellis-test-remesh-shell && ./build/trellis-test-remesh-shell
//
// A closed solid's unsigned narrow band is two walls. The remesh must emit
// only the outer one: watertight, outward, with face centroids outside the
// input solid. An open sheet must survive as one layer. A second real body
// must survive the existing pre-decimation component rules.
#include "remesh_dc.h"
#include "tri_bvh.h"
#include "uv_bake.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

int g_fails = 0;

void check(bool cond, const char* msg) {
    if (cond) return;
    std::fprintf(stderr, "FAIL: %s\n", msg);
    ++g_fails;
}

constexpr int kRes = 64;
constexpr int kBand = 1;

float shell_eps() {
    const float scale = (float)(kRes + 3 * kBand) / (float)kRes;
    return (float)kBand * scale / (float)kRes;
}

struct Built {
    std::vector<float> v;
    std::vector<int32_t> f;
    int n = 0;
    float h = 0;
};

void add_quad(std::vector<int32_t>& f, int a, int b, int c, int d) {
    f.insert(f.end(), {a, b, c, a, c, d});
}

Built make_cube(int n, float h, float cx, float cy, float cz) {
    Built b;
    b.n = n;
    b.h = h;
    const int N = n + 1;
    b.v.reserve((size_t)N * N * N * 3);
    for (int k = 0; k < N; ++k)
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i) {
                b.v.push_back(cx + (-h + 2.f * h * (float)i / (float)n));
                b.v.push_back(cy + (-h + 2.f * h * (float)j / (float)n));
                b.v.push_back(cz + (-h + 2.f * h * (float)k / (float)n));
            }
    auto id = [N](int i, int j, int k) { return (k * N + j) * N + i; };
    for (int j = 0; j < n; ++j) for (int k = 0; k < n; ++k) {
        add_quad(b.f, id(n, j, k), id(n, j + 1, k), id(n, j + 1, k + 1), id(n, j, k + 1));
        add_quad(b.f, id(0, j, k), id(0, j, k + 1), id(0, j + 1, k + 1), id(0, j + 1, k));
    }
    for (int i = 0; i < n; ++i) for (int k = 0; k < n; ++k) {
        add_quad(b.f, id(i, n, k), id(i, n, k + 1), id(i + 1, n, k + 1), id(i + 1, n, k));
        add_quad(b.f, id(i, 0, k), id(i + 1, 0, k), id(i + 1, 0, k + 1), id(i, 0, k + 1));
    }
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) {
        add_quad(b.f, id(i, j, n), id(i + 1, j, n), id(i + 1, j + 1, n), id(i, j + 1, n));
        add_quad(b.f, id(i, j, 0), id(i, j + 1, 0), id(i + 1, j + 1, 0), id(i + 1, j, 0));
    }
    return b;
}

void flip_faces(std::vector<int32_t>& f) {
    for (size_t i = 0; i + 2 < f.size(); i += 3) std::swap(f[i + 1], f[i + 2]);
}

void append_mesh(Built& dst, const Built& src) {
    const int base = (int)(dst.v.size() / 3);
    dst.v.insert(dst.v.end(), src.v.begin(), src.v.end());
    for (int idx : src.f) dst.f.push_back(idx + base);
}

void move_plus_x(Built& b, float cy, float cz, float ywin, float zwin, float new_x) {
    const int N = b.n + 1;
    for (int k = 0; k < N; ++k) for (int j = 0; j < N; ++j) {
        const int vi = (k * N + j) * N + b.n;
        const float y = b.v[3 * vi + 1], z = b.v[3 * vi + 2];
        if (std::fabs(y - cy) <= ywin && std::fabs(z - cz) <= zwin) b.v[3 * vi] = new_x;
    }
}

Built make_sheet(int n, float h) {
    Built b;
    b.n = n;
    b.h = h;
    const int N = n + 1;
    for (int j = 0; j < N; ++j) for (int i = 0; i < N; ++i) {
        b.v.push_back(-h + 2.f * h * (float)i / (float)n);
        b.v.push_back(-h + 2.f * h * (float)j / (float)n);
        b.v.push_back(0.f);
    }
    auto id = [N](int i, int j) { return j * N + i; };
    for (int j = 0; j < n; ++j) for (int i = 0; i < n; ++i)
        add_quad(b.f, id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1));
    return b;
}

Built make_sphere(float R, int stacks, int slices) {
    Built b;
    b.v.push_back(0.f); b.v.push_back(0.f); b.v.push_back(R);
    for (int i = 1; i < stacks; ++i) {
        const float theta = 3.14159265f * (float)i / (float)stacks;
        const float z = R * std::cos(theta);
        const float rs = R * std::sin(theta);
        for (int j = 0; j < slices; ++j) {
            const float phi = 6.2831853f * (float)j / (float)slices;
            b.v.push_back(rs * std::cos(phi));
            b.v.push_back(rs * std::sin(phi));
            b.v.push_back(z);
        }
    }
    b.v.push_back(0.f); b.v.push_back(0.f); b.v.push_back(-R);
    auto ring = [slices](int i, int j) { return 1 + (i - 1) * slices + (j % slices); };
    const int south = (int)(b.v.size() / 3) - 1;
    for (int j = 0; j < slices; ++j) {
        b.f.insert(b.f.end(), {0, ring(1, j), ring(1, j + 1)});
        const int i = stacks - 1;
        b.f.insert(b.f.end(), {south, ring(i, j + 1), ring(i, j)});
    }
    for (int i = 1; i < stacks - 1; ++i) for (int j = 0; j < slices; ++j) {
        const int a = ring(i, j), c = ring(i, j + 1), d = ring(i + 1, j), e = ring(i + 1, j + 1);
        add_quad(b.f, a, c, e, d);
    }
    // Force outward winding on this convex mesh so the input itself is sane.
    // The inward-cube case covers a fully flipped solid separately.
    for (size_t t = 0; t + 2 < b.f.size(); t += 3) {
        const float* a = &b.v[3 * b.f[t]];
        const float* c = &b.v[3 * b.f[t + 1]];
        const float* d = &b.v[3 * b.f[t + 2]];
        const float ab[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
        const float ac[3] = {d[0] - a[0], d[1] - a[1], d[2] - a[2]};
        const float n[3] = {ab[1] * ac[2] - ab[2] * ac[1], ab[2] * ac[0] - ab[0] * ac[2], ab[0] * ac[1] - ab[1] * ac[0]};
        const float cx = (a[0] + c[0] + d[0]) / 3.f, cy = (a[1] + c[1] + d[1]) / 3.f, cz = (a[2] + c[2] + d[2]) / 3.f;
        if (n[0] * cx + n[1] * cy + n[2] * cz < 0.f) std::swap(b.f[t + 1], b.f[t + 2]);
    }
    return b;
}

trellis::Mesh remesh(const std::vector<float>& v, const std::vector<int32_t>& f, int res = kRes) {
    const trellis::TriBvh bvh = trellis::TriBvh::build(v.data(), (int64_t)v.size() / 3,
                                                       f.data(), (int64_t)f.size() / 3);
    return trellis::remesh_narrow_band_dc(v.data(), (int64_t)v.size() / 3,
                                          f.data(), (int64_t)f.size() / 3,
                                          bvh, res, kBand, 0.f);
}

int boundary_edges(const std::vector<int32_t>& f) {
    std::vector<uint64_t> keys;
    keys.reserve(f.size());
    auto ek = [](int a, int b) {
        if (a > b) std::swap(a, b);
        return ((uint64_t)(uint32_t)a << 32) | (uint32_t)b;
    };
    for (size_t i = 0; i + 2 < f.size(); i += 3)
        for (int k = 0; k < 3; ++k) keys.push_back(ek(f[i + k], f[i + (k + 1) % 3]));
    std::sort(keys.begin(), keys.end());
    int boundary = 0;
    for (size_t i = 0; i < keys.size();) {
        size_t j = i + 1;
        while (j < keys.size() && keys[j] == keys[i]) ++j;
        if (j - i == 1) ++boundary;
        i = j;
    }
    return boundary;
}

int component_count(const std::vector<int32_t>& f, int V) {
    if (V <= 0 || f.empty()) return 0;
    std::vector<int> par((size_t)V);
    for (int i = 0; i < V; ++i) par[i] = i;
    auto find = [&](int x) {
        while (par[x] != x) { par[x] = par[par[x]]; x = par[x]; }
        return x;
    };
    for (size_t i = 0; i + 2 < f.size(); i += 3) {
        int a = find(f[i]), b = find(f[i + 1]), c = find(f[i + 2]);
        par[b] = a; par[c] = a;
    }
    std::vector<char> seen((size_t)V, 0);
    int n = 0;
    for (size_t i = 0; i < f.size(); ++i) {
        const int r = find(f[i]);
        if (!seen[r]) { seen[r] = 1; ++n; }
    }
    return n;
}

double signed_volume(const std::vector<float>& v, const std::vector<int32_t>& f) {
    double tp = 0;
    for (size_t i = 0; i + 2 < f.size(); i += 3) {
        const float* a = &v[3 * f[i]];
        const float* b = &v[3 * f[i + 1]];
        const float* c = &v[3 * f[i + 2]];
        const double abx = b[0] - a[0], aby = b[1] - a[1], abz = b[2] - a[2];
        const double acx = c[0] - a[0], acy = c[1] - a[1], acz = c[2] - a[2];
        const double nx = aby * acz - abz * acy;
        const double ny = abz * acx - abx * acz;
        const double nz = abx * acy - aby * acx;
        tp += a[0] * nx + a[1] * ny + a[2] * nz;
    }
    return tp / 6.0;
}

// Fraction of vertices whose chebyshev radius around `center` is inside the
// box of half-extent `h` by more than `margin`. The inner offset wall sits
// about one eps inside; the outer wall does not.
int deep_inside_box(const std::vector<float>& v, float cx, float cy, float cz, float h, float margin) {
    int n = 0;
    const float lim = h - margin;
    for (size_t i = 0; i + 2 < v.size(); i += 3) {
        const float m = std::max(std::fabs(v[i] - cx), std::max(std::fabs(v[i + 1] - cy), std::fabs(v[i + 2] - cz)));
        if (m < lim) ++n;
    }
    return n;
}

int count_x(const std::vector<float>& v, float x0, float x1) {
    int n = 0;
    for (size_t i = 0; i + 2 < v.size(); i += 3)
        if (v[i] >= x0 && v[i] <= x1) ++n;
    return n;
}

bool ray_hit(const float o[3], const float d[3], const float* a, const float* b, const float* c) {
    const float eps = 1e-7f;
    const float e1[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
    const float e2[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
    const float p[3] = {d[1] * e2[2] - d[2] * e2[1], d[2] * e2[0] - d[0] * e2[2], d[0] * e2[1] - d[1] * e2[0]};
    const float det = e1[0] * p[0] + e1[1] * p[1] + e1[2] * p[2];
    if (det > -eps && det < eps) return false;
    const float inv = 1.f / det;
    const float tvec[3] = {o[0] - a[0], o[1] - a[1], o[2] - a[2]};
    const float u = (tvec[0] * p[0] + tvec[1] * p[1] + tvec[2] * p[2]) * inv;
    if (u < 0.f || u > 1.f) return false;
    const float q[3] = {tvec[1] * e1[2] - tvec[2] * e1[1], tvec[2] * e1[0] - tvec[0] * e1[2], tvec[0] * e1[1] - tvec[1] * e1[0]};
    const float vv = (d[0] * q[0] + d[1] * q[1] + d[2] * q[2]) * inv;
    if (vv < 0.f || u + vv > 1.f) return false;
    const float t = (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]) * inv;
    return t > 1e-5f;
}

// Majority of three ray directions. Returns 1 inside, 0 outside, -1 if the
// three rays disagree (treat as unknown).
int parity_vote(const std::vector<float>& v, const std::vector<int32_t>& f, const float o[3]) {
    const float dirs[3][3] = {
        {1.f, 0.013f, 0.007f},
        {0.011f, 1.f, 0.017f},
        {0.019f, 0.008f, 1.f},
    };
    int insides = 0;
    for (const auto& d : dirs) {
        int hits = 0;
        for (size_t i = 0; i + 2 < f.size(); i += 3)
            if (ray_hit(o, d, &v[3 * f[i]], &v[3 * f[i + 1]], &v[3 * f[i + 2]])) ++hits;
        insides += hits & 1;
    }
    if (insides >= 2) return 1;
    if (insides == 0) return 0;
    return -1;
}

void expect_outer_box(const char* name, const trellis::Mesh& m, float cx, float cy, float cz, float h) {
    const float eps = shell_eps();
    std::printf("  [%s] V=%d F=%d boundary=%d comps=%d vol=%.4g deep_inside=%d\n",
                name, m.V(), m.F(), boundary_edges(m.faces), component_count(m.faces, m.V()),
                signed_volume(m.verts, m.faces), deep_inside_box(m.verts, cx, cy, cz, h, 0.45f * eps));
    check(m.F() > 200, (std::string(name) + ": empty or tiny shell").c_str());
    check(boundary_edges(m.faces) == 0, (std::string(name) + ": open edges (not watertight)").c_str());
    check(component_count(m.faces, m.V()) == 1, (std::string(name) + ": expected one shell").c_str());
    check(signed_volume(m.verts, m.faces) > 0.0, (std::string(name) + ": inward winding").c_str());
    check(deep_inside_box(m.verts, cx, cy, cz, h, 0.45f * eps) == 0,
          (std::string(name) + ": vertices on the inner wall").c_str());
    int outside = 0;
    for (size_t i = 0; i + 2 < m.verts.size(); i += 3) {
        const float mc = std::max(std::fabs(m.verts[i] - cx),
                                  std::max(std::fabs(m.verts[i + 1] - cy), std::fabs(m.verts[i + 2] - cz)));
        if (mc > h) ++outside;
    }
    check(outside * 10 > m.V() * 9, (std::string(name) + ": less than 90% of vertices outside the input box").c_str());
}

void test_cube(bool inward) {
    Built b = make_cube(10, 0.25f, 0.f, 0.f, 0.f);
    if (inward) flip_faces(b.f);
    std::printf("\n== %s cube ==\n", inward ? "inward" : "outward");
    const trellis::Mesh m = remesh(b.v, b.f);
    expect_outer_box(inward ? "inward-cube" : "cube", m, 0.f, 0.f, 0.f, 0.25f);
}

void test_sphere() {
    const float R = 0.22f;
    const float eps = shell_eps();
    Built b = make_sphere(R, 18, 28);
    std::printf("\n== sphere R=%.3f eps=%.4f ==\n", R, eps);
    const trellis::Mesh m = remesh(b.v, b.f);
    float min_r = 1e30f, max_r = 0.f;
    double sum_r = 0;
    int n = 0;
    for (size_t i = 0; i + 2 < m.verts.size(); i += 3) {
        const float r = std::sqrt(m.verts[i] * m.verts[i] + m.verts[i + 1] * m.verts[i + 1] + m.verts[i + 2] * m.verts[i + 2]);
        min_r = std::min(min_r, r);
        max_r = std::max(max_r, r);
        sum_r += r;
        ++n;
    }
    const double mean = n ? sum_r / n : 0;
    std::printf("  [sphere] V=%d F=%d boundary=%d min_r=%.4f mean_r=%.4f max_r=%.4f (outer~%.4f inner~%.4f)\n",
                m.V(), m.F(), boundary_edges(m.faces), min_r, mean, max_r, R + eps, R - eps);
    check(m.F() > 200, "sphere: empty shell");
    check(boundary_edges(m.faces) == 0, "sphere: open edges");
    check(signed_volume(m.verts, m.faces) > 0.0, "sphere: inward winding");
    check(min_r > R, "sphere: vertex inside the input sphere (inner wall)");
    check(mean > R + 0.4 * eps, "sphere: mean radius is not the outer offset");
    check(mean < R + 2.5 * eps, "sphere: shell blew outward");
}

void test_sheet() {
    const float h = 0.22f;
    const float eps = shell_eps();
    Built b = make_sheet(24, h);
    std::printf("\n== open sheet ==\n");
    const trellis::Mesh m = remesh(b.v, b.f);
    float zmin = 1e30f, zmax = -1e30f, xmin = 1e30f, xmax = -1e30f, ymin = 1e30f, ymax = -1e30f;
    double zsum = 0;
    const int n = m.V();
    for (int i = 0; i < n; ++i) {
        xmin = std::min(xmin, m.verts[3 * i]);
        xmax = std::max(xmax, m.verts[3 * i]);
        ymin = std::min(ymin, m.verts[3 * i + 1]);
        ymax = std::max(ymax, m.verts[3 * i + 1]);
        zmin = std::min(zmin, m.verts[3 * i + 2]);
        zmax = std::max(zmax, m.verts[3 * i + 2]);
        zsum += m.verts[3 * i + 2];
    }
    const double zmean = n ? zsum / n : 0;
    std::printf("  [sheet] V=%d F=%d z=[%.4f, %.4f] mean=%.4f span=%.4f eps=%.4f\n",
                m.V(), m.F(), zmin, zmax, zmean, zmax - zmin, eps);
    // An open sheet does not seal a volume, so the flood reaches both sides and
    // the unsigned tube is kept: one watertight slab about 2*eps thick.
    check(m.F() > 100, "sheet: surface was deleted");
    check(boundary_edges(m.faces) == 0, "sheet: open edges on the fallback tube");
    check(xmax - xmin > 2.f * h - 4.f * eps, "sheet: lost XY extent");
    check(ymax - ymin > 2.f * h - 4.f * eps, "sheet: lost XY extent");
    check(zmax - zmin > 1.4f * eps, "sheet: open surface lost a side");
    check(std::fabs(zmean) < 0.5f * eps, "sheet: slab drifted off the sheet");
    (void)zmin;
}

void test_recess() {
    const float h = 0.25f;
    // Deeper than eps so dilation does not close it, shallow enough that the
    // walls stay a couple of cells and the voxel barrier does not tear.
    const float depth = 0.05f;
    const float eps = shell_eps();
    Built b = make_cube(14, h, 0.f, 0.f, 0.f);
    move_plus_x(b, 0.f, 0.f, 0.09f, 0.09f, h - depth);
    std::printf("\n== recessed cube (floor x=%.3f) ==\n", h - depth);
    const int recess_res = kRes;
    const float origin[3] = {0.f, 0.f, 0.f};
    const float pocket[3] = {h - 0.5f * depth, 0.f, 0.f}; // between floor and opening
    const float solid[3] = {h - depth - 0.04f, 0.f, 0.f};
    const int p_origin = parity_vote(b.v, b.f, origin);
    const int p_pocket = parity_vote(b.v, b.f, pocket);
    const int p_solid = parity_vote(b.v, b.f, solid);
    std::printf("  [recess input] parity center=%d pocket=%d solid=%d boundary=%d\n",
                p_origin, p_pocket, p_solid, boundary_edges(b.f));
    check(p_origin == 1, "recess: parity harness says the cube center is outside");
    check(p_pocket == 0, "recess: parity harness says the pocket is solid");
    check(p_solid == 1, "recess: parity harness says the material behind the floor is air");

    const trellis::Mesh m = remesh(b.v, b.f, recess_res);
    const int out_pocket = parity_vote(m.verts, m.faces, pocket);
    int floor_verts = 0;
    for (int i = 0; i < m.V(); ++i) {
        const float x = m.verts[3 * i], y = m.verts[3 * i + 1], z = m.verts[3 * i + 2];
        if (x > (h - depth) + 0.15f * eps && x < (h - depth) + 1.8f * eps &&
            std::fabs(y) < 0.06f && std::fabs(z) < 0.06f) ++floor_verts;
    }
    std::printf("  [recess] V=%d F=%d boundary=%d pocket_parity=%d floor_verts=%d\n",
                m.V(), m.F(), boundary_edges(m.faces), out_pocket, floor_verts);
    check(m.F() > 200, "recess: empty shell");
    check(boundary_edges(m.faces) == 0, "recess: open edges");
    check(component_count(m.faces, m.V()) == 1, "recess: extra pieces");
    check(signed_volume(m.verts, m.faces) > 0.0, "recess: inward winding");
    check(out_pocket == 0, "recess: pocket was filled or sealed by the inner wall");
    check(floor_verts > 8, "recess: pocket floor was lost");
    // A full inner wall would put a large share of vertices eps inside the
    // undented back face. A few corner samples can sit there on a coarse dent.
    int deep_back = 0;
    for (int i = 0; i < m.V(); ++i) {
        const float x = m.verts[3 * i], y = m.verts[3 * i + 1], z = m.verts[3 * i + 2];
        const float mc = std::max(std::fabs(x), std::max(std::fabs(y), std::fabs(z)));
        if (x < -0.02f && mc < h - 0.5f * shell_eps()) ++deep_back;
    }
    std::printf("  [recess] deep_back=%d / %d\n", deep_back, m.V());
    check(deep_back * 8 < m.V(), "recess: inner wall behind the undented face");
    (void)eps;
}

void test_ridge() {
    const float h = 0.22f;
    const float push = 0.10f;
    const float eps = shell_eps();
    Built b = make_cube(16, h, 0.f, 0.f, 0.f);
    // One grid row on +X, pushed out into a thin ridge. Thickness at the root
    // is one grid step, under 2*eps, so an unsigned band has no inner cavity
    // there — the ridge must still come through on the outer shell.
    const float spacing = 2.f * h / 16.f;
    move_plus_x(b, 0.f, 0.f, 0.12f, spacing * 0.49f, h + push);
    std::printf("\n== thin ridge (tip x=%.3f) ==\n", h + push);
    const trellis::Mesh m = remesh(b.v, b.f);
    float max_x = -1e30f;
    for (int i = 0; i < m.V(); ++i) max_x = std::max(max_x, m.verts[3 * i]);
    std::printf("  [ridge] V=%d F=%d boundary=%d max_x=%.4f (tip %.3f, lost-if-below %.3f)\n",
                m.V(), m.F(), boundary_edges(m.faces), max_x, h + push, h + eps + 0.02f);
    check(m.F() > 200, "ridge: empty shell");
    check(boundary_edges(m.faces) == 0, "ridge: open edges");
    check(signed_volume(m.verts, m.faces) > 0.0, "ridge: inward winding");
    check(deep_inside_box(m.verts, 0.f, 0.f, 0.f, h, 0.45f * eps) == 0, "ridge: inner wall inside the cube");
    check(max_x > h + 0.55f * push, "ridge: thin feature was cut off");
}

void test_bodies() {
    // Big and medium are both well above the 3% keep. Tiny is a real separate
    // body; the pre-decimation rules decide it from its face share and gap.
    // Output face count follows the remesh grid, so the big body has to be
    // physically larger (not just finer) for the speck's offset blob to fall
    // under 2%. The speck sits more than 3 voxels away. Medium stays above 3%.
    Built big = make_cube(12, 0.16f, -0.22f, 0.f, 0.f);
    Built mid = make_cube(8, 0.09f, 0.16f, 0.f, 0.f);
    Built tiny = make_cube(2, 0.008f, 0.36f, 0.f, 0.f);
    Built all = big;
    append_mesh(all, mid);
    append_mesh(all, tiny);
    std::printf("\n== three bodies ==\n");
    trellis::Mesh m = remesh(all.v, all.f);
    const float eps = shell_eps();
    const float voxel = trellis::remesh_voxel_world(kRes, kBand);
    std::printf("  [bodies] V=%d F=%d comps=%d eps=%.4f voxel=%.4f\n",
                m.V(), m.F(), component_count(m.faces, m.V()), eps, voxel);
    check(m.F() > 200, "bodies: empty shell");
    const int before_big = count_x(m.verts, -0.45f, -0.02f);
    const int before_mid = count_x(m.verts, 0.05f, 0.28f);
    check(before_big > 80, "bodies: big body missing before filter");
    check(before_mid > 40, "bodies: medium body missing before filter");
    const int deep_big = deep_inside_box(m.verts, -0.22f, 0.f, 0.f, 0.16f, 0.45f * eps);
    const int deep_mid = deep_inside_box(m.verts, 0.16f, 0.f, 0.f, 0.09f, 0.45f * eps);
    std::printf("  [bodies] deep big=%d mid=%d\n", deep_big, deep_mid);
    check(deep_big == 0, "bodies: inner wall in the big box");
    check(deep_mid == 0, "bodies: inner wall in the medium box");

    const int before_tiny = count_x(m.verts, 0.30f, 0.44f);
    const int F_before = m.F();
    // Largest-component share, matching drop_pre_decimation_components: tiny
    // is not the largest. Use its vertex count only as a presence test; the
    // face share is measured by walking components below.
    std::vector<int> par((size_t)m.V());
    for (int i = 0; i < m.V(); ++i) par[i] = i;
    auto find = [&](int x) {
        while (par[x] != x) { par[x] = par[par[x]]; x = par[x]; }
        return x;
    };
    for (int fi = 0; fi < m.F(); ++fi) {
        const int a = find(m.faces[3 * fi]);
        par[find(m.faces[3 * fi + 1])] = a;
        par[find(m.faces[3 * fi + 2])] = a;
    }
    int tiny_faces = 0, max_faces = 0;
    std::vector<int> fc((size_t)m.V(), 0);
    for (int fi = 0; fi < m.F(); ++fi) fc[find(m.faces[3 * fi])]++;
    for (int c : fc) max_faces = std::max(max_faces, c);
    for (int fi = 0; fi < m.F(); ++fi) {
        const float x = (m.verts[3 * m.faces[3 * fi]] + m.verts[3 * m.faces[3 * fi + 1]] + m.verts[3 * m.faces[3 * fi + 2]]) / 3.f;
        if (x > 0.28f) tiny_faces++;
    }
    const double share = max_faces > 0 ? (double)tiny_faces / (double)max_faces : 0;
    const int dropped = trellis::drop_pre_decimation_components(m.verts, m.faces, voxel);
    const int after_big = count_x(m.verts, -0.45f, -0.02f);
    const int after_mid = count_x(m.verts, 0.05f, 0.28f);
    const int after_tiny = count_x(m.verts, 0.30f, 0.44f);
    std::printf("  [bodies filter] dropped=%d tiny_faces=%d share=%.2f%% of %d before_tiny_verts=%d after F=%d (was %d)\n",
                dropped, tiny_faces, 100.0 * share, max_faces, before_tiny, m.F(), F_before);
    std::printf("  [bodies filter] verts big %d -> %d, mid %d -> %d, tiny %d -> %d\n",
                before_big, after_big, before_mid, after_mid, before_tiny, after_tiny);
    check(after_big > 80, "bodies: filter removed the big body");
    check(after_mid > 40, "bodies: filter removed the medium body");
    const bool should_drop = share < 0.03; // <2% always, and 2–3% because the gap is > 3 voxels
    if (before_tiny == 0) {
        check(after_tiny == 0, "bodies: tiny appeared during filter");
    } else if (should_drop) {
        check(after_tiny == 0, "bodies: detached speck under 3% was kept");
    } else {
        check(after_tiny > 0, "bodies: real body at or above 3% was dropped");
    }
}

double outward_fraction(const std::vector<float>& v, const std::vector<int32_t>& f) {
    const int V = (int)v.size() / 3;
    if (V <= 0 || f.size() < 3) return 0;
    double cx = 0, cy = 0, cz = 0;
    for (int i = 0; i < V; ++i) { cx += v[3 * i]; cy += v[3 * i + 1]; cz += v[3 * i + 2]; }
    cx /= V; cy /= V; cz /= V;
    double out_a = 0, in_a = 0;
    for (size_t i = 0; i + 2 < f.size(); i += 3) {
        const float* a = &v[3 * f[i]];
        const float* b = &v[3 * f[i + 1]];
        const float* c = &v[3 * f[i + 2]];
        const float e1x = b[0] - a[0], e1y = b[1] - a[1], e1z = b[2] - a[2];
        const float e2x = c[0] - a[0], e2y = c[1] - a[1], e2z = c[2] - a[2];
        const float nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
        const double area = 0.5 * std::sqrt((double)nx * nx + (double)ny * ny + (double)nz * nz);
        const float mx = (a[0] + b[0] + c[0]) / 3.f - (float)cx;
        const float my = (a[1] + b[1] + c[1]) / 3.f - (float)cy;
        const float mz = (a[2] + b[2] + c[2]) / 3.f - (float)cz;
        if (nx * mx + ny * my + nz * mz > 0.f) out_a += area;
        else in_a += area;
    }
    const double tot = out_a + in_a;
    return tot > 0 ? out_a / tot : 0;
}

// Two concentric shells ~2.6 cells apart, joined by a couple of triangles, with
// garbled or fully inverted winding. That is the Pixal failure mode: a thin
// hollow band the component filter cannot split, and a sign that must not
// come from face winding. The outer wall stays; the inner one goes.
void test_hollow_band(bool invert) {
    const float scale = (float)(kRes + 3 * kBand) / (float)kRes;
    const float cell = scale / (float)kRes;
    const float outer_h = 0.25f;
    const float inner_h = outer_h - 2.6f * cell;
    Built mesh = make_cube(12, outer_h, 0.f, 0.f, 0.f);
    const int base = (int)(mesh.v.size() / 3);
    append_mesh(mesh, make_cube(10, inner_h, 0.f, 0.f, 0.f));
    mesh.f.insert(mesh.f.end(), {0, 1, base, 1, base + 1, base});
    if (invert) flip_faces(mesh.f);
    else {
        for (size_t i = 0; i + 5 < mesh.f.size(); i += 6) std::swap(mesh.f[i + 1], mesh.f[i + 2]);
    }
    std::printf("\n== hollow band %s (gap %.3f cells) ==\n", invert ? "inverted" : "garbled", 2.6f);
    trellis::Mesh m = remesh(mesh.v, mesh.f);
    const int boundary_before_clean = boundary_edges(m.faces);
    const int comps_before = component_count(m.faces, m.V());
    trellis::clean_mesh(m.verts.data(), m.V(), m.faces);
    const float eps = shell_eps();
    const double out_frac = outward_fraction(m.verts, m.faces);
    const int deep = deep_inside_box(m.verts, 0.f, 0.f, 0.f, outer_h, 0.5f * (outer_h - inner_h));
    const float naive = 6.f * (2.f * outer_h) * (2.f * outer_h) / (cell * cell) * 2.f;
    std::printf("  [hollow] V=%d F=%d boundary=%d->%d comps=%d vol=%.4g outward=%.2f deep=%d naive=%.0f\n",
                m.V(), m.F(), boundary_before_clean, boundary_edges(m.faces), comps_before,
                signed_volume(m.verts, m.faces), out_frac, deep, naive);
    check(m.F() > (int)(naive * 0.45f), "hollow: shell collapsed");
    check(m.F() < (int)(naive * 1.45f), "hollow: inner wall was not removed (~50% reduction)");
    check(boundary_edges(m.faces) == 0, "hollow: open edges");
    check(boundary_before_clean == 0, "hollow: remesh introduced open edges");
    check(component_count(m.faces, m.V()) == 1, "hollow: extra pieces");
    check(comps_before == 1, "hollow: remesh split the shell");
    check(signed_volume(m.verts, m.faces) > 0.0, "hollow: inside-out after clean");
    check(out_frac > 0.75, "hollow: outward area does not dominate");
    check(deep == 0, "hollow: vertices remain on the inner wall");
    (void)eps;
}

void test_clean_restores_outward() {
    Built b = make_cube(10, 0.25f, 0.f, 0.f, 0.f);
    trellis::Mesh m = remesh(b.v, b.f);
    flip_faces(m.faces);
    const double before = signed_volume(m.verts, m.faces);
    trellis::clean_mesh(m.verts.data(), m.V(), m.faces);
    const double after = signed_volume(m.verts, m.faces);
    const double out_frac = outward_fraction(m.verts, m.faces);
    std::printf("\n== clean_mesh on inverted shell ==\n");
    std::printf("  [clean] vol %.4g -> %.4g outward=%.2f boundary=%d\n",
                before, after, out_frac, boundary_edges(m.faces));
    check(before < 0.0, "clean: pre-flip was not inside-out");
    check(after > 0.0, "clean: left the shell inside-out");
    check(out_frac > 0.75, "clean: outward area does not dominate");
    check(boundary_edges(m.faces) == 0, "clean: introduced open edges");
    check(deep_inside_box(m.verts, 0.f, 0.f, 0.f, 0.25f, 0.45f * shell_eps()) == 0,
          "clean: moved geometry onto the inner wall");
}

}  // namespace

int main() {
    std::printf("remesh shell test res=%d band=%d eps=%.5f\n", kRes, kBand, shell_eps());
    test_cube(false);
    test_cube(true);
    test_sphere();
    test_sheet();
    test_recess();
    test_ridge();
    test_bodies();
    test_hollow_band(false);
    test_hollow_band(true);
    test_clean_restores_outward();
    if (g_fails) {
        std::fprintf(stderr, "\n%d check(s) failed\n", g_fails);
        return 1;
    }
    std::printf("\nall outer-shell checks passed\n");
    return 0;
}
