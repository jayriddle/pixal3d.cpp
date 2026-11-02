#include "remesh_dc.h"
#include "tri_bvh.h"
#include "uv_bake.h"
#include <algorithm>
#include <atomic>
#ifdef _MSC_VER
#include <intrin.h>
#endif
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <thread>
#include <unordered_map>
#include <vector>

namespace trellis {

namespace {

inline int ctz64(uint64_t v) {
#ifdef _MSC_VER
    unsigned long i;
    _BitScanForward64(&i, v);
    return (int)i;
#else
    return __builtin_ctzll(v);
#endif
}

inline uint64_t key3(int x, int y, int z) {
    return ((uint64_t)(uint32_t)x << 42) | ((uint64_t)(uint32_t)y << 21) | (uint32_t)z;
}

// スレッド数。Emscripten ビルドは pthread 無しでリンクしているので std::thread の生成が
// "thread constructor failed: Not supported" で throw する。さらに下の候補ビットセットは
// スレッドごとに res^3/8 バイト（res=1024 で 134 MB）を確保するため、wasm32 の 4 GiB
// ヒープでは並列度がそのままメモリ消費になる。ブラウザでは直列にする。
inline int remesh_threads() {
#ifdef __EMSCRIPTEN__
    return 1;
#else
    return (int)std::max(1u, std::thread::hardware_concurrency());
#endif
}

void parallel_for(int64_t n, const std::function<void(int64_t, int64_t)>& fn) {
    const int nt = remesh_threads();
    if (nt <= 1) { if (n > 0) fn(0, n); return; }   // 直列（pthread 無しの wasm ビルド）
    std::vector<std::thread> ts;
    const int64_t chunk = (n + nt - 1) / nt;
    for (int t = 0; t < nt; ++t) {
        const int64_t b = t * chunk, e = std::min(n, b + chunk);
        if (b >= e) break;
        ts.emplace_back(fn, b, e);
    }
    for (auto& t : ts) t.join();
}

// Separating-axis triangle vs AABB. `a,b,c` and the box are in the same space.
bool tri_hits_box(const float a[3], const float b[3], const float c[3],
                  const float bmin[3], const float bmax[3]) {
    float ctr[3], h[3], v0[3], v1[3], v2[3];
    for (int k = 0; k < 3; ++k) {
        ctr[k] = 0.5f * (bmin[k] + bmax[k]);
        h[k] = 0.5f * (bmax[k] - bmin[k]);
        v0[k] = a[k] - ctr[k];
        v1[k] = b[k] - ctr[k];
        v2[k] = c[k] - ctr[k];
    }
    float mn, mx;
    for (int k = 0; k < 3; ++k) {
        mn = std::min(v0[k], std::min(v1[k], v2[k]));
        mx = std::max(v0[k], std::max(v1[k], v2[k]));
        if (mn > h[k] || mx < -h[k]) return false;
    }
    float e0[3], e1[3];
    for (int k = 0; k < 3; ++k) { e0[k] = v1[k] - v0[k]; e1[k] = v2[k] - v0[k]; }
    const float nrm[3] = {e0[1]*e1[2] - e0[2]*e1[1], e0[2]*e1[0] - e0[0]*e1[2], e0[0]*e1[1] - e0[1]*e1[0]};
    auto axis_ok = [&](float ax, float ay, float az) -> bool {
        const float p0 = v0[0]*ax + v0[1]*ay + v0[2]*az;
        const float p1 = v1[0]*ax + v1[1]*ay + v1[2]*az;
        const float p2 = v2[0]*ax + v2[1]*ay + v2[2]*az;
        const float r = h[0]*std::fabs(ax) + h[1]*std::fabs(ay) + h[2]*std::fabs(az);
        const float lo = std::min(p0, std::min(p1, p2));
        const float hi = std::max(p0, std::max(p1, p2));
        return lo <= r && hi >= -r;
    };
    if (!axis_ok(nrm[0], nrm[1], nrm[2])) return false;
    float e2[3] = {v2[0]-v1[0], v2[1]-v1[1], v2[2]-v1[2]};
    const float* edges[3] = {e0, e1, e2};
    for (int i = 0; i < 3; ++i) {
        const float* e = edges[i];
        if (!axis_ok(0.f, -e[2], e[1])) return false;
        if (!axis_ok(e[2], 0.f, -e[0])) return false;
        if (!axis_ok(-e[1], e[0], 0.f)) return false;
    }
    return true;
}

// Exterior of the input surface on the remesh grid. Empty cells are flood-filled
// from the domain boundary; a cell the flood reaches is outside. Everything
// the surface seals off — the solid and the hollow interior of a thin shell —
// stays inside. Face winding is not consulted.
struct ExteriorGrid {
    int res = 0;
    float scale = 1.f;
    bool ready = false;
    std::vector<uint64_t> blocked;
    std::vector<uint64_t> exterior;

    int64_t ncells() const { return (int64_t)res * res * res; }
    int64_t index(int x, int y, int z) const {
        return ((int64_t)x * res + y) * res + z;
    }
    void setb(std::vector<uint64_t>& bits, int x, int y, int z) {
        if ((unsigned)x >= (unsigned)res || (unsigned)y >= (unsigned)res || (unsigned)z >= (unsigned)res) return;
        const int64_t i = index(x, y, z);
        bits[(size_t)(i >> 6)] |= 1ull << (unsigned)(i & 63);
    }
    bool getb(const std::vector<uint64_t>& bits, int x, int y, int z) const {
        if ((unsigned)x >= (unsigned)res || (unsigned)y >= (unsigned)res || (unsigned)z >= (unsigned)res) return false;
        const int64_t i = index(x, y, z);
        return ((bits[(size_t)(i >> 6)] >> (unsigned)(i & 63)) & 1ull) != 0;
    }
    void to_grid(const float p[3], float g[3]) const {
        for (int k = 0; k < 3; ++k) g[k] = (p[k] / scale + 0.5f) * (float)res;
    }
    // Flooded cell, or a blocked surface cell that touches the flood (so a
    // thick surface voxel does not swallow the outer offset).
    bool outside_cell(int x, int y, int z) const {
        if (getb(exterior, x, y, z)) return true;
        if (!getb(blocked, x, y, z)) return false;
        static const int D[6][3] = {{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
        for (int i = 0; i < 6; ++i)
            if (getb(exterior, x + D[i][0], y + D[i][1], z + D[i][2])) return true;
        return false;
    }
    bool is_outside(const float p[3]) const {
        if (!ready) return true;
        float g[3];
        to_grid(p, g);
        int base[3];
        for (int k = 0; k < 3; ++k) {
            int i = (int)std::floor(g[k]);
            if (i < 0) i = 0;
            if (i >= res) i = res - 1;
            base[k] = i;
        }
        if (outside_cell(base[0], base[1], base[2])) return true;
        // Grid vertices land on a cell corner. Look at the cell on the other
        // side of that plane too, so the corner isn't inside just because
        // floor() picked the inward cell.
        for (int k = 0; k < 3; ++k) {
            const float frac = g[k] - std::floor(g[k]);
            if (frac > 1e-3f && frac < 1.f - 1e-3f) continue;
            int nb[3] = {base[0], base[1], base[2]};
            nb[k] += (frac < 0.5f) ? -1 : 1;
            if (outside_cell(nb[0], nb[1], nb[2])) return true;
        }
        return false;
    }

    void voxelize(const float* verts, const int32_t* faces, int64_t nF, int nV) {
        for (int64_t fi = 0; fi < nF; ++fi) {
            const int ia = faces[3*fi], ib = faces[3*fi+1], ic = faces[3*fi+2];
            if (ia < 0 || ib < 0 || ic < 0 || ia >= nV || ib >= nV || ic >= nV) continue;
            if (ia == ib || ib == ic || ia == ic) continue;
            float g[3][3];
            to_grid(verts + 3*ia, g[0]);
            to_grid(verts + 3*ib, g[1]);
            to_grid(verts + 3*ic, g[2]);
            int c0[3], c1[3];
            for (int k = 0; k < 3; ++k) {
                const float mn = std::min(g[0][k], std::min(g[1][k], g[2][k]));
                const float mx = std::max(g[0][k], std::max(g[1][k], g[2][k]));
                c0[k] = std::max(0, (int)std::floor(mn) - 1);
                c1[k] = std::min(res - 1, (int)std::floor(mx) + 1);
            }
            // Grow the test box by a fraction of a cell so a triangle that only
            // clips a corner still seals that diagonal. A full extra cell of
            // blocker swallows the outer offset and opens holes, so this stays small.
            constexpr float grow = 0.05f;
            for (int x = c0[0]; x <= c1[0]; ++x)
                for (int y = c0[1]; y <= c1[1]; ++y)
                    for (int z = c0[2]; z <= c1[2]; ++z) {
                        const float bmin[3] = {(float)x - grow, (float)y - grow, (float)z - grow};
                        const float bmax[3] = {(float)x + 1.f + grow, (float)y + 1.f + grow, (float)z + 1.f + grow};
                        if (tri_hits_box(g[0], g[1], g[2], bmin, bmax)) setb(blocked, x, y, z);
                    }
            // Sample the triangle interior so a long thin face cannot skip a cell
            // the separating-axis test misses.
            float e1[3], e2[3];
            for (int k = 0; k < 3; ++k) { e1[k] = g[1][k] - g[0][k]; e2[k] = g[2][k] - g[0][k]; }
            const float l1 = std::sqrt(e1[0]*e1[0] + e1[1]*e1[1] + e1[2]*e1[2]);
            const float l2 = std::sqrt(e2[0]*e2[0] + e2[1]*e2[1] + e2[2]*e2[2]);
            const int n1 = std::max(1, (int)std::ceil(l1 / 0.35f));
            const int n2 = std::max(1, (int)std::ceil(l2 / 0.35f));
            for (int i = 0; i <= n1; ++i) for (int j = 0; j <= n2; ++j) {
                const float u = (float)i / (float)n1;
                const float v = (float)j / (float)n2;
                if (u + v > 1.f) continue;
                int c[3];
                bool okc = true;
                for (int k = 0; k < 3; ++k) {
                    const float p = g[0][k] + u * e1[k] + v * e2[k];
                    c[k] = (int)std::floor(p);
                    if ((unsigned)c[k] >= (unsigned)res) okc = false;
                }
                if (okc) setb(blocked, c[0], c[1], c[2]);
            }
        }
    }

    // Close single-cell cracks in the blocker so a watertight mesh cannot leak
    // the flood through a diagonal gap. Wider openings (a real hole, a recess)
    // stay open and the flood walks through them.
    void seal_cracks() {
        const std::vector<uint64_t> src = blocked;
        for (int axis = 0; axis < 3; ++axis) {
            for (int z = 0; z < res; ++z)
                for (int y = 0; y < res; ++y)
                    for (int x = 0; x < res; ++x) {
                        int a[3] = {x, y, z};
                        if (a[axis] + 2 >= res) continue;
                        int b[3] = {x, y, z};
                        int m[3] = {x, y, z};
                        b[axis] += 2;
                        m[axis] += 1;
                        if (getb(src, a[0], a[1], a[2]) && getb(src, b[0], b[1], b[2]) &&
                            !getb(blocked, m[0], m[1], m[2]))
                            setb(blocked, m[0], m[1], m[2]);
                    }
        }
    }

    void flood() {
        std::vector<int64_t> st;
        st.reserve(4096);
        auto push_empty = [&](int x, int y, int z) {
            if ((unsigned)x >= (unsigned)res || (unsigned)y >= (unsigned)res || (unsigned)z >= (unsigned)res) return;
            if (getb(blocked, x, y, z) || getb(exterior, x, y, z)) return;
            st.push_back(index(x, y, z));
        };
        for (int y = 0; y < res; ++y) for (int z = 0; z < res; ++z) {
            push_empty(0, y, z);
            push_empty(res - 1, y, z);
        }
        for (int x = 0; x < res; ++x) for (int z = 0; z < res; ++z) {
            push_empty(x, 0, z);
            push_empty(x, res - 1, z);
        }
        for (int x = 0; x < res; ++x) for (int y = 0; y < res; ++y) {
            push_empty(x, y, 0);
            push_empty(x, y, res - 1);
        }
        const int64_t plane = (int64_t)res * res;
        while (!st.empty()) {
            const int64_t id = st.back();
            st.pop_back();
            const int z = (int)(id / plane);
            const int y = (int)((id / res) % res);
            const int x = (int)(id % res);
            if (getb(exterior, x, y, z) || getb(blocked, x, y, z)) continue;
            int x0 = x;
            while (x0 > 0 && !getb(blocked, x0 - 1, y, z) && !getb(exterior, x0 - 1, y, z)) --x0;
            int x1 = x;
            while (x1 + 1 < res && !getb(blocked, x1 + 1, y, z) && !getb(exterior, x1 + 1, y, z)) ++x1;
            for (int xi = x0; xi <= x1; ++xi) setb(exterior, xi, y, z);
            for (int xi = x0; xi <= x1; ++xi) {
                auto step = [&](int nx, int ny, int nz, bool prev_open) {
                    if (prev_open) return;
                    push_empty(nx, ny, nz);
                };
                const bool prev_ym = xi > x0 && y > 0 && !getb(blocked, xi - 1, y - 1, z) && !getb(exterior, xi - 1, y - 1, z);
                const bool prev_yp = xi > x0 && y + 1 < res && !getb(blocked, xi - 1, y + 1, z) && !getb(exterior, xi - 1, y + 1, z);
                const bool prev_zm = xi > x0 && z > 0 && !getb(blocked, xi - 1, y, z - 1) && !getb(exterior, xi - 1, y, z - 1);
                const bool prev_zp = xi > x0 && z + 1 < res && !getb(blocked, xi - 1, y, z + 1) && !getb(exterior, xi - 1, y, z + 1);
                step(xi, y - 1, z, prev_ym);
                step(xi, y + 1, z, prev_yp);
                step(xi, y, z - 1, prev_zm);
                step(xi, y, z + 1, prev_zp);
            }
        }
    }

    void build(const float* verts, int nV, const int32_t* faces, int64_t nF, int resolution, float sc) {
        res = resolution;
        scale = sc;
        ready = false;
        if (res <= 1 || nF <= 0 || nV <= 0) return;
        const size_t words = (size_t)((ncells() + 63) / 64);
        blocked.assign(words, 0);
        exterior.assign(words, 0);
        voxelize(verts, faces, nF, nV);
        bool any_blocked = false;
        for (uint64_t w : blocked) if (w) { any_blocked = true; break; }
        if (!any_blocked) {
            blocked.clear();
            exterior.clear();
            return;
        }
        seal_cracks();
        flood();
        ready = true;
    }
};

bool ray_hits_tri(const float o[3], const float d[3], const float* a, const float* b, const float* c) {
    const float e1[3] = {b[0]-a[0], b[1]-a[1], b[2]-a[2]};
    const float e2[3] = {c[0]-a[0], c[1]-a[1], c[2]-a[2]};
    const float p[3] = {d[1]*e2[2]-d[2]*e2[1], d[2]*e2[0]-d[0]*e2[2], d[0]*e2[1]-d[1]*e2[0]};
    const float det = e1[0]*p[0] + e1[1]*p[1] + e1[2]*p[2];
    if (std::fabs(det) < 1e-12f) return false;
    const float inv = 1.f / det;
    const float tvec[3] = {o[0]-a[0], o[1]-a[1], o[2]-a[2]};
    const float u = (tvec[0]*p[0] + tvec[1]*p[1] + tvec[2]*p[2]) * inv;
    if (u < 0.f || u > 1.f) return false;
    const float qv[3] = {tvec[1]*e1[2]-tvec[2]*e1[1], tvec[2]*e1[0]-tvec[0]*e1[2], tvec[0]*e1[1]-tvec[1]*e1[0]};
    const float v = (d[0]*qv[0] + d[1]*qv[1] + d[2]*qv[2]) * inv;
    if (v < 0.f || u + v > 1.f) return false;
    const float t = (e2[0]*qv[0] + e2[1]*qv[1] + e2[2]*qv[2]) * inv;
    return t > 1e-5f;
}

// Odd parity along +X means the point is inside a closed shell. Used only to
// drop a small component trapped inside a larger one (an inner-wall fragment).
bool point_inside_component(const Mesh& m, const std::vector<int>& fcomp, int comp, const float o[3]) {
    const float dirs[3][3] = {{1.f, 0.017f, 0.011f}, {0.013f, 1.f, 0.019f}, {0.011f, 0.013f, 1.f}};
    int insides = 0;
    for (const auto& d : dirs) {
        int hits = 0;
        for (int f = 0; f < m.F(); ++f) {
            if (comp >= 0 && fcomp[f] != comp) continue;
            const float* a = &m.verts[3 * m.faces[3*f]];
            const float* b = &m.verts[3 * m.faces[3*f+1]];
            const float* c = &m.verts[3 * m.faces[3*f+2]];
            if (ray_hits_tri(o, d, a, b, c)) ++hits;
        }
        insides += hits & 1;
    }
    return insides >= 2;
}

struct FaceAgg {
    int faces = 0;
    int boundary = 0;
    double vol = 0;
    double ext = 0, inn = 0;
    double cx = 0, cy = 0, cz = 0;
    float bmin[3] = {1e30f, 1e30f, 1e30f};
    float bmax[3] = {-1e30f, -1e30f, -1e30f};
};

int find_root(std::vector<int>& par, int x) {
    while (par[x] != x) { par[x] = par[par[x]]; x = par[x]; }
    return x;
}

// Per vertex-connected component: drop a small piece whose centroids sit in
// the interior (an inner-wall fragment). A large piece in the interior means
// the sign field inverted — caller rebuilds the unsigned band for that bbox.
// Nearly-closed pieces with negative volume are flipped outward.
// Returns false when a large component failed and `failed_boxes` lists the
// boxes that must be replaced from the unsigned shell. `m` is updated.
bool screen_shell(Mesh& m, const ExteriorGrid& grid, std::vector<float>& failed_boxes) {
    failed_boxes.clear();
    const int V = m.V();
    const int F = m.F();
    if (V <= 0 || F <= 0) return false;
    std::vector<int> par((size_t)V);
    for (int i = 0; i < V; ++i) par[i] = i;
    auto uni = [&](int a, int b) {
        a = find_root(par, a); b = find_root(par, b);
        if (a != b) par[a] = b;
    };
    for (int f = 0; f < F; ++f) {
        const int a = m.faces[3*f], b = m.faces[3*f+1], c = m.faces[3*f+2];
        if (a < 0 || b < 0 || c < 0 || a >= V || b >= V || c >= V) continue;
        uni(a, b); uni(b, c);
    }
    std::unordered_map<int, int> cid;
    std::vector<int> fcomp((size_t)F, -1);
    for (int f = 0; f < F; ++f) {
        const int a = m.faces[3*f];
        if (a < 0 || a >= V) continue;
        const int r = find_root(par, a);
        auto it = cid.find(r);
        if (it == cid.end()) it = cid.emplace(r, (int)cid.size()).first;
        fcomp[f] = it->second;
    }
    const int nc = (int)cid.size();
    if (nc <= 0) return false;
    std::vector<FaceAgg> ag((size_t)nc);
    for (int f = 0; f < F; ++f) {
        const int c = fcomp[f];
        if (c < 0) continue;
        const int ia = m.faces[3*f], ib = m.faces[3*f+1], ic = m.faces[3*f+2];
        const float* pa = &m.verts[3*ia];
        const float* pb = &m.verts[3*ib];
        const float* pc = &m.verts[3*ic];
        float e1[3], e2[3];
        for (int k = 0; k < 3; ++k) { e1[k] = pb[k]-pa[k]; e2[k] = pc[k]-pa[k]; }
        const float nx = e1[1]*e2[2] - e1[2]*e2[1];
        const float ny = e1[2]*e2[0] - e1[0]*e2[2];
        const float nz = e1[0]*e2[1] - e1[1]*e2[0];
        const double area = 0.5 * std::sqrt((double)nx*nx + (double)ny*ny + (double)nz*nz);
        FaceAgg& A = ag[c];
        A.faces++;
        A.vol += (double)pa[0]*nx + (double)pa[1]*ny + (double)pa[2]*nz;
        const float cx = (pa[0]+pb[0]+pc[0]) / 3.f;
        const float cy = (pa[1]+pb[1]+pc[1]) / 3.f;
        const float cz = (pa[2]+pb[2]+pc[2]) / 3.f;
        const float q[3] = {cx, cy, cz};
        if (grid.is_outside(q)) A.ext += area; else A.inn += area;
        A.cx += cx * area; A.cy += cy * area; A.cz += cz * area;
        for (int k = 0; k < 3; ++k) {
            const float* p[3] = {pa, pb, pc};
            for (int j = 0; j < 3; ++j) {
                A.bmin[k] = std::min(A.bmin[k], p[j][k]);
                A.bmax[k] = std::max(A.bmax[k], p[j][k]);
            }
        }
    }
    struct EK { uint64_t k; int c; };
    std::vector<EK> edges;
    edges.reserve((size_t)F * 3);
    auto ek = [](int a, int b) -> uint64_t {
        if (a > b) std::swap(a, b);
        return ((uint64_t)(uint32_t)a << 32) | (uint32_t)b;
    };
    for (int f = 0; f < F; ++f) {
        if (fcomp[f] < 0) continue;
        for (int j = 0; j < 3; ++j) {
            const int a = m.faces[3*f+j], b = m.faces[3*f+(j+1)%3];
            edges.push_back({ek(a, b), fcomp[f]});
        }
    }
    std::sort(edges.begin(), edges.end(), [](const EK& p, const EK& q) {
        if (p.k != q.k) return p.k < q.k;
        return p.c < q.c;
    });
    for (size_t i = 0; i < edges.size();) {
        size_t j = i + 1;
        while (j < edges.size() && edges[j].k == edges[i].k && edges[j].c == edges[i].c) ++j;
        if (j - i == 1) ag[edges[i].c].boundary++;
        i = j;
    }
    int largest = 0;
    for (const FaceAgg& A : ag) largest = std::max(largest, A.faces);
    std::vector<char> drop((size_t)nc, 0), flip((size_t)nc, 0);
    bool need_unsigned = false;
    const float pad = grid.ready ? 4.f * grid.scale / (float)grid.res : 0.05f;
    int main_c = 0;
    for (int c = 1; c < nc; ++c) if (ag[c].faces > ag[main_c].faces) main_c = c;
    for (int c = 0; c < nc; ++c) {
        const FaceAgg& A = ag[c];
        const double tot = A.ext + A.inn;
        const double ext_frac = tot > 0 ? A.ext / tot : 1.0;
        const bool big = largest > 0 && A.faces * 5 >= largest;
        if (c != main_c && tot > 0) {
            const float q[3] = {(float)(A.cx / tot), (float)(A.cy / tot), (float)(A.cz / tot)};
            // A fragment whose centroid is inside the main shell is the inner wall.
            if (point_inside_component(m, fcomp, main_c, q)) { drop[c] = 1; continue; }
        }
        if (ext_frac < 0.45) {
            if (big) {
                need_unsigned = true;
                if (A.bmin[0] <= A.bmax[0]) {
                    failed_boxes.push_back(A.bmin[0] - pad);
                    failed_boxes.push_back(A.bmin[1] - pad);
                    failed_boxes.push_back(A.bmin[2] - pad);
                    failed_boxes.push_back(A.bmax[0] + pad);
                    failed_boxes.push_back(A.bmax[1] + pad);
                    failed_boxes.push_back(A.bmax[2] + pad);
                }
                drop[c] = 1;
            } else {
                drop[c] = 1; // stray inner-wall fragment
            }
            continue;
        }
        const bool closed = (int64_t)A.boundary * 100 <= (int64_t)std::max(A.faces, 1);
        if (closed && A.vol < 0.0) flip[c] = 1;
    }
    bool any_drop = false, any_flip = false;
    for (int c = 0; c < nc; ++c) { any_drop = any_drop || drop[c]; any_flip = any_flip || flip[c]; }
    if (any_flip) {
        for (int f = 0; f < F; ++f)
            if (fcomp[f] >= 0 && flip[fcomp[f]]) std::swap(m.faces[3*f+1], m.faces[3*f+2]);
    }
    if (any_drop) {
        std::vector<int32_t> nf;
        nf.reserve(m.faces.size());
        for (int f = 0; f < F; ++f)
            if (fcomp[f] < 0 || !drop[fcomp[f]])
                for (int k = 0; k < 3; ++k) nf.push_back(m.faces[3*f+k]);
        m.faces.swap(nf);
        std::vector<int> remap((size_t)V, -1);
        std::vector<float> nv;
        nv.reserve(m.verts.size());
        for (size_t i = 0; i < m.faces.size(); ++i) {
            int v = m.faces[i];
            if (v < 0 || v >= V) continue;
            if (remap[v] < 0) {
                remap[v] = (int)(nv.size() / 3);
                nv.insert(nv.end(), &m.verts[3*v], &m.verts[3*v] + 3);
            }
            m.faces[i] = remap[v];
        }
        m.verts.swap(nv);
    }
    return !need_unsigned;
}

}  // namespace

Mesh remesh_narrow_band_dc(const float* iverts, int64_t iV, const int32_t* ifaces, int64_t iF,
                           const TriBvh& bvh, int res, int band, float project_back) {
    (void)iV;
    Mesh out;
    if (iF == 0 || bvh.empty() || res <= 0) return out;

    // Reference domain: the world cube is inflated by (res+3·band)/res so the
    // offset shell never touches the boundary; eps is the offset distance.
    const float scale = (float)(res + 3 * band) / (float)res;
    const float cell = scale / (float)res;
    const float eps = (float)band * cell;
    const float keep = 0.87f * cell;

    // Candidate cells: conservative dilation of every triangle's AABB by the
    // band-plus-crossing radius, marked in a res^3 bitset.
    const int64_t nbits = (int64_t)res * res * res;
    std::vector<uint64_t> cand((size_t)((nbits + 63) / 64), 0);
    // A cell is active iff UDF(center) < (band+0.87)·cell, and its closest
    // surface point lies inside some triangle-marked cell, so the per-axis
    // index distance is < band+1.37, i.e. ≤ band+1.
    const int dil = band + 1;
    {
        const int F = (int)iF;
        std::vector<std::vector<uint64_t>> parts;
        const int nt = remesh_threads();
        parts.assign(nt, {});
        std::vector<std::thread> ts;
        const int chunk = (F + nt - 1) / nt;
        for (int t = 0; t < nt; ++t) {
            const int b = t * chunk, e = std::min(F, b + chunk);
            if (b >= e) break;
            parts[t].assign(cand.size(), 0);
            auto job = [&, t, b, e]() {
                auto& bits = parts[t];
                auto setb = [&bits, res](int x, int y, int z) {
                    const int64_t i = ((int64_t)x * res + y) * res + z;
                    bits[(size_t)(i >> 6)] |= 1ull << (i & 63);
                };
                for (int f = b; f < e; ++f) {
                    float bmin[3] = {1e30f, 1e30f, 1e30f}, bmax[3] = {-1e30f, -1e30f, -1e30f};
                    for (int j = 0; j < 3; ++j) {
                        const float* p = &iverts[3 * ifaces[3*f+j]];
                        for (int k = 0; k < 3; ++k) {
                            bmin[k] = std::min(bmin[k], p[k]);
                            bmax[k] = std::max(bmax[k], p[k]);
                        }
                    }
                    int c0[3], c1[3];
                    for (int k = 0; k < 3; ++k) {
                        c0[k] = std::max(0, (int)std::floor((bmin[k] / scale + 0.5f) * res) - dil);
                        c1[k] = std::min(res - 1, (int)std::floor((bmax[k] / scale + 0.5f) * res) + dil);
                    }
                    for (int x = c0[0]; x <= c1[0]; ++x)
                        for (int y = c0[1]; y <= c1[1]; ++y)
                            for (int z = c0[2]; z <= c1[2]; ++z) setb(x, y, z);
                }
            };
            if (nt <= 1) job(); else ts.emplace_back(std::move(job));   // 直列（pthread 無しの wasm ビルド）
        }
        for (auto& th : ts) th.join();
        for (auto& bits : parts)
            if (!bits.empty())
                for (size_t i = 0; i < cand.size(); ++i) cand[i] |= bits[i];
    }

    // Candidate indices, then drop the res³ bitset before the shell orientation
    // tables (those are the new peak, and they must not sit on top of the bitset).
    std::vector<int64_t> cand_cells;
    for (int64_t w = 0; w < (int64_t)cand.size(); ++w) {
        uint64_t bits = cand[w];
        while (bits) {
            const int b = ctz64(bits);
            bits &= bits - 1;
            cand_cells.push_back((w << 6) | b);
        }
    }
    cand.clear(); cand.shrink_to_fit();

    const auto t_shell = std::chrono::steady_clock::now();
    ExteriorGrid grid;
    grid.build(iverts, (int)iV, ifaces, (int)iF, res, scale);
    const double flood_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_shell).count();

    // Active voxels. |UDF−eps| < 0.87·cell is the unsigned tube (both walls).
    // The outer wall is the same band restricted to cells the flood reached.
    int64_t n_udf = 0, n_sdf = 0;
    int64_t active_voxels = 0;
    bool built_ok = false;
    for (int pass = 0; pass < 2 && !built_ok; ++pass) {
        const bool use_signed = grid.ready && pass == 0;
        std::vector<int> acoord;
        {
            std::vector<uint8_t> act_u(cand_cells.size(), 0), act_s(cand_cells.size(), 0);
            parallel_for((int64_t)cand_cells.size(), [&](int64_t b, int64_t e) {
                for (int64_t i = b; i < e; ++i) {
                    const int64_t c = cand_cells[i];
                    const int x = (int)(c / ((int64_t)res * res)), y = (int)((c / res) % res), z = (int)(c % res);
                    const float p[3] = { ((x + 0.5f) / res - 0.5f) * scale,
                                         ((y + 0.5f) / res - 0.5f) * scale,
                                         ((z + 0.5f) / res - 0.5f) * scale };
                    const TriBvh::Hit h = bvh.closest(p, eps + keep);
                    if (h.face < 0) continue;
                    const float udf = std::sqrt(h.dist2);
                    if (std::fabs(udf - eps) < keep) act_u[i] = 1;
                    if (use_signed && grid.is_outside(p) && std::fabs(udf - eps) < keep) act_s[i] = 1;
                }
            });
            if (pass == 0) {
                n_udf = 0; n_sdf = 0;
                for (size_t i = 0; i < cand_cells.size(); ++i) {
                    if (act_u[i]) ++n_udf;
                    if (act_s[i]) ++n_sdf;
                }
            }
            for (size_t i = 0; i < cand_cells.size(); ++i) {
                if (use_signed ? !act_s[i] : !act_u[i]) continue;
                const int64_t c = cand_cells[i];
                acoord.push_back((int)(c / ((int64_t)res * res)));
                acoord.push_back((int)((c / res) % res));
                acoord.push_back((int)(c % res));
            }
        }
        const int64_t Na = (int64_t)acoord.size() / 3;
        if (Na < 100) {
            if (use_signed && n_udf >= 100) continue;
            fprintf(stderr, "  remesh: only %lld active voxels; skipping remesh\n", (long long)Na);
            return out;
        }

        std::unordered_map<uint64_t, int> vox;
        vox.reserve((size_t)Na * 2);
        for (int64_t i = 0; i < Na; ++i) vox.emplace(key3(acoord[3*i], acoord[3*i+1], acoord[3*i+2]), (int)i);

        std::vector<int> vcoord;
        std::unordered_map<uint64_t, int> vmap;
        vmap.reserve((size_t)Na * 3);
        for (int64_t i = 0; i < Na; ++i)
            for (int dx = 0; dx < 2; ++dx) for (int dy = 0; dy < 2; ++dy) for (int dz = 0; dz < 2; ++dz) {
                const int x = acoord[3*i] + dx, y = acoord[3*i+1] + dy, z = acoord[3*i+2] + dz;
                if (vmap.emplace(key3(x, y, z), (int)(vcoord.size() / 3)).second) {
                    vcoord.push_back(x); vcoord.push_back(y); vcoord.push_back(z);
                }
            }
        const int64_t Nv = (int64_t)vcoord.size() / 3;
        std::vector<float> fvert((size_t)Nv);
        parallel_for(Nv, [&](int64_t b, int64_t e) {
            for (int64_t i = b; i < e; ++i) {
                const float p[3] = { ((float)vcoord[3*i]   / res - 0.5f) * scale,
                                     ((float)vcoord[3*i+1] / res - 0.5f) * scale,
                                     ((float)vcoord[3*i+2] / res - 0.5f) * scale };
                const TriBvh::Hit h = bvh.closest(p, eps + 2 * cell);
                const float udf = h.face >= 0 ? std::sqrt(h.dist2) : eps + 2 * cell;
                float sdf = udf;
                if (use_signed && h.face >= 0 && !grid.is_outside(p)) sdf = -udf;
                fvert[i] = sdf - eps;
            }
        });
        auto fval = [&](int x, int y, int z) -> float {
            auto it = vmap.find(key3(x, y, z));
            return it == vmap.end() ? 1e9f : fvert[it->second];
        };

        std::vector<float> dual((size_t)Na * 3);
        std::vector<int8_t> owned((size_t)Na * 3, 0);
        parallel_for(Na, [&](int64_t b, int64_t e) {
            for (int64_t i = b; i < e; ++i) {
                const int vx = acoord[3*i], vy = acoord[3*i+1], vz = acoord[3*i+2];
                double sum[3] = {0, 0, 0};
                int cnt = 0;
                for (int axis = 0; axis < 3; ++axis)
                    for (int u = 0; u < 2; ++u) for (int v = 0; v < 2; ++v) {
                        int a0[3] = {vx, vy, vz}, a1[3];
                        a0[(axis + 1) % 3] += u;
                        a0[(axis + 2) % 3] += v;
                        a1[0] = a0[0]; a1[1] = a0[1]; a1[2] = a0[2];
                        a1[axis] += 1;
                        const float v1 = fval(a0[0], a0[1], a0[2]);
                        const float v2 = fval(a1[0], a1[1], a1[2]);
                        const bool c12 = v1 < 0 && v2 >= 0, c21 = v1 >= 0 && v2 < 0;
                        if (c12 || c21) {
                            const float t = -v1 / (v2 - v1);
                            double pt[3] = {(double)a0[0], (double)a0[1], (double)a0[2]};
                            pt[axis] += t;
                            for (int k = 0; k < 3; ++k) sum[k] += pt[k];
                            ++cnt;
                        }
                        if (u == 1 && v == 1) owned[3*i + axis] = c12 ? 1 : (c21 ? -1 : 0);
                    }
                if (cnt) for (int k = 0; k < 3; ++k) dual[3*i+k] = (float)(sum[k] / cnt);
                else { dual[3*i] = vx + 0.5f; dual[3*i+1] = vy + 0.5f; dual[3*i+2] = vz + 0.5f; }
            }
        });

        static const int OFF[3][4][3] = {
            {{0,0,0}, {0,0,1}, {0,1,1}, {0,1,0}},
            {{0,0,0}, {1,0,0}, {1,0,1}, {0,0,1}},
            {{0,0,0}, {0,1,0}, {1,1,0}, {1,0,0}},
        };
        std::vector<int32_t> qfaces;
        qfaces.reserve((size_t)Na * 6);
        std::vector<uint8_t> used((size_t)Na, 0);
        for (int64_t i = 0; i < Na; ++i) {
            const int vx = acoord[3*i], vy = acoord[3*i+1], vz = acoord[3*i+2];
            for (int axis = 0; axis < 3; ++axis) {
                const int dir = owned[3*i + axis];
                if (!dir) continue;
                int q[4];
                bool ok = true;
                for (int k = 0; k < 4 && ok; ++k) {
                    auto it = vox.find(key3(vx + OFF[axis][k][0], vy + OFF[axis][k][1], vz + OFF[axis][k][2]));
                    if (it == vox.end()) ok = false;
                    else q[k] = it->second;
                }
                if (!ok) continue;
                static const int S1N[6] = {0, 1, 2, 0, 2, 3};
                static const int S1P[6] = {0, 2, 1, 0, 3, 2};
                const int* sp = dir > 0 ? S1P : S1N;
                for (int k = 0; k < 6; ++k) qfaces.push_back(q[sp[k]]);
                for (int k = 0; k < 4; ++k) used[q[k]] = 1;
            }
        }
        if (qfaces.empty()) {
            if (use_signed && n_udf >= 100) continue;
            return out;
        }

        Mesh built;
        std::vector<int32_t> remap((size_t)Na, -1);
        int nv2 = 0;
        for (int64_t i = 0; i < Na; ++i)
            if (used[i]) {
                remap[i] = nv2++;
                built.verts.push_back((dual[3*i]   / res - 0.5f) * scale);
                built.verts.push_back((dual[3*i+1] / res - 0.5f) * scale);
                built.verts.push_back((dual[3*i+2] / res - 0.5f) * scale);
            }
        built.faces.resize(qfaces.size());
        for (size_t k = 0; k < qfaces.size(); ++k) built.faces[k] = remap[qfaces[k]];

        if (use_signed) {
            // Faces whose centroids sit in the interior are the inner wall,
            // including where a few edges still join it to the outer shell.
            {
                std::vector<int32_t> kept;
                kept.reserve(built.faces.size());
                const int V = built.V();
                for (int f = 0; f < built.F(); ++f) {
                    const int ia = built.faces[3*f], ib = built.faces[3*f+1], ic = built.faces[3*f+2];
                    if (ia < 0 || ib < 0 || ic < 0 || ia >= V || ib >= V || ic >= V) continue;
                    const float q[3] = {
                        (built.verts[3*ia] + built.verts[3*ib] + built.verts[3*ic]) / 3.f,
                        (built.verts[3*ia+1] + built.verts[3*ib+1] + built.verts[3*ic+1]) / 3.f,
                        (built.verts[3*ia+2] + built.verts[3*ib+2] + built.verts[3*ic+2]) / 3.f,
                    };
                    if (!grid.is_outside(q)) continue;
                    kept.push_back(ia); kept.push_back(ib); kept.push_back(ic);
                }
                if ((int)kept.size() / 3 > built.F() / 5) built.faces.swap(kept);
                std::vector<int> remap((size_t)V, -1);
                std::vector<float> nv;
                for (size_t i = 0; i < built.faces.size(); ++i) {
                    const int v = built.faces[i];
                    if (remap[v] < 0) {
                        remap[v] = (int)(nv.size() / 3);
                        nv.insert(nv.end(), &built.verts[3*v], &built.verts[3*v] + 3);
                    }
                    built.faces[i] = remap[v];
                }
                built.verts.swap(nv);
            }
            std::vector<float> failed_boxes;
            if (!screen_shell(built, grid, failed_boxes)) {
                // Dominant component sits in the interior (sign came out inverted).
                // A cell-count cutoff does not catch this — an inverted field still
                // keeps about half the band. Rebuild the unsigned tube.
                continue;
            }
        }
        out = std::move(built);
        active_voxels = Na;
        built_ok = true;

        const int faces_after = out.F();
        int faces_before = faces_after;
        int removed_faces = 0;
        if (use_signed && n_sdf > 0 && n_udf > n_sdf) {
            faces_before = (int)std::llround((double)faces_after * (double)n_udf / (double)n_sdf);
            if (faces_before < faces_after) faces_before = faces_after;
            removed_faces = faces_before - faces_after;
        }
        const double shell_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_shell).count();
        printf("  remesh postproc: outer shell faces %d -> %d (removed %d inner-wall faces) in %.3fs (flood %.3fs)%s\n",
               faces_before, faces_after, removed_faces, shell_s, flood_s,
               use_signed ? "" : " (unsigned fallback)");
        fflush(stdout);
    }
    if (!built_ok) return out;

    // Voxel cracks leave a handful of boundary edges on curved shells. Close
    // only short loops; a real opening is longer than this and stays open.
    if (out.F() > 0) {
        const int filled = fill_small_holes(out.faces, 192);
        if (filled) {
            printf("  remesh postproc: filled %d short remesh cracks\n", filled);
            fflush(stdout);
        }
    }

    // Project the dual vertices back onto the input surface (reference:
    // remesh_project=0.9, o_voxel/postprocess.py::to_glb -> remeshing.py §8).
    // Dual contouring places each vertex at the plain MEAN of its cell's edge
    // crossings, on the eps-offset shell — so sharp features come out rounded and
    // the shell keeps its own offset noise. Lerping each vertex back toward its
    // closest point on the original surface restores those features. The BVH's
    // closest point is exactly the reference's barycentric interpolation of the
    // hit triangle, so no uvw round-trip is needed.
    if (project_back > 0.0f) {
        const int64_t NV = (int64_t)out.verts.size() / 3;
        std::atomic<int64_t> missed{0};
        parallel_for(NV, [&](int64_t b, int64_t e) {
            int64_t local = 0;
            for (int64_t i = b; i < e; ++i) {
                float* v = &out.verts[3 * i];
                const float p[3] = {v[0], v[1], v[2]};
                // Vertices sit on the eps shell, so the surface is ~eps away; the
                // same bound the field pass uses is comfortably sufficient.
                const TriBvh::Hit h = bvh.closest(p, eps + 2 * cell);
                if (h.face < 0) { ++local; continue; }   // no hit in range: leave as-is
                for (int k = 0; k < 3; ++k) v[k] = p[k] - project_back * (p[k] - h.point[k]);
            }
            if (local) missed.fetch_add(local, std::memory_order_relaxed);
        });
        if (missed.load())
            printf("  remesh_dc: project_back %.2f (%lld/%lld vertices had no hit in range)\n",
                   project_back, (long long)missed.load(), (long long)NV);
    }

    printf("  remesh_dc: %lld active voxels -> V=%d F=%d (eps=%.4g, project_back=%.2f)\n",
           (long long)active_voxels, out.V(), out.F(), eps, project_back);
    fflush(stdout);
    return out;
}

}  // namespace trellis
