#include "remesh_dc.h"
#include "tri_bvh.h"
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
    double seal_s = 0;
    double flood_s = 0;
    // Inclusive cell range of the input vertices. Cells strictly outside this
    // box cannot be inside the solid.
    int amin[3] = {0, 0, 0};
    int amax[3] = {-1, -1, -1};
    // Coarse empty-region labels. 0 = sealed inside, 1 = surface (use the fine
    // bitset), 2 = empty and reachable from outside the grid. Block size 4 keeps
    // the volume flood on a 4³-smaller grid; the fine flood stays in the surface
    // blocks only.
    static constexpr int kCoarse = 4;
    int cn = 0;
    std::vector<uint8_t> ckind;
    std::vector<uint64_t> blocked;
    std::vector<uint64_t> exterior;
    std::vector<int64_t> blocked_ids;

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
    // First write records the id so sealing can walk the surface, not res³.
    bool mark_blocked(int x, int y, int z) {
        if ((unsigned)x >= (unsigned)res || (unsigned)y >= (unsigned)res || (unsigned)z >= (unsigned)res) return false;
        const int64_t i = index(x, y, z);
        uint64_t& w = blocked[(size_t)(i >> 6)];
        const uint64_t mask = 1ull << (unsigned)(i & 63);
        if (w & mask) return false;
        w |= mask;
        blocked_ids.push_back(i);
        return true;
    }
    void to_grid(const float p[3], float g[3]) const {
        for (int k = 0; k < 3; ++k) g[k] = (p[k] / scale + 0.5f) * (float)res;
    }
    bool cell_oob(int x, int y, int z) const {
        return (unsigned)x >= (unsigned)res || (unsigned)y >= (unsigned)res || (unsigned)z >= (unsigned)res;
    }
    bool outside_aabb_cell(int x, int y, int z) const {
        return x < amin[0] || x > amax[0] || y < amin[1] || y > amax[1] || z < amin[2] || z > amax[2];
    }
    int coarse_index(int x, int y, int z) const {
        const int cx = x / kCoarse, cy = y / kCoarse, cz = z / kCoarse;
        return (cx * cn + cy) * cn + cz;
    }
    // Empty coarse cell already proven reachable from outside the grid.
    bool coarse_out(int x, int y, int z) const {
        if (cn <= 0 || cell_oob(x, y, z)) return false;
        const int cx = x / kCoarse, cy = y / kCoarse, cz = z / kCoarse;
        if ((unsigned)cx >= (unsigned)cn || (unsigned)cy >= (unsigned)cn || (unsigned)cz >= (unsigned)cn)
            return false;
        return ckind[(size_t)coarse_index(x, y, z)] == 2;
    }
    bool coarse_mixed(int x, int y, int z) const {
        if (cn <= 0 || cell_oob(x, y, z)) return false;
        return ckind[(size_t)coarse_index(x, y, z)] == 1;
    }
    bool reached_out(int x, int y, int z) const {
        if (cell_oob(x, y, z) || outside_aabb_cell(x, y, z)) return true;
        if (getb(exterior, x, y, z) || coarse_out(x, y, z)) return true;
        return false;
    }
    // A surface voxel is outside only on the side that faces the flood. Treating
    // the whole voxel as outside pulls inner-wall samples onto the outer sheet.
    bool blocked_outside(int x, int y, int z, const float g[3]) const {
        static const int D[6][3] = {{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
        const float frac[3] = {g[0] - (float)x, g[1] - (float)y, g[2] - (float)z};
        float best_ext = 1e30f, best_in = 1e30f;
        for (int i = 0; i < 6; ++i) {
            const int nx = x + D[i][0], ny = y + D[i][1], nz = z + D[i][2];
            const int ax = i >> 1;
            const float face_dist = (i & 1) ? frac[ax] : (1.f - frac[ax]);
            const bool ext = reached_out(nx, ny, nz);
            if (ext) best_ext = std::min(best_ext, face_dist);
            else if (!getb(blocked, nx, ny, nz)) best_in = std::min(best_in, face_dist);
        }
        if (best_ext >= 1e29f) return false;
        if (best_in >= 1e29f) return true;
        return best_ext <= best_in;
    }
    bool outside_cell(int x, int y, int z, const float g[3]) const {
        if (cell_oob(x, y, z)) return true;
        if (outside_aabb_cell(x, y, z)) return true;
        if (!getb(blocked, x, y, z) && coarse_out(x, y, z)) return true;
        if (getb(exterior, x, y, z)) return true;
        if (cn > 0 && !coarse_mixed(x, y, z) && !getb(blocked, x, y, z)) return false;
        if (getb(blocked, x, y, z)) return blocked_outside(x, y, z, g);
        // One-cell halo of the flooded surface. A sealed cavity is not adjacent
        // to a flooded cell; deep exterior is a coarse-outside block.
        static const int D[6][3] = {{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
        for (int i = 0; i < 6; ++i) {
            const int nx = x + D[i][0], ny = y + D[i][1], nz = z + D[i][2];
            if (reached_out(nx, ny, nz)) return true;
        }
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
        if (outside_cell(base[0], base[1], base[2], g)) return true;
        // Grid vertices land on a cell corner. Look at the cell on the other
        // side of that plane too, so the corner isn't inside just because
        // floor() picked the inward cell.
        for (int k = 0; k < 3; ++k) {
            const float frac = g[k] - std::floor(g[k]);
            if (frac > 1e-3f && frac < 1.f - 1e-3f) continue;
            int nb[3] = {base[0], base[1], base[2]};
            nb[k] += (frac < 0.5f) ? -1 : 1;
            if (outside_cell(nb[0], nb[1], nb[2], g)) return true;
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
                        if (tri_hits_box(g[0], g[1], g[2], bmin, bmax)) mark_blocked(x, y, z);
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
                if (okc) mark_blocked(c[0], c[1], c[2]);
            }
        }
    }

    // Close single-cell cracks between blocked surface cells so a watertight
    // mesh cannot leak the flood through a one-cell gap. Only the blocked-cell
    // list is visited — never the dense res³ grid. Wider openings (a real hole,
    // a recess, a gap between feet) stay open.
    void seal_cracks() {
        const auto t0 = std::chrono::steady_clock::now();
        const size_t n = blocked_ids.size();
        const int64_t plane = (int64_t)res * res;
        std::vector<int> fill;
        fill.reserve(n / 4 + 16);
        static const int D[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        for (size_t s = 0; s < n; ++s) {
            const int64_t id = blocked_ids[s];
            const int x = (int)(id / plane);
            const int y = (int)((id / res) % res);
            const int z = (int)(id % res);
            for (int a = 0; a < 3; ++a) {
                const int x2 = x + 2 * D[a][0], y2 = y + 2 * D[a][1], z2 = z + 2 * D[a][2];
                const int xm = x + D[a][0], ym = y + D[a][1], zm = z + D[a][2];
                if (getb(blocked, x2, y2, z2) && !getb(blocked, xm, ym, zm)) {
                    fill.push_back(xm); fill.push_back(ym); fill.push_back(zm);
                }
            }
        }
        for (size_t i = 0; i + 2 < fill.size(); i += 3) mark_blocked(fill[i], fill[i + 1], fill[i + 2]);
        seal_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }

    // id = x*res^2 + y*res + z. x is the high axis; z is the low bits.
    void decode_id(int64_t id, int& x, int& y, int& z) const {
        const int64_t plane = (int64_t)res * res;
        x = (int)(id / plane);
        y = (int)((id / res) % res);
        z = (int)(id % res);
    }

    // Reachability from outside the grid. Empty 4^3 blocks flood on the coarse
    // grid (so the deep exterior is not walked cell by cell). The fine flood
    // stays inside surface blocks, which is what keeps a sealed cavity inside
    // and still reaches a second body sitting in that exterior.
    void flood() {
        const auto t0 = std::chrono::steady_clock::now();
        cn = (res + kCoarse - 1) / kCoarse;
        ckind.assign((size_t)cn * cn * cn, 0);
        std::vector<int> mixed;
        mixed.reserve(blocked_ids.size() / 8 + 16);
        for (int64_t id : blocked_ids) {
            int x, y, z;
            decode_id(id, x, y, z);
            const int ci = coarse_index(x, y, z);
            if (ckind[(size_t)ci] == 0) {
                ckind[(size_t)ci] = 1;
                mixed.push_back(ci);
            }
        }
        auto c_oob = [&](int cx, int cy, int cz) {
            return (unsigned)cx >= (unsigned)cn || (unsigned)cy >= (unsigned)cn || (unsigned)cz >= (unsigned)cn;
        };
        auto c_at = [&](int cx, int cy, int cz) { return (cx * cn + cy) * cn + cz; };
        std::vector<int> cq;
        cq.reserve(1024);
        auto cpush = [&](int cx, int cy, int cz) {
            if (c_oob(cx, cy, cz)) return;
            const int ci = c_at(cx, cy, cz);
            if (ckind[(size_t)ci] != 0) return;
            ckind[(size_t)ci] = 2;
            cq.push_back(ci);
        };
        for (int cy = 0; cy < cn; ++cy) for (int cz = 0; cz < cn; ++cz) {
            cpush(0, cy, cz);
            cpush(cn - 1, cy, cz);
        }
        for (int cx = 0; cx < cn; ++cx) for (int cz = 0; cz < cn; ++cz) {
            cpush(cx, 0, cz);
            cpush(cx, cn - 1, cz);
        }
        for (int cx = 0; cx < cn; ++cx) for (int cy = 0; cy < cn; ++cy) {
            cpush(cx, cy, 0);
            cpush(cx, cy, cn - 1);
        }
        static const int D[6][3] = {{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
        auto c_drain = [&]() {
            while (!cq.empty()) {
                const int ci = cq.back();
                cq.pop_back();
                const int cx = ci / (cn * cn);
                const int cy = (ci / cn) % cn;
                const int cz = ci % cn;
                for (int i = 0; i < 6; ++i) cpush(cx + D[i][0], cy + D[i][1], cz + D[i][2]);
            }
        };
        c_drain();

        auto allow = [&](int x, int y, int z) -> bool {
            return !cell_oob(x, y, z) && coarse_mixed(x, y, z) &&
                   !getb(blocked, x, y, z) && !getb(exterior, x, y, z);
        };
        std::vector<int> st;
        st.reserve(4096);
        auto push = [&](int x, int y, int z) {
            if (!allow(x, y, z)) return;
            st.push_back(x); st.push_back(y); st.push_back(z);
        };
        for (int ci : mixed) {
            const int cx = ci / (cn * cn);
            const int cy = (ci / cn) % cn;
            const int cz = ci % cn;
            const int x0 = cx * kCoarse, y0 = cy * kCoarse, z0 = cz * kCoarse;
            const int x1 = std::min(res, x0 + kCoarse) - 1;
            const int y1 = std::min(res, y0 + kCoarse) - 1;
            const int z1 = std::min(res, z0 + kCoarse) - 1;
            for (int i = 0; i < 6; ++i) {
                const int nx = cx + D[i][0], ny = cy + D[i][1], nz = cz + D[i][2];
                if (c_oob(nx, ny, nz) || ckind[(size_t)c_at(nx, ny, nz)] != 2) continue;
                int fx0 = x0, fy0 = y0, fz0 = z0, fx1 = x1, fy1 = y1, fz1 = z1;
                if (D[i][0] > 0) fx0 = fx1 = x1;
                else if (D[i][0] < 0) fx0 = fx1 = x0;
                if (D[i][1] > 0) fy0 = fy1 = y1;
                else if (D[i][1] < 0) fy0 = fy1 = y0;
                if (D[i][2] > 0) fz0 = fz1 = z1;
                else if (D[i][2] < 0) fz0 = fz1 = z0;
                for (int x = fx0; x <= fx1; ++x)
                    for (int y = fy0; y <= fy1; ++y)
                        for (int z = fz0; z <= fz1; ++z) push(x, y, z);
            }
        }
        // A solid that fills the remesh cube makes every boundary block mixed, so
        // the coarse flood never starts. The outside of the grid is still exterior:
        // seed the empty cells on the six domain faces.
        for (int axis = 0; axis < 3; ++axis) {
            for (int sign = -1; sign <= 1; sign += 2) {
                for (int a = 0; a < res; ++a) for (int b = 0; b < res; ++b) {
                    int c[3];
                    c[axis] = sign < 0 ? 0 : res - 1;
                    c[(axis + 1) % 3] = a;
                    c[(axis + 2) % 3] = b;
                    if (getb(blocked, c[0], c[1], c[2])) continue;
                    if (coarse_mixed(c[0], c[1], c[2])) push(c[0], c[1], c[2]);
                }
            }
        }
        while (!st.empty()) {
            const int z = st.back(); st.pop_back();
            const int y = st.back(); st.pop_back();
            const int x = st.back(); st.pop_back();
            if (!allow(x, y, z)) continue;
            int z0 = z;
            while (z0 > 0 && allow(x, y, z0 - 1)) --z0;
            int z1 = z;
            while (z1 + 1 < res && allow(x, y, z1 + 1)) ++z1;
            for (int zi = z0; zi <= z1; ++zi) setb(exterior, x, y, zi);
            for (int zi = z0; zi <= z1; ++zi) {
                auto step = [&](int nx, int ny, int nz, bool prev_open) {
                    if (prev_open) return;
                    if (cell_oob(nx, ny, nz)) return;
                    if (!getb(blocked, nx, ny, nz)) {
                        const int nci = coarse_index(nx, ny, nz);
                        if (ckind[(size_t)nci] == 0) cpush(nx / kCoarse, ny / kCoarse, nz / kCoarse);
                    }
                    push(nx, ny, nz);
                };
                const bool prev_xm = zi > z0 && x > 0 && allow(x - 1, y, zi - 1);
                const bool prev_xp = zi > z0 && x + 1 < res && allow(x + 1, y, zi - 1);
                const bool prev_ym = zi > z0 && y > 0 && allow(x, y - 1, zi - 1);
                const bool prev_yp = zi > z0 && y + 1 < res && allow(x, y + 1, zi - 1);
                step(x - 1, y, zi, prev_xm);
                step(x + 1, y, zi, prev_xp);
                step(x, y - 1, zi, prev_ym);
                step(x, y + 1, zi, prev_yp);
            }
            // The span already walked ±Z inside this mixed block. A cavity that
            // opens along Z continues into the next block, which is not a mixed
            // neighbour the X/Y steps can see.
            auto touch_z = [&](int nz) {
                if ((unsigned)nz >= (unsigned)res) return;
                if (!getb(blocked, x, y, nz)) {
                    const int nci = coarse_index(x, y, nz);
                    if (ckind[(size_t)nci] == 0) cpush(x / kCoarse, y / kCoarse, nz / kCoarse);
                }
                push(x, y, nz);
            };
            if (z0 > 0) touch_z(z0 - 1);
            if (z1 + 1 < res) touch_z(z1 + 1);
        }
        // Empty blocks touched by the fine flood (a recess entered through a
        // neck thinner than one coarse block) join the exterior.
        c_drain();
        flood_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }

    void build(const float* verts, int nV, const int32_t* faces, int64_t nF, int resolution, float sc) {
        res = resolution;
        scale = sc;
        ready = false;
        seal_s = 0;
        flood_s = 0;
        if (res <= 1 || nF <= 0 || nV <= 0) return;
        amin[0] = amin[1] = amin[2] = res;
        amax[0] = amax[1] = amax[2] = -1;
        for (int i = 0; i < nV; ++i) {
            float g[3];
            to_grid(verts + 3 * i, g);
            for (int k = 0; k < 3; ++k) {
                int c = (int)std::floor(g[k]);
                if (c < 0) c = 0;
                if (c >= res) c = res - 1;
                amin[k] = std::min(amin[k], c);
                amax[k] = std::max(amax[k], c);
            }
        }
        const size_t words = (size_t)((ncells() + 63) / 64);
        blocked.assign(words, 0);
        exterior.assign(words, 0);
        blocked_ids.clear();
        voxelize(verts, faces, nF, nV);
        if (blocked_ids.empty()) {
            blocked.clear();
            exterior.clear();
            return;
        }
        seal_cracks();
        flood();
        ready = true;
    }
};

int find_root(std::vector<int>& par, int x) {
    while (par[x] != x) { par[x] = par[par[x]]; x = par[x]; }
    return x;
}

void unite(std::vector<int>& par, int a, int b) {
    a = find_root(par, a);
    b = find_root(par, b);
    if (a != b) par[a] = b;
}

uint64_t edge_key(int a, int b) {
    if (a > b) std::swap(a, b);
    return ((uint64_t)(uint32_t)a << 32) | (uint32_t)b;
}

// Dual-contour one active set. `vmap` / `fvert` cover every corner of `acoord`
// (the unsigned cell set is a superset of the signed one, so one vertex field
// serves both shells).
Mesh contour_mesh(const std::vector<int>& acoord,
                  const std::unordered_map<uint64_t, int>& vmap,
                  const std::vector<float>& fvert,
                  int res, float scale) {
    const int64_t Na = (int64_t)acoord.size() / 3;
    Mesh built;
    if (Na <= 0) return built;
    std::unordered_map<uint64_t, int> vox;
    vox.reserve((size_t)Na * 2);
    for (int64_t i = 0; i < Na; ++i)
        vox.emplace(key3(acoord[3 * i], acoord[3 * i + 1], acoord[3 * i + 2]), (int)i);
    auto fval = [&](int x, int y, int z) -> float {
        auto it = vmap.find(key3(x, y, z));
        return it == vmap.end() ? 1e9f : fvert[(size_t)it->second];
    };
    std::vector<float> dual((size_t)Na * 3);
    std::vector<int8_t> owned((size_t)Na * 3, 0);
    parallel_for(Na, [&](int64_t b, int64_t e) {
        for (int64_t i = b; i < e; ++i) {
            const int vx = acoord[3 * i], vy = acoord[3 * i + 1], vz = acoord[3 * i + 2];
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
                    if (u == 1 && v == 1) owned[3 * i + axis] = c12 ? 1 : (c21 ? -1 : 0);
                }
            if (cnt) for (int k = 0; k < 3; ++k) dual[3 * i + k] = (float)(sum[k] / cnt);
            else { dual[3 * i] = vx + 0.5f; dual[3 * i + 1] = vy + 0.5f; dual[3 * i + 2] = vz + 0.5f; }
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
        const int vx = acoord[3 * i], vy = acoord[3 * i + 1], vz = acoord[3 * i + 2];
        for (int axis = 0; axis < 3; ++axis) {
            const int dir = owned[3 * i + axis];
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
    std::vector<int32_t> remap((size_t)Na, -1);
    for (int64_t i = 0; i < Na; ++i)
        if (used[i]) {
            remap[i] = built.V();
            built.verts.push_back((dual[3 * i]     / res - 0.5f) * scale);
            built.verts.push_back((dual[3 * i + 1] / res - 0.5f) * scale);
            built.verts.push_back((dual[3 * i + 2] / res - 0.5f) * scale);
        }
    built.faces.resize(qfaces.size());
    for (size_t k = 0; k < qfaces.size(); ++k) built.faces[k] = remap[qfaces[k]];
    return built;
}

// Vertex-connected components. face_comp[f] and vert_comp[v] are dense ids, or -1.
int component_ids(const Mesh& m, std::vector<int>& face_comp, std::vector<int>& vert_comp) {
    const int V = m.V(), F = m.F();
    face_comp.assign((size_t)std::max(F, 0), -1);
    vert_comp.assign((size_t)std::max(V, 0), -1);
    if (V <= 0 || F <= 0) return 0;
    std::vector<int> par((size_t)V);
    for (int i = 0; i < V; ++i) par[i] = i;
    for (int f = 0; f < F; ++f) {
        const int a = m.faces[3 * f], b = m.faces[3 * f + 1], c = m.faces[3 * f + 2];
        if (a < 0 || b < 0 || c < 0 || a >= V || b >= V || c >= V) continue;
        unite(par, a, b);
        unite(par, b, c);
    }
    std::unordered_map<int, int> cid;
    cid.reserve(64);
    for (int f = 0; f < F; ++f) {
        const int a = m.faces[3 * f], b = m.faces[3 * f + 1], c = m.faces[3 * f + 2];
        if (a < 0 || a >= V) continue;
        const int r = find_root(par, a);
        auto it = cid.find(r);
        if (it == cid.end()) it = cid.emplace(r, (int)cid.size()).first;
        face_comp[f] = it->second;
        vert_comp[a] = it->second;
        if (b >= 0 && b < V) vert_comp[b] = it->second;
        if (c >= 0 && c < V) vert_comp[c] = it->second;
    }
    return (int)cid.size();
}

int real_piece_count(const Mesh& m) {
    std::vector<int> face_comp, vert_comp;
    const int nc = component_ids(m, face_comp, vert_comp);
    if (nc <= 0) return 0;
    std::vector<int> cnt((size_t)nc, 0);
    for (int c : face_comp) if (c >= 0) cnt[c]++;
    int largest = 0;
    for (int c : cnt) largest = std::max(largest, c);
    int real = 0;
    for (int c : cnt) if (largest > 0 && (int64_t)c * 50 >= largest) ++real;
    return real;
}

void count_boundary_per_comp(const Mesh& m, const std::vector<int>& face_comp, int nc, std::vector<int>& boundary) {
    boundary.assign((size_t)std::max(nc, 0), 0);
    if (nc <= 0 || m.F() <= 0) return;
    struct EK { uint64_t k; int c; };
    std::vector<EK> edges;
    edges.reserve((size_t)m.F() * 3);
    const int V = m.V();
    for (int f = 0; f < m.F(); ++f) {
        const int c = f < (int)face_comp.size() ? face_comp[f] : -1;
        if (c < 0 || c >= nc) continue;
        for (int j = 0; j < 3; ++j) {
            const int a = m.faces[3 * f + j], b = m.faces[3 * f + (j + 1) % 3];
            if (a < 0 || b < 0 || a >= V || b >= V) continue;
            edges.push_back({edge_key(a, b), c});
        }
    }
    std::sort(edges.begin(), edges.end(), [](const EK& p, const EK& q) {
        if (p.k != q.k) return p.k < q.k;
        return p.c < q.c;
    });
    for (size_t i = 0; i < edges.size();) {
        size_t j = i + 1;
        while (j < edges.size() && edges[j].k == edges[i].k && edges[j].c == edges[i].c) ++j;
        if (j - i == 1) boundary[edges[i].c]++;
        i = j;
    }
}

// Grid hash of mesh vertices. Lookups stay inside a few cells of the query,
// which is the 1.5·eps acceptance neighbourhood — not a full BVH pass.
struct NearGrid {
    int res = 0;
    float scale = 1.f;
    int rad = 2;
    std::vector<uint64_t> bits;
    std::unordered_map<uint64_t, std::vector<int>> cells;

    void reset(int resolution, float sc, int radius) {
        res = std::max(resolution, 1);
        scale = sc;
        rad = std::max(radius, 1);
        const int64_t nbits = (int64_t)res * res * res;
        bits.assign((size_t)((nbits + 63) / 64), 0);
        cells.clear();
    }
    void cell_of(const float p[3], int c[3]) const {
        for (int k = 0; k < 3; ++k) {
            const float g = (p[k] / scale + 0.5f) * (float)res;
            int i = (int)std::floor(g);
            if (i < 0) i = 0;
            if (i >= res) i = res - 1;
            c[k] = i;
        }
    }
    void add(const float p[3], int idx) {
        int c[3];
        cell_of(p, c);
        const int64_t i = ((int64_t)c[0] * res + c[1]) * res + c[2];
        bits[(size_t)(i >> 6)] |= 1ull << (unsigned)(i & 63);
        cells[key3(c[0], c[1], c[2])].push_back(idx);
    }
    bool occupied(int x, int y, int z) const {
        if ((unsigned)x >= (unsigned)res || (unsigned)y >= (unsigned)res || (unsigned)z >= (unsigned)res) return false;
        const int64_t i = ((int64_t)x * res + y) * res + z;
        return ((bits[(size_t)(i >> 6)] >> (unsigned)(i & 63)) & 1ull) != 0;
    }
    int nearest(const float p[3], const std::vector<float>& verts, float& dist, float stop) const {
        int c[3];
        cell_of(p, c);
        float best2 = 1e30f;
        int best = -1;
        const float stop2 = stop * stop;
        for (int r = 0; r <= rad && best2 > stop2; ++r) {
            for (int dz = -r; dz <= r; ++dz)
                for (int dy = -r; dy <= r; ++dy)
                    for (int dx = -r; dx <= r; ++dx) {
                        const int adx = dx < 0 ? -dx : dx, ady = dy < 0 ? -dy : dy, adz = dz < 0 ? -dz : dz;
                        if (r > 0 && std::max(adx, std::max(ady, adz)) != r) continue;
                        const int x = c[0] + dx, y = c[1] + dy, z = c[2] + dz;
                        if (!occupied(x, y, z)) continue;
                        auto it = cells.find(key3(x, y, z));
                        if (it == cells.end()) continue;
                        for (int vi : it->second) {
                            const float* q = &verts[(size_t)3 * vi];
                            const float d0 = q[0] - p[0], d1 = q[1] - p[1], d2 = q[2] - p[2];
                            const float d2s = d0 * d0 + d1 * d1 + d2 * d2;
                            if (d2s < best2) { best2 = d2s; best = vi; }
                        }
                    }
        }
        dist = best < 0 ? 1e30f : std::sqrt(best2);
        return best;
    }
};

void append_faces(Mesh& dst, const Mesh& src, const std::vector<int>& face_ids) {
    std::unordered_map<int, int> remap;
    remap.reserve(face_ids.size() * 2 + 8);
    for (int f : face_ids) {
        if (f < 0 || f >= src.F()) continue;
        for (int k = 0; k < 3; ++k) {
            const int v = src.faces[3 * f + k];
            if (v < 0 || v >= src.V()) { dst.faces.push_back(0); continue; }
            int nv;
            auto it = remap.find(v);
            if (it == remap.end()) {
                nv = dst.V();
                remap.emplace(v, nv);
                dst.verts.insert(dst.verts.end(), &src.verts[(size_t)3 * v], &src.verts[(size_t)3 * v] + 3);
            } else nv = it->second;
            dst.faces.push_back(nv);
        }
    }
}

int count_open_edges(const std::vector<int32_t>& faces) {
    struct EK { uint64_t k; int n; };
    std::vector<uint64_t> keys;
    keys.reserve(faces.size());
    for (size_t i = 0; i + 2 < faces.size(); i += 3) {
        for (int j = 0; j < 3; ++j) {
            int a = faces[i + j], b = faces[i + (j + 1) % 3];
            if (a > b) std::swap(a, b);
            keys.push_back(((uint64_t)(uint32_t)a << 32) | (uint32_t)b);
        }
    }
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

// Fan-fill every clean boundary loop. Pinch contacts between the inner and outer
// walls leave short seams; a loop that is not a simple cycle is left open so the
// component can fall back instead of being papered over.
struct RepairStats { int loops = 0; int edges_before = 0; int edges_after = 0; };

RepairStats close_boundary_loops(std::vector<int32_t>& faces) {
    RepairStats st;
    st.edges_before = count_open_edges(faces);
    if (st.edges_before == 0) return st;
    const int max_loop = (int)faces.size();
    auto ekey = [](int a, int b) { return ((uint64_t)(uint32_t)a << 32) | (uint32_t)b; };
    auto fill_loops = [&](bool directed) {
        const size_t F = faces.size() / 3;
        std::unordered_map<uint64_t, int> dir;
        dir.reserve(F * 3);
        for (size_t f = 0; f < F; ++f)
            for (int j = 0; j < 3; ++j)
                dir[ekey(faces[3 * f + j], faces[3 * f + (j + 1) % 3])]++;
        if (directed) {
            std::unordered_map<int, int> nxt, outd, ind;
            for (const auto& kv : dir) {
                const int a = (int)(kv.first >> 32), b = (int)(uint32_t)kv.first;
                if (kv.second == 1 && dir.find(ekey(b, a)) == dir.end()) {
                    nxt[b] = a; outd[b]++; ind[a]++;
                }
            }
            std::unordered_map<int, char> used;
            for (const auto& kv : nxt) {
                const int start = kv.first;
                if (used[start] || outd[start] != 1 || ind[start] != 1) continue;
                std::vector<int> loop = {start};
                int cur = start;
                bool cycle = false, clean = true;
                for (int steps = 0; steps <= max_loop; ++steps) {
                    auto it = nxt.find(cur);
                    if (it == nxt.end() || used[cur]) { clean = false; break; }
                    cur = it->second;
                    if (outd[cur] != 1 || ind[cur] != 1) { clean = false; break; }
                    if (cur == start) { cycle = true; break; }
                    loop.push_back(cur);
                }
                for (int v : loop) used[v] = 1;
                if (!clean || !cycle || loop.size() < 3) continue;
                for (size_t i = 1; i + 1 < loop.size(); ++i) {
                    faces.push_back(loop[0]); faces.push_back(loop[i]); faces.push_back(loop[i + 1]);
                }
                ++st.loops;
            }
        } else {
            std::unordered_map<int, std::vector<int>> adj;
            std::unordered_map<uint64_t, int> und;
            for (size_t f = 0; f < faces.size() / 3; ++f)
                for (int j = 0; j < 3; ++j) {
                    int a = faces[3 * f + j], b = faces[3 * f + (j + 1) % 3];
                    if (a > b) std::swap(a, b);
                    und[ekey(a, b)]++;
                }
            for (const auto& kv : und) {
                if (kv.second != 1) continue;
                const int a = (int)(kv.first >> 32), b = (int)(uint32_t)kv.first;
                adj[a].push_back(b); adj[b].push_back(a);
            }
            std::unordered_map<int, char> used;
            for (const auto& kv : adj) {
                const int start = kv.first;
                if (used[start] || kv.second.size() != 2) continue;
                std::vector<int> loop = {start};
                int prev = start, cur = kv.second[0];
                bool cycle = false, clean = true;
                for (int steps = 0; steps <= max_loop; ++steps) {
                    auto it = adj.find(cur);
                    if (it == adj.end() || it->second.size() != 2 || used[cur]) { clean = false; break; }
                    if (cur == start) { cycle = true; break; }
                    loop.push_back(cur);
                    const int nx = it->second[0] == prev ? it->second[1] : it->second[0];
                    prev = cur; cur = nx;
                }
                for (int v : loop) used[v] = 1;
                if (!clean || !cycle || loop.size() < 3) continue;
                for (size_t i = 1; i + 1 < loop.size(); ++i) {
                    faces.push_back(loop[0]); faces.push_back(loop[i]); faces.push_back(loop[i + 1]);
                }
                ++st.loops;
            }
        }
    };
    fill_loops(true);
    fill_loops(false);
    st.edges_after = count_open_edges(faces);
    return st;
}

enum ShellFail { kKeep = 0, kCoverage = 1, kOpen = 2, kPieces = 3, kDropInterior = 4, kFloater = 5 };

// Signed shell S replaces the unsigned double shell U only where S still covers
// every outer vertex of U (within 1.5·eps), adds no open edges, and keeps that
// component as one real piece. Anything else keeps U's double shell for that
// component. A component with no outer vertex is the inner wall and is dropped.
Mesh choose_shell(Mesh U, Mesh S, const ExteriorGrid& grid, float eps, int band,
                  int& faces_before, int& removed, const char*& note) {
    faces_before = U.F();
    removed = 0;
    note = " (unsigned fallback)";
    if (U.F() <= 0) return U;
    if (S.F() <= 0 || !grid.ready) {
        printf("  remesh postproc: outer shell fallback component faces=%d reason=empty-signed\n", U.F());
        fflush(stdout);
        return U;
    }
    // Pinch seams are boundary loops where the inner and outer walls touched.
    // Close those locally, then keep the component only if nothing is left open.
    const RepairStats rep = close_boundary_loops(S.faces);
    if (rep.edges_before > 0) {
        printf("  remesh postproc: outer shell repair loops=%d edges %d -> %d\n",
               rep.loops, rep.edges_before, rep.edges_after);
        fflush(stdout);
    }
    std::vector<int> u_face, u_vert;
    const int nu = component_ids(U, u_face, u_vert);
    if (nu <= 0) return U;
    std::vector<int> u_boundary, u_faces((size_t)nu, 0);
    count_boundary_per_comp(U, u_face, nu, u_boundary);
    for (int f = 0; f < U.F(); ++f) if (u_face[f] >= 0) u_faces[u_face[f]]++;

    std::vector<char> outer((size_t)U.V(), 0);
    for (int i = 0; i < U.V(); ++i) {
        if (u_vert[i] < 0) continue;
        const float p[3] = {U.verts[3 * i], U.verts[3 * i + 1], U.verts[3 * i + 2]};
        if (grid.is_outside(p)) outer[i] = 1;
    }

    const int rad = (int)std::ceil(1.5f * (float)std::max(band, 1)) + 1;
    const float gap_lim = 1.5f * eps;
    NearGrid occ;
    occ.reset(grid.res, grid.scale, rad);
    for (int i = 0; i < S.V(); ++i) occ.add(&S.verts[3 * i], i);

    std::vector<int> why((size_t)nu, kKeep);
    std::vector<int> misses((size_t)nu, 0), checked((size_t)nu, 0);
    std::vector<float> max_gap((size_t)nu, 0.f);
    for (int i = 0; i < U.V(); ++i) {
        if (!outer[i]) continue;
        const int c = u_vert[i];
        if (c < 0 || c >= nu) continue;
        ++checked[c];
        float d = 0.f;
        occ.nearest(&U.verts[3 * i], S.verts, d, gap_lim);
        max_gap[c] = std::max(max_gap[c], d);
        if (d > gap_lim) ++misses[c];
    }
    occ.bits.clear();
    occ.cells.clear();
    for (int c = 0; c < nu; ++c) {
        if (checked[c] == 0) { why[c] = kDropInterior; continue; }
        // A handful of vertices past 1.5·eps (a few ppm, and at least 4) is
        // numerical noise. A real missing region is far more than that.
        const int64_t allow = std::max<int64_t>(4, (int64_t)checked[c] * 5 / 1000000);
        if ((int64_t)misses[c] > allow) why[c] = kCoverage;
        else if (misses[c] > 0)
            printf("  remesh postproc: outer shell coverage kept faces=%d misses=%d/%d max_gap=%.4g eps=%.4g\n",
                   u_faces[c], misses[c], checked[c], max_gap[c], eps);
    }

    NearGrid outer_occ;
    outer_occ.reset(grid.res, grid.scale, rad);
    for (int i = 0; i < U.V(); ++i)
        if (outer[i]) outer_occ.add(&U.verts[3 * i], i);
    std::vector<int> s_asg((size_t)S.F(), -1);
    for (int f = 0; f < S.F(); ++f) {
        const int ia = S.faces[3 * f], ib = S.faces[3 * f + 1], ic = S.faces[3 * f + 2];
        if (ia < 0 || ib < 0 || ic < 0 || ia >= S.V() || ib >= S.V() || ic >= S.V()) continue;
        const float q[3] = {
            (S.verts[3 * ia] + S.verts[3 * ib] + S.verts[3 * ic]) / 3.f,
            (S.verts[3 * ia + 1] + S.verts[3 * ib + 1] + S.verts[3 * ic + 1]) / 3.f,
            (S.verts[3 * ia + 2] + S.verts[3 * ib + 2] + S.verts[3 * ic + 2]) / 3.f,
        };
        float d = 0.f;
        const int vi = outer_occ.nearest(q, U.verts, d, gap_lim);
        if (vi >= 0 && d <= gap_lim && u_vert[vi] >= 0) s_asg[f] = u_vert[vi];
    }
    // Repair fans and the seam faces they border sit off the outer sheet, so a
    // 1.5·eps centroid test leaves them unassigned. Their edges then count as
    // open and assembly drops the patch. Grow the assignment across shared edges.
    {
        std::unordered_map<uint64_t, std::vector<int>> edge_faces;
        edge_faces.reserve((size_t)S.F() * 2);
        for (int f = 0; f < S.F(); ++f) {
            for (int j = 0; j < 3; ++j) {
                int a = S.faces[3 * f + j], b = S.faces[3 * f + (j + 1) % 3];
                if (a > b) std::swap(a, b);
                edge_faces[edge_key(a, b)].push_back(f);
            }
        }
        std::vector<int> grow;
        grow.reserve((size_t)S.F());
        for (int f = 0; f < S.F(); ++f) if (s_asg[f] >= 0) grow.push_back(f);
        for (size_t qi = 0; qi < grow.size(); ++qi) {
            const int f = grow[qi];
            const int c = s_asg[f];
            if (c < 0) continue;
            for (int j = 0; j < 3; ++j) {
                int a = S.faces[3 * f + j], b = S.faces[3 * f + (j + 1) % 3];
                if (a > b) std::swap(a, b);
                auto it = edge_faces.find(edge_key(a, b));
                if (it == edge_faces.end()) continue;
                for (int o : it->second) {
                    if (s_asg[o] >= 0) continue;
                    s_asg[o] = c;
                    grow.push_back(o);
                }
            }
        }
    }

    std::vector<int> s_boundary;
    count_boundary_per_comp(S, s_asg, nu, s_boundary);

    std::vector<int> par((size_t)std::max(S.V(), 0));
    for (int i = 0; i < S.V(); ++i) par[i] = i;
    for (int f = 0; f < S.F(); ++f) {
        if (s_asg[f] < 0) continue;
        const int a = S.faces[3 * f], b = S.faces[3 * f + 1], c = S.faces[3 * f + 2];
        if (a < 0 || b < 0 || c < 0 || a >= S.V() || b >= S.V() || c >= S.V()) continue;
        unite(par, a, b);
        unite(par, b, c);
    }
    std::vector<int> pieces((size_t)nu, 0), s_faces((size_t)nu, 0);
    std::vector<float> cmin((size_t)nu * 3, 1e30f), cmax((size_t)nu * 3, -1e30f);
    for (int i = 0; i < U.V(); ++i) {
        const int c = u_vert[i];
        if (c < 0 || c >= nu) continue;
        for (int k = 0; k < 3; ++k) {
            cmin[(size_t)c * 3 + k] = std::min(cmin[(size_t)c * 3 + k], U.verts[3 * i + k]);
            cmax[(size_t)c * 3 + k] = std::max(cmax[(size_t)c * 3 + k], U.verts[3 * i + k]);
        }
    }
    for (int c = 0; c < nu; ++c) {
        if (why[c] == kDropInterior) continue;
        if (why[c] == kKeep && s_boundary[c] > u_boundary[c]) why[c] = kOpen;
        std::unordered_map<int, int> roots;
        int assigned = 0;
        for (int f = 0; f < S.F(); ++f) {
            if (s_asg[f] != c) continue;
            ++assigned;
            const int a = S.faces[3 * f];
            if (a < 0 || a >= S.V()) continue;
            roots[find_root(par, a)]++;
        }
        s_faces[c] = assigned;
        int significant = 0;
        const int thresh = std::max(1, assigned / 50);
        for (const auto& kv : roots) if (kv.second >= thresh) ++significant;
        pieces[c] = significant;
        if (why[c] == kKeep && significant != 1) why[c] = kPieces;
    }

    // Same cuts as drop_pre_decimation_components, applied to the shell we are
    // about to emit. Single-shelling must not push a piece that the double shell
    // would have kept under the 2% cut, or under the 3% cut when it is detached.
    {
        const float voxel = grid.scale / (float)std::max(grid.res, 1);
        auto box_gap = [&](int c, int body) -> float {
            float acc = 0.f;
            for (int k = 0; k < 3; ++k) {
                const float lo = cmin[(size_t)c * 3 + k], hi = cmax[(size_t)c * 3 + k];
                const float blo = cmin[(size_t)body * 3 + k], bhi = cmax[(size_t)body * 3 + k];
                float d = 0.f;
                if (hi < blo) d = blo - hi;
                else if (lo > bhi) d = lo - bhi;
                acc += d * d;
            }
            return std::sqrt(acc);
        };
        // Cuts match drop_pre_decimation_components against the unsigned body.
        // Using the already single-shelled face count shrinks the denominator
        // and lets a remnant under 2% of the real body clear the cut.
        int body_faces = 0, body = 0;
        for (int c = 0; c < nu; ++c) {
            if (why[c] == kDropInterior) continue;
            if (u_faces[c] > body_faces) { body_faces = u_faces[c]; body = c; }
        }
        const int cut2 = (int)(0.02 * body_faces);
        const int cut3 = (int)(0.03 * body_faces);
        for (int c = 0; c < nu; ++c) {
            if (why[c] != kKeep || c == body || body_faces <= 0) continue;
            const int sf = s_faces[c], uf = u_faces[c];
            if (uf < cut2) continue;
            const bool lose2 = sf < cut2;
            const bool lose3 = sf < cut3 && uf >= cut3 && box_gap(c, body) > 3.f * voxel;
            if (lose2 || lose3) why[c] = kFloater;
        }
    }

    bool any_keep = false, any_fallback = false;
    int expect_largest = 0;
    for (int c = 0; c < nu; ++c) {
        if (why[c] == kDropInterior) {
            printf("  remesh postproc: outer shell dropped interior component faces=%d\n", u_faces[c]);
            continue;
        }
        const int out_faces = why[c] == kKeep ? s_faces[c] : u_faces[c];
        expect_largest = std::max(expect_largest, out_faces);
        if (why[c] == kKeep) { any_keep = true; continue; }
        any_fallback = true;
        if (why[c] == kFloater) {
            printf("  remesh postproc: outer shell keep double shell component faces=%d reason=floater-share signed=%d unsigned=%d\n",
                   u_faces[c], s_faces[c], u_faces[c]);
        } else if (why[c] == kCoverage) {
            printf("  remesh postproc: outer shell fallback component faces=%d reason=coverage misses=%d/%d max_gap=%.4g eps=%.4g\n",
                   u_faces[c], misses[c], checked[c], max_gap[c], eps);
        } else if (why[c] == kOpen) {
            printf("  remesh postproc: outer shell fallback component faces=%d reason=open-edges %d (unsigned %d) after repair loops=%d edges %d -> %d\n",
                   u_faces[c], s_boundary[c], u_boundary[c], rep.loops, rep.edges_before, rep.edges_after);
        } else {
            printf("  remesh postproc: outer shell fallback component faces=%d reason=pieces signed=%d\n",
                   u_faces[c], pieces[c]);
        }
    }
    fflush(stdout);

    auto expect_real = [&]() {
        int n = 0;
        for (int c = 0; c < nu; ++c) {
            if (why[c] == kDropInterior) continue;
            const int out_faces = why[c] == kKeep ? s_faces[c] : u_faces[c];
            if (expect_largest > 0 && (int64_t)out_faces * 50 >= expect_largest) ++n;
        }
        return n;
    };

    if (!any_keep) {
        note = " (unsigned fallback)";
        removed = 0;
        return U;
    }

    Mesh mixed;
    for (int c = 0; c < nu; ++c) {
        if (why[c] == kDropInterior) continue;
        std::vector<int> ids;
        if (why[c] != kKeep) {
            ids.reserve((size_t)u_faces[c]);
            for (int f = 0; f < U.F(); ++f) if (u_face[f] == c) ids.push_back(f);
            append_faces(mixed, U, ids);
        } else {
            for (int f = 0; f < S.F(); ++f) if (s_asg[f] == c) ids.push_back(f);
            append_faces(mixed, S, ids);
        }
    }
    const int got = real_piece_count(mixed);
    const int want = expect_real();
    if (got < want) {
        printf("  remesh postproc: outer shell fallback component faces=%d reason=pieces signed=%d unsigned=%d\n",
               U.F(), got, want);
        fflush(stdout);
        note = " (unsigned fallback)";
        removed = 0;
        return U;
    }
    note = any_fallback ? " (partial fallback)" : "";
    removed = U.F() - mixed.F();
    if (removed < 0) removed = 0;
    return mixed;
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

    // Candidate indices, then drop the res³ bitset before the shell tables.
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

    // One closest-point pass classifies the unsigned tube and the outer sheet.
    // Both shells are contoured from that field. Faces are never deleted: the
    // signed shell replaces the double shell only where it still covers it.
    std::vector<uint8_t> act_u(cand_cells.size(), 0);
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
        }
    });
    std::vector<int> acoord_u;
    for (size_t i = 0; i < cand_cells.size(); ++i) {
        if (!act_u[i]) continue;
        const int64_t c = cand_cells[i];
        acoord_u.push_back((int)(c / ((int64_t)res * res)));
        acoord_u.push_back((int)((c / res) % res));
        acoord_u.push_back((int)(c % res));
    }
    cand_cells.clear(); cand_cells.shrink_to_fit();
    const int64_t Na = (int64_t)acoord_u.size() / 3;
    if (Na < 100) {
        fprintf(stderr, "  remesh: only %lld active voxels; skipping remesh\n", (long long)Na);
        return out;
    }

    std::vector<int> vcoord;
    std::unordered_map<uint64_t, int> vmap;
    vmap.reserve((size_t)Na * 3);
    for (int64_t i = 0; i < Na; ++i)
        for (int dx = 0; dx < 2; ++dx) for (int dy = 0; dy < 2; ++dy) for (int dz = 0; dz < 2; ++dz) {
            const int x = acoord_u[3 * i] + dx, y = acoord_u[3 * i + 1] + dy, z = acoord_u[3 * i + 2] + dz;
            if (vmap.emplace(key3(x, y, z), (int)(vcoord.size() / 3)).second) {
                vcoord.push_back(x); vcoord.push_back(y); vcoord.push_back(z);
            }
        }
    const int64_t Nv = (int64_t)vcoord.size() / 3;
    std::vector<float> fu((size_t)Nv), fs((size_t)Nv);
    parallel_for(Nv, [&](int64_t b, int64_t e) {
        for (int64_t i = b; i < e; ++i) {
            const float p[3] = { ((float)vcoord[3 * i]     / res - 0.5f) * scale,
                                 ((float)vcoord[3 * i + 1] / res - 0.5f) * scale,
                                 ((float)vcoord[3 * i + 2] / res - 0.5f) * scale };
            const TriBvh::Hit h = bvh.closest(p, eps + 2 * cell);
            const float udf = h.face >= 0 ? std::sqrt(h.dist2) : eps + 2 * cell;
            fu[i] = udf - eps;
            float sdf = udf;
            if (grid.ready && h.face >= 0 && !grid.is_outside(p)) sdf = -udf;
            fs[i] = sdf - eps;
        }
    });
    vcoord.clear(); vcoord.shrink_to_fit();

    Mesh unsigned_shell = contour_mesh(acoord_u, vmap, fu, res, scale);
    if (unsigned_shell.F() == 0) return out;
    int faces_before = unsigned_shell.F();
    int removed_faces = 0;
    const char* note = " (unsigned fallback)";
    const int64_t active_voxels = Na;
    if (grid.ready) {
        // Same cells as the unsigned tube. The signed field is negative on both
        // sides of the inner wall, so those cells emit nothing; sign changes only
        // on the outer sheet. Filtering by the cell center drops outer cells
        // whose center fell inside a surface voxel and opens the shell.
        Mesh signed_shell = contour_mesh(acoord_u, vmap, fs, res, scale);
        out = choose_shell(std::move(unsigned_shell), std::move(signed_shell), grid, eps, band,
                           faces_before, removed_faces, note);
    } else {
        out = std::move(unsigned_shell);
        removed_faces = 0;
        note = " (unsigned fallback)";
    }
    const double shell_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_shell).count();
    printf("  remesh postproc: outer shell faces %d -> %d (removed %d inner-wall faces) in %.3fs (flood %.3fs, seal %.3fs)%s\n",
           faces_before, out.F(), removed_faces, shell_s, grid.flood_s, grid.seal_s, note);
    fflush(stdout);
    if (out.F() == 0) return out;

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
