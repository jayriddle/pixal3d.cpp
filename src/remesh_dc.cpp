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

inline uint64_t ekey(int a, int b) {
    const uint32_t lo = (uint32_t)std::min(a, b);
    const uint32_t hi = (uint32_t)std::max(a, b);
    return ((uint64_t)lo << 32) | hi;
}

inline void cross3(const float* a, const float* b, float* o) {
    o[0] = a[1]*b[2] - a[2]*b[1];
    o[1] = a[2]*b[0] - a[0]*b[2];
    o[2] = a[0]*b[1] - a[1]*b[0];
}

inline float dot3(const float* a, const float* b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }

inline float len3(const float* a) { return std::sqrt(dot3(a, a)); }

// Outward angle-weighted pseudonormals for the input mesh. `signed_distance`
// is +UDF outside the solid and −UDF inside, which turns the narrow-band
// isosurface f = dist − eps into the outer offset only. A component we cannot
// orient (non-orientable loop) is left unsigned so the caller keeps both
// sides of that patch instead of deleting it.
struct OutwardField {
    const float* verts = nullptr;
    const int32_t* faces = nullptr;
    int nV = 0, nF = 0;
    bool apply_sign = true;
    int trusted_faces = 0;
    std::vector<uint8_t> trust;     // per face
    std::vector<float> face_n;      // F*3 unit outward
    std::vector<float> vert_n;      // V*3 unit, 0 if unused
    std::vector<uint64_t> edge_key; // sorted undirected edge
    std::vector<int32_t> edge_f0, edge_f1;

    void clear() {
        trust.clear(); face_n.clear(); vert_n.clear();
        edge_key.clear(); edge_f0.clear(); edge_f1.clear();
        trust.shrink_to_fit(); face_n.shrink_to_fit(); vert_n.shrink_to_fit();
        edge_key.shrink_to_fit(); edge_f0.shrink_to_fit(); edge_f1.shrink_to_fit();
        trusted_faces = 0;
    }

    bool face_ok(int f) const {
        if (f < 0 || f >= nF) return false;
        const int a = faces[3*f], b = faces[3*f+1], c = faces[3*f+2];
        return a >= 0 && b >= 0 && c >= 0 && a < nV && b < nV && c < nV && a != b && b != c && a != c;
    }

    int find_edge(uint64_t k) const {
        const auto it = std::lower_bound(edge_key.begin(), edge_key.end(), k);
        if (it == edge_key.end() || *it != k) return -1;
        return (int)(it - edge_key.begin());
    }

    // Unit normal of the closest feature (face, edge, or vertex). False when
    // the hit face is untrusted or degenerate.
    bool feature_normal(int face, const float q[3], float n[3]) const {
        if (!face_ok(face) || !trust[face]) return false;
        const int ia = faces[3*face], ib = faces[3*face+1], ic = faces[3*face+2];
        const float* a = verts + 3*ia;
        const float* b = verts + 3*ib;
        const float* c = verts + 3*ic;
        float ab[3], ac[3], aq[3];
        for (int k = 0; k < 3; ++k) { ab[k] = b[k]-a[k]; ac[k] = c[k]-a[k]; aq[k] = q[k]-a[k]; }
        const float d00 = dot3(ab, ab), d01 = dot3(ab, ac), d11 = dot3(ac, ac);
        const float d20 = dot3(aq, ab), d21 = dot3(aq, ac);
        const float denom = d00*d11 - d01*d01;
        float u = 1.f, v = 0.f, w = 0.f; // u at a, v at b, w at c
        if (std::fabs(denom) > 1e-20f) {
            v = (d11*d20 - d01*d21) / denom;
            w = (d00*d21 - d01*d20) / denom;
            u = 1.f - v - w;
        }
        const float lim = 2e-4f;
        auto use_face = [&]() {
            n[0] = face_n[3*face]; n[1] = face_n[3*face+1]; n[2] = face_n[3*face+2];
            return len3(n) > 0.5f;
        };
        auto use_vert = [&](int vi) {
            n[0] = vert_n[3*vi]; n[1] = vert_n[3*vi+1]; n[2] = vert_n[3*vi+2];
            if (len3(n) > 0.5f) return true;
            return use_face();
        };
        auto use_edge = [&](int i0, int i1) {
            const int e = find_edge(ekey(i0, i1));
            if (e < 0) return use_face();
            float acc[3] = {0, 0, 0};
            int cnt = 0;
            const int fs[2] = {edge_f0[e], edge_f1[e]};
            for (int t = 0; t < 2; ++t) {
                const int g = fs[t];
                if (g < 0 || !trust[g]) continue;
                acc[0] += face_n[3*g]; acc[1] += face_n[3*g+1]; acc[2] += face_n[3*g+2];
                ++cnt;
            }
            if (!cnt) return use_face();
            const float L = len3(acc);
            if (!(L > 1e-12f)) return use_face();
            n[0] = acc[0]/L; n[1] = acc[1]/L; n[2] = acc[2]/L;
            return true;
        };
        if (u <= lim && v <= lim) return use_vert(ic);
        if (u <= lim && w <= lim) return use_vert(ib);
        if (v <= lim && w <= lim) return use_vert(ia);
        if (u <= lim) return use_edge(ib, ic);
        if (v <= lim) return use_edge(ia, ic);
        if (w <= lim) return use_edge(ia, ib);
        return use_face();
    }

    float signed_distance(const float p[3], const TriBvh::Hit& h, float udf) const {
        if (!apply_sign || h.face < 0) return udf;
        float n[3];
        if (!feature_normal(h.face, h.point, n)) return udf;
        const float s = (p[0]-h.point[0])*n[0] + (p[1]-h.point[1])*n[1] + (p[2]-h.point[2])*n[2];
        return s < 0.f ? -udf : udf;
    }

    void build(const float* v, int V, const int32_t* f, int F, const TriBvh& bvh, float eps) {
        verts = v; faces = f; nV = V; nF = F;
        trust.assign((size_t)F, 0);
        face_n.assign((size_t)F * 3, 0.f);
        vert_n.assign((size_t)V * 3, 0.f);
        trusted_faces = 0;
        apply_sign = true;
        if (V <= 0 || F <= 0) return;

        struct Rec { uint64_t key; int face, a, b; };
        std::vector<Rec> recs;
        recs.reserve((size_t)F * 3);
        for (int fi = 0; fi < F; ++fi) {
            if (!face_ok(fi)) continue;
            const int ix[3] = {f[3*fi], f[3*fi+1], f[3*fi+2]};
            for (int k = 0; k < 3; ++k) {
                const int a = ix[k], b = ix[(k+1)%3];
                recs.push_back({ekey(a, b), fi, a, b});
            }
        }
        std::sort(recs.begin(), recs.end(), [](const Rec& p, const Rec& q) {
            if (p.key != q.key) return p.key < q.key;
            return p.face < q.face;
        });

        std::vector<int> deg((size_t)F, 0);
        for (size_t i = 0; i < recs.size();) {
            size_t j = i + 1;
            while (j < recs.size() && recs[j].key == recs[i].key) ++j;
            if (j - i == 2 && recs[i].face != recs[i+1].face) {
                deg[recs[i].face]++; deg[recs[i+1].face]++;
            }
            i = j;
        }
        std::vector<int> off((size_t)F + 1, 0);
        for (int fi = 0; fi < F; ++fi) off[fi+1] = off[fi] + deg[fi];
        std::vector<int> adj_nb(off.back());
        std::vector<uint8_t> adj_same(off.back());
        std::vector<int> cursor = off;
        for (size_t i = 0; i < recs.size();) {
            size_t j = i + 1;
            while (j < recs.size() && recs[j].key == recs[i].key) ++j;
            if (j - i == 2 && recs[i].face != recs[i+1].face) {
                const Rec& r0 = recs[i];
                const Rec& r1 = recs[i+1];
                const uint8_t same = (r0.a == r1.a && r0.b == r1.b) ? 1 : 0;
                int c0 = cursor[r0.face]++;
                int c1 = cursor[r1.face]++;
                adj_nb[c0] = r1.face; adj_same[c0] = same;
                adj_nb[c1] = r0.face; adj_same[c1] = same;
            }
            i = j;
        }

        std::vector<int> comp((size_t)F, -1);
        std::vector<uint8_t> flip((size_t)F, 0);
        std::vector<uint8_t> conflict;
        std::vector<int> stack;
        for (int seed = 0; seed < F; ++seed) {
            if (comp[seed] >= 0 || !face_ok(seed)) continue;
            const int c = (int)conflict.size();
            conflict.push_back(0);
            comp[seed] = c;
            stack.push_back(seed);
            while (!stack.empty()) {
                const int fi = stack.back(); stack.pop_back();
                for (int e = off[fi]; e < off[fi+1]; ++e) {
                    const int nb = adj_nb[e];
                    const uint8_t need = (uint8_t)(flip[fi] ^ adj_same[e]);
                    if (comp[nb] < 0) {
                        comp[nb] = c;
                        flip[nb] = need;
                        stack.push_back(nb);
                    } else if (flip[nb] != need) {
                        conflict[c] = 1;
                    }
                }
            }
        }
        const int nc = (int)conflict.size();
        adj_nb.clear(); adj_same.clear(); off.clear(); deg.clear();
        adj_nb.shrink_to_fit(); adj_same.shrink_to_fit();

        std::vector<int> nfaces(nc, 0), nboundary(nc, 0);
        std::vector<double> vol(nc, 0.0);
        std::vector<double> bmin(nc * 3, 1e300), bmax(nc * 3, -1e300);
        for (int fi = 0; fi < F; ++fi) {
            if (comp[fi] < 0) continue;
            const int c = comp[fi];
            nfaces[c]++;
            const int ia = f[3*fi], ib = f[3*fi+1], ic = f[3*fi+2];
            const float* a = v + 3*ia; const float* b = v + 3*ib; const float* cpt = v + 3*ic;
            for (int k = 0; k < 3; ++k) {
                const float m = std::min(a[k], std::min(b[k], cpt[k]));
                const float M = std::max(a[k], std::max(b[k], cpt[k]));
                bmin[3*c+k] = std::min(bmin[3*c+k], (double)m);
                bmax[3*c+k] = std::max(bmax[3*c+k], (double)M);
            }
            const double ax = a[0], ay = a[1], az = a[2];
            const double bx = b[0], by = b[1], bz = b[2];
            const double cx = cpt[0], cy = cpt[1], cz = cpt[2];
            const double cxp = (by-ay)*(cz-az) - (bz-az)*(cy-ay);
            const double cyp = (bz-az)*(cx-ax) - (bx-ax)*(cz-az);
            const double czp = (bx-ax)*(cy-ay) - (by-ay)*(cx-ax);
            double tp = ax*cxp + ay*cyp + az*czp;
            if (flip[fi]) tp = -tp;
            vol[c] += tp;
        }
        for (size_t i = 0; i < recs.size();) {
            size_t j = i + 1;
            while (j < recs.size() && recs[j].key == recs[i].key) ++j;
            if (!(j - i == 2 && recs[i].face != recs[i+1].face)) {
                for (size_t k = i; k < j; ++k) {
                    const int c = comp[recs[k].face];
                    if (c >= 0) nboundary[c]++;
                }
            }
            i = j;
        }

        for (int fi = 0; fi < F; ++fi) {
            if (!face_ok(fi) || comp[fi] < 0) continue;
            const float* a = v + 3*f[3*fi];
            const float* b = v + 3*f[3*fi+1];
            const float* cpt = v + 3*f[3*fi+2];
            float ab[3], ac[3], nrm[3];
            for (int k = 0; k < 3; ++k) { ab[k] = b[k]-a[k]; ac[k] = cpt[k]-a[k]; }
            cross3(ab, ac, nrm);
            if (flip[fi]) { nrm[0] = -nrm[0]; nrm[1] = -nrm[1]; nrm[2] = -nrm[2]; }
            const float L = len3(nrm);
            if (!(L > 1e-20f)) continue;
            face_n[3*fi] = nrm[0]/L; face_n[3*fi+1] = nrm[1]/L; face_n[3*fi+2] = nrm[2]/L;
            trust[fi] = 1;
        }

        std::vector<int> placed;
        std::vector<int> begin(nc + 1, 0);
        if (nc > 0) {
            for (int fi = 0; fi < F; ++fi) if (comp[fi] >= 0) begin[comp[fi]+1]++;
            for (int c = 0; c < nc; ++c) begin[c+1] += begin[c];
            placed.assign((size_t)F, -1);
            std::vector<int> cur = begin;
            for (int fi = 0; fi < F; ++fi) if (comp[fi] >= 0) placed[cur[comp[fi]]++] = fi;
        }

        auto negate_comp = [&](int c) {
            for (int i = begin[c]; i < begin[c+1]; ++i) {
                const int fi = placed[i];
                face_n[3*fi] = -face_n[3*fi];
                face_n[3*fi+1] = -face_n[3*fi+1];
                face_n[3*fi+2] = -face_n[3*fi+2];
            }
        };
        const float dirs[4][3] = {
            {0.91f, 0.24f, 0.13f},
            {0.16f, 0.93f, 0.27f},
            {0.19f, 0.22f, 0.96f},
            {0.62f, 0.51f, 0.37f},
        };
        for (int c = 0; c < nc; ++c) {
            if (conflict[c] || nfaces[c] < 4) {
                for (int i = begin[c]; i < begin[c+1]; ++i) trust[placed[i]] = 0;
                continue;
            }
            const double dx = bmax[3*c] - bmin[3*c];
            const double dy = bmax[3*c+1] - bmin[3*c+1];
            const double dz = bmax[3*c+2] - bmin[3*c+2];
            const double bvol = std::max(0.0, dx) * std::max(0.0, dy) * std::max(0.0, dz);
            // Volume is the enclosed-solid sign, so it is only used when the
            // component is watertight or has a crack fringe under 1% of its
            // faces. A real opening (a sheet rim, a cup lip) fails this and
            // falls through to the support-vertex test; a flat sheet's signed
            // volume is ~0 either way and would not flip on the sign of noise.
            const bool closed = (int64_t)nboundary[c] * 100 <= (int64_t)nfaces[c];
            const bool vol_ok = bvol > 0.0 && std::fabs(vol[c]) > 1e-3 * bvol;
            if (closed && vol_ok) {
                if (vol[c] < 0.0) negate_comp(c);
            } else {
                int votes = 0, seen = 0;
                for (int di = 0; di < 4; ++di) {
                    const float* dir = dirs[di];
                    int best_v = -1;
                    float best_d = -1e30f;
                    for (int i = begin[c]; i < begin[c+1]; ++i) {
                        const int fi = placed[i];
                        if (!trust[fi]) continue;
                        for (int k = 0; k < 3; ++k) {
                            const int vi = f[3*fi+k];
                            const float d = v[3*vi]*dir[0] + v[3*vi+1]*dir[1] + v[3*vi+2]*dir[2];
                            if (d > best_d) { best_d = d; best_v = vi; }
                        }
                    }
                    if (best_v < 0) continue;
                    float acc[3] = {0, 0, 0};
                    int cnt = 0;
                    for (int i = begin[c]; i < begin[c+1]; ++i) {
                        const int fi = placed[i];
                        if (!trust[fi]) continue;
                        bool inc = false;
                        for (int k = 0; k < 3; ++k) if (f[3*fi+k] == best_v) inc = true;
                        if (!inc) continue;
                        acc[0] += face_n[3*fi]; acc[1] += face_n[3*fi+1]; acc[2] += face_n[3*fi+2];
                        ++cnt;
                    }
                    const float L = len3(acc);
                    if (!cnt || !(L > 1e-12f)) continue;
                    ++seen;
                    if ((acc[0]*dir[0] + acc[1]*dir[1] + acc[2]*dir[2]) / L < 0.f) ++votes;
                }
                if (seen > 0 && votes * 2 > seen) negate_comp(c);

                // Support vertex can still point inward on a concave open patch.
                // A point just outside the bbox must come back positive; a clearly
                // negative sign flips the component. Closed solids already took
                // the volume sign above and are not re-flipped here.
                if (!bvh.empty()) {
                    const double mid[3] = {
                        0.5*(bmin[3*c]+bmax[3*c]),
                        0.5*(bmin[3*c+1]+bmax[3*c+1]),
                        0.5*(bmin[3*c+2]+bmax[3*c+2]),
                    };
                    const float pad = std::max(2.f * eps, 1e-3f);
                    const float probes[3][3] = {
                        {(float)bmax[3*c] + pad, (float)mid[1], (float)mid[2]},
                        {(float)mid[0], (float)bmax[3*c+1] + pad, (float)mid[2]},
                        {(float)mid[0], (float)mid[1], (float)bmax[3*c+2] + pad},
                    };
                    int neg = 0, pos = 0;
                    for (int pi = 0; pi < 3; ++pi) {
                        const float* pr = probes[pi];
                        const TriBvh::Hit h = bvh.closest(pr);
                        if (h.face < 0 || comp[h.face] != c || !trust[h.face]) continue;
                        const float udf = std::sqrt(h.dist2);
                        if (!(udf > 1e-8f)) continue;
                        const float side = (pr[0]-h.point[0])*face_n[3*h.face]
                                         + (pr[1]-h.point[1])*face_n[3*h.face+1]
                                         + (pr[2]-h.point[2])*face_n[3*h.face+2];
                        if (side < -0.25f * udf) ++neg;
                        else if (side > 0.25f * udf) ++pos;
                    }
                    if (neg > pos) negate_comp(c);
                }
            }
        }

        for (int fi = 0; fi < F; ++fi) {
            if (!trust[fi]) continue;
            const int ia = f[3*fi], ib = f[3*fi+1], ic = f[3*fi+2];
            const float* a = v + 3*ia; const float* b = v + 3*ib; const float* cpt = v + 3*ic;
            const int ids[3] = {ia, ib, ic};
            const float* pts[3] = {a, b, cpt};
            for (int k = 0; k < 3; ++k) {
                float u[3], w[3];
                const float* p0 = pts[k];
                const float* p1 = pts[(k+1)%3];
                const float* p2 = pts[(k+2)%3];
                for (int t = 0; t < 3; ++t) { u[t] = p1[t]-p0[t]; w[t] = p2[t]-p0[t]; }
                const float lu = len3(u), lw = len3(w);
                if (!(lu > 1e-20f) || !(lw > 1e-20f)) continue;
                float cs = dot3(u, w) / (lu * lw);
                cs = std::max(-1.f, std::min(1.f, cs));
                const float ang = std::acos(cs);
                vert_n[3*ids[k]]   += ang * face_n[3*fi];
                vert_n[3*ids[k]+1] += ang * face_n[3*fi+1];
                vert_n[3*ids[k]+2] += ang * face_n[3*fi+2];
            }
        }
        for (int vi = 0; vi < V; ++vi) {
            float* n = &vert_n[3*vi];
            const float L = len3(n);
            if (L > 1e-20f) { n[0] /= L; n[1] /= L; n[2] /= L; }
            else { n[0] = n[1] = n[2] = 0.f; }
        }

        edge_key.clear(); edge_f0.clear(); edge_f1.clear();
        edge_key.reserve(recs.size() / 2 + 1);
        for (size_t i = 0; i < recs.size();) {
            size_t j = i + 1;
            while (j < recs.size() && recs[j].key == recs[i].key) ++j;
            const size_t cnt = j - i;
            if (cnt == 1) {
                edge_key.push_back(recs[i].key);
                edge_f0.push_back(recs[i].face);
                edge_f1.push_back(-1);
            } else if (cnt == 2 && recs[i].face != recs[i+1].face) {
                edge_key.push_back(recs[i].key);
                edge_f0.push_back(recs[i].face);
                edge_f1.push_back(recs[i+1].face);
            }
            i = j;
        }
        for (int fi = 0; fi < F; ++fi) if (trust[fi]) ++trusted_faces;
    }
};

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
    OutwardField field;
    field.build(iverts, (int)iV, ifaces, (int)iF, bvh, eps);
    const double orient_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_shell).count();

    // Active voxels. The unsigned band |UDF−eps| < 0.87·cell is the reference
    // predicate and covers BOTH offset walls. The signed predicate |SDF−eps|
    // keeps only cells on the outer wall (SDF > 0 outside the solid).
    std::vector<int> acoord;
    int64_t n_udf = 0, n_sdf = 0;
    bool use_signed = field.trusted_faces > 0;
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
                const float sdf = field.signed_distance(p, h, udf);
                if (std::fabs(udf - eps) < keep) act_u[i] = 1;
                if (std::fabs(sdf - eps) < keep) act_s[i] = 1;
            }
        });
        for (size_t i = 0; i < cand_cells.size(); ++i) {
            if (act_u[i]) ++n_udf;
            if (act_s[i]) ++n_sdf;
        }
        // Sign collapse (almost every band cell classified inside) means the
        // orientation is not usable. Keep the unsigned shell rather than emit
        // a fragment or only the inner wall.
        if (n_udf >= 100 && (n_sdf < 100 || n_sdf * 50 < n_udf)) use_signed = false;
        field.apply_sign = use_signed;
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
        fprintf(stderr, "  remesh: only %lld active voxels; skipping remesh\n", (long long)Na);
        return out;
    }

    std::unordered_map<uint64_t, int> vox;
    vox.reserve((size_t)Na * 2);
    for (int64_t i = 0; i < Na; ++i) vox.emplace(key3(acoord[3*i], acoord[3*i+1], acoord[3*i+2]), (int)i);

    // f = UDF - eps at the grid VERTICES (corner mapping v/res, spec 27 §4.3).
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
            // Crossing edges always have their far endpoint under eps+cell
            // (f changes at most one cell-length per edge), so this bound
            // never clips a value that feeds the crossing interpolation.
            const TriBvh::Hit h = bvh.closest(p, eps + 2 * cell);
            const float udf = h.face >= 0 ? std::sqrt(h.dist2) : eps + 2 * cell;
            const float sdf = h.face >= 0 ? field.signed_distance(p, h, udf) : udf;
            fvert[i] = sdf - eps;
        }
    });
    field.clear();
    auto fval = [&](int x, int y, int z) -> float {
        auto it = vmap.find(key3(x, y, z));
        return it == vmap.end() ? 1e9f : fvert[it->second];
    };

    // Dual vertices: plain mean of edge crossings, cell-center fallback; per
    // voxel, ownership of the 3 "far" edges records crossing direction
    // (spec 27 §4.4).
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

    // Quad assembly per owned crossing edge (spec 27 §4.5); winding from the
    // crossing direction. The reference's "planar diagonal" selection is a
    // latent no-op upstream (always diagonal q0-q2 unless triangle q0q1q2 is
    // exactly degenerate) — reproduced faithfully here as split 1 always.
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
    if (qfaces.empty()) return out;

    // Compact used dual vertices; map back to world coordinates.
    std::vector<int32_t> remap((size_t)Na, -1);
    int nv2 = 0;
    for (int64_t i = 0; i < Na; ++i)
        if (used[i]) {
            remap[i] = nv2++;
            out.verts.push_back((dual[3*i]   / res - 0.5f) * scale);
            out.verts.push_back((dual[3*i+1] / res - 0.5f) * scale);
            out.verts.push_back((dual[3*i+2] / res - 0.5f) * scale);
        }
    out.faces.resize(qfaces.size());
    for (size_t k = 0; k < qfaces.size(); ++k) out.faces[k] = remap[qfaces[k]];

    // Face ratio tracks the active-cell ratio: the inner and outer bands are
    // the same triangulation of the same surface, one cell thick.
    const int faces_after = out.F();
    int faces_before = faces_after;
    int removed_faces = 0;
    if (use_signed && n_sdf > 0 && n_udf > n_sdf) {
        faces_before = (int)std::llround((double)faces_after * (double)n_udf / (double)n_sdf);
        if (faces_before < faces_after) faces_before = faces_after;
        removed_faces = faces_before - faces_after;
    }
    const double shell_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_shell).count();
    printf("  remesh postproc: outer shell faces %d -> %d (removed %d inner-wall faces) in %.3fs (orient %.3fs)%s\n",
           faces_before, faces_after, removed_faces, shell_s, orient_s,
           use_signed ? "" : " (unsigned fallback)");
    fflush(stdout);

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
           (long long)Na, out.V(), out.F(), eps, project_back);
    fflush(stdout);
    return out;
}

}  // namespace trellis
