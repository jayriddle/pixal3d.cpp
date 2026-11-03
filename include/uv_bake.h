// UV unwrap (xatlas) + bake voxel-volume PBR into texture images.
#pragma once
#include <array>
#include <vector>
#include <cstdint>

namespace trellis {

struct BakedMesh {
    std::vector<float>   verts;   // [Vo*3] atlas vertices (positions, TRELLIS space)
    std::vector<float>   uv;      // [Vo*2] normalized [0,1]
    std::vector<int32_t> faces;   // [Fo*3]
    std::vector<uint8_t> base;    // [T*T*4] RGBA base color
    std::vector<uint8_t> mr;      // [T*T*4] RGBA glTF metallic-roughness (G=rough, B=metal)
    int T = 0;
    bool ok() const { return T > 0 && !faces.empty(); }
};

class TriBvh;

// Sparse per-voxel PBR field at grid resolution `res` (mesh space [-0.5,0.5]^3;
// voxel i covers [i/res-0.5, (i+1)/res-0.5)). feats layout [N*6] in [0,1]:
// base RGB, metallic, roughness, alpha. Matches the reference bake, which
// trilinearly samples this volume per texel (texturing.py grid_sample_3d)
// instead of interpolating decimation-averaged per-vertex colors. When `snap`
// is set (a BVH over the ORIGINAL hole-filled mesh), texels whose position
// falls off the voxel shell are first snapped to the closest surface point —
// the reference's cuBVH unsigned_distance correction.
struct VoxelPbr {
    const std::vector<std::array<int,3>>* coords = nullptr;
    const std::vector<float>* feats = nullptr;
    int res = 0;
    const TriBvh* snap = nullptr;
    bool ok() const { return coords && feats && res > 0 && !coords->empty(); }
};

// Vertex-clustering decimation: snap verts to a `grid`-cell lattice over [-0.5,0.5]^3, average
// position + pbr per cell, drop degenerate faces. Reduces a dense voxel-surface mesh enough for
// xatlas. Outputs new verts/faces/pbr6 (in place via the out vectors).
void decimate_cluster(const std::vector<float>& verts, int V, const std::vector<int32_t>& faces, int F,
                      const std::vector<float>& pbr6, int grid,
                      std::vector<float>& ov, std::vector<int32_t>& of, std::vector<float>& op);

// Quadric-error simplification (meshoptimizer) to ~target_faces, compacting the vertex buffer.
// Smoothness-preserving, unlike the voxel-lattice clustering above (which stair-steps).
void decimate_simplify(const std::vector<float>& verts, int V, const std::vector<int32_t>& faces, int F,
                       int target_faces, std::vector<float>& ov, std::vector<int32_t>& of);

// Faithful CPU port of CuMesh's QEM edge-collapse simplifier (refs/CuMesh/src/simplify.cu):
// Garland-Heckbert quadrics + a skinny-triangle shape penalty + flip rejection + boundary
// weighting, driven by the reference threshold ladder. Produces the reference's adaptive,
// low-sliver triangulation from a dense dual-contour mesh, unlike the meshopt/FQMS path.
void decimate_qem(const std::vector<float>& verts, int V, const std::vector<int32_t>& faces, int F,
                  int target_faces, std::vector<float>& ov, std::vector<int32_t>& of);

// Drop connected components (shared-vertex face adjacency) whose face count is below
// frac*(largest component's). Removes decode floaters + spurious ground fragments (which
// shatter our mesh into 50+ pieces) while keeping legitimately large secondary shells
// (a hollow vase's inner+outer surface). In place; returns the number of components dropped.
//
// Face-count-only cut. The post-QEM pass uses this at 3% so an inward skin
// that sits against the wall (under 3% of the body) is removed, matching
// shipped. The pre-decimation pass is drop_pre_decimation_components.
int drop_small_components(std::vector<float>& verts, std::vector<int32_t>& faces, float frac = 0.02f);

// World length of one remesh cell. Matches remesh_narrow_band_dc: output
// vertices are (grid/res - 0.5) * (res + 3*band)/res, so one grid step is
// (res + 3*band) / res^2.
inline float remesh_voxel_world(int res, int band) {
    if (res < 1) res = 1;
    const int b = band < 1 ? 1 : band;
    return (float)(res + 3 * b) / ((float)res * (float)res);
}

// A component whose closest vertex is within this many remesh voxels of the
// main body is touching it or sitting right against it (a remesh crack, an
// ear, an antenna). Farther than this, and under 3% of the body's faces, it
// is a detached shell.
constexpr float kPreDecimAttachVoxels = 3.f;

// Pre-decimation filter. Largest component is the body.
// - face share < 2%: dropped
// - face share < 3% and gap > kPreDecimAttachVoxels: dropped
// - face share >= 3% (feet at ~3.9%): kept even when detached
// - face share in [2%, 3%) and within the attach distance: kept
// `voxel_world` is remesh_voxel_world(res, band). Logs every 2–3% decision
// with face share and gap, and summarizes pieces under 2% (those are dropped
// with no gap search). Prints
// `remesh postproc: components took X s (gap search Y s)`.
// Returns components dropped. Pre-decimation only; after QEM use
// drop_small_components(..., 0.03f).
int drop_pre_decimation_components(std::vector<float>& verts, std::vector<int32_t>& faces,
                                   float voxel_world);

// xatlas PackOptions::padding, in packer pixels. Full gutter when the atlas has
// about 4+ texels per face (the 250k / 1024 bake). Fades to 0 as the mesh
// approaches 1 texel of atlas per face, which is the historical padding-0 pack:
// a fixed padding on a chart-heavy mesh inflates the packer atlas and the
// downscale into the bake atlas throws away texels.
uint32_t xatlas_pack_padding(int texsize, int faces);

// Fill small boundary loops (holes) with a centroid fan, port of CuMesh fill_holes
// (clean_up.cu): loops whose perimeter is below max_perimeter get one new vertex at the
// mean of their edge midpoints plus one triangle per rim edge. The reference postprocess
// runs this on the decoded mesh BEFORE the remesh (max_hole_perimeter=3e-2 in the
// [-0.5,0.5] world cube) -- open cracks make the narrow-band in/out test ambiguous and
// leak holes into the remeshed surface. In place; returns the number of holes filled.
int fill_holes(std::vector<float>& verts, std::vector<int32_t>& faces, float max_perimeter = 3e-2f);

// Taubin (lambda/mu) shrink-free Laplacian smoothing. Strips the ~1-voxel stair-step noise
// of the dual-contour surface so the quadric simplifier stays curvature-adaptive instead of
// emitting a uniform-dense sliver mesh. Boundary verts pinned. In place.
void taubin_smooth(std::vector<float>& verts, const std::vector<int32_t>& faces,
                   int iters = 5, float lambda = 0.5f, float mu = -0.53f);

// In-place cleanup of a welded surface mesh so meshopt's guarded quadric collapse
// reaches the target without the roughening FQMS fallback: drop degenerate faces and
// unify face orientations by BFS over manifold-edge adjacency. The narrow-band DC output is ~11%
// inconsistently wound + non-manifold, which otherwise stalls meshopt and forces FQMS.
// `verts` is required: after the relative BFS, each nearly-closed component whose
// signed volume is negative is flipped so a consistent shell cannot be left inside-out.
void clean_mesh(const float* verts, int V, std::vector<int32_t>& faces);

// Fan-fill boundary loops of at most max_loop edges in place; returns the number of holes filled.
int fill_small_holes(std::vector<int32_t>& faces, int max_loop = 64);

// Merge vertices within `step` of each other in place (optionally remapping per-vertex RGB);
// returns the number of duplicates removed. The dual-grid decoder emits epsilon-different
// positions for corners shared across cells, so the raw mesh is full of hairline cracks that
// read as borders: they produce pinhole boundary loops and block edge-collapse simplification.
int weld_vertices(std::vector<float>& verts, std::vector<int32_t>& faces,
                  std::vector<float>* colors3 = nullptr, float step = 1.0f / 8192.0f);

// verts [V*3], faces [F*3], pbr6 [V*6] per-vertex (base3, metallic, roughness, alpha) in [0,1].
// Unwraps with xatlas, shades texels from `vox` (trilinear volume sampling) when provided,
// else from interpolated per-vertex PBR; dilates seams.
BakedMesh uv_bake(const std::vector<float>& verts, int V, const std::vector<int32_t>& faces, int F,
                  const std::vector<float>& pbr6, int texsize, const VoxelPbr* vox = nullptr);

// Voxel-native box (cube) projection atlas — no xatlas, no chart computation. Faces are
// assigned to the axis-aligned plane (of 6) along which they are visible, with a second-layer
// chart set for occluded faces; texels are shaded from `vox` when provided (else per-vertex
// PBR) with a depth-tested raster. O(F) — fast even for the undecimated res-1024 cascade mesh.
BakedMesh uv_box_project(const std::vector<float>& verts, int V, const std::vector<int32_t>& faces, int F,
                         const std::vector<float>& pbr6, int texsize, const VoxelPbr* vox = nullptr);

// Normal-clustered planar charts + shelf packing: deterministic unwrap with
// area-proportional texel density. Fallback when xatlas times out or is skipped.
BakedMesh uv_chart_project(const std::vector<float>& verts, int V, const std::vector<int32_t>& faces, int F,
                           const std::vector<float>& pbr6, int texsize, const VoxelPbr* vox = nullptr);

} // namespace trellis
