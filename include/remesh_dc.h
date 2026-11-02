// Narrow-band dual-contouring remesh, ported from the reference
// (CuMesh remeshing.py::remesh_narrow_band_dc + simple_dual_contour.cu; see
// docs/spec/27-reference-postprocess.md §4). Rebuilds a noisy voxel-derived
// mesh as a consistently-oriented offset surface at distance
// eps = band·scale/res.
//
// The reference implicit function is f = UDF − eps. That zero set is the
// boundary of a tubular neighborhood, so a closed solid comes out as two
// walls: the outer offset and a second surface ~2·eps inside. Downstream
// decimation and the atlas then spend about half their budget on the hidden
// wall. This port signs the field (positive outside the solid) and extracts
// f = SDF − eps, whose only zero is the outer offset. Components whose
// winding cannot be oriented fall back to the unsigned band so an open
// sheet is not deleted. The quad assembly, winding tables, and project_back
// lerp are otherwise the reference algorithm.
#pragma once
#include "dual_grid.h"

namespace trellis {

class TriBvh;

// `bvh` must be built over (verts, faces). Returns an empty mesh on failure
// (caller keeps the un-remeshed path).
// `project_back` lerps each dual vertex toward its closest point on the input
// surface (o_voxel postprocess `remesh_project`; 0 disables).
//
// Default 0 to match the HF space — the config that produced the reference GLBs
// users compare against — which passes remesh_project=0 (trellis2-space/app.py).
// o_voxel's own to_glb default is 0.9 and TRELLIS.2's app.py inherits it, so the
// knob is kept: measured on the issue #1 goblin, 0.9 improves the surface
// (dihedral 8.08->7.41deg, >60deg spikes 1.12%->0.64%) but quadruples our UV
// merge clusters (2519->10248), so it is NOT a free win in this pipeline.
Mesh remesh_narrow_band_dc(const float* verts, int64_t V, const int32_t* faces, int64_t F,
                           const TriBvh& bvh, int res, int band = 1,
                           float project_back = 0.0f);

}  // namespace trellis
