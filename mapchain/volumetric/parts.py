"""Target parts: reference meshes, the complex CSG part, normalisation and inside tests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import trimesh

from mapchain.store import ROOT
from mapchain.volumetric.transport import contains_chunked, tetrahedralize

MeshCreator = Callable[[], trimesh.Trimesh]


TORUS_MAJOR_RADIUS = 1.5


TORUS_MINOR_RADIUS = 1.0


def create_box() -> trimesh.Trimesh:
    """Convex control: a box centered at the origin."""
    return trimesh.creation.box(extents=(2.0, 2.0, 2.0))


def create_cylinder() -> trimesh.Trimesh:
    """Convex, watertight reference geometry (continuity guaranteed)."""
    return trimesh.creation.cylinder(radius=1.0, height=2.0, sections=96)


def create_sphere() -> trimesh.Trimesh:
    """Convex, curved-boundary reference geometry."""

    return trimesh.creation.icosphere(subdivisions=4, radius=1.0)


def create_torus() -> trimesh.Trimesh:
    """Non-convex, genus-1 geometry: a solid torus centered at the origin."""
    return trimesh.creation.torus(
        major_radius=TORUS_MAJOR_RADIUS,
        minor_radius=TORUS_MINOR_RADIUS,
        major_sections=96,
        minor_sections=48,
    )


def create_l_box() -> trimesh.Trimesh:
    """Non-convex, genus-0 geometry: one connected, watertight L-prism."""
    width = 3.0
    depth = 3.0
    height = 1.0
    thickness = 1.0

    vertices_2d = np.array(
        [
            [0.0, 0.0],
            [width, 0.0],
            [width, thickness],
            [thickness, thickness],
            [thickness, depth],
            [0.0, depth],
        ],
        dtype=float,
    )
    faces_2d = np.array(
        [[0, 1, 2], [0, 2, 3], [0, 3, 5], [3, 4, 5]],
        dtype=int,
    )
    mesh = trimesh.creation.extrude_triangulation(
        vertices=vertices_2d,
        faces=faces_2d,
        height=height,
    )
    mesh.apply_translation(-mesh.bounding_box.centroid)
    return mesh


MESH_CREATORS: dict[str, MeshCreator] = {
    "box": create_box,
    "cylinder": create_cylinder,
    "sphere": create_sphere,
    "torus": create_torus,
    "l_box": create_l_box,
}


def load_or_create_mesh(
    name: str,
    models_directory: str | Path = "models",
) -> trimesh.Trimesh:
    """Load ``models/<name>.stl`` or create and cache it if missing."""
    normalized_name = name.strip().lower()
    directory = Path(models_directory)
    directory.mkdir(parents=True, exist_ok=True)
    stl_path = directory / f"{normalized_name}.stl"

    if stl_path.is_file():
        loaded = trimesh.load(stl_path, force="mesh")
        if not isinstance(loaded, trimesh.Trimesh):
            raise TypeError(f"'{stl_path}' did not load as one mesh.")
        return loaded

    try:
        creator = MESH_CREATORS[normalized_name]
    except KeyError as error:
        available = ", ".join(sorted(MESH_CREATORS))
        raise ValueError(
            f"Unknown mesh '{name}'. Available generated meshes: {available}."
        ) from error

    mesh = creator()
    mesh.export(stl_path)
    return mesh


def normalize_mesh_to_unit_cube(
    mesh: trimesh.Trimesh,
    *,
    padding: float = 0.05,
) -> trimesh.Trimesh:
    """Return a uniformly scaled copy centered inside ``[0, 1]^3``.

    Uniform scaling preserves proportions and puts the source cube and the
    target geometry in the same coordinate frame before the quadratic OT cost
    is computed.
    """
    if not 0.0 <= padding < 0.5:
        raise ValueError("padding must satisfy 0 <= padding < 0.5.")

    normalized = mesh.copy()
    extents = normalized.extents
    largest_extent = float(np.max(extents))
    if largest_extent <= 0.0:
        raise ValueError("The mesh has zero spatial extent.")

    center = normalized.bounds.mean(axis=0)
    usable_size = 1.0 - 2.0 * padding
    scale = usable_size / largest_extent

    normalized.apply_translation(-center)
    normalized.apply_scale(scale)
    normalized.apply_translation(np.full(3, 0.5))
    return normalized


def normalization_params(mesh: trimesh.Trimesh, *, padding: float = 0.05):
    largest = float(np.max(mesh.extents))
    center = mesh.bounds.mean(axis=0)
    scale = (1.0 - 2.0 * padding) / largest
    return center, scale


def _primitive_test(name: str):
    n = name.strip().lower()
    eps = 1e-9
    if n == "box":
        return lambda o: np.all(np.abs(o) <= 1.0 + eps, axis=1)
    if n == "cylinder":
        return lambda o: (o[:, 0] ** 2 + o[:, 1] ** 2 <= 1.0 + eps) & (np.abs(o[:, 2]) <= 1.0 + eps)
    if n == "sphere":
        return lambda o: o[:, 0] ** 2 + o[:, 1] ** 2 + o[:, 2] ** 2 <= 1.0 + eps
    if n == "torus":
        return lambda o: (
            (np.sqrt(o[:, 0] ** 2 + o[:, 1] ** 2) - TORUS_MAJOR_RADIUS) ** 2 + o[:, 2] ** 2
            <= TORUS_MINOR_RADIUS**2 + eps
        )
    if n == "l_box":

        def test(o):
            u = o[:, 0] + 1.5
            v = o[:, 1] + 1.5
            in_l = ((u >= -eps) & (u <= 3.0 + eps) & (v >= -eps) & (v <= 1.0 + eps)) | (
                (u >= -eps) & (u <= 1.0 + eps) & (v >= -eps) & (v <= 3.0 + eps)
            )
            return in_l & (np.abs(o[:, 2]) <= 0.5 + eps)

        return test
    raise ValueError(f"No analytic interior test for '{name}'.")


def make_inside_fn(name: str, center: np.ndarray, scale: float):
    """Return f(points_in_[0,1]^3) -> bool mask, exact for the named primitive."""
    test = _primitive_test(name)
    center = np.asarray(center, dtype=float)

    def inside(points_normalized: np.ndarray) -> np.ndarray:
        p = np.asarray(points_normalized, dtype=float)
        original = (p - 0.5) / scale + center
        return test(original)

    return inside


def sample_inside(
    inside_fn,
    n_points: int,
    *,
    seed: int | None = None,
    lower=(0.0, 0.0, 0.0),
    upper=(1.0, 1.0, 1.0),
    batch: int = 200_000,
    max_batches: int = 2_000,
) -> np.ndarray:
    """Rejection-sample points in [0,1]^3 accepted by a vectorized inside_fn."""
    rng = np.random.default_rng(seed)
    lower = np.asarray(lower, float)
    upper = np.asarray(upper, float)
    accepted: list[np.ndarray] = []
    total = 0
    for _ in range(max_batches):
        remaining = n_points - total
        if remaining <= 0:
            break
        cand = rng.uniform(lower, upper, size=(batch, 3))
        good = cand[inside_fn(cand)]
        if len(good):
            take = good[:remaining]
            accepted.append(take)
            total += len(take)
    if total < n_points:
        raise RuntimeError(f"Sampled only {total}/{n_points} interior points.")
    return np.concatenate(accepted, axis=0)


MODELS = ROOT / "models"


N_TARGET_EVAL = 10000


POOL_SIZE = 60000


@dataclass
class SampleBundle:
    name: str
    inside_fn: object
    pool: np.ndarray
    target_eval: np.ndarray


_CACHE: dict[str, SampleBundle] = {}


def sample_bundle(model_name: str, *, padding: float = 0.05) -> SampleBundle:
    if model_name in _CACHE:
        return _CACHE[model_name]
    mesh = load_or_create_mesh(model_name, MODELS)
    _ = normalize_mesh_to_unit_cube(mesh, padding=padding)
    center, scale = normalization_params(mesh, padding=padding)
    inside_fn = make_inside_fn(model_name, center, scale)
    pool = sample_inside(inside_fn, POOL_SIZE, seed=1000)
    pool = pool[np.random.default_rng(1).permutation(len(pool))]
    b = SampleBundle(model_name, inside_fn, pool, pool[:N_TARGET_EVAL])
    _CACHE[model_name] = b
    return b


PARTS = ("cylinder", "sphere", "torus", "l_box", "complex")


MESH_PARTS = ("bunny", "box")


AVAILABLE_PARTS = PARTS + MESH_PARTS


def complex_part():
    from manifold3d import Manifold

    n = 96
    solid = Manifold.cube((1.0, 0.6, 0.3))
    holes = [(0.18, 0.17, 0.06), (0.48, 0.17, 0.08), (0.78, 0.17, 0.1)]
    for x, y, r in holes:
        solid = solid - Manifold.cylinder(0.5, r, circular_segments=n).translate((x, y, -0.1))
    solid = solid - Manifold.cylinder(0.15, 0.055, circular_segments=n).translate((0.2, 0.44, 0.16))
    solid = solid + Manifold.cylinder(0.14, 0.105, circular_segments=n).translate((0.7, 0.44, 0.29))
    data = solid.to_mesh()
    mesh = trimesh.Trimesh(
        vertices=np.asarray(data.vert_properties)[:, :3].astype(float),
        faces=np.asarray(data.tri_verts),
        process=True,
    )
    if not mesh.is_watertight or mesh.euler_number != -4 or solid.genus() != 3:
        raise RuntimeError("Complex geometry is not watertight genus 3")
    definition = dict(
        block_bounds=[[0, 0, 0], [1, 0.6, 0.3]],
        through_holes=[
            dict(axis="z", centre_xy=[x, y], diameter=2 * r, cutter_z=[-0.1, 0.4])
            for x, y, r in holes
        ],
        blind_hole=dict(centre_xy=[0.2, 0.44], diameter=0.11, cutter_z=[0.16, 0.31], depth=0.14),
        boss=dict(centre_xy=[0.7, 0.44], diameter=0.21, z=[0.29, 0.43], overlap=0.01),
        circular_segments=n,
        operations="Subtract three through cutters, subtract blind cutter, union boss",
        watertight=bool(mesh.is_watertight),
        euler_number=mesh.euler_number,
        genus=solid.genus(),
        vertex_count=len(mesh.vertices),
        face_count=len(mesh.faces),
    )
    return (mesh, definition)


def polygon_disk(points, x, y, r, n=96):
    delta = points[:, :2] - [x, y]
    angle = np.arctan2(delta[:, 1], delta[:, 0])
    half = np.pi / n
    phase = angle % (2 * half) - half
    boundary = r * np.cos(half) / np.cos(phase)
    return np.linalg.norm(delta, axis=1) <= boundary + 1e-09


def geometry(part):
    if part in MESH_PARTS:
        raw = load_or_create_mesh(part, MODELS)
        if not raw.is_volume or len(raw.split(only_watertight=False)) != 1:
            raise ValueError(f"{part} must be one watertight, consistently oriented solid")
        mesh = normalize_mesh_to_unit_cube(raw)

        def inside(points):
            points = np.asarray(points, dtype=float)
            return contains_chunked(mesh, points) if len(points) else np.empty(0, dtype=bool)

        vertices, tets, _ = tetrahedralize(mesh, max_volume=None)
        corners = vertices[tets]
        volumes = np.abs(np.linalg.det(corners[:, 1:] - corners[:, 0, None])) / 6
        volume_tolerance = 1e-05
        if not np.isclose(volumes.sum(), mesh.volume, rtol=volume_tolerance):
            raise ValueError(f"{part} tetrahedralization does not preserve mesh volume")
        rng = np.random.default_rng(1000)
        selected = rng.choice(len(tets), size=POOL_SIZE, p=volumes / volumes.sum())
        weights = rng.dirichlet(np.ones(4), size=POOL_SIZE)
        pool = np.einsum("ni,nij->nj", weights, corners[selected])
        pool = pool[np.random.default_rng(1).permutation(len(pool))]
        provenance_path = MODELS / f"{part}_source.json"
        provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
        definition = dict(
            existing_model=f"models/{part}.stl",
            padding=0.05,
            source=provenance.get("source", "User-provided mesh"),
            model_sha256=hashlib.sha256((MODELS / f"{part}.stl").read_bytes()).hexdigest(),
            vertex_count=len(raw.vertices),
            face_count=len(raw.faces),
            tetrahedral_volume_relative_error=float(abs(volumes.sum() / mesh.volume - 1)),
            tetrahedral_volume_relative_tolerance=volume_tolerance,
            evaluation_sampler="Volume-weighted tetrahedra; uniform barycentric samples; seed 1000; permutation seed 1; first 10000",
            containment="Chunked ray casting against the normalized polygonal mesh",
        )
        return (
            mesh,
            SimpleNamespace(inside_fn=inside, target_eval=pool[:N_TARGET_EVAL], pool=pool),
            definition,
        )
    if part != "complex":
        raw = load_or_create_mesh(part, MODELS)
        return (
            normalize_mesh_to_unit_cube(raw),
            sample_bundle(part),
            dict(
                existing_model=f"models/{part}.stl",
                model_sha256=hashlib.sha256((MODELS / f"{part}.stl").read_bytes()).hexdigest(),
            ),
        )
    raw, definition = complex_part()
    centre, scale = normalization_params(raw)
    mesh = normalize_mesh_to_unit_cube(raw)

    def inside(p):
        p = (np.asarray(p) - 0.5) / scale + centre
        block = np.all(
            (p >= [-1e-09, -1e-09, -1e-09]) & (p <= [1 + 1e-09, 0.6 + 1e-09, 0.3 + 1e-09]), axis=1
        )
        boss = (
            polygon_disk(p, 0.7, 0.44, 0.105)
            & (p[:, 2] >= 0.29 - 1e-09)
            & (p[:, 2] <= 0.43 + 1e-09)
        )
        removed = np.zeros(len(p), bool)
        for x, y, r in [(0.18, 0.17, 0.06), (0.48, 0.17, 0.08), (0.78, 0.17, 0.1)]:
            removed |= polygon_disk(p, x, y, r) & (p[:, 2] >= -0.1) & (p[:, 2] <= 0.4)
        removed |= polygon_disk(p, 0.2, 0.44, 0.055) & (p[:, 2] >= 0.16) & (p[:, 2] <= 0.31)
        return (block | boss) & ~removed

    test = np.random.default_rng(2026).uniform(0, 1, (4096, 3))
    if not np.array_equal(inside(test), mesh.contains(test)):
        raise RuntimeError("Complex CSG membership disagrees with watertight mesh")
    pool = sample_inside(inside, POOL_SIZE, seed=1000)
    pool = pool[np.random.default_rng(1).permutation(len(pool))]
    definition.update(
        normalization_centre=centre,
        normalization_scale=scale,
        padding=0.05,
        evaluation_sampler="Existing sample_inside; seed 1000, pool 60000; permutation seed 1; first 10000",
    )
    return (
        mesh,
        SimpleNamespace(inside_fn=inside, target_eval=pool[:N_TARGET_EVAL], pool=pool),
        definition,
    )
