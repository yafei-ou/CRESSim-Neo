"""
Generate a tetrahedral mesh for ``container.obj``.

This mirrors the TetGen export flow used by
``examples/physics/fixtures/generate_super_toroid.py`` while operating on the
container model in this directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyvista as pv
import tetgen


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir / "container.obj"

    parser = argparse.ArgumentParser(
        description="Generate TetGen .node/.ele files for a container OBJ mesh."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help=f"Input OBJ surface mesh (default: {default_input.name}).",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Output file prefix. Defaults to the input path without its extension.",
    )
    parser.add_argument(
        "--mindihedral",
        type=float,
        default=20.0,
        help="TetGen minimum dihedral angle quality target.",
    )
    parser.add_argument(
        "--minratio",
        type=float,
        default=1.5,
        help="TetGen radius-edge ratio quality target.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show the generated tetrahedral grid after export.",
    )
    return parser.parse_args()


def load_surface_mesh(obj_path: Path) -> pv.PolyData:
    mesh = pv.read(obj_path)
    surface = mesh.extract_surface().triangulate().clean(tolerance=1e-9)
    if surface.n_points == 0 or surface.n_cells == 0:
        raise RuntimeError(f"'{obj_path}' did not produce a valid triangulated surface mesh.")
    return surface


def export_surface(surface: pv.PolyData, output_obj: Path) -> None:
    oriented = surface.compute_normals(
        cell_normals=True,
        point_normals=True,
        consistent_normals=True,
        auto_orient_normals=True,
        split_vertices=False,
        inplace=False,
    )
    oriented.save(output_obj)


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_prefix = (
        args.output_prefix.resolve()
        if args.output_prefix is not None
        else input_path.with_suffix("")
    )

    surface = load_surface_mesh(input_path)

    tet = tetgen.TetGen(surface)
    tet.tetrahedralize(order=1, mindihedral=args.mindihedral, minratio=args.minratio)
    grid = tet.grid

    node_path = output_prefix.with_suffix(".node")
    surface_obj_path = output_prefix.parent / f"{output_prefix.name}_surface.obj"

    pv.save_meshio(node_path, grid)
    export_surface(grid.extract_surface().triangulate(), surface_obj_path)

    print(f"Input surface:   {input_path}")
    print(f"Tet nodes:       {node_path}")
    print(f"Tet elements:    {output_prefix.with_suffix('.ele')}")
    print(f"Render surface:  {surface_obj_path}")
    print(f"Points / tets:   {grid.n_points} / {grid.n_cells}")

    if args.plot:
        grid.plot(show_edges=True)


if __name__ == "__main__":
    main()
