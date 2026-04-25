from dataclasses import dataclass


@dataclass
class TileLayout:
    rows: int
    cols: int
    data_per_tile: int
    ancilla_per_tile: int

    @property
    def num_tiles(self) -> int:
        return self.rows * self.cols

    @property
    def total_data_qubits(self) -> int:
        return self.num_tiles * self.data_per_tile

    @property
    def total_ancilla_qubits(self) -> int:
        return self.num_tiles * self.ancilla_per_tile


@dataclass
class RoutedGeometry:
    layout: TileLayout
    max_move_distance: int

    def is_local(self, tile_a: tuple, tile_b: tuple) -> bool:
        dr = abs(tile_a[0] - tile_b[0])
        dc = abs(tile_a[1] - tile_b[1])
        return max(dr, dc) <= self.max_move_distance

    def cross_tile_edge_count(self) -> int:
        count = 0
        tiles = [(r, c) for r in range(self.layout.rows)
                         for c in range(self.layout.cols)]
        for i, ta in enumerate(tiles):
            for tb in tiles[i+1:]:
                if not self.is_local(ta, tb):
                    count += 1
        return count


MovementConstraint = RoutedGeometry
