# Soft Neighbor Pipeline

The soft contact pipeline now uses a scan/compaction flow instead of a global atomic append.

## Shape

1. Build particle broad-phase entries and sorted cell ranges.
2. Count soft-soft candidates per soft particle.
3. Exclusive-scan those counts to per-particle offsets.
4. Finalize stream totals into `GpuSoftNeighborMeta`.
5. Emit the compact soft-soft candidate stream.
6. Repeat the same count/scan/finalize/emit flow for soft-rigid candidates.
7. For each solver iteration:
   - generate soft-soft contacts from the compact soft-soft candidate stream
   - compact active soft-soft contacts
   - generate soft-rigid contacts from the compact soft-rigid candidate stream
   - compact active soft-rigid contacts
   - solve using the compact active contact streams

## Invariants

- `softRadixBitFlagsBuffer` and `softRadixBitOffsetsBuffer` are reused as generic scratch buffers for count/scan/active-flag work in the soft pipeline.
- `GpuSoftNeighborMeta` is the source of truth for:
  - emitted soft-soft candidate count
  - emitted soft-rigid candidate count
  - required counts before clipping
  - overflow flags
  - active contact counts after compaction
- Candidate streams are split:
  - `softSoftCandidatePairsBuffer`
  - `softRigidCandidatePairsBuffer`
- Solve passes consume compact active contact buffers:
  - `activeSoftContactsBuffer`
  - `activeSoftRigidContactsBuffer`

## Future Fluids

Fluids should reuse the same pattern:

- per-source count buffer
- exclusive scan to offsets
- compact emitted neighbor list
- small meta buffer for required/emitted counts

The payload layout can stay domain-specific. The shared concept is the count/scan/emit orchestration, not a single unified payload type.
