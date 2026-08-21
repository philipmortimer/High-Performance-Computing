# Distributed Lattice Boltzmann Fluid Solver

High-performance C implementation of a D2Q9-BGK lattice Boltzmann fluid solver, optimised from a serial baseline and scaled to **112 CPU cores across 4 HPC nodes**.

For the 1024 × 1024 benchmark, runtime was reduced from **537.8s to 1.63s (~330× speedup)** through SIMD vectorisation, memory-layout optimisation, OpenMP and distributed-memory parallelism with MPI.

## Optimisations

The project was developed in two stages:

### [Part 1 — Single-node optimisation](Part1)

Optimised and parallelised the original serial implementation using:

* loop fusion and arithmetic optimisation
* structure-of-arrays memory layout to enable SIMD vectorisation
* cache-aware memory alignment
* OpenMP parallelisation across 28 cores
* NUMA-aware data placement and thread pinning
* compiler and architecture-specific optimisation

On the 1024 × 1024 benchmark, these changes reduced runtime from **537.8s to 11.5s** on a single 28-core node.

See the [Part 1 report](Part1/report.pdf) for profiling and performance analysis.

### [Part 2 — Multi-node scaling](Part2)

Extended the optimised solver across multiple compute nodes using MPI, including:

* distributed domain decomposition
* non-blocking halo exchange with `MPI_Isend` / `MPI_Irecv`
* MPI collectives and parallel file I/O
* experiments with hybrid MPI + OpenMP execution
* cache, communication and scaling analysis across up to 112 cores

The final MPI implementation reduced the 1024 × 1024 runtime to **1.63s across 4 nodes / 112 cores**.

Roofline analysis showed the solver to be memory-bandwidth bound, reaching **64% of peak STREAM DRAM bandwidth**.

See the [Part 2 report](Part2/report.pdf) for implementation details, scaling experiments and profiling results.

## Performance

| Implementation           | 1024 × 1024 runtime |
| ------------------------ | ------------------: |
| Serial baseline          |              537.8s |
| OpenMP, 28 cores         |               11.5s |
| MPI, 112 cores / 4 nodes |           **1.63s** |

**Overall speedup: ~330×**

## Context

Originally developed as part of a coursework and run on the BlueCrystal supercomputing cluster.
