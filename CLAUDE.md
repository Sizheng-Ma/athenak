# AthenaK — Newtonian Star Project

This file is the resume context for Claude Code. It captures the full state of the
ongoing project so any future session can pick up immediately.

---

## Project Goal

Simulate a **Newtonian polytropic star** in AthenaK with:
1. Lane-Emden equilibrium initial conditions
2. Full 3D self-gravity via a multigrid Poisson solver

---

## Files Created / Modified

### New files (all written from scratch in this project)

| File | Purpose |
|---|---|
| `src/pgen/newtonian_star.cpp` | Problem generator: ICs and gravity source term |
| `src/gravity/mg_gravity.hpp` | MGGravity class interface |
| `src/gravity/mg_gravity.cpp` | V-cycle multigrid Poisson solver implementation |
| `inputs/hydro/newtonian_star.athinput` | Runtime parameter file |

### Modified files

| File | Change |
|---|---|
| `src/CMakeLists.txt` | Added `gravity/mg_gravity.cpp` to the source list |

---

## Cluster Build (Perimeter Institute)

The cluster at `/mnt/beegfs/sma2/` uses Intel oneAPI MPI (OpenMPI at `/usr/mpi/gcc/openmpi-4.1.7rc1` is NOT built with SLURM PMI support — don't use it). Job launcher is `srun`.

```bash
source /cm/shared/opt/intel/oneapi/setvars.sh
cd /mnt/beegfs/sma2/athenak
rm -rf build && mkdir build && cd build
cmake -D PROBLEM=newtonian_star \
      -D Athena_ENABLE_MPI=ON \
      -D Athena_ENABLE_OPENMP=ON \
      -D CMAKE_CXX_COMPILER=/cm/shared/opt/intel/oneapi/mpi/2021.15/bin/mpicxx \
      ..
make -j$(nproc)
cp src/athena /mnt/beegfs/sma2/GMC/athena
```

**Note:** AthenaK does NOT support HDF5 output. Use `file_type = vtk` (or `bin`) in the athinput.

### SLURM job script (`run.sh`) — 1 node, 40 cores (8 MPI × 5 OpenMP)

```bash
#!/bin/bash
#SBATCH -J GMC
#SBATCH -o res
#SBATCH -e err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --cpus-per-task=5
#SBATCH -t 24:00:00
#SBATCH -p defq
#SBATCH --mail-user=sma2@perimeterinstitute.ca
#SBATCH --mail-type=END

source /cm/shared/opt/intel/oneapi/setvars.sh
export OMP_NUM_THREADS=5
export OMP_PROC_BIND=close
export OMP_PLACES=cores

srun -n 8 ./athena -i newtonian_star.athinput -d output
```

Run directory: `/mnt/beegfs/sma2/GMC/`

---

## How to Build Locally

```bash
cd /Users/sizhengma/athenak
mkdir -p build && cd build
cmake -D PROBLEM=newtonian_star ..
make -j$(nproc)
./src/athena -i ../inputs/hydro/newtonian_star.athinput
```

---

## Physics Summary

### Lane-Emden equilibrium
- Polytropic EOS: `P = K * rho^gamma`, with `gamma = 5/3` => polytropic index `n = 3/2`
- Lane-Emden ODE solved with RK4 on the host at startup
- Scale radius: `alpha = sqrt((n+1)*K*rho_c^(gamma-2) / (4*pi*G))`
- With `G=K=rho_c=1`: `R_star ~ 1.331`, `M_star ~ 0.564`
- Outside `R_star`: low-density atmosphere at `rho_atmo = 1e-6`

### Multigrid Poisson solver (`MGGravity`)
Solves `∇²Φ = 4πGρ` with **Dirichlet BC** (Φ=0 on all 6 faces).

- Smoother: red-black Gauss-Seidel (2 pre, 2 post sweeps per level; 20 at coarsest)
- Restriction: full-weighting (average of 8 fine cells per coarse cell)
- Prolongation: piecewise-constant injection (coarse value copied to all 8 fine children)
- Dirichlet stencil: boundary cells use denominator `6 + N_bnd` (where `N_bnd` = number of adjacent boundary faces), equivalent to image-ghost approach
- Grid hierarchy built at construction; coarsens until any dimension < 4 or odd
- Host-only C++ (no Kokkos); operates on `std::vector<Real>`

### Self-gravity workflow (every RK sub-step)
1. `deep_copy` primitives `w0` from device → host mirror
2. Loop over MeshBlocks: map `rho` into global `rho_global[]` array using `mb_size.h_view(m).x1min` offsets
3. `MPI_Allreduce(MPI_IN_PLACE, ..., MPI_SUM)` to combine contributions across MPI ranks
4. `MGGravity::SetRHS(4*pi*G*rho_global)` + `MGGravity::Solve(n_vcycles)`
5. Copy `phi_[0]` → device `d_phi` via `deep_copy`
6. `par_for` kernel on device: central-difference `-∇Φ` → update `u0(IM1,IM2,IM3,IEN)`

---

## Key AthenaK Conventions

- **`par_for` signature**: `(int m, int k, int j, int i)` — always 4 indices for 3D+block loops
- **Conserved variable indices**: `IDN=0, IM1=1(IVX), IM2=2(IVY), IM3=3(IVZ), IEN=4`
- **Primitive variable indices**: `IDN=0, IVX=1, IVY=2, IVZ=3, IPR=4`
- **Device arrays**: `DvceArray5D<Real> u0(block, var, k, j, i)`
- **`mb_size`**: per-block geometry; `.d_view(m)` on device, `.h_view(m)` on host
- **`mesh_indcs`**: global grid; `nx1,nx2,nx3` are active cell counts (no ghosts); `ng` = ghost count
- **`mesh_size`**: `dx1,dx2,dx3` = cell spacing; `x1min/x1max` = domain bounds
- **Math in KOKKOS_LAMBDA**: use `#include <math.h>` and unqualified `sqrt`, `pow`, `sin` — NOT `std::` versions (GPU device code incompatibility)
- **User source terms**: register `user_srcs_func = MyFunc` where `MyFunc` has signature `void(Mesh*, const Real bdt)`
- **MPI**: wrap with `#if MPI_PARALLEL_ENABLED` / `#endif`; use `MPI_ATHENA_REAL` for the datatype
- **MeshBlocks**: total blocks = `(mesh.nx / meshblock.nx)` per dimension multiplied. Must divide evenly; meshblock ≥ 4 cells and divisible by 2 in each dim.
- **Output formats**: supported are `vtk`, `pvtk`, `bin`, `cbin`, `tab`, `hst`, `rst`, `trk`, `log`. HDF5 is NOT supported.

---

## StarData Struct (global state in pgen)

```cpp
struct StarData {
  Real G_newt, gamma, poly_n, rho_c, K_poly;
  Real alpha, R_star, M_star, rho_atmo;
  MGGravity *mg;          // heap-allocated multigrid solver
  int n_vcycles;
  int Nx, Ny, Nz;         // global active cell counts
  Real x1min, x2min, x3min, h;  // domain origin and cell spacing
  std::vector<Real> rho_global; // host density buffer (Nx*Ny*Nz)
  Real rho_center = 0.0;        // central density (updated each gravity step)
  DvceArray1D<Real> d_phi;      // device potential array
  int n_le;
  DvceArray1D<Real> d_r_le, d_rho_le, d_prs_le;  // Lane-Emden IC tables
};
```

---

## Input File Parameters (`<problem>` block)

| Parameter | Default | Meaning |
|---|---|---|
| `G_newt` | 1.0 | Newton's constant in code units |
| `rho_c` | 1.0 | Central density |
| `K_poly` | 1.0 | Polytropic constant |
| `rho_atmo` | 1e-6 | Atmosphere density floor |
| `v_pert` | 0.0 | Radial velocity perturbation amplitude |
| `n_vcycles` | 4 | Multigrid V-cycles per RK sub-step |

Grid: 64³ cells, box [-2,2]³, 32³ cells per MeshBlock, outflow BCs on all faces.

---

## Known Issues / Gotchas Fixed in Past Sessions

1. **`std::sqrt` in KOKKOS_LAMBDA fails on GPU** — use `#include <math.h>` and unqualified `sqrt`, `pow`, `sin` inside all device kernels. Host code can still use `std::`.
2. **`MDRangePolicy` is not used in AthenaK** — always use `par_for` with 4-index signature.
3. **Global index offset**: map MeshBlock to global grid with `I0 = round((x1min_mb - x1min_global) / h)`.
4. **Dirichlet without ghost cells**: denominator is `6 + N_bnd` not `6`; boundary neighbors contribute 0 to neighbor sum.
5. **Kokkos global destructor crash**: device arrays (`DvceArray1D`) in a global `StarData` struct are destroyed after `Kokkos::finalize()`. Fix: call `Kokkos::push_finalize_hook([]() { star.d_phi = DvceArray1D<Real>(); ... })` at the end of `UserProblem` to reset them before Kokkos shuts down.
6. **Cluster MPI**: OpenMPI at `/usr/mpi/gcc/openmpi-4.1.7rc1` is not built with SLURM PMI — causes abort at `MPI_Init`. Use Intel MPI (`/cm/shared/opt/intel/oneapi/mpi/2021.15/bin/mpicxx`) instead. Use `srun` (not `mpirun`) as the job launcher.
7. **rho_center reads ~8x too high** (`~7.95` instead of `~1.0`): persists despite two attempted fixes. Fix attempt 1: read from `rho_global` after `MPI_Allreduce(SUM)` — gave 8x. Fix attempt 2: search `w0_mirror` per-block with `MPI_Allreduce(MPI_MAX)` — still gives 7.95. A debug print was added (prints to `err` file on the first gravity call) to determine: whether the center block is found, what cell indices are used, what raw value is read, and whether the new binary is actually running. **Next session: check the `err` output from the cluster to diagnose.** Also, `rho_center = 0` at `t=0` because the gravity source term hasn't been called yet when the first history write happens.

---

## Self-Gravity Verification Plan

The goal is to verify self-gravity is correct by exciting the fundamental radial oscillation mode and comparing its period to the analytical value.

- **Method**: set `v_pert = 0.05` in `<problem>` block, run with `tlim = 30`
- **Diagnostic**: `newt_star.user.hst` column `rho_c` should oscillate sinusoidally around `~1.0`
- **Analytical period**: `T = 2π / (σ · ω_dyn)` where `ω_dyn = sqrt(GM/R³) ≈ 0.489` (code units) and `σ` is the dimensionless eigenvalue for n=3/2, γ=5/3 polytrope (from Cox 1980 or numerical solution of linearised oscillation equations)
- **Blocker**: `rho_center` is currently reading `~7.95` instead of `~1.0` (see known issue #7 above) — must fix before the period comparison is meaningful

---

## Status (as of last session)

- [x] Lane-Emden initial conditions
- [x] MGGravity multigrid Poisson solver (`src/gravity/`)
- [x] `inputs/hydro/newtonian_star.athinput` with multigrid params and VTK output
- [x] `src/CMakeLists.txt` updated to include `gravity/mg_gravity.cpp`
- [x] Lagrangian particle removed (user does not need it)
- [x] Kokkos finalize-hook fix for clean shutdown
- [x] Central density history output (`NewtonianStarHistory` → `newt_star.user.hst`)
- [x] Successfully ran to `tlim=10` on cluster (mass conserved, ~5% energy loss over 35 time units)
- [ ] **Fix rho_center bug** — debug print added; check `err` file on cluster for `[DEBUG rho_center]` line
- [ ] Once rho_center is fixed: run oscillation test (`v_pert=0.05`, `tlim=30`) and compare period to analytical value
- [ ] Investigate star drift/symmetry breaking after t~60 (KE components become asymmetric, star develops net z-momentum ~0.43 at t=90, 28% mass loss by t=102) — likely from MeshBlock boundary asymmetry; test with single MeshBlock (meshblock=64³)
