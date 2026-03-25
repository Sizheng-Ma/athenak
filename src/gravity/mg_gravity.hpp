#ifndef GRAVITY_MG_GRAVITY_HPP_
#define GRAVITY_MG_GRAVITY_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mg_gravity.hpp
//! \brief Multigrid Poisson solver for Newtonian self-gravity.
//!
//! Solves: nabla^2 Phi = 4*pi*G*rho
//! on a uniform Cartesian global grid using a V-cycle multigrid method.
//!
//! Boundary conditions: Dirichlet Phi = 0 on all six faces.
//! Smoother:            Red-black Gauss-Seidel.
//! Restriction:         Full-weighting (average of 8 fine cells).
//! Prolongation:        Piecewise-constant injection.
//!
//! The class is host-only and does not depend on Kokkos.  The caller is responsible
//! for gathering density onto the global grid (and MPI-reducing it across ranks),
//! and for scattering the resulting gradient back to the device after Solve().

#include <vector>
#include "athena.hpp" // Real

class MGGravity
{
public:
  // nx, ny, nz : number of active (interior) cells in each direction
  // h          : uniform cell spacing (must satisfy hx == hy == hz)
  // G_newt     : Newton's constant in code units
  MGGravity(int nx, int ny, int nz, Real h, Real G_newt);

  // ---- caller interface -------------------------------------------------------

  // Fill the source term S = 4*pi*G*rho.
  // src must hold nx*ny*nz values, indexed src[k*ny*nx + j*nx + i] (0-based, k slow).
  void SetRHS(const std::vector<Real> &src);

  // Zero the potential (call before the first solve, or to reset).
  void ZeroPhi();

  // Run n_vcycles V-cycles.  On convergence, Phi holds the solution.
  void Solve(int n_vcycles = 4);

  // Return residual L2-norm (useful for diagnostics).
  Real ResidualNorm() const;

  // Gravitational acceleration at global cell (I,J,K), 0-based.
  // ax = -dPhi/dx,  ay = -dPhi/dy,  az = -dPhi/dz.
  void GetAccel(int I, int J, int K, Real &ax, Real &ay, Real &az) const;

  // Read-only access to the finest-level potential (Nx*Ny*Nz elements, 0-based).
  const std::vector<Real> &Phi() const { return phi_[0]; }

  // Grid dimensions
  int Nx() const { return nx_l_[0]; }
  int Ny() const { return ny_l_[0]; }
  int Nz() const { return nz_l_[0]; }

private:
  int n_levels_;
  Real h0_; // cell size at finest level
  Real G_;

  std::vector<int> nx_l_, ny_l_, nz_l_; // dimensions at each level
  std::vector<Real> h_l_;               // cell size at each level

  // Solution, RHS, and residual at each level (size nx*ny*nz, 0-based, no ghosts)
  std::vector<std::vector<Real>> phi_;
  std::vector<std::vector<Real>> rhs_;
  std::vector<std::vector<Real>> res_;

  // Flat index (no ghost cells)
  int Idx(int i, int j, int k, int nx, int ny) const
  {
    return k * ny * nx + j * nx + i;
  }

  // Number of boundary faces adjacent to interior cell (i,j,k)
  int NBnd(int i, int j, int k, int nx, int ny, int nz) const
  {
    return (i == 0) + (i == nx - 1) + (j == 0) + (j == ny - 1) + (k == 0) + (k == nz - 1);
  }

  // Sum of the six neighbors that exist (boundary neighbors contribute 0 via Dirichlet)
  Real SumNeighbors(const std::vector<Real> &phi,
                    int i, int j, int k, int nx, int ny, int nz) const;

  // Core algorithmic steps
  void Smooth(std::vector<Real> &phi, const std::vector<Real> &rhs,
              int nx, int ny, int nz, Real h, int n_iters);
  void CalcResidual(const std::vector<Real> &phi, const std::vector<Real> &rhs,
                    std::vector<Real> &res, int nx, int ny, int nz, Real h) const;
  void Restrict(const std::vector<Real> &fine, std::vector<Real> &coarse,
                int nx_f, int ny_f, int nz_f) const;
  void Prolongate(const std::vector<Real> &coarse, std::vector<Real> &fine,
                  int nx_c, int ny_c, int nz_c) const;
  void VCycle(int level);
};

#endif // GRAVITY_MG_GRAVITY_HPP_
