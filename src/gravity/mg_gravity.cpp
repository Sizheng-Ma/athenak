//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mg_gravity.cpp
//! \brief Multigrid Poisson solver for Newtonian self-gravity.
//! See mg_gravity.hpp for algorithm description.

#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>

#include "mg_gravity.hpp"

// ---------------------------------------------------------------------------
// Constructor: build the grid hierarchy
// ---------------------------------------------------------------------------
MGGravity::MGGravity(int nx, int ny, int nz, Real h, Real G_newt)
    : h0_(h), G_(G_newt) {
  // Build coarsening levels until any dimension reaches 2 or is odd
  int lnx = nx, lny = ny, lnz = nz;
  Real lh = h;
  for (;;) {
    nx_l_.push_back(lnx);
    ny_l_.push_back(lny);
    nz_l_.push_back(lnz);
    h_l_.push_back(lh);
    phi_.push_back(std::vector<Real>(lnx*lny*lnz, 0.0));
    rhs_.push_back(std::vector<Real>(lnx*lny*lnz, 0.0));
    res_.push_back(std::vector<Real>(lnx*lny*lnz, 0.0));

    // Stop if any dimension can no longer be halved evenly or is too small
    if (lnx < 4 || lny < 4 || lnz < 4 ||
        lnx % 2 != 0 || lny % 2 != 0 || lnz % 2 != 0) break;

    lnx /= 2; lny /= 2; lnz /= 2;
    lh  *= 2.0;
  }
  n_levels_ = static_cast<int>(nx_l_.size());
}

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------
void MGGravity::SetRHS(const std::vector<Real>& src) {
  rhs_[0] = src;   // copy caller's source term into the finest level
}

void MGGravity::ZeroPhi() {
  for (auto& v : phi_) std::fill(v.begin(), v.end(), 0.0);
}

void MGGravity::Solve(int n_vcycles) {
  for (int cyc = 0; cyc < n_vcycles; cyc++) {
    VCycle(0);
  }
}

Real MGGravity::ResidualNorm() const {
  const int nx = nx_l_[0], ny = ny_l_[0], nz = nz_l_[0];
  const Real h = h_l_[0];
  std::vector<Real> res(nx*ny*nz);
  CalcResidual(phi_[0], rhs_[0], res, nx, ny, nz, h);
  Real norm2 = 0.0;
  for (Real v : res) norm2 += v*v;
  return std::sqrt(norm2 / res.size());
}

void MGGravity::GetAccel(int I, int J, int K,
                         Real& ax, Real& ay, Real& az) const {
  const int nx = nx_l_[0], ny = ny_l_[0], nz = nz_l_[0];
  const Real h2 = 2.0 * h0_;
  const auto& phi = phi_[0];

  // For Dirichlet Phi=0, the image ghost cell beyond a boundary face has value
  // -phi[boundary_cell].  That makes the central-difference gradient:
  //   At I=0:    dPhi/dx = (phi[1,J,K] + phi[0,J,K]) / (2h)
  //   At I=Nx-1: dPhi/dx = (-phi[Nx-1,J,K] - phi[Nx-2,J,K]) / (2h)
  //   Interior:  dPhi/dx = (phi[I+1,J,K] - phi[I-1,J,K]) / (2h)

  Real phiXp = (I < nx-1) ? phi[Idx(I+1,J,K,nx,ny)] : -phi[Idx(I,J,K,nx,ny)];
  Real phiXm = (I > 0)    ? phi[Idx(I-1,J,K,nx,ny)] : -phi[Idx(I,J,K,nx,ny)];
  Real phiYp = (J < ny-1) ? phi[Idx(I,J+1,K,nx,ny)] : -phi[Idx(I,J,K,nx,ny)];
  Real phiYm = (J > 0)    ? phi[Idx(I,J-1,K,nx,ny)] : -phi[Idx(I,J,K,nx,ny)];
  Real phiZp = (K < nz-1) ? phi[Idx(I,J,K+1,nx,ny)] : -phi[Idx(I,J,K,nx,ny)];
  Real phiZm = (K > 0)    ? phi[Idx(I,J,K-1,nx,ny)] : -phi[Idx(I,J,K,nx,ny)];

  ax = -(phiXp - phiXm) / h2;   // -dPhi/dx
  ay = -(phiYp - phiYm) / h2;
  az = -(phiZp - phiZm) / h2;
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------
Real MGGravity::SumNeighbors(const std::vector<Real>& phi,
                              int i, int j, int k,
                              int nx, int ny, int nz) const {
  Real s = 0.0;
  if (i > 0)    s += phi[Idx(i-1,j,k,nx,ny)];
  if (i < nx-1) s += phi[Idx(i+1,j,k,nx,ny)];
  if (j > 0)    s += phi[Idx(i,j-1,k,nx,ny)];
  if (j < ny-1) s += phi[Idx(i,j+1,k,nx,ny)];
  if (k > 0)    s += phi[Idx(i,j,k-1,nx,ny)];
  if (k < nz-1) s += phi[Idx(i,j,k+1,nx,ny)];
  return s;
}

// Red-black Gauss-Seidel smoother
// Dirichlet BC: boundary neighbors contribute 0 (image = -phi, so net stencil
// coefficient becomes 6 + nbnd instead of 6).
void MGGravity::Smooth(std::vector<Real>& phi, const std::vector<Real>& rhs,
                       int nx, int ny, int nz, Real h, int n_iters) {
  const Real h2 = h * h;
  for (int iter = 0; iter < n_iters; iter++) {
    for (int color = 0; color < 2; color++) {
      for (int k = 0; k < nz; k++)
      for (int j = 0; j < ny; j++)
      for (int i = 0; i < nx; i++) {
        if ((i + j + k) % 2 != color) continue;
        Real denom = 6.0 + NBnd(i, j, k, nx, ny, nz);
        Real s = SumNeighbors(phi, i, j, k, nx, ny, nz);
        phi[Idx(i,j,k,nx,ny)] = (s - h2 * rhs[Idx(i,j,k,nx,ny)]) / denom;
      }
    }
  }
}

// Residual r = S - L[Phi] where L is the 6-point Laplacian with Dirichlet BC
void MGGravity::CalcResidual(const std::vector<Real>& phi,
                              const std::vector<Real>& rhs,
                              std::vector<Real>& res,
                              int nx, int ny, int nz, Real h) const {
  const Real inv_h2 = 1.0 / (h * h);
  for (int k = 0; k < nz; k++)
  for (int j = 0; j < ny; j++)
  for (int i = 0; i < nx; i++) {
    Real denom = 6.0 + NBnd(i, j, k, nx, ny, nz);
    Real s = SumNeighbors(phi, i, j, k, nx, ny, nz);
    Real lap = (s - denom * phi[Idx(i,j,k,nx,ny)]) * inv_h2;
    res[Idx(i,j,k,nx,ny)] = rhs[Idx(i,j,k,nx,ny)] - lap;
  }
}

// Full-weighting restriction: average of 2^3 = 8 fine children per coarse cell
void MGGravity::Restrict(const std::vector<Real>& fine, std::vector<Real>& coarse,
                         int nx_f, int ny_f, int nz_f) const {
  const int nx_c = nx_f/2, ny_c = ny_f/2, nz_c = nz_f/2;
  for (int K = 0; K < nz_c; K++)
  for (int J = 0; J < ny_c; J++)
  for (int I = 0; I < nx_c; I++) {
    Real sum = 0.0;
    for (int dk = 0; dk < 2; dk++)
    for (int dj = 0; dj < 2; dj++)
    for (int di = 0; di < 2; di++) {
      sum += fine[Idx(2*I+di, 2*J+dj, 2*K+dk, nx_f, ny_f)];
    }
    coarse[Idx(I,J,K,nx_c,ny_c)] = sum * 0.125;
  }
}

// Piecewise-constant (injection) prolongation: one coarse cell -> 8 fine cells
void MGGravity::Prolongate(const std::vector<Real>& coarse, std::vector<Real>& fine,
                           int nx_c, int ny_c, int nz_c) const {
  const int nx_f = 2*nx_c, ny_f = 2*ny_c;
  std::fill(fine.begin(), fine.end(), 0.0);
  for (int K = 0; K < nz_c; K++)
  for (int J = 0; J < ny_c; J++)
  for (int I = 0; I < nx_c; I++) {
    Real val = coarse[Idx(I,J,K,nx_c,ny_c)];
    for (int dk = 0; dk < 2; dk++)
    for (int dj = 0; dj < 2; dj++)
    for (int di = 0; di < 2; di++) {
      fine[Idx(2*I+di, 2*J+dj, 2*K+dk, nx_f, ny_f)] = val;
    }
  }
}

// V-cycle at the given multigrid level
void MGGravity::VCycle(int lev) {
  const int nx = nx_l_[lev], ny = ny_l_[lev], nz = nz_l_[lev];
  const Real h  = h_l_[lev];

  // Coarsest level: smooth many times and return
  if (lev == n_levels_ - 1) {
    Smooth(phi_[lev], rhs_[lev], nx, ny, nz, h, 20);
    return;
  }

  // Pre-smooth
  Smooth(phi_[lev], rhs_[lev], nx, ny, nz, h, 2);

  // Compute residual
  CalcResidual(phi_[lev], rhs_[lev], res_[lev], nx, ny, nz, h);

  // Restrict residual to coarse-level RHS
  Restrict(res_[lev], rhs_[lev+1], nx, ny, nz);

  // Zero the coarse-level solution (it represents the error correction)
  std::fill(phi_[lev+1].begin(), phi_[lev+1].end(), 0.0);

  // Recurse
  VCycle(lev + 1);

  // Prolongate correction and add to current solution
  const int nx_c = nx_l_[lev+1], ny_c = ny_l_[lev+1], nz_c = nz_l_[lev+1];
  std::vector<Real> correction(nx*ny*nz);
  Prolongate(phi_[lev+1], correction, nx_c, ny_c, nz_c);
  for (int idx = 0; idx < nx*ny*nz; idx++) phi_[lev][idx] += correction[idx];

  // Post-smooth
  Smooth(phi_[lev], rhs_[lev], nx, ny, nz, h, 2);
}
