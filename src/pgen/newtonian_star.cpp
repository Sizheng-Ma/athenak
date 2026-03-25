//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file newtonian_star.cpp
//! \brief Problem generator: Newtonian polytropic star with multigrid self-gravity
//!        and one Lagrangian tracer particle.
//!
//! Self-gravity algorithm (every RK sub-step):
//!   1. Gather density rho from all MeshBlocks onto a single host grid.
//!   2. MPI_Allreduce across ranks.
//!   3. Solve  nabla^2 Phi = 4*pi*G*rho  with Dirichlet BC (Phi=0 on faces)
//!      using n_vcycles V-cycles of red-black Gauss-Seidel multigrid.
//!   4. Compute -grad Phi at each cell centre; add to conserved momenta/energy.
//!
//! Particle:
//!   - Position (px, py, pz) and velocity (pvx, pvy, pvz) stored in StarData.
//!   - Leapfrog integration: at each sub-step, interpolate -grad Phi at the
//!     particle position and kick the velocity; drift the position.
//!   - Particle trajectory printed to a separate history file via user_hist_func.
//!
//! Compile: cmake -D PROBLEM=newtonian_star <source_dir>
//!
//! Input parameters (<problem> block):
//!   rho_c      : central density                    (default 1.0)
//!   K_poly     : polytropic constant                (default 1.0)
//!   G_newt     : Newton's constant in code units    (default 1.0)
//!   rho_atmo   : atmosphere density floor           (default rho_c * 1e-6)
//!   v_pert     : radial velocity perturbation       (default 0.0)
//!   n_vcycles  : multigrid V-cycles per sub-step    (default 4)
//!   px0,py0,pz0        : particle initial position  (default 0.5*R_star, 0, 0)
//!   pvx0,pvy0,pvz0     : particle initial velocity  (default 0)
//!
//! Requires <hydro> block with eos = ideal.

#include <math.h>
#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "outputs/outputs.hpp"
#include "pgen/pgen.hpp"
#include "gravity/mg_gravity.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

// ---------------------------------------------------------------------------
namespace {

struct StarData {
  // Physical parameters
  Real G_newt, gamma, poly_n, rho_c, K_poly;
  Real alpha, R_star, M_star, rho_atmo;

  // Multigrid solver (heap-allocated, owned by this struct)
  MGGravity *mg = nullptr;
  int n_vcycles;

  // Global grid layout (set from Mesh at startup)
  int Nx, Ny, Nz;       // total active cells in each direction
  Real x1min, x2min, x3min;  // global domain origin
  Real h;               // uniform cell spacing

  // Host buffer for density source term (Nx*Ny*Nz)
  std::vector<Real> rho_global;

  // Device array holding the potential (Nx*Ny*Nz, flat), for the apply kernel
  DvceArray1D<Real> d_phi;

  // Lane-Emden IC tables (device)
  int n_le;
  DvceArray1D<Real> d_r_le, d_rho_le, d_prs_le;

  // Particle state (leapfrog; updated on host, applied to device indirectly)
  Real px, py, pz;        // position
  Real pvx, pvy, pvz;     // velocity (half-step leapfrog)
  bool particle_active;

  ~StarData() { delete mg; }
};

StarData star;

} // namespace

// Forward declarations (global scope, matching their definitions below)
static void NewtonianStarGravity(Mesh *pm, const Real bdt);
static void NewtonianStarHistory(HistoryData *pdata, Mesh *pm);

// ---------------------------------------------------------------------------
// Lane-Emden solver (RK4, host)
// ---------------------------------------------------------------------------
static void SolveLaneEmden(Real n_poly,
                           std::vector<Real>& xi_tab,
                           std::vector<Real>& theta_tab,
                           std::vector<Real>& phi_tab) {
  xi_tab.clear(); theta_tab.clear(); phi_tab.clear();
  const Real dxi = 1.0e-3;
  xi_tab.push_back(0.0); theta_tab.push_back(1.0); phi_tab.push_back(0.0);

  Real xi = dxi;
  Real theta = 1.0 - xi*xi/6.0;
  Real phi   = -xi*xi*xi/3.0;
  xi_tab.push_back(xi); theta_tab.push_back(theta); phi_tab.push_back(phi);

  auto rhs = [n_poly](Real xi_, Real theta_, Real phi_)
      -> std::pair<Real,Real> {
    Real dtheta = (xi_ > 1e-12) ? phi_/(xi_*xi_) : 0.0;
    Real th_n   = std::pow(std::max(theta_, 0.0), n_poly);
    return {dtheta, -xi_*xi_*th_n};
  };

  while (theta > 0.0) {
    auto [k1t,k1p] = rhs(xi,          theta,             phi);
    auto [k2t,k2p] = rhs(xi+dxi/2.0,  theta+dxi/2.0*k1t, phi+dxi/2.0*k1p);
    auto [k3t,k3p] = rhs(xi+dxi/2.0,  theta+dxi/2.0*k2t, phi+dxi/2.0*k2p);
    auto [k4t,k4p] = rhs(xi+dxi,      theta+dxi*k3t,      phi+dxi*k3p);
    theta += dxi/6.0*(k1t+2.0*k2t+2.0*k3t+k4t);
    phi   += dxi/6.0*(k1p+2.0*k2p+2.0*k3p+k4p);
    xi    += dxi;
    xi_tab.push_back(xi);
    theta_tab.push_back(std::max(theta, 0.0));
    phi_tab.push_back(phi);
  }
}

// ---------------------------------------------------------------------------
// UserProblem
// ---------------------------------------------------------------------------
void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->phydro == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << "\nNewtonian star requires <hydro> block.\n";
    std::exit(EXIT_FAILURE);
  }

  // ---- Read parameters -------------------------------------------------------
  star.G_newt   = pin->GetOrAddReal("problem", "G_newt",  1.0);
  star.rho_c    = pin->GetOrAddReal("problem", "rho_c",   1.0);
  star.K_poly   = pin->GetOrAddReal("problem", "K_poly",  1.0);
  star.gamma    = pin->GetReal("hydro", "gamma");
  star.poly_n   = 1.0 / (star.gamma - 1.0);
  star.n_vcycles= pin->GetOrAddInteger("problem", "n_vcycles", 4);

  const Real G = star.G_newt, rho_c = star.rho_c, K = star.K_poly;
  const Real Gamma = star.gamma, n = star.poly_n;
  const Real v_pert = pin->GetOrAddReal("problem", "v_pert", 0.0);

  star.alpha = std::sqrt((n+1.0)*K*std::pow(rho_c, Gamma-2.0) / (4.0*M_PI*G));

  // ---- Solve Lane-Emden -------------------------------------------------------
  std::vector<Real> xi_tab, theta_tab, phi_tab;
  SolveLaneEmden(n, xi_tab, theta_tab, phi_tab);
  const int N_le = static_cast<int>(xi_tab.size());
  star.R_star  = star.alpha * xi_tab[N_le-1];
  star.M_star  = -4.0*M_PI*std::pow(star.alpha,3)*rho_c*phi_tab[N_le-1];
  star.rho_atmo= pin->GetOrAddReal("problem", "rho_atmo", rho_c*1.0e-6);

  // ---- Global grid layout ----------------------------------------------------
  const auto &ms  = pmy_mesh_->mesh_size;
  const auto &mi  = pmy_mesh_->mesh_indcs;
  star.Nx     = mi.nx1;  star.Ny = mi.nx2;  star.Nz = mi.nx3;
  star.x1min  = ms.x1min;
  star.x2min  = ms.x2min;
  star.x3min  = ms.x3min;
  star.h      = ms.dx1;   // assumed cubic

  std::cout << "\n### Newtonian Star (multigrid gravity) ###\n"
            << "  gamma     = " << Gamma       << "\n"
            << "  R_star    = " << star.R_star << "\n"
            << "  M_star    = " << star.M_star << "\n"
            << "  G_newt    = " << G           << "\n"
            << "  grid      = " << star.Nx << "x" << star.Ny << "x" << star.Nz << "\n"
            << "  h         = " << star.h      << "\n"
            << "  n_vcycles = " << star.n_vcycles << "\n"
            << "##########################################\n\n";

  // ---- Allocate multigrid solver ----------------------------------------------
  delete star.mg;
  star.mg = new MGGravity(star.Nx, star.Ny, star.Nz, star.h, G);
  star.mg->ZeroPhi();

  star.rho_global.assign(star.Nx * star.Ny * star.Nz, 0.0);
  star.d_phi = DvceArray1D<Real>("d_phi", star.Nx * star.Ny * star.Nz);

  // ---- Lane-Emden IC tables (device) -----------------------------------------
  star.n_le = N_le;
  HostArray1D<Real> h_r_le("h_r_le",N_le), h_rho_le("h_rho_le",N_le),
                    h_prs_le("h_prs_le",N_le);
  for (int i = 0; i < N_le; i++) {
    h_r_le(i)   = star.alpha * xi_tab[i];
    h_rho_le(i) = rho_c * std::pow(theta_tab[i], n);
    h_prs_le(i) = K * std::pow(h_rho_le(i), Gamma);
  }
  star.d_r_le   = DvceArray1D<Real>("d_r_le",  N_le);
  star.d_rho_le = DvceArray1D<Real>("d_rho_le",N_le);
  star.d_prs_le = DvceArray1D<Real>("d_prs_le",N_le);
  Kokkos::deep_copy(star.d_r_le,   h_r_le);
  Kokkos::deep_copy(star.d_rho_le, h_rho_le);
  Kokkos::deep_copy(star.d_prs_le, h_prs_le);

  // ---- Particle initial conditions -------------------------------------------
  star.particle_active = true;
  star.px  = pin->GetOrAddReal("problem", "px0",  0.5*star.R_star);
  star.py  = pin->GetOrAddReal("problem", "py0",  0.0);
  star.pz  = pin->GetOrAddReal("problem", "pz0",  0.0);
  star.pvx = pin->GetOrAddReal("problem", "pvx0", 0.0);
  star.pvy = pin->GetOrAddReal("problem", "pvy0", 0.0);
  star.pvz = pin->GetOrAddReal("problem", "pvz0", 0.0);

  // ---- Enroll source term and history ----------------------------------------
  user_srcs_func = NewtonianStarGravity;
  user_hist      = true;
  user_hist_func = NewtonianStarHistory;

  if (restart) return;

  // ---- Set initial conditions (fluid) ----------------------------------------
  auto &indcs = pmy_mesh_->mb_indcs;
  const int is=indcs.is, ie=indcs.ie, js=indcs.js, je=indcs.je, ks=indcs.ks, ke=indcs.ke;
  auto &size = pmbp->pmb->mb_size;
  const int nmb1 = pmbp->nmb_thispack - 1;
  auto &u0 = pmbp->phydro->u0, &w0 = pmbp->phydro->w0;

  auto d_r   = star.d_r_le, d_rho = star.d_rho_le, d_prs = star.d_prs_le;
  const int n_le_tab = star.n_le;
  const Real R_star = star.R_star, rho_atmo = star.rho_atmo;
  const Real prs_atmo = K * std::pow(rho_atmo, Gamma);

  par_for("pgen_newt_star", DevExeSpace(), 0,nmb1, ks,ke, js,je, is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min; Real &x1max = size.d_view(m).x1max;
    Real x = CellCenterX(i-is, indcs.nx1, x1min, x1max);
    Real &x2min = size.d_view(m).x2min; Real &x2max = size.d_view(m).x2max;
    Real y = CellCenterX(j-js, indcs.nx2, x2min, x2max);
    Real &x3min = size.d_view(m).x3min; Real &x3max = size.d_view(m).x3max;
    Real z = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
    Real r = sqrt(x*x + y*y + z*z);

    Real rho, prs;
    if (r >= R_star) {
      rho = rho_atmo; prs = prs_atmo;
    } else {
      int lo = 0, hi = n_le_tab - 1;
      while (hi - lo > 1) { int mid=(lo+hi)/2; if(d_r(mid)<=r) lo=mid; else hi=mid; }
      Real t = (d_r(hi)-d_r(lo) > 1e-30) ? (r-d_r(lo))/(d_r(hi)-d_r(lo)) : 0.0;
      rho = (1-t)*d_rho(lo) + t*d_rho(hi);
      prs = (1-t)*d_prs(lo) + t*d_prs(hi);
      rho = (rho > rho_atmo) ? rho : rho_atmo;
      prs = K * pow(rho, Gamma);
    }
    Real vr = (v_pert != 0.0 && r < R_star && r > 1e-15)
              ? v_pert * sin(M_PI*r/R_star) : 0.0;
    Real vx = (r>1e-15) ? vr*x/r : 0.0;
    Real vy = (r>1e-15) ? vr*y/r : 0.0;
    Real vz = (r>1e-15) ? vr*z/r : 0.0;

    w0(m,IDN,k,j,i)=rho; w0(m,IVX,k,j,i)=vx; w0(m,IVY,k,j,i)=vy;
    w0(m,IVZ,k,j,i)=vz; w0(m,IPR,k,j,i)=prs;
    u0(m,IDN,k,j,i)=rho;
    u0(m,IM1,k,j,i)=rho*vx; u0(m,IM2,k,j,i)=rho*vy; u0(m,IM3,k,j,i)=rho*vz;
    u0(m,IEN,k,j,i)=prs/(Gamma-1.0) + 0.5*rho*(vx*vx+vy*vy+vz*vz);
  });
}

// ---------------------------------------------------------------------------
// NewtonianStarGravity — user source term called every RK sub-step
// ---------------------------------------------------------------------------
void NewtonianStarGravity(Mesh *pm, const Real bdt) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs  = pm->mb_indcs;
  const int is=indcs.is, ie=indcs.ie, js=indcs.js, je=indcs.je, ks=indcs.ks, ke=indcs.ke;
  auto &mbsize = pmbp->pmb->mb_size;
  const int nmb  = pmbp->nmb_thispack;
  const int nmb1 = nmb - 1;
  auto &w0 = pmbp->phydro->w0;
  auto &u0 = pmbp->phydro->u0;

  const int Nx = star.Nx, Ny = star.Ny, Nz = star.Nz;
  const Real h  = star.h;
  const Real G  = star.G_newt;
  const Real Gamma = star.gamma;

  // ==========================================================================
  // Step 1: Gather density from device -> host global array
  // ==========================================================================
  auto w0_mirror = Kokkos::create_mirror_view(w0);
  Kokkos::deep_copy(w0_mirror, w0);

  std::fill(star.rho_global.begin(), star.rho_global.end(), 0.0);

  const Real x1min = star.x1min, x2min = star.x2min, x3min = star.x3min;
  for (int m = 0; m < nmb; m++) {
    const int I0 = static_cast<int>(std::round(
                     (mbsize.h_view(m).x1min - x1min) / h));
    const int J0 = static_cast<int>(std::round(
                     (mbsize.h_view(m).x2min - x2min) / h));
    const int K0 = static_cast<int>(std::round(
                     (mbsize.h_view(m).x3min - x3min) / h));
    for (int k = ks; k <= ke; k++)
    for (int j = js; j <= je; j++)
    for (int i = is; i <= ie; i++) {
      int I = I0 + (i - is);
      int J = J0 + (j - js);
      int K = K0 + (k - ks);
      star.rho_global[K*Ny*Nx + J*Nx + I] = w0_mirror(m, IDN, k, j, i);
    }
  }

  // ==========================================================================
  // Step 2: MPI reduce (sum contributions from all ranks)
  // ==========================================================================
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, star.rho_global.data(), Nx*Ny*Nz,
                MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif

  // ==========================================================================
  // Step 3: Build source term  S = 4*pi*G*rho  and solve Poisson equation
  // ==========================================================================
  const Real four_pi_G = 4.0 * M_PI * G;
  std::vector<Real> src(Nx*Ny*Nz);
  for (int idx = 0; idx < Nx*Ny*Nz; idx++) src[idx] = four_pi_G * star.rho_global[idx];

  star.mg->SetRHS(src);
  star.mg->Solve(star.n_vcycles);

  // ==========================================================================
  // Step 4: Copy potential to device
  // ==========================================================================
  const auto& phi_host = star.mg->Phi();  // std::vector<Real>, size Nx*Ny*Nz
  {
    HostArray1D<Real> h_phi("h_phi", Nx*Ny*Nz);
    for (int idx = 0; idx < Nx*Ny*Nz; idx++) h_phi(idx) = phi_host[idx];
    Kokkos::deep_copy(star.d_phi, h_phi);
  }

  // ==========================================================================
  // Step 5: Apply -grad Phi to fluid conserved variables  (device kernel)
  // ==========================================================================
  auto d_phi = star.d_phi;

  par_for("apply_mg_grav", DevExeSpace(), 0,nmb1, ks,ke, js,je, is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // Global cell indices
    int I = static_cast<int>((mbsize.d_view(m).x1min - x1min) / h + 0.5) + (i - is);
    int J = static_cast<int>((mbsize.d_view(m).x2min - x2min) / h + 0.5) + (j - js);
    int K = static_cast<int>((mbsize.d_view(m).x3min - x3min) / h + 0.5) + (k - ks);

    // Central-difference gradient with Dirichlet image ghost (phi_ghost = -phi_bnd)
    Real h2 = 2.0 * h;
    Real phiXp = (I < Nx-1) ? d_phi(K*Ny*Nx + J*Nx + (I+1)) : -d_phi(K*Ny*Nx + J*Nx + I);
    Real phiXm = (I > 0)    ? d_phi(K*Ny*Nx + J*Nx + (I-1)) : -d_phi(K*Ny*Nx + J*Nx + I);
    Real phiYp = (J < Ny-1) ? d_phi(K*Ny*Nx + (J+1)*Nx + I) : -d_phi(K*Ny*Nx + J*Nx + I);
    Real phiYm = (J > 0)    ? d_phi(K*Ny*Nx + (J-1)*Nx + I) : -d_phi(K*Ny*Nx + J*Nx + I);
    Real phiZp = (K < Nz-1) ? d_phi((K+1)*Ny*Nx + J*Nx + I) : -d_phi(K*Ny*Nx + J*Nx + I);
    Real phiZm = (K > 0)    ? d_phi((K-1)*Ny*Nx + J*Nx + I) : -d_phi(K*Ny*Nx + J*Nx + I);

    Real ax = -(phiXp - phiXm) / h2;
    Real ay = -(phiYp - phiYm) / h2;
    Real az = -(phiZp - phiZm) / h2;

    Real rho = w0(m, IDN, k, j, i);
    Real vx  = w0(m, IVX, k, j, i);
    Real vy  = w0(m, IVY, k, j, i);
    Real vz  = w0(m, IVZ, k, j, i);

    u0(m, IM1, k, j, i) += bdt * rho * ax;
    u0(m, IM2, k, j, i) += bdt * rho * ay;
    u0(m, IM3, k, j, i) += bdt * rho * az;
    u0(m, IEN, k, j, i) += bdt * rho * (vx*ax + vy*ay + vz*az);
  });

  // ==========================================================================
  // Step 6: Particle leapfrog kick + drift
  // Only do this on one rank (all ranks have the same phi after allreduce)
  // ==========================================================================
  if (!star.particle_active) return;

  // Interpolate -grad Phi at the particle position (trilinear, host-side)
  // Continuous coordinate -> fractional global index
  auto interp_accel = [&](Real px, Real py, Real pz,
                          Real &ax, Real &ay, Real &az) {
    // fractional global index (0-based)
    Real fi = (px - x1min) / h - 0.5;
    Real fj = (py - x2min) / h - 0.5;
    Real fk = (pz - x3min) / h - 0.5;

    int i0 = static_cast<int>(std::floor(fi));
    int j0 = static_cast<int>(std::floor(fj));
    int k0 = static_cast<int>(std::floor(fk));

    Real tx = fi - i0, ty = fj - j0, tz = fk - k0;

    ax = ay = az = 0.0;
    for (int dk = 0; dk <= 1; dk++)
    for (int dj = 0; dj <= 1; dj++)
    for (int di = 0; di <= 1; di++) {
      int I = i0+di, J = j0+dj, K = k0+dk;
      // clamp to valid range
      I = std::max(0, std::min(Nx-1, I));
      J = std::max(0, std::min(Ny-1, J));
      K = std::max(0, std::min(Nz-1, K));

      Real wt = (di ? tx : 1.0-tx) * (dj ? ty : 1.0-ty) * (dk ? tz : 1.0-tz);
      Real lax, lay, laz;
      star.mg->GetAccel(I, J, K, lax, lay, laz);
      ax += wt * lax;  ay += wt * lay;  az += wt * laz;
    }
  };

  Real ax_p, ay_p, az_p;
  interp_accel(star.px, star.py, star.pz, ax_p, ay_p, az_p);

  // Velocity kick (full leapfrog: first half-kick is done at t=0 in IC setup,
  // but here we do a full kick each sub-step, which is exact for uniform accel)
  star.pvx += bdt * ax_p;
  star.pvy += bdt * ay_p;
  star.pvz += bdt * az_p;

  // Position drift
  star.px += bdt * star.pvx;
  star.py += bdt * star.pvy;
  star.pz += bdt * star.pvz;
}

// ---------------------------------------------------------------------------
// NewtonianStarHistory — output particle trajectory to history file
// ---------------------------------------------------------------------------
void NewtonianStarHistory(HistoryData *pdata, Mesh *pm) {
  // Append 6 custom columns: px, py, pz, pvx, pvy, pvz
  pdata->nhist = 6;
  pdata->label[0] = "px";   pdata->hdata[0] = star.px;
  pdata->label[1] = "py";   pdata->hdata[1] = star.py;
  pdata->label[2] = "pz";   pdata->hdata[2] = star.pz;
  pdata->label[3] = "pvx";  pdata->hdata[3] = star.pvx;
  pdata->label[4] = "pvy";  pdata->hdata[4] = star.pvy;
  pdata->label[5] = "pvz";  pdata->hdata[5] = star.pvz;
}
