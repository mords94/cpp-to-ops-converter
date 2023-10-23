#define malloc(x) NULL

#include "pom2k_c_header.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define ACC2_jk(j, k) (j) + (k)*jm

int im;
int jm;
int kb;
int imm1;
int jmm1;
int kbm1;
int imm2;
int jmm2;
int kbm2;

float alpha, dte, dte2, dti, dti2;
float grav, hmax, pi;
float ramp, rfe, rfn, rfs;
float rfw, rhoref, sbias, slmax;
float small, tbias, time2, tprni;
float umol, vmaxl;
float horcon;

float period, time0, time1;

const float kappa = 0.4f;    // von Karman's constant
const float z0b = .01f;      // Bottom roughness (metres)
const float cbcmin = .0025f; // Minimum bottom friction coeff.
const float cbcmax = 1.0f;   // Maximum bottom friction coeff.

int iint, iprint, iskp, jskp;
int kl1, kl2, mode, ntp;
int kbm3;
int kb;

void example(float *in1, float *in2, float *out1, float *out2, int arg1) {
  float a = 13.0f;
  float b = 1.2f;
  for (int k = kb - 3; k > 0; k--) {
    for (int j = 0; j < jm + 1; j++) {
      for (int i = 0; i < im; i++) {
        out1[ACC3(i, j, k)] = out1[ACC3(i, j, k - 1)] * a;
        out2[ACC3(i, j, k)] = out2[ACC3(i, j, k - 1)] * b;
        a += 0.1f;
      }
    }
  }
}

void example_locals(float *out, float *in) {
  float local = 0.0f;
  float local2 = 0.1f;

  for (int j = 0; j < jm; j++) {
    for (int i = 0; i < im; i++) {
      if (fabs(in[ACC2(i, j)]) < local2) {
        local = fabs(in[ACC2(i, j)]);
        local2 = fabs(in[ACC2(i, j)]) * 2.0f;
      }
    }
  }

  *out = local;
}

void ext_comp_vamax_(float *_vamax, float *vaf, int *_imax, int *_jmax) {
  float vamax = 0.0f;
  float bla = *_vamax;
  int imax;
  int jmaxxx;

  for (int j = 0; j < jm; j++) {
    for (int i = 0; i < im; i++) {
      if (fabs(vaf[ACC2(i, j)] * bla) > vamax) {
        vamax = fabs(vaf[ACC2(i, j)]);
        imax = i;
        jmaxxx = j;
      }
    }
  }
  *_vamax = vamax;
  *_imax = imax;
  *_jmax = jmaxxx;
}

// for
// i = 11 to kbm3

void ext_profq_(float *sm, float *sh, float *dh, float *cc, float *h,
                float *etf, float *a, float *c, float *kq, float *dz,
                float *dzz, float *ee, float *gg, float *wusurf, float *wvsurf,
                float *uf, float *wubot, float *wvbot, float *t, float *s,
                float *zz, float *q2b, float *q2lb, float *l, float *z,
                float *km, float *u, float *v, float *kh, float *vf, float *fsm,
                float *q2, float *dt, float *rho, float *dtef, float *l0,
                float *gh, float *boygr, float *stf, float *prod, float *zmin,
                float *zmax) {

  float a1 = 0.92f;
  float a2 = 0.74f;
  float b1 = 16.6f;
  float b2 = 10.1f;
  float c1 = 0.08f;
  float e1 = 1.8f;
  float e2 = 1.33f;
  float sef = 1.0f;
  float cbcnst = 100.0f;
  float surfl = 2e5;
  float shiw = 0.0f;
  float coef1, coef2, coef3 = .99, coef4, coef5;
  float const1, ghc;
  float p, sp, tp;
  float utau2, df0, df1, df2;
  //   float * l0;
  //   float * gh;
  //   float * boygr;
  //   float * stf;
  //   float * prod;
  //   float * kn;

  //   l0 = (float*)malloc(im * jm * sizeof(float));
  //   gh = (float*)malloc(im * jm * kb * sizeof(float));
  //   boygr = (float*)malloc(im * jm * kb * sizeof(float));
  //   stf = (float*)malloc(im * jm * kb * sizeof(float));

  //   prod = kn = (float*)malloc(im * jm * kb * sizeof(float));
  //  kétszer kéne dtef-et hívni???
  //  float * dtef = cc;

  /*
    union {
      float prod[im*jm*kb];
      float kn[im*jm*kb];
    } equivalence;  //equivalence (prod,kn)
    */

  /*
        real sm(im,jm,kb),sh(im,jm,kb),cc(im,jm,kb)
        real gh(im,jm,kb),boygr(im,jm,kb),dh(im,jm),stf(im,jm,kb)
        real prod(im,jm,kb),kn(im,jm,kb)
        real a1,a2,b1,b2,c1
        real coef1,coef2,coef3,coef4,coef5
        real const1,e1,e2,ghc
        real p,sef,sp,tp
        real l0(im,jm)
        real cbcnst,surfl,shiw
        real utau2, df0,df1,df2
  C
        integer i,j,k,ki
  C
        equivalence (prod,kn)
  C
        data a1,b1,a2,b2,c1/0.92e0,16.6e0,0.74e0,10.1e0,0.08e0/
        data e1/1.8e0/,e2/1.33e0/
        data sef/1.e0/
        data cbcnst/100./surfl/2.e5/shiw/0.0/
  */
  /*       do j=1,jm
          do i=1,im
            dh(i,j)=h(i,j)+etf(i,j)
          end do
        end do
        */
  //   for (int j = 0; j < jm; j++) {
  //     for (int i = 0; i < im; i++) {
  //       dh[ACC2(i, j)] = h[ACC2(i, j)] + etf[ACC2(i, j)];
  //     }
  //   }

  //   /*       do k=2,kbm1
  //           do j=1,jm
  //             do i=1,im
  //               a(i,j,k)=-dti2*(kq(i,j,k+1)+kq(i,j,k)+2.e0*umol)*.5e0
  //        $                /(dzz(k-1)*dz(k)*dh(i,j)*dh(i,j))
  //               c(i,j,k)=-dti2*(kq(i,j,k-1)+kq(i,j,k)+2.e0*umol)*.5e0
  //        $                /(dzz(k-1)*dz(k-1)*dh(i,j)*dh(i,j))
  //             end do
  //           end do
  //         end do */
  //   for (int k = 1; k < kbm1; k++) {
  //     for (int j = 0; j < jm; j++) {
  //       for (int i = 0; i < im; i++) {
  //         a[ACC3(i, j, k)] =
  //             -dti2 * (kq[ACC3(i, j, k + 1)] + kq[ACC3(i, j, k)] + 2.0f *
  //             umol) * 0.5f / (dzz[k - 1] * dz[k] * dh[ACC2(i, j)] *
  //             dh[ACC2(i, j)]);
  //         c[ACC3(i, j, k)] =
  //             -dti2 * (kq[ACC3(i, j, k - 1)] + kq[ACC3(i, j, k)] + 2.0f *
  //             umol) * 0.5f / (dzz[k - 1] * dz[k - 1] * dh[ACC2(i, j)] *
  //             dh[ACC2(i, j)]);
  //       }
  //     }
  //   }

  //   // C
  //   //
  //   C-----------------------------------------------------------------------
  //   // C
  //   // C     The following section solves the equation:
  //   // C
  //   // C       dti2*(kq*q2')' - q2*(2.*dti2*dtef+1.) = -q2b
  //   // C
  //   // C     Surface and bottom boundary conditions:
  //   // C
  //   const1 = (powf(16.6f, (2.0f / 3.0f))) * sef;
  //   /*       const1=(16.6e0**(2.e0/3.e0))*sef */
  //   // C
  //   // C initialize fields that are not calculated on all boundaries
  //   // C but are later used there
  //   /*       do i=1,im
  //           ee(i,jm,1)=0.
  //           gg(i,jm,1)=0.
  //           l0(i,jm)=0.
  //         end do
  //         do j=1,jm
  //           ee(im,j,1)=0.
  //           gg(im,j,1)=0.
  //           l0(im,j)=0.
  //         end do
  //         do i=1,im
  //         do j=1,jm
  //          do k=2,kbm1
  //           prod(i,j,k)=0.
  //          end do
  //         end do
  //         end do */
  //   for (int i = 0; i < im; i++) {
  //     ee[ACC3(i, jm - 1, 0)] =
  //         0.0f; // A k tag C-ben 0-tól számozódik, F-ben 1-től, ezért k az 0
  //         itt
  //     gg[ACC3(i, jm - 1, 0)] =
  //         0.0f; // A k tag C-ben 0-tól számozódik, F-ben 1-től, ezért k az 0
  //         itt
  //     l0[ACC2(i, jm - 1)] = 0.0f;
  //   }
  //   for (int j = 0; j < jm; j++) {
  //     ee[ACC3(im - 1, j, 0)] =
  //         0.0f; // A k tag C-ben 0-tól számozódik, F-ben 1-től, ezért k az 0
  //         itt
  //     gg[ACC3(im - 1, j, 0)] =
  //         0.0f; // A k tag C-ben 0-tól számozódik, F-ben 1-től, ezért k az 0
  //         itt
  //     l0[ACC2(im - 1, j)] = 0.0f;
  //   }
  //   for (int i = 0; i < im; i++) {
  //     for (int j = 0; j < jm; j++) {
  //       for (int k = 1; k < kbm1; k++) {
  //         prod[ACC3(i, j, k)] = 0.0f;
  //       }
  //     }
  //   }

  //   /*       do j=1,jmm1
  //           do i=1,imm1
  //             utau2=sqrt((.5e0*(wusurf(i,j)+wusurf(i+1,j)))**2
  //        $                  +(.5e0*(wvsurf(i,j)+wvsurf(i,j+1)))**2) */
  //   for (int j = 0; j < jmm1; j++) {
  //     for (int i = 0; i < imm1; i++) {
  //       utau2 = sqrtf((0.5f * (wusurf[ACC2(i, j)] + wusurf[ACC2(i + 1, j)]))
  //       *
  //                         (0.5f * (wusurf[ACC2(i, j)] + wusurf[ACC2(i + 1,
  //                         j)])) +
  //                     (0.5f * (wvsurf[ACC2(i, j)] + wvsurf[ACC2(i, j + 1)]))
  //                     *
  //                         (0.5f * (wvsurf[ACC2(i, j)] + wvsurf[ACC2(i, j +
  //                         1)])));

  //       // C Wave breaking energy- a variant of Craig & Banner (1994)
  //       // C see Mellor and Blumberg, 2003.
  //       /*           ee(i,j,1)=0.e0
  //                 gg(i,j,1)=(15.8*cbcnst)**(2./3.)*utau2  */
  //       ee[ACC3(i, j, 0)] =
  //           0.0f; // A k tag C-ben 0-tól számozódik, F-ben 1-től, ezért k az
  //           0 itt
  //       gg[ACC3(i, j, 0)] = powf(15.8f * cbcnst, (2.0f / 3.0f)) *
  //                           utau2; // A k tag C-ben 0-tól számozódik, F-ben
  //                           1-től,
  //                                  // ezért k az 0 itt

  //       // C Surface length scale following Stacey (1999).
  //       /*           l0(i,j)=surfl*utau2/grav */
  //       l0[ACC2(i, j)] = surfl * utau2 / grav;
  //       /*           uf(i,j,kb)=sqrt((.5e0*(wubot(i,j)+wubot(i+1,j)))**2
  //            $                   +(.5e0*(wvbot(i,j)+wvbot(i,j+1)))**2)*const1
  //               end do
  //             end do */
  //       uf[ACC3(i, j, kb - 1)] =
  //           sqrtf((0.5f * (wubot[ACC2(i, j)] + wubot[ACC2(i + 1, j)])) *
  //                     (0.5f * (wubot[ACC2(i, j)] + wubot[ACC2(i + 1, j)])) +
  //                 (0.5f * (wvbot[ACC2(i, j)] + wvbot[ACC2(i, j + 1)])) *
  //                     (0.5f * (wvbot[ACC2(i, j)] + wvbot[ACC2(i, j + 1)]))) *
  //           const1;
  //     }
  //   }

  //   // C
  //   // C    Calculate speed of sound squared:
  //   // C
  //   /*       do k=1,kbm1
  //           do j=1,jm
  //             do i=1,im
  //               tp=t(i,j,k)+tbias
  //               sp=s(i,j,k)+sbias */
  //   for (int k = 0; k < kbm1; k++) {
  //     for (int j = 0; j < jm; j++) {
  //       for (int i = 0; i < im; i++) {
  //         tp = t[ACC3(i, j, k)] + tbias;
  //         sp = s[ACC3(i, j, k)] + sbias;

  //         // C
  //         // C     Calculate pressure in units of decibars:
  //         // C
  //         /*             p=grav*rhoref*(-zz(k)* h(i,j))*1.e-4
  //                     cc(i,j,k)=1449.1e0+.00821e0*p+4.55e0*tp -.045e0*tp**2
  //              $                 +1.34e0*(sp-35.0e0)
  //                     cc(i,j,k)=cc(i,j,k)
  //              $                 /sqrt((1.e0-.01642e0*p/cc(i,j,k))
  //              $                   *(1.e0-0.40e0*p/cc(i,j,k)**2))
  //                   end do
  //                 end do
  //               end do */
  //         p = grav * rhoref * (-zz[k] * h[ACC2(i, j)]) * (0.0001f);
  //         cc[ACC3(i, j, k)] = 1449.1f + 0.00821f * p + 4.55f * tp -
  //                             0.045f * tp * tp + 1.34f * (sp - 35.0f);
  //         cc[ACC3(i, j, k)] =
  //             cc[ACC3(i, j, k)] /
  //             sqrtf((1.0f - 0.01642f * p / cc[ACC3(i, j, k)]) *
  //                   (1.0f -
  //                    0.4f * p / ((cc[ACC3(i, j, k)]) * (cc[ACC3(i, j,
  //                    k)]))));
  //       }
  //     }
  //   }
  //   // C
  //   // C     Calculate buoyancy gradient:
  //   // C
  //   /*       do k=2,kbm1
  //           do j=1,jm
  //             do i=1,im
  //               q2b(i,j,k)=abs(q2b(i,j,k))
  //               q2lb(i,j,k)=abs(q2lb(i,j,k))
  //               boygr(i,j,k)=grav*(rho(i,j,k-1)-rho(i,j,k))
  //        $                    /(dzz(k-1)* h(i,j)) */
  //   for (int k = 1; k < kbm1; k++) {
  //     for (int j = 0; j < jm; j++) {
  //       for (int i = 0; i < im; i++) {
  //         q2b[ACC3(i, j, k)] = fabsf(q2b[ACC3(i, j, k)]);
  //         q2lb[ACC3(i, j, k)] = fabsf(q2lb[ACC3(i, j, k)]);
  //         boygr[ACC3(i, j, k)] =
  //             grav * (rho[ACC3(i, j, k - 1)] - rho[ACC3(i, j, k)]) /
  //                 (dzz[k - 1] * h[ACC2(i, j)])

  //             // C *** NOTE: comment out next line if dens does not include
  //             // pressure
  //             /*      $      +(grav**2)*2.e0/(cc(i,j,k-1)**2+cc(i,j,k)**2)
  //                       end do
  //                     end do
  //                   end do */
  //             + (grav * grav) * 2.0f /
  //                   ((cc[ACC3(i, j, k - 1)] * cc[ACC3(i, j, k - 1)]) +
  //                    cc[ACC3(i, j, k)] * cc[ACC3(i, j, k)]);
  //       }
  //     }
  //   }
  //   /*       do k=2,kbm1
  //         do j=1,jm
  //           do i=1,im
  //             l(i,j,k)=abs(q2lb(i,j,k)/q2b(i,j,k))
  //             if(z(k).gt.-0.5) l(i,j,k)=max(l(i,j,k),kappa*l0(i,j))
  //             gh(i,j,k)=(l(i,j,k)**2)*boygr(i,j,k)/q2b(i,j,k)
  //             gh(i,j,k)=min(gh(i,j,k),.028e0)
  //           end do
  //         end do
  //       end do */
  //   for (int k = 1; k < kbm1; k++) {
  //     for (int j = 0; j < jm; j++) {
  //       for (int i = 0; i < im; i++) {
  //         l[ACC3(i, j, k)] = fabsf(q2lb[ACC3(i, j, k)] / q2b[ACC3(i, j, k)]);
  //         if (z[k] > -0.5f)
  //           l[ACC3(i, j, k)] = fmaxf(l[ACC3(i, j, k)], kappa * l0[ACC2(i,
  //           j)]);
  //         gh[ACC3(i, j, k)] = (l[ACC3(i, j, k)] * l[ACC3(i, j, k)]) *
  //                             boygr[ACC3(i, j, k)] / q2b[ACC3(i, j, k)];
  //         gh[ACC3(i, j, k)] = fminf(gh[ACC3(i, j, k)], 0.028f);
  //       }
  //     }
  //   }
  //   /*      do j=1,jm
  //           do i=1,im
  //             l(i,j,1)=kappa*l0(i,j)
  //             l(i,j,kb)=0.e0
  //             gh(i,j,1)=0.e0
  //             gh(i,j,kb)=0.e0
  //           end do
  //         end do */
  //   for (int j = 0; j < jm; j++) {
  //     for (int i = 0; i < im; i++) {
  //       l[ACC3(i, j, 0)] =
  //           kappa * l0[ACC2(i, j)]; // A k tag C-ben 0-tól számozódik, F-ben
  //                                   // 1-től, ezért k az 0 itt
  //       l[ACC3(i, j, kb - 1)] = 0.0f;
  //       gh[ACC3(i, j, 0)] =
  //           0.0f; // A k tag C-ben 0-tól számozódik, F-ben 1-től, ezért k az
  //           0 itt
  //       gh[ACC3(i, j, kb - 1)] = 0.0f;
  //     }
  //   }

  //   // C
  //   // C    Calculate production of turbulent kinetic energy:
  //   // C
  //   /*       do k=2,kbm1
  //           do j=2,jmm1
  //             do i=2,imm1
  //               prod(i,j,k)=km(i,j,k)*.25e0*sef
  //        $                   *((u(i,j,k)-u(i,j,k-1)
  //        $                      +u(i+1,j,k)-u(i+1,j,k-1))**2
  //        $                     +(v(i,j,k)-v(i,j,k-1)
  //        $                      +v(i,j+1,k)-v(i,j+1,k-1))**2)
  //        $                   /(dzz(k-1)*dh(i,j))**2 */
  //   for (int k = 1; k < kbm1; k++) {
  //     for (int j = 1; j < jmm1; j++) {
  //       for (int i = 1; i < imm1; i++) {
  //         prod[ACC3(i, j, k)] =
  //             km[ACC3(i, j, k)] * 0.25f * sef *
  //                 ((u[ACC3(i, j, k)] - u[ACC3(i, j, k - 1)] +
  //                   u[ACC3(i + 1, j, k)] - u[ACC3(i + 1, j, k - 1)]) *
  //                      (u[ACC3(i, j, k)] - u[ACC3(i, j, k - 1)] +
  //                       u[ACC3(i + 1, j, k)] - u[ACC3(i + 1, j, k - 1)]) +
  //                  (v[ACC3(i, j, k)] - v[ACC3(i, j, k - 1)] +
  //                   v[ACC3(i, j + 1, k)] - v[ACC3(i, j + 1, k - 1)]) *
  //                      (v[ACC3(i, j, k)] - v[ACC3(i, j, k - 1)] +
  //                       v[ACC3(i, j + 1, k)] - v[ACC3(i, j + 1, k - 1)])) /
  //                 ((dzz[k - 1] * dh[ACC2(i, j)]) * (dzz[k - 1] * dh[ACC2(i,
  //                 j)]))
  //             // C   Add shear due to internal wave field
  //             /*      $             -shiw*km(i,j,k)*boygr(i,j,k)
  //                         prod(i,j,k)=prod(i,j,k)+kh(i,j,k)*boygr(i,j,k)
  //                       end do
  //                     end do
  //                   end do */
  //             - shiw * km[ACC3(i, j, k)] * boygr[ACC3(i, j, k)];
  //         prod[ACC3(i, j, k)] =
  //             prod[ACC3(i, j, k)] + kh[ACC3(i, j, k)] * boygr[ACC3(i, j, k)];
  //       }
  //     }
  //   }
  //   // C
  //   // C  NOTE: Richardson # dep. dissipation correction (Mellor, 2001; Ezer,
  //   // 2000), C  depends on ghc the critical number (empirical -6 to -2) to
  //   // increase mixing.
  //   /*       ghc=-6.0e0
  //         do k=1,kb
  //           do j=1,jm
  //             do i=1,im
  //               stf(i,j,k)=1.e0 */
  //   ghc = -6.0f;
  //   for (int k = 0; k < kb; k++) {
  //     for (int j = 0; j < jm; j++) {
  //       for (int i = 0; i < im; i++) {
  //         stf[ACC3(i, j, k)] = 1.0f;

  //         // C It is unclear yet if diss. corr. is needed when surf. waves
  //         are
  //         // included. c           if(gh(i,j,k).lt.0.e0) c    $
  //         // stf(i,j,k)=1.0e0-0.9e0*(gh(i,j,k)/ghc)**1.5e0 c
  //         if(gh(i,j,k).lt.ghc)
  //         // stf(i,j,k)=0.1e0
  //         /*             dtef(i,j,k)=sqrt(abs(q2b(i,j,k)))*stf(i,j,k)
  //              $                   /(b1*l(i,j,k)+small)
  //                   end do
  //                 end do
  //               end do */
  //         dtef[ACC3(i, j, k)] = sqrtf(fabsf(q2b[ACC3(i, j, k)])) *
  //                               stf[ACC3(i, j, k)] /
  //                               (b1 * l[ACC3(i, j, k)] + small);
  //       }
  //     }
  //   }
  //   /*       do k=2,kbm1
  //         do j=1,jm
  //           do i=1,im
  //             gg(i,j,k)=1.e0/(a(i,j,k)+c(i,j,k)*(1.e0-ee(i,j,k-1))
  //      $                      -(2.e0*dti2*dtef(i,j,k)+1.e0))
  //             ee(i,j,k)=a(i,j,k)*gg(i,j,k)
  //             gg(i,j,k)=(-2.e0*dti2*prod(i,j,k)+c(i,j,k)*gg(i,j,k-1)
  //      $                 -uf(i,j,k))*gg(i,j,k)
  //           end do
  //         end do
  //       end do
  //  */
  //   for (int k = 1; k < kbm1; k++) {
  //     for (int j = 0; j < jm; j++) {
  //       for (int i = 0; i < im; i++) {
  //         gg[ACC3(i, j, k)] =
  //             1.0f / (a[ACC3(i, j, k)] +
  //                     c[ACC3(i, j, k)] * (1.0f - ee[ACC3(i, j, k - 1)]) -
  //                     (2.0f * dti2 * dtef[ACC3(i, j, k)] + 1.0f));
  //         ee[ACC3(i, j, k)] = a[ACC3(i, j, k)] * gg[ACC3(i, j, k)];
  //         gg[ACC3(i, j, k)] =
  //             (-2.0f * dti2 * prod[ACC3(i, j, k)] +
  //              c[ACC3(i, j, k)] * gg[ACC3(i, j, k - 1)] - uf[ACC3(i, j, k)])
  //              *
  //             gg[ACC3(i, j, k)];
  //       }
  //     }
  //   }

  //   /*       do k=1,kbm1
  //           ki=kb-k
  //           do j=1,jm
  //             do i=1,im
  //               uf(i,j,ki)=ee(i,j,ki)*uf(i,j,ki+1)+gg(i,j,ki)
  //             end do
  //           end do
  //         end do */
  //   for (int k = kb - 1; k >= 0; k--) {
  //     for (int j = 0; j < jm; j++) {
  //       for (int i = 0; i < im; i++) {
  //         uf[ACC3(i, j, k)] =
  //             ee[ACC3(i, j, k)] * uf[ACC3(i, j, k + 1)] + gg[ACC3(i, j, k)];
  //       }
  //     }
  //   }

  //   // C
  //   //
  //   C-----------------------------------------------------------------------
  //   // C
  //   // C     The following section solves the equation:
  //   // C
  //   // C       dti2(kq*q2l')' - q2l*(dti2*dtef+1.) = -q2lb
  //   /*       do j=1,jm
  //           do i=1,im
  //             ee(i,j,2)=0.e0
  //             gg(i,j,2)=0.e0
  //             vf(i,j,kb)=0.e0
  //           end do
  //         end do */
  //   for (int j = 0; j < jm; j++) {
  //     for (int i = 0; i < im; i++) {
  //       ee[ACC3(i, j, 1)] =
  //           0.0f; // A k tag C-ben 0-tól számozódik, F-ben 1-től, ezért k az
  //           1 itt
  //       gg[ACC3(i, j, 1)] =
  //           0.0f; // A k tag C-ben 0-tól számozódik, F-ben 1-től, ezért k az
  //           1 itt
  //       vf[ACC3(i, j, kb - 1)] = 0.0f;
  //     }
  //   }

  //   /*       do k=2,kbm1
  //           do j=1,jm
  //             do i=1,im
  //               dtef(i,j,k)=dtef(i,j,k)
  //        $                   *(1.e0+e2*((1.e0/abs(z(k)-z(1))
  //        $                               +1.e0/abs(z(k)-z(kb)))
  //        $                                *l(i,j,k)/(dh(i,j)*kappa))**2)
  //               gg(i,j,k)=1.e0/(a(i,j,k)+c(i,j,k)*(1.e0-ee(i,j,k-1))
  //        $                      -(dti2*dtef(i,j,k)+1.e0))
  //               ee(i,j,k)=a(i,j,k)*gg(i,j,k)
  //               gg(i,j,k)=(dti2*(-prod(i,j,k)*l(i,j,k)*e1)
  //        $                 +c(i,j,k)*gg(i,j,k-1)-vf(i,j,k))*gg(i,j,k)
  //             end do
  //           end do
  //         end do */
  //   for (int k = 1; k < kbm1; k++) {
  //     for (int j = 0; j < jm; j++) {
  //       for (int i = 0; i < im; i++) {
  //         dtef[ACC3(i, j, k)] =
  //             dtef[ACC3(i, j, k)] *
  //             (1.0f +
  //              e2 *
  //                  ((1.0f / fabsf(z[k] - *zmin) + 1.0f / fabsf(z[k] - *zmax))
  //                  *
  //                   l[ACC3(i, j, k)] / (dh[ACC2(i, j)] * kappa)) *
  //                  ((1.0f / fabsf(z[k] - *zmin) + 1.0f / fabsf(z[k] - *zmax))
  //                  *
  //                   l[ACC3(i, j, k)] / (dh[ACC2(i, j)] * kappa)));
  //         gg[ACC3(i, j, k)] =
  //             1.0f / (a[ACC3(i, j, k)] +
  //                     c[ACC3(i, j, k)] * (1.0f - ee[ACC3(i, j, k - 1)]) -
  //                     (dti2 * dtef[ACC3(i, j, k)] + 1.0f));
  //         ee[ACC3(i, j, k)] = a[ACC3(i, j, k)] * gg[ACC3(i, j, k)];
  //         gg[ACC3(i, j, k)] =
  //             (dti2 * (-prod[ACC3(i, j, k)] * l[ACC3(i, j, k)] * e1) +
  //              c[ACC3(i, j, k)] * gg[ACC3(i, j, k - 1)] - vf[ACC3(i, j, k)])
  //              *
  //             gg[ACC3(i, j, k)];
  //       }
  //     }
  //   }

  //   /*       do k=1,kb-2
  //           ki=kb-k
  //           do j=1,jm
  //             do i=1,im
  //               vf(i,j,ki)=ee(i,j,ki)*vf(i,j,ki+1)+gg(i,j,ki)
  //             end do
  //           end do
  //         end do */
  //   for (int k = kb - 2; k >= 0; k--) {
  //     for (int j = 0; j < jm; j++) {
  //       for (int i = 0; i < im; i++) {
  //         vf[ACC3(i, j, k)] =
  //             ee[ACC3(i, j, k)] * vf[ACC3(i, j, k + 1)] + gg[ACC3(i, j, k)];
  //       }
  //     }
  //   }

  //   /*       do k=2,kbm1
  //           do j=1,jm
  //             do i=1,im
  //               if(uf(i,j,k).le.small.or.vf(i,j,k).le.small) then
  //                 uf(i,j,k)=small
  //                 vf(i,j,k)=0.1*dt(i,j)*small
  //               endif
  //             end do
  //           end do
  //         end do */
  //   for (int k = 1; k < kbm1; k++) {
  //     for (int j = 0; j < jm; j++) {
  //       for (int i = 0; i < im; i++) {
  //         if ((uf[ACC3(i, j, k)] <= small) || (vf[ACC3(i, j, k)] <= small)) {
  //           uf[ACC3(i, j, k)] = small;
  //           vf[ACC3(i, j, k)] = 0.1f * dt[ACC2(i, j)] * small;
  //         }
  //       }
  //     }
  //   }

  // C
  // C-----------------------------------------------------------------------
  // C
  // C     The following section solves for km and kh:
  // C
  /*       coef4=18.e0*a1*a1+9.e0*a1*a2
        coef5=9.e0*a1*a2 */
  coef4 = 18.0f * a1 * a1 + 9.0f * a1 * a2;
  coef5 = 9.0f * a1 * a2;

  // C
  // C     Note that sm and sh limit to infinity when gh approaches 0.0288:
  // C
  /*       do k=1,kb
          do j=1,jm
            do i=1,im
              coef1=a2*(1.e0-6.e0*a1/b1*stf(i,j,k))
              coef2=3.e0*a2*b2/stf(i,j,k)+18.e0*a1*a2
              coef3=a1*(1.e0-3.e0*c1-6.e0*a1/b1*stf(i,j,k))
              sh(i,j,k)=coef1/(1.e0-coef2*gh(i,j,k))
              sm(i,j,k)=coef3+sh(i,j,k)*coef4*gh(i,j,k)
              sm(i,j,k)=sm(i,j,k)/(1.e0-coef5*gh(i,j,k))
            end do
          end do
        end do */
  for (int k = 0; k < kb; k++) {
    for (int j = 0; j < jm; j++) {
      for (int i = 0; i < im; i++) {
        // coef1 = a2 * (1.0f - 6.0f * a1 / b1 * stf[ACC3(i, j, k)]);
        // coef2 = 3.0f * a2 * b2 / stf[ACC3(i, j, k)] + 18.0f * a1 * a2;
        coef3 = coe3 * b1 * stf[ACC3(i, j, k)];
        sm[ACC3(i, j, k)] = coef3 + sh[ACC3(i, j, k)];
        // sm[ACC3(i, j, k)] =
        // sm[ACC3(i, j, k)] / (1.0f - coef5 * gh[ACC3(i, j, k)]);
      }
    }
  }

  // /*       do k=1,kb
  //         do j=1,jm
  //           do i=1,im
  //             kn(i,j,k)=l(i,j,k)*sqrt(abs(q2(i,j,k)))
  //             kq(i,j,k)=(kn(i,j,k)*.41e0*sh(i,j,k)+kq(i,j,k))*.5e0
  //             km(i,j,k)=(kn(i,j,k)*sm(i,j,k)+km(i,j,k))*.5e0
  //             kh(i,j,k)=(kn(i,j,k)*sh(i,j,k)+kh(i,j,k))*.5e0
  //           end do
  //         end do
  //       end do */
  // for (int k = 0; k < kb; k++) {
  //   for (int j = 0; j < jm; j++) {
  //     for (int i = 0; i < im; i++) {
  //       prod[ACC3(i, j, k)] =
  //           l[ACC3(i, j, k)] * sqrtf(fabsf(q2[ACC3(i, j, k)]));
  //       kq[ACC3(i, j, k)] = (prod[ACC3(i, j, k)] * 0.41f * sh[ACC3(i, j, k)]
  //       +
  //                            kq[ACC3(i, j, k)]) *
  //                           0.5f;
  //       km[ACC3(i, j, k)] =
  //           (prod[ACC3(i, j, k)] * sm[ACC3(i, j, k)] + km[ACC3(i, j, k)]) *
  //           0.5f;
  //       kh[ACC3(i, j, k)] =
  //           (prod[ACC3(i, j, k)] * sh[ACC3(i, j, k)] + kh[ACC3(i, j, k)]) *
  //           0.5f;
  //     }
  //   }
  // }

  // // C cosmetics: make boundr. values as interior
  // // C (even if not used, printout otherwise may show strange values)
  // /*       do k=1,kb
  //         do i=1,im
  //            km(i,jm,k)=km(i,jmm1,k)*fsm(i,jm)
  //            kh(i,jm,k)=kh(i,jmm1,k)*fsm(i,jm)
  //            km(i,1,k)=km(i,2,k)*fsm(i,1)
  //            kh(i,1,k)=kh(i,2,k)*fsm(i,1)
  //         end do
  //         do j=1,jm
  //            km(im,j,k)=km(imm1,j,k)*fsm(im,j)
  //            kh(im,j,k)=kh(imm1,j,k)*fsm(im,j)
  //            km(1,j,k)=km(2,j,k)*fsm(1,j)
  //            kh(1,j,k)=kh(2,j,k)*fsm(1,j)
  //         end do
  //       end do
  //  */
  // for (int k = 0; k < kb; k++) {
  //   for (int i = 0; i < im; i++) {
  //     km[ACC3(i, jm - 1, k)] = km[ACC3(i, jmm1 - 1, k)] * fsm[ACC2(i, jm -
  //     1)]; kh[ACC3(i, jm - 1, k)] = kh[ACC3(i, jmm1 - 1, k)] * fsm[ACC2(i, jm
  //     - 1)]; km[ACC3(i, 0, k)] =
  //         km[ACC3(i, 1, k)] *
  //         fsm[ACC2(i, 0)]; // j 0-tól számozódik C-ben, nem 1-től
  //     kh[ACC3(i, 0, k)] =
  //         kh[ACC3(i, 1, k)] *
  //         fsm[ACC2(i, 0)]; // j 0-tól számozódik C-ben, nem 1-től
  //   }
  // }

  // for (int k = 0; k < kb; k++) {
  //   for (int j = 0; j < jm; j++) {
  //     km[ACC3(im - 1, j, k)] = km[ACC3(imm1 - 1, j, k)] * fsm[ACC2(im - 1,
  //     j)]; kh[ACC3(im - 1, j, k)] = kh[ACC3(imm1 - 1, j, k)] * fsm[ACC2(im -
  //     1, j)]; km[ACC3(0, j, k)] =
  //         km[ACC3(1, j, k)] *
  //         fsm[ACC2(0, j)]; // i 0-tól számozódik C-ben, nem 1-től
  //     kh[ACC3(0, j, k)] =
  //         kh[ACC3(1, j, k)] *
  //         fsm[ACC2(0, j)]; // i 0-tól számozódik C-ben, nem 1-től
  //   }
  // }

  // free(l0);
  // free(gh);
  // free(boygr);
  // free(stf);
  //  free(dtef);
}