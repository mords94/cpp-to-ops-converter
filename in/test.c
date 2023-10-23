#define malloc(x) NULL

#include "pom2k_c_header.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define ACC2_jk(j, k) (j) + (k)*jm

#define ACC3M2(i, j, k) (i) + (j) * (im - 2) + (k) * (im - 2) * (jm - 1)

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
        b = powf(1.111f + a, 2.0f);
        out2[ACC3M2(i, j, k)] = out1[ACC3M2(i, j, k - 1)] * 2.0f;
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

void ext_comp_vamax_(real_t *_vamax, real_t *vaf, int *_imax, int *_jmax) {
  float vamax = 0.0f;
  real_t bla = *_vamax;
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

typedef float real_t2;

void red3(real_t *in1, real_t *out1) {
  real_t sol = 11.00f;
  float a = 0.0f;
  float b = 1.1;
  float c = 1.2;
  float d = 1.0f;
  for (int k = 0; k < kb; k++) {
    for (int j = 0; j < jm; j++) {
      for (int i = 0; i < im; i++) {
        a = d * 2;
        d = abs(sol - out1[ACC3(i, j, k)]);
        out1[ACC3(i, j, k)] = in1[ACC3(i, j, k)] + d;
        a = abs(sol - out1[ACC3(i, j, k)]);
      }
    }
  }
}

void red1(real_t *in1) {
  real_t sol = 11.00f;
  int e = 0;
  real_t *out1;

  out1 = (real_t *)malloc(sizeof(real_t) * im * jm * kb);

  float a = 0.0f;
  for (int k = 0; k < kb; k++) {
    for (int j = 0; j < jm; j++) {
      for (int i = 0; i < im; i++) {
        out1[ACC3(i, j, k)] = e + in1[ACC3(i, j, k)];
        a = abs(sol - out1[ACC3(i, j, k)]);
        e = a + 0.5f + e;
      }
    }
  }
}

void red2(float *in1, float *out1, float *e2) {

  float sol = 11.00f;
  float e = *e2;

  for (int k = 0; k < kb; k++) {
    for (int j = 0; j < jm; j++) {
      for (int i = 0; i < im; i++) {
        out1[ACC3(i, j, k)] = e + in1[ACC3(i, j, k)];
        e = abs(sol - out1[ACC3(i, j, k)]);
      }
    }
  }
}

void ext_profq_(float *sm, float *sh, float *dh, float *cc, float *h,
                float *etf, float *a, float *c, float *kq, float *dz,
                float *dzz, float *ee, float *gg, float *wusurf, float *wvsurf,
                float *uf, float *wubot, float *wvbot, float *t, float *s,
                float *zz, float *q2b, float *q2lb, float *l, float *z,
                float *km, float *u, float *v, float *kh, float *vf, float *fsm,
                float *q2, float *dt, float *rho, float *dtef, float *l0,
                float *gh, float *boygr, float *stf, float *prod, float *zmin,
                float *zmax) {

  float b1 = 16.6f;
  float coef3 = .99;
  float coef4;

  for (int k = 0; k < kb; k++) {
    for (int j = 0; j < jm; j++) {
      for (int i = 0; i < im; i++) {
        // coef1 = a2 * (1.0f - 6.0f * a1 / b1 * stf[ACC3(i, j, k)]);
        // coef2 = 3.0f * a2 * b2 / stf[ACC3(i, j, k)] + 18.0f * a1 * a2;
        coef3 = coef3 * b1 * stf[ACC3(i, j, k)];
        sm[ACC3(i, j, k)] = coef3 + sh[ACC3(i, j, k)];
        // sm[ACC3(i, j, k)] =
        // sm[ACC3(i, j, k)] / (1.0f - coef5 * gh[ACC3(i, j, k)]);
      }
    }
  }
}

void ext_profq_alt(real_t *sm, real_t *sh, real_t *dh, real_t *cc, real_t *h,
                   real_t *etf, real_t *a, real_t *c, real_t *kq, real_t *dz,
                   real_t *dzz, real_t *ee, real_t *gg, real_t *wusurf,
                   real_t *wvsurf, real_t *uf, real_t *wubot, real_t *wvbot,
                   real_t *t, real_t *s, real_t *zz, real_t *q2b, real_t *q2lb,
                   real_t *l, real_t *z, real_t *km, real_t *u, real_t *v,
                   real_t *kh, real_t *vf, real_t *fsm, real_t *q2, real_t *dt,
                   real_t *rho, real_t *dtef, real_t *l0, real_t *gh,
                   real_t *boygr, real_t *stf, real_t *prod, real_t *zmin,
                   real_t *zmax) {

  real_t a1 = 0.92f;
  real_t a2 = 0.74f;
  real_t b1 = 16.6f;
  real_t b2 = 10.1f;
  real_t c1 = 0.08f;
  real_t e1 = 1.8f;
  real_t e2 = 1.33f;
  real_t sef = 1.0f;
  real_t cbcnst = 100.0f;
  real_t surfl = 2e5;
  real_t shiw = 0.0f;
  real_t coef1;
  real_t coef2;
  real_t coef3;
  real_t coef4;
  real_t coef5;
  real_t const1;
  real_t ghc;
  real_t p;
  real_t sp;
  real_t tp;
  real_t utau2;
  real_t df0;
  real_t df1;
  real_t df2;

  for (int k = 0; k < kbm1; k++) {
    for (int j = 0; j < jm; j++) {
      for (int i = 0; i < im; i++) {
        tp = t[ACC3(i, j, k)] + tbias;
        sp = s[ACC3(i, j, k)] + sbias;

        p = grav * rhoref * (-zz[k] * h[ACC2(i, j)]) * (0.0001f) * p;
        cc[ACC3(i, j, k)] = 1449.1f + 0.00821f * p + 4.55f * tp -
                            0.045f * tp * tp + 1.34f * (sp - 35.0f);
        cc[ACC3(i, j, k)] =
            cc[ACC3(i, j, k)] /
            sqrtf((1.0f - 0.01642f * p / cc[ACC3(i, j, k)]) *
                  (1.0f -
                   0.4f * p / ((cc[ACC3(i, j, k)]) * (cc[ACC3(i, j, k)]))));
      }
    }
  }
}

void ext_dens_(real_t *si, real_t *ti, real_t *rhoo, real_t *h, real_t *fsm,
               real_t *zz, real_t *tbias, real_t *sbias) {
  for (int k = 0; k < kbm1; k++) {
    for (int j = 0; j < jm; j++) {
      for (int i = 0; i < im; i++) {
        real_t cr, p, rhor, sr, tr, tr2, tr3, tr4;
        tr = ti[ACC3(i, j, k)] + *tbias;
        sr = si[ACC3(i, j, k)] + *sbias;
        tr2 = tr * tr;
        tr3 = tr2 * tr;
        tr4 = tr3 * tr;
        // C
        // C     Approximate pressure in units of bars:
        // C
        p = grav * rhoref * (-zz[k] * h[ACC2(i, j)]) * 1.e-5f;
        rhor = -0.157406f + 6.793952e-2f * tr - 9.095290e-3f * tr2 +
               1.001685e-4f * tr3 - 1.120083e-6f * tr4 +
               6.536332e-9f * tr4 * tr;
        rhor = rhor +
               (0.824493f - 4.0899e-3f * tr + 7.6438e-5f * tr2 -
                8.2467e-7f * tr3 + 5.3875e-9f * tr4) *
                   sr +
               (-5.72466e-3f + 1.0227e-4f * tr - 1.6546e-6f * tr2) *
                   powf(fabs(sr), 1.5f) +
               4.8314e-4f * sr * sr;
        cr = 1449.1f + .0821f * p + 4.55f * tr - .045f * tr2 +
             1.34f * (sr - 35.0f);
        rhor = rhor + 1.e5f * p / (cr * cr) * (1.0f - 2.0f * p / (cr * cr));
        rhoo[ACC3(i, j, k)] = rhor / rhoref * fsm[ACC2(i, j)];
      }
    }
  }
}

void cbc(real_t *sm, real_t *sh, real_t *dh, real_t *cc, real_t *h, real_t *etf,
         real_t *a, real_t *c, real_t *kq, real_t *dz, real_t *dzz, real_t *ee,
         real_t *gg, real_t *wusurf, real_t *wvsurf, real_t *uf, real_t *wubot,
         real_t *wvbot, real_t *t, real_t *s, real_t *zz, real_t *q2b,
         real_t *q2lb, real_t *l, real_t *z, real_t *km, real_t *u, real_t *v,
         real_t *kh, real_t *vf, real_t *fsm, real_t *q2, real_t *dt,
         real_t *rho, real_t *dtef, real_t *l0, real_t *gh, real_t *boygr,
         real_t *stf, real_t *prod, real_t *zmin, real_t *zmax) {

  real_t a1 = 0.92f;
  real_t a2 = 0.74f;
  real_t b1 = 16.6f;
  real_t b2 = 10.1f;
  real_t c1 = 0.08f;
  real_t e1 = 1.8f;
  real_t e2 = 1.33f;
  real_t sef = 1.0f;
  real_t cbcnst = 100.0f;
  real_t surfl = 2e5;
  real_t shiw = 0.0f;
  real_t ghc = -6.0f;
  real_t df0;
  real_t df1;
  real_t df2;
  real_t const1 = 0.5f;

  real_t coef4 = 18.0f * a1 * a1 + 9.0f * a1 * a2;
  real_t coef5 = 9.0f * a1 * a2;

  for (int k = 0; k < kb; k++) {
    for (int j = 0; j < jm; j++) {
      for (int i = 0; i < im; i++) {
        real_t coef1 = a2 * (1.0f - 6.0f * a1 / b1 * stf[ACC3(i, j, k)]);
        real_t coef2 = 3.0f * a2 * b2 / stf[ACC3(i, j, k)] + 18.0f * a1 * a2;
        real_t coef3 =
            a1 * (1.0f - 3.0f * c1 - 6.0f * a1 / b1 * stf[ACC3(i, j, k)]);
        sh[ACC3(i, j, k)] = coef1 / (1.0f - coef2 * gh[ACC3(i, j, k)]);
        sm[ACC3(i, j, k)] =
            coef3 + sh[ACC3(i, j, k)] * coef4 * gh[ACC3(i, j, k)];
        sm[ACC3(i, j, k)] =
            sm[ACC3(i, j, k)] / (1.0f - coef5 * gh[ACC3(i, j, k)]);
      }
    }
  }
}