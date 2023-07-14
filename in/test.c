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

real_t alpha, dte, dte2, dti, dti2;
real_t grav, hmax, pi;
real_t ramp, rfe, rfn, rfs;
real_t rfw, rhoref, sbias, slmax;
real_t small, tbias, time2, tprni;
real_t umol, vmaxl;
real_t horcon;

real_t period, time0, time1;

const real_t kappa = 0.4f;    // von Karman's constant
const real_t z0b = .01f;      // Bottom roughness (metres)
const real_t cbcmin = .0025f; // Minimum bottom friction coeff.
const real_t cbcmax = 1.0f;   // Maximum bottom friction coeff.

int iint, iprint, iskp, jskp;
int kl1, kl2, mode, ntp;
int kbm3;

int LEFT_EDGE = 0;
void example(real_t *in1, real_t *in2, real_t *out1, real_t *out2, int arg1) {
  for (int k = kbm3; k > 0; k--) {
    for (int j = 0; j < jm; j++) {
      for (int i = 0; i < im; i++) {
        out1[ACC3(i, j, k)] = in1[ACC3(i, j, k - 1)];
      }
    }
  }
}

// for
// i = 11 to kbm3