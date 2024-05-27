// Elliptic curve operations (Short Weierstrass Jacobian form)

#define POINT_ZERO ((POINT_jacobian){BASE_ZERO, BASE_ONE, BASE_ZERO})

typedef struct {
  BASE x;
  BASE y;
} POINT_affine;

typedef struct {
  BASE x;
  BASE y;
  BASE z;
} POINT_jacobian;

// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#doubling-dbl-2009-l
DEVICE POINT_jacobian POINT_double(POINT_jacobian inp) {
  const BASE local_zero = BASE_ZERO;
  if(BASE_eq(inp.z, local_zero)) {
      return inp;
  }

  const BASE a = BASE_sqr(inp.x); // A = X1^2
  const BASE b = BASE_sqr(inp.y); // B = Y1^2
  BASE c = BASE_sqr(b); // C = B^2

  // D = 2*((X1+B)2-A-C)
  BASE d = BASE_add(inp.x, b);
  d = BASE_sqr(d); d = BASE_sub(BASE_sub(d, a), c); d = BASE_double(d);

  const BASE e = BASE_add(BASE_double(a), a); // E = 3*A
  const BASE f = BASE_sqr(e);

  inp.z = BASE_mul(inp.y, inp.z); inp.z = BASE_double(inp.z); // Z3 = 2*Y1*Z1
  inp.x = BASE_sub(BASE_sub(f, d), d); // X3 = F-2*D

  // Y3 = E*(D-X3)-8*C
  c = BASE_double(c); c = BASE_double(c); c = BASE_double(c);
  inp.y = BASE_sub(BASE_mul(BASE_sub(d, inp.x), e), c);

  return inp;
}

// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-madd-2007-bl
DEVICE POINT_jacobian POINT_add_mixed(POINT_jacobian a, POINT_affine b) {
  const BASE local_zero = BASE_ZERO;
  if(BASE_eq(a.z, local_zero)) {
    const BASE local_one = BASE_ONE;
    a.x = b.x;
    a.y = b.y;
    a.z = local_one;
    return a;
  }

  const BASE z1z1 = BASE_sqr(a.z);
  const BASE u2 = BASE_mul(b.x, z1z1);
  const BASE s2 = BASE_mul(BASE_mul(b.y, a.z), z1z1);

  if(BASE_eq(a.x, u2) && BASE_eq(a.y, s2)) {
      return POINT_double(a);
  }

  const BASE h = BASE_sub(u2, a.x); // H = U2-X1
  const BASE hh = BASE_sqr(h); // HH = H^2
  BASE i = BASE_double(hh); i = BASE_double(i); // I = 4*HH
  BASE j = BASE_mul(h, i); // J = H*I
  BASE r = BASE_sub(s2, a.y); r = BASE_double(r); // r = 2*(S2-Y1)
  const BASE v = BASE_mul(a.x, i);

  POINT_jacobian ret;

  // X3 = r^2 - J - 2*V
  ret.x = BASE_sub(BASE_sub(BASE_sqr(r), j), BASE_double(v));

  // Y3 = r*(V-X3)-2*Y1*J
  j = BASE_mul(a.y, j); j = BASE_double(j);
  ret.y = BASE_sub(BASE_mul(BASE_sub(v, ret.x), r), j);

  // Z3 = (Z1+H)^2-Z1Z1-HH
  ret.z = BASE_add(a.z, h); ret.z = BASE_sub(BASE_sub(BASE_sqr(ret.z), z1z1), hh);
  return ret;
}

// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-add-2007-bl
DEVICE POINT_jacobian POINT_add(POINT_jacobian a, POINT_jacobian b) {

  const BASE local_zero = BASE_ZERO;
  if(BASE_eq(a.z, local_zero)) return b;
  if(BASE_eq(b.z, local_zero)) return a;

  const BASE z1z1 = BASE_sqr(a.z); // Z1Z1 = Z1^2
  const BASE z2z2 = BASE_sqr(b.z); // Z2Z2 = Z2^2
  const BASE u1 = BASE_mul(a.x, z2z2); // U1 = X1*Z2Z2
  const BASE u2 = BASE_mul(b.x, z1z1); // U2 = X2*Z1Z1
  BASE s1 = BASE_mul(BASE_mul(a.y, b.z), z2z2); // S1 = Y1*Z2*Z2Z2
  const BASE s2 = BASE_mul(BASE_mul(b.y, a.z), z1z1); // S2 = Y2*Z1*Z1Z1

  if(BASE_eq(u1, u2) && BASE_eq(s1, s2))
    return POINT_double(a);
  else {
    const BASE h = BASE_sub(u2, u1); // H = U2-U1
    BASE i = BASE_double(h); i = BASE_sqr(i); // I = (2*H)^2
    const BASE j = BASE_mul(h, i); // J = H*I
    BASE r = BASE_sub(s2, s1); r = BASE_double(r); // r = 2*(S2-S1)
    const BASE v = BASE_mul(u1, i); // V = U1*I
    a.x = BASE_sub(BASE_sub(BASE_sub(BASE_sqr(r), j), v), v); // X3 = r^2 - J - 2*V

    // Y3 = r*(V - X3) - 2*S1*J
    a.y = BASE_mul(BASE_sub(v, a.x), r);
    s1 = BASE_mul(s1, j); s1 = BASE_double(s1); // S1 = S1 * J * 2
    a.y = BASE_sub(a.y, s1);

    // Z3 = ((Z1+Z2)^2 - Z1Z1 - Z2Z2)*H
    a.z = BASE_add(a.z, b.z); a.z = BASE_sqr(a.z);
    a.z = BASE_sub(BASE_sub(a.z, z1z1), z2z2);
    a.z = BASE_mul(a.z, h);

    return a;
  }
}

DEVICE POINT_jacobian POINT_neg(POINT_jacobian a) {
  a.y = BASE_sub(BASE_ZERO, a.y);
  return a;
}

DEVICE POINT_affine POINT_affine_neg(POINT_affine a) {
  a.y = BASE_sub(BASE_ZERO, a.y);
  return a;
}

DEVICE POINT_jacobian POINT_sub(POINT_jacobian a, POINT_jacobian b) {
  return POINT_add(a, POINT_neg(b));
}

DEVICE POINT_jacobian POINT_mul_exponent(POINT_jacobian base, SCALAR_repr exp) {
  POINT_jacobian res = POINT_ZERO;
  for(uint i = 0; i < SCALAR_BITS; i++) {
    res = POINT_double(res);
    bool exp_bit_i = SCALAR_get_bit(exp, i);
    if(exp_bit_i) res = POINT_add(res, base);
  }
  return res;
}

DEVICE POINT_jacobian POINT_mul(POINT_jacobian base, SCALAR exp) {
  return POINT_mul_exponent(base, SCALAR_unmont(exp));
}