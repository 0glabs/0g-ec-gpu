KERNEL void test_ec(POINT_jacobian a, SCALAR b, GLOBAL POINT_jacobian *result) {
  *result = POINT_mul(a, b);
}

KERNEL void test_add(SCALAR a, SCALAR b, GLOBAL SCALAR *result) {
  *result = SCALAR_add(a, b);
}

KERNEL void test_mul(SCALAR a, SCALAR b, GLOBAL SCALAR *result) {
  *result = SCALAR_mul(a, b);
}

KERNEL void test_sub(SCALAR a, SCALAR b, GLOBAL SCALAR *result) {
  *result = SCALAR_sub(a, b);
}

KERNEL void test_pow(SCALAR a, uint b, GLOBAL SCALAR *result) {
  *result = SCALAR_pow(a, b);
}

KERNEL void test_mont(SCALAR_repr a, GLOBAL SCALAR *result) {
  *result = SCALAR_mont(a);
}

KERNEL void test_unmont(SCALAR a, GLOBAL SCALAR_repr *result) {
  *result = SCALAR_unmont(a);
}

KERNEL void test_sqr(SCALAR a, GLOBAL SCALAR *result) {
  *result = SCALAR_sqr(a);
}

KERNEL void test_double(SCALAR a, GLOBAL SCALAR *result) {
  *result = SCALAR_double(a);
}
