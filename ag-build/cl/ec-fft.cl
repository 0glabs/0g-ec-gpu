/*
 * FFT algorithm for G1 is inspired from: http://www.bealto.com/gpu-fft_group-1.html
 */
KERNEL void POINT_radix_fft(GLOBAL POINT_jacobian* x, // Source buffer
                      GLOBAL POINT_jacobian* y, // Destination buffer
                      GLOBAL SCALAR* pq, // Precalculated twiddle factors
                      GLOBAL SCALAR* omegas, // [omega, omega^2, omega^4, ...]
                      LOCAL POINT_jacobian* u_arg, // Local buffer to store intermediary values
                      uint n, // Number of elements
                      uint lgp, // Log2 of `p` (Read more in the link above)
                      uint deg, // 1=>radix2, 2=>radix4, 3=>radix8, ...
                      uint vbs, // Virtual block size, the algorithm may require a small block size, which will damage the performance
                      uint max_deg) // Maximum degree supported, according to `pq` and `omegas`
{
// CUDA doesn't support local buffers ("shared memory" in CUDA lingo) as function arguments,
// ignore that argument and use the globally defined extern memory instead.
#ifdef CUDA
  // There can only be a single dynamic shared memory item, hence cast it to the type we need.
  POINT_jacobian* u = (POINT_jacobian*)cuda_shared;
#else
  LOCAL POINT_jacobian* u = u_arg;
#endif

  uint lid = GET_LOCAL_ID();
  uint lsize = GET_LOCAL_SIZE();
  uint index = GET_GROUP_ID();

  index = index * (lsize / vbs) + (lid / vbs);
  u += (lid / vbs) * vbs * 2;
  lid = lid & (vbs - 1);
  lsize = vbs;
   
  uint t = n >> deg;
  uint p = 1 << lgp;
  uint k = index & (p - 1);

  x += index;
  y += ((index - k) << deg) + k;

  uint count = 1 << deg; // 2^deg
  uint counth = count >> 1; // Half of count

  uint counts = count / lsize * lid;
  uint counte = counts + count / lsize;

  // Compute powers of twiddle
  const SCALAR twiddle = SCALAR_pow_lookup(omegas, (n >> lgp >> deg) * k);
  SCALAR tmp = SCALAR_pow(twiddle, counts);
  for(uint i = counts; i < counte; i++) {
    u[i] = POINT_mul(x[i*t], tmp);
    tmp = SCALAR_mul(tmp, twiddle);
  }
  BARRIER_LOCAL();

  const uint pqshift = max_deg - deg;
  for(uint rnd = 0; rnd < deg; rnd++) {
    const uint bit = counth >> rnd;
    for(uint i = counts >> 1; i < counte >> 1; i++) {
      const uint di = i & (bit - 1);
      const uint i0 = (i << 1) - di;
      const uint i1 = i0 + bit;
      POINT_jacobian tmp_point = u[i0];
      u[i0] = POINT_add(u[i0], u[i1]);
      u[i1] = POINT_sub(tmp_point, u[i1]);
      SCALAR pq_pow = SCALAR_pow(pq[0], di << rnd << pqshift);
      u[i1] = POINT_mul(u[i1], pq_pow);
    }

    BARRIER_LOCAL();
  }

  for(uint i = counts >> 1; i < counte >> 1; i++) {
    y[i*p] = u[bitreverse(i, deg)];
    y[(i+counth)*p] = u[bitreverse(i + counth, deg)];
  }
}

/// Multiplies all of the elements by `g ^ i * c`
KERNEL void POINT_mul_by_field(GLOBAL POINT_jacobian* elements,
                        uint n,
                        SCALAR g,
                        SCALAR c,
                        uint flags
                        ) {
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) {
    return;
  }
  SCALAR field = c;

  bool g_is_one = (flags & 0x2) == 0;
  bool c_is_one = (flags & 0x1) == 0;
  
  if (!g_is_one && !c_is_one) {
    field = SCALAR_mul(SCALAR_pow(g, gid), c);
  } else if (!g_is_one) {
    // c == one
    field = SCALAR_pow(g, gid);
  }
  
  elements[gid] = POINT_mul(elements[gid], field);
}
