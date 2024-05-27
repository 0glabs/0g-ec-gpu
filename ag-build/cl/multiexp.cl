/**
 * This code computes the Multi-Scalar Multiplication (MSM) operation, where the input is the dot product
 * of elliptic curve points and a vector of scalars for the elliptic curve. Each scalar is treated as a large integer,
 * and the product of a scalar and an elliptic curve point is equivalent to repeatedly adding the elliptic curve point
 * that many times.
 *
 * The trivial approach to compute the scalar multiplication of an elliptic curve point and a large integer is to
 * repeatedly double the elliptic curve point P to obtain P, 2P, 4P, ..., perform a binary expansion of the integer,
 * and then combine the doubling results with the binary expansion results.
 *
 * MSM computation has an optimization. For example, to calculate 5P + 7Q, we can transform it into (4P + P) + (4Q + 2Q + Q),
 * and then combine the results based on powers of two to get 4(Q+P) + 2P + (P+Q). Here, 4, 2, 1 are viewed as three buckets,
 * and Q+P, P, P+Q are the contents of the three buckets, respectively. Each bucket accumulates elliptic curve points that 
 * need to be multiplied by the same scalar integer. After traversing the entire input vector, it performs a 
 * single scalar multiplication, avoiding the overhead of computing separate scalar multiplications for each point.
 *
 * There are multiple ways to set up the buckets. One approach is to divide the binary representation of the large integers into several windows,
 * and each non-zero binary value within a window corresponds to a bucket. For example, when the large integer is 256 bits and the window size is 8,
 * it can be divided into 32 windows, and each window has 255 buckets.
 *
 * One optimization technique is to treat the numbers within a window as signed integers called WNAF (windowed non-adjacent form). 
 * For example, with a 4-bit window, 93 can be represented as 5 × 16 + 13, but also 6 × 16 - 3. For -3 × P, the negated point -P is 
 * placed into the bucket corresponding to 3. This optimization is suitable when negating an elliptic curve point is cheap, and it can 
 * save approximately half of the bucket.
 *
 * This code takes a row of large integer vectors and several rows of elliptic curve points as input. All the rows share the same length.
 * The code divides each row into several chunks based on the input parameters and computes the MSM for each chunk separately. When there 
 * are m rows and each row is divided into k chunks, there will be m * k independent MSM tasks.
 *
 * For each MSM task, each bucket window corresponds to a thread, the threads belongs to the same task should be in the same cuda block. 
 * Each thread traverses each large integer scalar within the task range and maintains the buckets of its cluster. When all the data has
 * been traversed, the points in the buckets are multiplied by the corresponding bucket values and added together.
 *
 * One optimization technique is to treat the numbers within a window as signed integers called WNAF (windowed non-adjacent form). 
 * For example, with a 4-bit window, 93 can be represented as 5 × 16 + 13, but also 6 × 16 - 3. For -3 × P, the negated point -P is 
 * placed into the bucket corresponding to 3. This optimization is suitable when negating an elliptic curve point is cheap, and it can 
 * save approximately half of the bucket.
 */


/**
 * @brief Computes a chunk of the Multi-Scalar Multiplication (MSM) operation using multiple threads. Each thread is responsible for 
 *        computing the results within its assigned bucket window for the current MSM task.
 *
 * @param bases The vector of elliptic curve points.
 * @param exps The vector of large integer scalars.
 * @param buckets The buckets allocated for the current MSM task.
 * @param tid The thread ID within the current MSM task.
 * @param chunk_len The length of each vector in the MSM task.
 * @param n_chunk_threads The number of threads assigned to the current MSM task.
 * @param n_thread_buckets The number of buckets owned by each thread, i.e., 2^window_bits.
 * @param window_bits The number of bits in each bucket window.
 * @param signed_window Indicates whether each window is treated as a signed or unsigned integer,
 *                      related to the WNAF optimization.
 *
 * 
 * The function accumulates the elliptic curve points in the respective buckets based on the corresponding scalar values.
 * After processing all the points, it performs a single scalar multiplication for each bucket to obtain the final results.
 * 
 * TODO: for WNAF optimization, the current code assumes the actual large integer bits is SCALAR_BITS - 1.
 */
DEVICE void POINT_multiexp_chunk(
  GLOBAL POINT_affine *bases,
  GLOBAL SCALAR_repr *exps,
  GLOBAL POINT_jacobian *buckets,
  uint tid,
  uint chunk_len,
  uint n_chunk_threads,
  uint n_thread_buckets,
  uint window_bits,
  bool signed_window
)
{
  // When the large integer bits number is not divisible by window_bits, some threads may have an actual 
  // window size smaller than window_bits. Here, we calculate the effective window size for those threads.
  const ushort w = min((ushort)window_bits, (ushort)(SCALAR_BITS - tid * window_bits));

  // The WNAF optimization needs to check if the next less significant window generates a carry. 
  // Here, we calculate the size of the next window.
  ushort w_next = 0;
  if (SCALAR_BITS >= (tid + 1) * window_bits) {
    w_next = min((ushort)window_bits, (ushort)(SCALAR_BITS - (tid + 1) * window_bits));
  }

  // Init buckets belongs to the current thread
  POINT_jacobian* t_buckets = &buckets[tid * n_thread_buckets];
  for(uint i = 0; i < n_thread_buckets; i++) {
    t_buckets[i] = POINT_ZERO;
  }

  uint half_bucket = 1 << (window_bits - 1);
  uint full_bucket = 1 << window_bits;
  
  // Process each input element  
  for(uint i = 0; i < chunk_len; i++) {
    // Scalar for the thread's window
    uint ind = SCALAR_get_bits(exps[i], tid * window_bits, w);

    // Check if the current window generates a carry for the next more significant window.
    bool carry = (ind >= half_bucket);

    // Check if the next less significant window generets a carry for the current window.
    if (signed_window && w_next == window_bits) {
      uint ind_next = SCALAR_get_bits(exps[i], tid * window_bits + window_bits, w_next);
      if (ind_next >= half_bucket) {
        ind += 1;
      }
    }

    bool compute_neg = carry && signed_window;
    
    if (ind > 0 && !compute_neg) {
      POINT_jacobian* bucket = &t_buckets[ind - 1];
      *bucket = POINT_add_mixed(*bucket, bases[i]);
    } else if (full_bucket > ind && compute_neg) {
      POINT_jacobian* bucket = &t_buckets[full_bucket - ind - 1];
      *bucket = POINT_add_mixed(*bucket, POINT_affine_neg(bases[i]));
    }
  }

  // Compute summation of t_buckets[i] * (n_thread_buckets - i)
  // Optimization. 3a + 2b + 1c = a +
  //                             (a) + b +
  //                             ((a) + b) + c
  POINT_jacobian acc = t_buckets[n_thread_buckets - 1];
  POINT_jacobian res = acc;
  for(int j = n_thread_buckets - 1; j >= 1; j--) {
    acc = POINT_add(acc, t_buckets[j - 1]);
    res = POINT_add(res, acc);
  }
  t_buckets[0] = res;
  
  BARRIER_LOCAL();
}

// A utility function for `POINT_aggregate_chunk`
DEVICE uint POINT_bucket_scalar_exp(uint index, uint window_bits, uint height) {
  uint x = (index + (1 << height)) * window_bits;
  if (x >= SCALAR_BITS) {
    return 0;
  } else {
    return SCALAR_BITS - x;
  }
}

/**
 * @brief Aggregates the results of each thread bottom-up using a binary tree structure.
 *
 * @param buckets The buckets used by the current MSM task. The result of each thread is stored at thread_id * n_thread_buckets.
 * @param tid The thread ID within the current MSM task.
 * @param n_chunk_threads The number of threads assigned to the current MSM task.
 * @param n_thread_buckets The number of buckets owned by each thread, i.e., 2^window_bits.
 * @param window_bits The number of bits in each bucket window.
 * 
 * Note: After each thread finishes its computation, each bucket corresponds to a scalar (a power of 2) to be multiplied. 
 * And scalar_exp in the code represents the exponent of this power.
 */

DEVICE void POINT_aggregate_chunk(
  GLOBAL POINT_jacobian *buckets,
  uint tid,
  uint n_chunk_threads,
  uint n_thread_buckets,
  uint window_bits
)
{
  // The current processing height.
  uint h = 0;
  while(n_chunk_threads > (1 << h)) {
    // Only the thread whose id divides 2^(h + 1) applies the following work.
    uint lead_id = tid >> (h + 1) << (h + 1);
    if (tid != lead_id) {
      return;
    }

    // The sibling node id to be added for the current thread.
    uint sib_id = tid + (1 << h);
    if (sib_id >= n_chunk_threads) {
      return;
    }
    
    // The bucket_scalar_exp keeps changes in each height.
    uint my_scalar_exp = POINT_bucket_scalar_exp(tid, window_bits, h);
    uint sib_scalar_exp = POINT_bucket_scalar_exp(sib_id, window_bits, h);


    POINT_jacobian res = buckets[tid * n_thread_buckets];
    for(uint i = 0; i < my_scalar_exp - sib_scalar_exp; i++) {
      res = POINT_double(res);
    }
    buckets[tid * n_thread_buckets] = POINT_add(res, buckets[sib_id * n_thread_buckets]); // 8

    h += 1;
  
    BARRIER_LOCAL();
  }
}

/**
 * @brief Computes the Multi-Scalar Multiplication (MSM) operation for multiple lines of elliptic curve points and a row of large integer scalars.
 *
 * @param bases Multiple lines of elliptic curve points used for computation, with a size of line_len * n_lines.
 * @param results The computation results, with a size of n_lines * n_chunks. The results are stored sequentially for each line.
 * @param exps A row of large integer scalars, with a size of line_len.
 * @param buckets Uninitialized memory allocated for the bucket computations.
 * @param line_len The length of each line of elliptic curve points and the row of large integer scalars, which must be a power of two.
 * @param n_lines The number of lines of elliptic curve points.
 * @param n_chunks The number of chunks each line is divided into for parallel computation.
 * @param n_chunk_threads The number of threads assigned to each chunk, representing the number of windows.
 * @param window_bits The number of bits in each bucket window.
 * @param neg_is_cheap Indicates whether the affine negation operation is relatively cheap, controlling the WNAF optimization.
 *
 * This function receives a row of large integer scalars and multiple rows of elliptic curve points. The length of each line of elliptic curve points
 * equals to the length of the large integer row and must be a power of two. The code divides each line into several chunks based on the input parameters
 * and computes the MSM for each chunk separately.
 */
KERNEL void POINT_multiexp(
    GLOBAL POINT_affine *bases,
    GLOBAL POINT_jacobian *results,
    GLOBAL SCALAR_repr *exps,
    GLOBAL POINT_jacobian *buckets,
    uint line_len,
    uint n_lines,
    uint n_chunks,
    uint n_chunk_threads,
    uint window_bits,
    bool neg_is_cheap
) 
{
  const uint gid = GET_GLOBAL_ID();
  if(gid >= n_lines * n_chunks * n_chunk_threads) return;

  // POINT_jacobian* buckets = (POINT_jacobian*)cuda_shared;

  const uint chunk_len = line_len / n_chunks;
  
  // task_id ∈ [0, n_lines * n_chunks)
  const uint task_id = gid / n_chunk_threads;
  const uint local_thread_id = gid % n_chunk_threads;

  const uint chunk_id = task_id / n_lines;
  const uint line_id = task_id % n_lines;

  const bool signed_window = neg_is_cheap && window_bits > 1;
  uint n_thread_buckets;
  if (signed_window) {
    n_thread_buckets = 1 << (window_bits - 1);
  } else {
    n_thread_buckets = (1 << window_bits) - 1;
  }
  
  POINT_affine *bases_line = &bases[line_id * line_len];
  POINT_affine *bases_chunk = &bases_line[chunk_id * chunk_len];
  SCALAR_repr *exps_chunk = &exps[chunk_id * chunk_len];
  POINT_jacobian *buckets_chunk = &buckets[task_id * n_chunk_threads * n_thread_buckets];

  POINT_multiexp_chunk(bases_chunk, exps_chunk, buckets_chunk, local_thread_id, chunk_len, n_chunk_threads, n_thread_buckets, window_bits, signed_window);

  POINT_aggregate_chunk(buckets_chunk, local_thread_id, n_chunk_threads, n_thread_buckets, window_bits);

  if (local_thread_id == 0) {
    results[line_id * n_chunks + chunk_id] = buckets_chunk[0];
  }
}