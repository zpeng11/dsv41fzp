namespace bb {

// Candidate for FMA contraction; whether it is fused depends only on flags.
float probe_mul_add(float a, float b, float c) { return a * b + c; }

}  // namespace bb
