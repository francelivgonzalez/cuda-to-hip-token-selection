# CUDA-to-HIP Token Selection

I’ll write a CUDA kernel that finds the five highest-scoring tokens in a language model’s vocabulary, then use AMD’s HIPIFY tool to port it to HIP C++. I’ll test both versions on my NVIDIA Quadro T2000 to see whether the port gives me the same answers and how its runtime compares.

**Status:** Planned
**Projected timeline:** July 2027
**Hardware:** HP ZBook 15 G6 with an NVIDIA Quadro T2000, 4 GB
**Tools:** CUDA C++, AMD HIPIFY, HIP C++, NVIDIA `nvcc`, Python, and PyTorch

## What the kernel does

A language model produces a score for each token it could generate next. If I have 128,000 scores and only want the top five, do I need to sort the whole list?

My kernel will split the scores into chunks. Each chunk will keep its five highest-scoring tokens. A second kernel will merge those candidates and return the final five token IDs and their scores.

I’ll use generated scores with a fixed random seed so I can repeat the same tests without downloading a model.

## How I’ll use CUDA and AMD ROCm tools

1. Write the original kernel in CUDA C++ and run it on my T2000.
2. Check its output against a CPU reference.
3. Use HIPIFY to convert the CUDA source to HIP C++. Record what converted automatically and what I had to change by hand.
4. Build the HIP version using NVIDIA’s CUDA backend and rerun the same tests on the T2000.
5. When I have access to a supported AMD GPU, build the HIP version for AMD and repeat the experiment with ROCm.

The T2000 runs are my first milestone. Using the same GPU for both versions lets me check the port while holding the hardware constant.

## What I’ll test

- **Known answer:** Start with 16 scores whose top five I can check by hand.
- **Chunk boundaries:** Put all five winners in one chunk, then spread them across different chunks. Include an input whose final chunk is only partly full.
- **Negative scores and ties:** For my CUDA and HIP implementations, the smaller token ID wins when two scores are equal. When checking PyTorch results, I’ll allow different IDs tied at the cutoff.
- **Vocabulary size:** Test 32,000 and 128,000 scores per request.
- **Batch size:** Test one request at a time, then batches of 8 and 32.

## How I’ll measure performance

I’ll compare four methods on the T2000: a full GPU sort, PyTorch’s `torch.topk`, my CUDA kernel, and my HIP port.

For each method, I’ll put the same scores on the GPU before timing, warm it up, and time repeated runs with GPU events. For my kernels, the timing will include both the initial selection and the final merge. I’ll report median runtime in microseconds for each vocabulary and batch size.

## Timeline

- **Week 1:** Build the CPU reference, generate repeatable inputs, and write the correctness tests.
- **Week 2:** Implement the CUDA kernel and verify its output on the T2000.
- **Week 3:** Port the kernel with HIPIFY, document manual changes, and rerun the tests.
- **Week 4:** Benchmark all four methods and publish the T2000 results.
- **AMD follow-up:** Run the HIP version on a supported AMD GPU when I have access. Record the GPU model, ROCm version, and results.

## Results I’ll publish

- A test summary showing whether the CUDA and HIP versions returned the expected token IDs and scores.
- A short porting log with examples of automatic conversions and manual fixes.
- A graph showing how runtime changes with vocabulary size and batch size.
- Build and run commands, software versions, and raw timing data so someone else can repeat the experiment.

## Planned folders

- `cuda/` - original CUDA implementation
- `hip/` - HIP port
- `reference/` - CPU implementation
- `tests/` - correctness and edge-case tests
- `benchmarks/` - timing scripts and baseline comparisons
- `docs/` - build instructions and porting notes
- `results/` - test summaries, raw timings, and plots
