# Performance Benchmarks

This directory contains benchmark scripts for evaluating the performance of the CUDA-accelerated LOF implementation compared to other implementations like scikit-learn's LOF.

## Scripts

- `benchmark_lof.py` - Compares the performance of CUDA LOF against scikit-learn's LOF across various dataset sizes and dimensionalities

## Running Benchmarks

To run the benchmarks:

```bash
# Run with default settings
python benchmarks/benchmark_lof.py

# Run with custom parameters
python benchmarks/benchmark_lof.py --min-samples 1000 --max-samples 100000 --n-features 20
```

## Benchmark Results

Benchmark results are saved as images in the `img/` directory. The main benchmark comparison chart can be found at `img/benchmark_results.png`.

For detailed analysis of the benchmark results, see [Benchmark Results](../docs/benchmark_results.md). 