# Matrix Elimination Optimizations

High-performance C and MPI implementations of Gaussian elimination with partial pivoting, using cache blocking, loop unrolling, OpenMP, and MPI block-cyclic distribution. Includes performance reports and usage examples.

---

## Projects

### Fast GEPP (Shared-Memory)
- **Techniques:** Loop unrolling, cache blocking, OpenMP  
- **Code:** `shared_gepp/`  
- **Usage:**
  ```bash
  cd shared_gepp
  make
  ./gepp_block <matrix_size>
  ./gepp_omp <matrix_size> <num_threads>
  ```
- **Report:** `shared_gepp/report_shared.pdf`

### Distributed GEPP (MPI)
- **Techniques:** 1D column block-cyclic partitioning, optional loop unrolling
- **Code:** `distributed_gepp/`
- **Usage:**
  ```bash
  cd distributed_gepp
  make
  mpirun -np <P> ./cbc <matrix_size> <block_size>    # baseline
  mpirun -np <P> ./cbc_lu <matrix_size>              # unrolled (b=8)
  ```
- **Report:** `distributed_gepp/report_dist.pdf`

## Requirements
- **Compiler:** GCC (with `-fopenmp`)
- **MPI:** OpenMPI or MPICH
- **OS:** Linux or Azure VM

## Performance Highlights
- **Shared-Memory:** Up to 4× speedup vs. baseline for large matrices
- **MPI:** Nearly 6× reduction in runtime with 5 processes (block-cyclic)
- Detailed charts and analysis are in each subfolder's PDF report.

## Repository Structure

```
.
├── shared_gepp/
│   ├── gepp_block.c
│   ├── gepp_omp.c
│   └── report_shared.pdf
├── distributed_gepp/
│   ├── cbc.c
│   ├── cbc_lu.c
│   └── report_dist.pdf
├── .gitignore
└── README.md
```

## Getting Started

1. **Clone this repo**
   ```bash
   git clone git@github.com:nisisiyishou/matrix-elimination-optimizations.git
   cd matrix-elimination-optimizations
   ```

2. **Build & run shared-memory version**
   ```bash
   cd shared_gepp
   make
   ./gepp_block 1000
   ./gepp_omp 1000 4
   ```

3. **Build & run MPI version**
   ```bash
   cd ../distributed_gepp
   make
   mpirun -np 5 ./cbc 5008 4
   mpirun -np 5 ./cbc_lu 10008
   ```

## License & Credits

All code and analyses are authored by Yi Si as part of academic coursework. Feel free to fork and adapt for your own research or performance studies.

## Contact

**Yi Si**  
Email: siyiwh98@gmail.com  
GitHub: nisisiyishou
