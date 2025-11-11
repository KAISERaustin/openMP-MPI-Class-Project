# Build

## MPI version 
mpic++ -O2 -o r1 r1.cpp

## OpenMP version 
g++-15 -fopenmp r2.cpp -o r2

# Run

## MPI version
mpirun -np [P] ./r1 [N] [SEED]

## OpenMP version
./r2 [N] [SEED] [THREADS] [SCHEDULE] [CHUNK]

# Sample Commands

## 4 ranks, 512x512 matrices, 2345 is seed
mpirun -np 4 ./r1 512 2345

## 1024x1024 matrices, 2345 is seed, 8 OpenMP threads, guided schedule, chunk size 32
./r2 1024 2345 8 guided 32