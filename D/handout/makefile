.PHONY: clean bench

CFLAGS = -O3 -std=c++20 -flto -march=native -fopenmp -pthread

seed=42

all: sssp

sssp: sssp.o main.o
	g++ $(CFLAGS) sssp.o main.o -o sssp

%.o: %.cpp
	g++ $(CFLAGS) -c $< -o $@

clean:
	rm -rf *.o sssp

bench: sssp
	OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close ./sssp 100000 200000 ${seed} 1.out
	OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close ./sssp 1000000 20000000 ${seed} 2.out
	OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close ./sssp 1000000 50000000 ${seed} 3.out
	OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close ./sssp 10000000 50000000 ${seed} 4.out
	OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close ./sssp 1000000 200000000 ${seed} 5.out
	OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close ./sssp 10000000 200000000 ${seed} 6.out
	OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close ./sssp 100000000 200000000 ${seed} 7.out

