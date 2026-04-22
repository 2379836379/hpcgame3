# Makefile for LU Solver Competition
CC = g++
CFLAGS = -O3 -Wall -march=native -std=c++17 -fopenmp
LDFLAGS = -lm -fopenmp

# 默认目标
all: solver checker gen

solver: driver.cpp solver.cpp
	$(CC) $(CFLAGS) -o solver driver.cpp solver.cpp $(LDFLAGS)

checker: checker.cpp
	$(CC) $(CFLAGS) -o checker checker.cpp $(LDFLAGS)

gen: gen.cpp
	$(CC) $(CFLAGS) -o gen gen.cpp $(LDFLAGS)

# 生成数据
data:
	@chmod +x ./gen
	./gen 1025
	./gen 2049
	./gen 4096
	./gen 8192

clean:
	rm -f solver solver.o *.bin checker gen

.PHONY: all data clean