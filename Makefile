ARCH = $(shell uname -m)
OS   = $(shell uname -s)

CXX      = g++
CXXFLAGS = -O3 -std=c++11 -Wall -Wextra -fopenmp -march=native
LDFLAGS  = -fopenmp


TARGET     = matmul1
SRC        = matmul1.cpp
RESULT_DIR = results

OUTPUT_CSV       = $(RESULT_DIR)/results.csv
SCHEDULING_CSV   = $(RESULT_DIR)/scheduling_results.csv
TILING_CSV        = $(RESULT_DIR)/tiling_results.csv
THREAD_SCALING_CSV = $(RESULT_DIR)/thread_scaling.csv

all: directories $(TARGET)

directories:
	@mkdir -p $(RESULT_DIR)

$(TARGET): $(SRC)
	@echo "========================================"
	@echo "Compiling for $(OS) $(ARCH)"
	@echo "========================================"
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)
	@echo "Compilation successful!"

run: $(TARGET)
	@echo "=== Running with default settings (size=512, threads=8) ==="
	./$(TARGET) 512 8

# Base test: sequential + base parallel, varying size and threads
test: $(TARGET) directories
	@echo "=== Starting performance tests (size vs threads) ==="
	@echo "SIZE,THREADS,SEQ_TIME,PAR_TIME,SPEEDUP,EFFICIENCY,VERIFIED,STDDEV" > $(OUTPUT_CSV)
	@for size in 512 850 1024 1520 2048; do \
		echo "  Testing size: $$size x $$size"; \
		seq_time=$$(./$(TARGET) $$size 0 seq 10); \
		echo "    Sequential baseline: $$seq_time s"; \
		for threads in 2 4 6 8 10 12 14 16 20 24 32 64 128 256; do \
			echo "    Threads: $$threads"; \
			./$(TARGET) $$size $$threads par_only $$seq_time >> $(OUTPUT_CSV); \
		done; \
	done
	@echo "Results saved in: $(OUTPUT_CSV)"

# Scheduling test
test-scheduling: $(TARGET) directories
	@echo "=== Scheduling tests (static/dynamic/guided) ==="
	@echo "SCHEDULE_TYPE,SIZE,THREADS,TIME,SPEEDUP,EFFICIENCY,VERIFIED,STDDEV" > $(SCHEDULING_CSV)
	@for size in 512 1024 2048; do \
		echo "  Sequential baseline for size=$$size..."; \
		seq_time=$$(./$(TARGET) $$size 0 seq 10); \
		echo "  Baseline: $$seq_time s"; \
		for sched in static dynamic guided; do \
			echo "  Schedule: $$sched"; \
			for threads in 2 4 6 8 10 12 14 16 20 24 32; do \
				./$(TARGET) $$size $$threads $${sched}_only $$seq_time >> $(SCHEDULING_CSV); \
			done; \
		done; \
	done
	@echo "Results saved in: $(SCHEDULING_CSV)"

# Tiling test
test-tiling: $(TARGET) directories
	@echo "=== Tiling/Blocking tests ==="
	@echo "MODE,BLOCK_SIZE,SIZE,THREADS,TIME,SPEEDUP,EFFICIENCY,VERIFIED,STDDEV" > $(TILING_CSV)
	@for size in 512 1024 2048; do \
		echo "  Sequential baseline for size=$$size..."; \
		seq_time=$$(./$(TARGET) $$size 0 seq 10); \
		echo "  Baseline: $$seq_time s"; \
		for block in 32 64 128 256; do \
			echo "  Block: $$block"; \
			for threads in 2 4 8 12 16 20 24 32; do \
				./$(TARGET) $$size $$threads tiled_only $$block $$seq_time >> $(TILING_CSV); \
			done; \
		done; \
	done
	@echo "Results saved in: $(TILING_CSV)"

# Optimized tiling test
test-tiling-opt: $(TARGET) directories
	@echo "=== Optimized Tiling tests ==="
	@echo "MODE,BLOCK_SIZE,SIZE,THREADS,TIME,SPEEDUP,EFFICIENCY,VERIFIED,STDDEV" >> $(TILING_CSV)
	@for size in 512 1024 2048; do \
		seq_time=$$(./$(TARGET) $$size 0 seq 10); \
		for block in 32 64 128 256; do \
			for threads in 2 4 8 12 16 20 24 32; do \
				./$(TARGET) $$size $$threads tiled_opt_only $$block $$seq_time >> $(TILING_CSV); \
			done; \
		done; \
	done
	@echo "Optimized tiling results appended to: $(TILING_CSV)"

# Thread scaling study
thread-scaling: $(TARGET) directories
	@echo "=== Thread scaling study ==="
	@echo "SIZE,THREADS,SEQ_TIME,PAR_TIME,SPEEDUP,EFFICIENCY,VERIFIED,STDDEV" > $(THREAD_SCALING_CSV)
	@for size in 512 850 1024 1520 2048; do \
		echo "  Sequential baseline for size=$$size..."; \
		seq_time=$$(./$(TARGET) $$size 0 seq 10); \
		echo "  Baseline: $$seq_time s"; \
		max_threads=$$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8); \
		for threads in 2 4 6 8 10 12 14 16 20 24 28 32; do \
			if [ $$threads -le $$max_threads ]; then \
				./$(TARGET) $$size $$threads par_only $$seq_time >> $(THREAD_SCALING_CSV); \
			fi; \
		done; \
	done
	@echo "Results saved in: $(THREAD_SCALING_CSV)"

# Quick test
quick-test: $(TARGET)
	@echo "=== Quick verification test ==="
	@echo "Size	Threads	SeqTime	ParTime	Speedup	Efficiency"
	@echo "--------------------------------------------------------"
	@for size in 64 128 256; do \
		for threads in 2 4 6 8 10; do \
			./$(TARGET) $$size $$threads quick; \
		done; \
	done

# Correctness verification with small matrices, compared against the sequential result
verify: $(TARGET)
	@echo "=== Correctness verification ==="
	@echo ""
	@echo "Size=32, Threads=1:"
	./$(TARGET) 32 1 verify
	@echo ""
	@echo "Size=64, Threads=4 (static):"
	./$(TARGET) 64 4 static
	@echo ""
	@echo "Size=64, Threads=4 (dynamic):"
	./$(TARGET) 64 4 dynamic
	@echo ""
	@echo "Size=64, Threads=4 (guided):"
	./$(TARGET) 64 4 guided
	@echo ""
	@echo "Size=128, Threads=8 (tiled, block=64):"
	./$(TARGET) 128 8 tiled 64
	@echo ""
	@echo "Size=128, Threads=8 (tiled_opt, block=64):"
	./$(TARGET) 128 8 tiled_opt 64

# Test with different matrix sizes
test-sizes: $(TARGET)
	@echo "=== Testing different matrix sizes (threads=8) ==="
	@echo "Size	Time_Seq	Time_Par	Speedup	Efficiency"
	@echo "--------------------------------------------------------"
	@for size in 64 128 256 512 768 1024 1280 1536 1792 2048; do \
		echo -n "$$size	"; \
		./$(TARGET) $$size 8 quick; \
	done

# All tests
full-test: test test-scheduling test-tiling test-tiling-opt thread-scaling test-sizes
	@echo "=== All tests completed. Results in $(RESULT_DIR)/ ==="

# Full clean (removes executable, object files, and results directory)
clean:
	rm -f $(TARGET) *.o
	rm -rf $(RESULT_DIR)
	@echo "Cleaned."

# Light clean (removes only the executable, keeps results)
clean-light:
	rm -f $(TARGET)
	@echo "Removed executable only."

help:
	@echo "========================================"
	@echo "  MATMUL - OpenMP Matrix Multiplication"
	@echo "========================================"
	@echo "Targets:"
	@echo "  make                - Build"
	@echo "  make run            - Run with default settings (size=512, threads=8)"
	@echo "  make verify         - Verify correctness"
	@echo "  make test           - Size vs Threads -> results.csv"
	@echo "  make test-sizes     - Various sizes (threads=8)"
	@echo "  make thread-scaling - Thread scaling -> thread_scaling.csv"
	@echo "  make test-scheduling- static/dynamic/guided -> scheduling_results.csv"
	@echo "  make test-tiling    - Tiling -> tiling_results.csv"
	@echo "  make test-tiling-opt- Optimized tiling -> tiling_results.csv (append)"
	@echo "  make quick-test     - Quick test"
	@echo "  make full-test      - All tests"
	@echo "  make clean          - Remove everything"
	@echo "  make clean-light    - Remove executable only"
	@echo ""
	@echo "Manual usage: ./matmul1 <size> <threads> [mode] [extra]"
	@echo "  ./matmul1 512 0 seq 10          # sequential baseline (median of 10 runs)"
	@echo "  ./matmul1 512 8 par_only 0.85   # parallel only, seq time passed in"
	@echo "  ./matmul1 512 8 static_only 0.85"
	@echo "  ./matmul1 512 8 tiled_only 64 0.85"
	@echo "========================================"

.PHONY: all run test test-scheduling test-tiling test-tiling-opt thread-scaling \
        quick-test verify test-sizes full-test clean clean-light help directories