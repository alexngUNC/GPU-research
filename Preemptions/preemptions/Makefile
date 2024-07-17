.PHONY: all clean
NVCC ?= nvcc

all: constant_cycles_kernel pf_spin pf_thrasher crash preemption_logger shared no_shared constant no_constant

memory: shared constant_cycles_kernel preemption_logger

shared: shared.cu testbench.h
	$(NVCC) shared.cu -o bin/shared

no_shared: no_shared.cu testbench.h
	$(NVCC) no_shared.cu -o bin/no_shared

constant: constant.cu testbench.h
	$(NVCC) constant.cu -o bin/constant

no_constant: no_constant.cu testbench.h
	$(NVCC) no_constant.cu -o bin/no_constant

constant_cycles_kernel: constant_cycles_kernel.cu testbench.h
	$(NVCC) constant_cycles_kernel.cu -o bin/constant_cycles_kernel

pf_spin: pf_spin.cu task_host_utilities.cu testbench.h
	$(NVCC) pf_spin.cu -o bin/pf_spin -lcuda

pf_thrasher: pf_thrasher.cu testbench.h
	$(NVCC) pf_thrasher.cu -o bin/pf_thrasher -lcuda

crash: crash_ts.cu task_host_utilities.cu testbench.h
	$(NVCC) crash_ts.cu -o bin/crash_ts -lcuda

preemption_logger: task_host_utilities.cu preemption_logger.cu testbench.h
	$(NVCC) preemption_logger.cu -o bin/preemption_logger -g

clean:
	rm -f bin/pf_spin bin/crash bin/constant_cycles_kernel bin/preemption_logger bin/pf_thrasher bin/crash_ts bin/shared bin/no_shared bin/constant bin/no_constant
