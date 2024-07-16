.PHONY: all clean
NVCC ?= nvcc

all: constant_cycles_kernel pf_spin pf_thrasher crash preemption_logger 

memory: shared constant_cycles_kernel preemption_logger

shared: shared.cu testbench.h
	$(NVCC) shared.cu -o shared

no_shared: no_shared.cu testbench.h
	$(NVCC) no_shared.cu -o no_shared

constant_cycles_kernel: constant_cycles_kernel.cu testbench.h
	$(NVCC) constant_cycles_kernel.cu -o constant_cycles_kernel

pf_spin: pf_spin.cu task_host_utilities.cu testbench.h
	$(NVCC) pf_spin.cu -o pf_spin -lcuda

pf_thrasher: pf_thrasher.cu testbench.h
	$(NVCC) pf_thrasher.cu -o pf_thrasher -lcuda

crash: crash_ts.cu task_host_utilities.cu testbench.h
	$(NVCC) crash_ts.cu -o crash_ts -lcuda

preemption_logger: task_host_utilities.cu preemption_logger.cu testbench.h
	$(NVCC) preemption_logger.cu -o preemption_logger -g

clean:
	rm -f pf_spin crash constant_cycles_kernel preemption_logger pf_thrasher shared no_shared
