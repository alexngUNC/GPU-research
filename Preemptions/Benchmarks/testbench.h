/* Copyright 2021-2023 Joshua Bakita
 * Header for miscellaneous experimental helper functions.
 */

#define SAFE(x) \
	if ((err = (cudaError_t)(x)) != 0) { \
		printf("CUDA error %d! %s\n", err, cudaGetErrorString(err)); \
		printf("Suspect line: %s\n", #x); \
		exit(1); \
	}

#define s2ns(s) ((s)*1000l*1000l*1000l)
#define ns2ms(s) ((s)/(1000l*1000l))

// Return the difference between two timestamps in nanoseconds
#define timediff(start, end) ((s2ns((end).tv_sec) + (end).tv_nsec) - (s2ns((start).tv_sec) + (start).tv_nsec))
#define time2ns(time) (s2ns((time).tv_sec) + (time).tv_nsec)
