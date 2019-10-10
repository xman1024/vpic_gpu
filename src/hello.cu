#include <cstdio>

__global__ void kernelik() {
	printf("Hello from cuda!\n");
}

void fun()
{
	kernelik<<<1,1>>>();
}
