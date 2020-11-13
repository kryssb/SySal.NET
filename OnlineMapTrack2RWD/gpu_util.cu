#include "gpu_util.h"
#include "cuda_runtime.h"

namespace SySal {
namespace GPU {
int GetAvailableGPUs()
{
	cudaError_t err;
	int count = 0;
	cudaGetDeviceCount(&count);
	return count;
}
};
};