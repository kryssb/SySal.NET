#ifndef _GPU_INTERRUPTIBLE_KERNELS_H_
#define _GPU_INTERRUPTIBLE_KERNELS_H_

#include <stdio.h>

namespace SySal { namespace GPU {

namespace InterruptibleKernels
{
	////////////////////////////////////////////////////////////////////////////////
	/// Every interruptible kernel has arguments.                                ///
	/// Each thread receives a pointer to a structure that is derived from this. ///
	/// Parameters cannot be modified, since a single copy of the structure is   ///
	///  shared among all threads.                                               ///
	////////////////////////////////////////////////////////////////////////////////
	struct Args
	{
		////////////////////////////////////////////////////////////////////////
		/// Set to 0 if this is the first launch of the kernel, 1 otherwise. ///
		/// Reserved: threads must not modify this variable.                 ///
		////////////////////////////////////////////////////////////////////////
		int Started;
		////////////////////////////////////////////////////////
		/// Set to 0 if no thread requests relaunch.         ///
		/// Reserved: threads must not modify this variable. ///
		////////////////////////////////////////////////////////
		int Continue;
		/////////////////////////////////////////////////////////////////
		/// Maximum breakpoints to meet before relaunch is requested. ///
		/// Reserved: threads must not modify this variable.          ///
		/////////////////////////////////////////////////////////////////
		int Limiter;
	};

	////////////////////////////////////////////////////////////////////////////////////////////////////////
	/// Every interruptible kernel must store its persistent variables in a structure derived from this. ///
	/// Variables not stored here are forgotten between one launch and the next.                         ///
	/// Each threads receives a different copy of this structure.                                        ///
	////////////////////////////////////////////////////////////////////////////////////////////////////////
	struct Status
	{
		///////////////////////////////////////////////////////////////////////////////////////////////
		/// Set to 0 if this is the first execution, or to -1 if the thread has completed its task. ///
		/// Other numbers denot the breakpoint from which execution is to be restarted.             ///
		///////////////////////////////////////////////////////////////////////////////////////////////
		int Interrupt;
	};		

	////////////////////////////////////////////////////////////////////////////////////////////////////////
	/// Template kernel launcher class.                                                                  ///
	/// Each run of an interruptible kernel with a certain set of parameters is managed by one instance. ///
	/// Instances can be reused changing parameters or with the same set of parameters.                  ///
	/// Each instance must be created with the correct class for the arguments and for the permanent     ///
	///  kernel variables. The function to be specified is the kernel function itself.                   ///
	/// Example:                                                                                         ///
	///                                                                                                  ///
	///  IntKernel<myargs, myvars, mykernel> launcher;													 ///
	///  launcher.Arguments = userargs;																	 ///
	///  cout << "Launches needed: " << launcher.Launch(dim3(5,1,1), dim3(42,1,1));                      ///
	////////////////////////////////////////////////////////////////////////////////////////////////////////
	template <class TArg, class TStatus, void F(TArg *, TStatus *)> class IntKernel
	{
	public:
		TArg Arguments;
	private:
		TStatus Status;		

	public:		
		///////////////////////////////////////////////////////////////////////////////////////////////////
		/// Runs the kernel, relaunching it as needed.                                                  ///
		/// The kernel will be relaunched when the limit on the number of interrupt checkpoints is hit. ///
		/// The method returns the number of launches needed.                                           ///
		///////////////////////////////////////////////////////////////////////////////////////////////////
		int Launch(dim3 iblocks, dim3 ithreads, int interrupt_checkpoints = 10, int max_launches = -1)
		{
			static_cast<Args &>(Arguments).Started = 1;
			static_cast<Args &>(Arguments).Continue = 0;
			static_cast<Args &>(Arguments).Limiter = interrupt_checkpoints;
			cudaError_t err;		
			static char LastError[512];
			TArg *pArgs = 0;
			TStatus *pStatus = 0;
			if (err = cudaMalloc((void **)&pArgs, sizeof(TArg) ))
			{
				strcpy(LastError, cudaGetErrorString(err));			
				throw LastError;
			}
			if (err = cudaMalloc((void **)&pStatus, sizeof(TStatus) * (iblocks.x * iblocks.y * iblocks.z * ithreads.x * ithreads.y * ithreads.z) ))
			{
				strcpy(LastError, cudaGetErrorString(err));
				if (pArgs) cudaFree(pArgs);
				throw LastError;
			}			
			cudaMemcpy(pArgs, &Arguments, sizeof(TArg), cudaMemcpyHostToDevice);
			int shouldcontinue;
			int launches = 0;
			do
			{
				if (max_launches >= 0 && launches >= max_launches)
				{
					cudaFree(pArgs);
					cudaFree(pStatus);
					throw "Maximum number of launches exceeded.";
				}
				shouldcontinue = 0;				
				cudaMemcpy(&pArgs->Continue, &shouldcontinue, sizeof(int), cudaMemcpyHostToDevice);				
				F<<<iblocks, ithreads>>>(pArgs, pStatus);				
				cudaMemcpy(&shouldcontinue, &pArgs->Continue, sizeof(int), cudaMemcpyDeviceToHost);				
				if (launches == 0) 
				{
					Arguments.Started = 0;
					cudaMemcpy(&pArgs->Started, &Arguments.Started, sizeof(int), cudaMemcpyHostToDevice);
				}
				launches++;
			}
			while(shouldcontinue);			
			cudaMemcpy(&Arguments, pArgs, sizeof(TArg), cudaMemcpyDeviceToHost);
			cudaFree(pArgs);
			cudaFree(pStatus);
			return launches;
		};
	};

/************* The following macros help writing interruptible kernels ********************/

/////////////////////////////////////////////////////////
/// This must be the first instruction of the kernel. ///
/////////////////////////////////////////////////////////
#define _IKGPU_PROLOG(pargs, pstatus) { pstatus += (threadIdx.x + (threadIdx.y + threadIdx.z * blockDim.y) * blockDim.x + blockDim.x * blockDim.y * blockDim.z * (blockIdx.x + (blockIdx.y + blockIdx.z * gridDim.y) * gridDim.x)); if (pargs->Started) pstatus->Interrupt = 0; else if (pstatus->Interrupt == -1) return; } int __ikgpu_counter__ = 0;

////////////////////////////////////////////////////////////////////////////////////
/// This must follow _IKGPU_PROLOG in a sequence at the beginning of the kernel. ///
/// You must have as many of these as interrupt checkpoints in the kernel body.  ///
/// Example:                                                                     ///
///                                                                              ///
/// __global__ mykernel(myargs *args, myvars *status)                            ///
/// {                                                                            ///
///    _IKGPU_PROLOG(args, status)                                               ///
///    _IKGPU_RESUMEFROM(1, status)                                              ///
///    _IKGPU_RESUMEFROM(2, status)                                              ///
///    for (status->i = 0; status->i < args->maxcount; i++)                      ///
///    {                                                                         ///
///			_IKGPU_INTERRUPT(1, status)                                          ///
///         for (status->j = 0; status->j < args->maxcount; j++)                 ///
///         {                                                                    ///
///			      _IKGPU_INTERRUPT(2, status)                                    ///
///               /* do something */                                             ///
///         }                                                                    ///
///     }                                                                        ///
///     _IKGPU_END                                                               ///
/// }                                                                            ///
////////////////////////////////////////////////////////////////////////////////////
#define _IKGPU_RESUMEFROM(at, pstatus) { if (pstatus->Interrupt == at) goto BREAKPOINT ## at; }

/////////////////////////////////////////////////////////////////
/// This macro must be used at each exit point of the kernel. ///
/////////////////////////////////////////////////////////////////
#define _IKGPU_END(pstatus) { pstatus->Interrupt = -1; return; }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Put this macro at each interrupt checkpoint.                                                             ///
/// Notice that the first parameter must be a unique number that identifies the checkpoint in the kernel.    ///
/// The other two parameters must point to the structure holding the persistent variables and the arguments. ///
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define _IKGPU_INTERRUPT(at, pstatus, pargs)\
	{ if (++__ikgpu_counter__ >= pargs->Limiter) { pstatus->Interrupt = at; pargs->Continue = 1; return; } }\
	BREAKPOINT ## at:;

}

}}

#endif
