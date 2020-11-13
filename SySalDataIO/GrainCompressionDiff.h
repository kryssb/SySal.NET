#ifndef _SYSAL_GRAINCOMPRESSION_WITH_DIFF_METHOD_
#define _SYSAL_GRAINCOMPRESSION_WITH_DIFF_METHOD_

#define USE_MIDL_INCOMPATIBLE_STRUCTS
#include "..\Common\datastructs.h"

bool CompressGrains_DifferentialMethod(IO_BULK_Tracks *pData, float TransvPrec, float DepthPrec, BYTE **pCompressedGrains, unsigned *pSize);
bool DecompressGrains_DifferentialMethod(IO_BULK_Tracks *pData, BYTE *pCompressedDataBlock);

#endif