#include "stdafx.h"
#include "GrainCompressionDiff.h"
#include <memory.h>
#include <math.h>


// Internal functions

static unsigned Pack(unsigned count, int *psource, BYTE *pdata, int bits, int *pbitpos);
static unsigned Unpack(unsigned count, BYTE *psource, int *pdata, int bits, int *pbitpos, bool *pException);

static int EncodeCoord(IO_BULK_Compressed_GrainSection_TrackHeader::t_Coding *pCoding, unsigned CoordOffset, float Precision, Track *pTrack, BYTE *pdata, int *pbitpos);
static int DecodeCoord(IO_BULK_Compressed_GrainSection_TrackHeader::t_Coding Coding, unsigned CoordOffset, float Precision, Track *pTrack, BYTE *pdata, int *pbitpos);



// Exported functions: Compress / Decompress

bool CompressGrains_DifferentialMethod(IO_BULK_Tracks *pData, float TransvPrec, float DepthPrec, BYTE **pCompressedGrains, unsigned *pSize)
{
	if (TransvPrec <= 0.f || DepthPrec <= 0.f) return false;
	unsigned totaltks = pData->Hdr.TCount[0] + pData->Hdr.TCount[1];
	BYTE *pC = (BYTE *)malloc(sizeof(IO_BULK_Compressed_GrainSection_TrackHeader) * totaltks + sizeof(IO_BULK_DiffCompression_Header));
	if (!pC) return false;
	unsigned CurrPos = 0;
	unsigned CurrMax = 16384;
	IO_BULK_DiffCompression_Header *pH = (IO_BULK_DiffCompression_Header *)pC;
	pH->Hdr.Size = 0;
	pH->DepthPrecision = DepthPrec;
	pH->TransversePrecision = TransvPrec;
	IO_BULK_Compressed_GrainSection_TrackHeader *pT = (IO_BULK_Compressed_GrainSection_TrackHeader *)(pC + sizeof(IO_BULK_DiffCompression_Header));
	BYTE *pG = (BYTE *)malloc(CurrMax);
	if (!pG)
	{
		free(pC);
		return false;
		};
	int bitpos = 0;
	unsigned s, t;
	for (s = 0; s < 2; s++)
		for (t = 0; t < pData->Hdr.TCount[s]; t++)
		{
			pT->XCoding.Bits = pT->YCoding.Bits = pT->ZCoding.Bits = 0;
			pT->XCoding.Polarization = pT->YCoding.Polarization = pT->ZCoding.Polarization = 0;
			if (!pData->pTracks[s][t].PointsN)
			{
				pT->XCoding.Base = pT->YCoding.Base = pT->ZCoding.Base = 0.f;
				}
			else
			{				
				Track *pTrack = pData->pTracks[s] + t;
				if ((CurrMax - CurrPos) < (sizeof(TVector) * pTrack->PointsN))
				{
					void *p = realloc(pG, CurrMax + (sizeof(TVector) * pTrack->PointsN));
					if (!p)
					{
						free(pG);
						free(pC);
						return false;
						};
					pG = (BYTE *)p;
					CurrMax += (sizeof(TVector) * pTrack->PointsN);
					};
				int sizeinc;
				if ((sizeinc = EncodeCoord(&pT->XCoding, 0, TransvPrec, pTrack, pG + CurrPos, &bitpos)) < 0)
				{
					free(pC);
					free(pG);
					return false;
					}
				else CurrPos += sizeinc;
				if ((sizeinc = EncodeCoord(&pT->YCoding, 1, TransvPrec, pTrack, pG + CurrPos, &bitpos)) < 0)
				{
					free(pC);
					free(pG);
					return false;
					}
				else CurrPos += sizeinc;
				if ((sizeinc = EncodeCoord(&pT->ZCoding, 2, DepthPrec, pTrack, pG + CurrPos, &bitpos)) < 0)
				{
					free(pC);
					free(pG);
					return false;
					}
				else CurrPos += sizeinc;
				};
			pT++;
			};
	void *p = realloc(pC, CurrMax + sizeof(IO_BULK_Compressed_GrainSection_TrackHeader) * totaltks + sizeof(IO_BULK_DiffCompression_Header));
	if (!p)
	{
		free(pG);
		free(pC);
		return false;
		};
	pC = (BYTE *)p;	
	CurrPos++;
	memcpy(pC + sizeof(IO_BULK_Compressed_GrainSection_TrackHeader) * totaltks + sizeof(IO_BULK_DiffCompression_Header), pG, CurrPos);
	*pCompressedGrains = pC;
	free(pG);
	pH = (IO_BULK_DiffCompression_Header *)pC;
	pH->Hdr.Size = (*pSize = CurrPos + sizeof(IO_BULK_Compressed_GrainSection_TrackHeader) * totaltks + sizeof(IO_BULK_DiffCompression_Header));	
	return true;
	};

bool DecompressGrains_DifferentialMethod(IO_BULK_Tracks *pData, BYTE *pCompressedDataBlock)
{
	IO_BULK_DiffCompression_Header *pH = (IO_BULK_DiffCompression_Header *)pCompressedDataBlock;
	unsigned totaltks = pData->Hdr.TCount[0] + pData->Hdr.TCount[1];
	IO_BULK_Compressed_GrainSection_TrackHeader *pT = (IO_BULK_Compressed_GrainSection_TrackHeader *)(pCompressedDataBlock + sizeof(IO_BULK_DiffCompression_Header));
	BYTE *pG = pCompressedDataBlock + sizeof(IO_BULK_DiffCompression_Header) + totaltks * sizeof(IO_BULK_Compressed_GrainSection_TrackHeader);
	try
	{
		unsigned s, t;
		int bitpos = 0;
		for (s = 0; s < 2; s++)
			for (t = 0; t < pData->Hdr.TCount[s]; t++)
			{
				int sizeinc;
				if ((sizeinc = DecodeCoord(pT->XCoding, 0, pH->TransversePrecision, &pData->pTracks[s][t], pG, &bitpos)) < 0) return false;
				pG += sizeinc;
				if ((sizeinc = DecodeCoord(pT->YCoding, 1, pH->TransversePrecision, &pData->pTracks[s][t], pG, &bitpos)) < 0) return false;
				pG += sizeinc;
				if ((sizeinc = DecodeCoord(pT->ZCoding, 2, pH->DepthPrecision, &pData->pTracks[s][t], pG, &bitpos)) < 0) return false;
				pG += sizeinc;
				pT++;
				};
		}
	catch (...)
	{
		return false;
		};
	return true;
	};



// Encoding / Decoding functions

int EncodeCoord(IO_BULK_Compressed_GrainSection_TrackHeader::t_Coding *pCoding, unsigned CoordOffset, float Precision, Track *pTrack, BYTE *pdata, int *pbitpos)
{
	unsigned N = pTrack->PointsN;
	int *pDiffs = (int *)malloc(sizeof(int) * N/*(N - 1)*/);
	if (N && !pDiffs)
	{
		if (pDiffs) free(pDiffs);
		return -1;
		};
	pCoding->Base = *(float *)((BYTE *)&pTrack->pPoints[0] + CoordOffset * sizeof(float));	
	if (N > 1)
	{
		unsigned i;
		unsigned tester;
		
		for (i = 1; i < pTrack->PointsN; i++)
			pDiffs[i - 1] = floor((*(float *)((BYTE *)&pTrack->pPoints[i] + CoordOffset * sizeof(float)) - pCoding->Base) / Precision + 0.5f);
		for (i = pTrack->PointsN - 2; i > 0; i--) pDiffs[i] -= pDiffs[i - 1];
/*
		pDiffs[0] = (*(float *)((BYTE *)&pTrack->pPoints[1] + CoordOffset * sizeof(float)) - pCoding->Base) / Precision;
		for (i = 2; i < pTrack->PointsN; i++)
			pDiffs[i - 1] = (*(float *)((BYTE *)&pTrack->pPoints[i] + CoordOffset * sizeof(float)) - pCoding->Base) / Precision - pDiffs[i - 2];
*/
		int mind, maxd;
		unsigned range;
		mind = maxd = pDiffs[0];
		for (i = 1; i < (N - 1); i++)
		{
			if (mind > pDiffs[i]) mind = pDiffs[i];
			else if (maxd < pDiffs[i]) maxd = pDiffs[i];
			};
		pCoding->Polarization = mind;
		range = maxd - mind;
		tester = 1;
		pCoding->Bits = 1;
		while (tester && range > tester)
		{
			tester <<= 1;
			pCoding->Bits++;
			};
		if (!tester)
		{
			if (pDiffs) free(pDiffs);
			return false;
			};
		for (i = 0; i < (pTrack->PointsN - 1); i++)
			pDiffs[i] -= mind;
		};
	int index = Pack(N - 1, pDiffs, pdata, pCoding->Bits, pbitpos);
	free(pDiffs);
	return index;
	};

static int DecodeCoord(IO_BULK_Compressed_GrainSection_TrackHeader::t_Coding Coding, unsigned CoordOffset, float Precision, Track *pTrack, BYTE *pdata, int *pbitpos)
{
	int N = pTrack->PointsN;
	if (!N) return 0;
	int *pDiffs	= (int *)malloc(sizeof(int) * (N - 1));
	if (!pDiffs) return -1;
	bool CheckException = false;
	unsigned index = Unpack(N - 1, pdata, pDiffs, Coding.Bits, pbitpos, &CheckException);
	if (CheckException) return -1;
	int i;
	for (i = 0; i < (N - 1); i++)
		pDiffs[i] += Coding.Polarization;
	for (i = 1; i < (N - 1); i++)
		pDiffs[i] += pDiffs[i - 1];
	*(float *)((BYTE *)&pTrack->pPoints[0] + sizeof(float) * CoordOffset) = Coding.Base;
	for (i = 1; i < N; i++)
		*(float *)((BYTE *)&pTrack->pPoints[i] + sizeof(float) * CoordOffset) = Coding.Base + Precision * pDiffs[i - 1];
	free(pDiffs);
	return index;
	};



// Packing / Unpacking functions

unsigned Pack(unsigned count, int *psource, BYTE *pdata, int bits, int *pbitpos)
{	
	memset(pdata, 0, count * sizeof(unsigned));
	unsigned index = 0;
	int bitpos = *pbitpos;
	if (!count) return 0;
	unsigned data = 0;
	int avbits = 0;
	int i = 0;
	while (i < count || avbits)
	{		
		if (!avbits)
		{
			data = psource[i++];
			avbits = bits;
			};
		if ((bitpos + avbits) >= 8)
		{
			unsigned w = 8 - bitpos;
			pdata[index++] |= (data >> (avbits - w));
			if (avbits == w) data = 0;
			else data &= ((0xffffffff) >> (32 - avbits + w));
			avbits -= w;
			bitpos = 0;
			}
		else
		{
			pdata[index] |= (data << (8 - bitpos - avbits));
			bitpos += avbits;
			avbits = 0;
			};
		};	
	*pbitpos = bitpos;
	return index;
	};

static unsigned Unpack(unsigned count, BYTE *psource, int *pdata, int bits, int *pbitpos, bool *pException)
{
	*pException = false;
	int avbits = (8 - *pbitpos);
	unsigned index = 0;
	try
	{		
		if (!count) return 0;
		int i = 0;		
		unsigned data = psource[0] & (0xff >> *pbitpos);		
		while (i < count)
		{
			if (avbits < bits)
			{
				data = (data << 8) | psource[++index];
				avbits += 8;				
				}
			else
			{
				pdata[i++] = (data >> (avbits - bits));
				if (avbits == bits) data = 0;
				else data &= (0xffffffff >> (32 - avbits + bits));
				avbits -= bits;
				};
			};
		}
	catch (...)
	{
		*pException = true;
		return 0;
		};
	if (avbits) *pbitpos = 8 - avbits;
	else 
	{
		*pbitpos = 0;
		index++;
		};
	return index;
	};