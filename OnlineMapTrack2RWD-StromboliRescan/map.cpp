#include "Stdafx.h"
#include "map.h"
#include <stdio.h>
#include <malloc.h>
#include <memory.h>

using namespace SySal;

inline void throw_on_fread(void *ptr, size_t size, size_t count, FILE *stream, void * &pData, const char *filename) 
{
	if (fread(ptr, size, count, stream) != count)
	{
		printf("\nFILEREADERROR on file %s\n");
		if (pData)
		{
			free(pData);
			pData = 0;
		}
		throw "Can't read cluster file!";
	}
}

inline void throw_on_fseek(FILE *stream, long int offset, int origin, void * &pData, const char *filename) 
{
	if (fseek(stream, offset, origin) != 0)
	{
		printf("\nFILESEEKERROR on file %s\n");
		if (pData)
		{
			free(pData);
			pData = 0;
		}
		throw "Can't read cluster file!";
	}
}

IntClusterFile::IntClusterFile(const char *filename, bool loadclusters) : pData(0)
{
	FILE *f = fopen(filename, "rb");
	if (f == 0) 
	{
		printf("\nFILEOPENERROR on file %s\n", filename);
		throw "Can't open cluster file!";
	}
	int imgs;
	throw_on_fread(&imgs, sizeof(imgs), 1, f, pData, filename);
	fseek(f, 2 * sizeof(int) + 2 * sizeof(short) + 16 + (2 + 3 * imgs) * sizeof(double), SEEK_SET);
	int i, c, total;	
	for (i = total = 0; i < imgs; i++)
	{
		throw_on_fread(&c, sizeof(int), 1, f, pData, filename);
		total += c;
	}
	TotalSize = 2 * sizeof(int) + 2 * sizeof(short) + 16 + (2 + 3 * imgs) * sizeof(double) + imgs * sizeof(int) + total * (loadclusters ? sizeof(IntCluster) : 0);		
	pData = malloc(TotalSize);
	fseek(f, 0, SEEK_SET);
	throw_on_fread(pData, TotalSize, 1, f, pData, filename);
	pImages = (int *)pData;

	pScale = (int *)pData + 1;
	pWidth = (short *)(void *)(pScale + 1);
	pHeight = pWidth + 1;
	pPixMicronX = (double *)((char *)(void *)(pHeight + 1) + 16);
	pPixMicronY = pPixMicronX + 1;
	pStagePos = pPixMicronY + 1;
	pImageClusterCounts = (int *)(void *)(pStagePos + 3 * imgs);
	pClusters = (IntCluster *)(void *)(pImageClusterCounts + imgs);

	fclose(f);
}

IntClusterFile::IntClusterFile()
{
	int TotalSize = 2 * sizeof(int) + 2 * sizeof(short) + 16;
	pData = malloc(TotalSize);
	pImages = (int *)pData;

	pScale = (int *)pData + 1;
	pWidth = (short *)(void *)(pScale + 1);
	pHeight = pWidth + 1;
	pPixMicronX = (double *)((char *)(void *)(pHeight + 1) + 16);
	pPixMicronY = pPixMicronX + 1;
	pStagePos = 0;
	pImageClusterCounts = 0;
	pClusters = 0;
}

void IntClusterFile::CopyEmpty(IntClusterFile &ic)
{
	int TotalSize = 2 * sizeof(int) + 2 * sizeof(short) + 16 + (2 + 3 * ic.Images()) * sizeof(double) + ic.Images() * sizeof(int);
	free(pData);

	pData = malloc(TotalSize);
	pImages = (int *)pData;

	pScale = (int *)pData + 1;
	pWidth = (short *)(void *)(pScale + 1);
	pHeight = pWidth + 1;
	pPixMicronX = (double *)((char *)(void *)(pHeight + 1) + 16);
	pPixMicronY = pPixMicronX + 1;
	pStagePos = pPixMicronY + 1;
	pImageClusterCounts = (int *)(void *)(pStagePos + 3 * ic.Images());
	pClusters = (IntCluster *)(void *)(pImageClusterCounts + ic.Images());

	*pImages = ic.Images();
	*pScale = ic.Scale();
	*pWidth = ic.Width();
	*pHeight = ic.Height();
	*pPixMicronX = ic.PixMicronX();
	*pPixMicronY = ic.PixMicronY();

	memcpy(pStagePos, ic.pStagePos, sizeof(double) * 3 * ic.Images());
	int i;
	for (i = 0; i < ic.Images(); i++)	
		pImageClusterCounts[i] = 0;			
}

IntClusterFile::~IntClusterFile()
{
	if (pData) free(pData);
}

void IntClusterFile::Dump()
{
	printf("\n\nClusterFile dump:\nImages: %d\nScale: %d\nWidth: %d\nHeight: %d\nPix/MicronX: %f\nPix/MicronY: %f\nCluster counts:",
		Images(), Scale(), Width(), Height(), PixMicronX(), PixMicronY());
	int i;
	for (i = 0; i < Images(); i++)
		printf("\n  Image %d: %d, (%.1f %.1f %.1f)", i, pImageClusterCounts[i], pStagePos[3 * i], pStagePos[3 * i + 1], pStagePos[3 * i + 2]);
	printf("\nTotalSize: %d", TotalSize);
	printf("\nEnd dump.");
}