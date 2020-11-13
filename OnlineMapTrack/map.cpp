#include "map.h"
#include <stdio.h>
#include <malloc.h>

using namespace SySal;

IntClusterFile::IntClusterFile(const char *filename, bool loadclusters) : pData(0)
{
	FILE *f = fopen(filename, "rb");
	if (f == 0) throw "Can't open cluster file!";
	int imgs;
	fread(&imgs, sizeof(imgs), 1, f);
	fseek(f, 2 * sizeof(int) + 2 * sizeof(short) + 16 + (2 + 3 * imgs) * sizeof(double), SEEK_SET);
	int i, c, total;	
	for (i = total = 0; i < imgs; i++)
	{
		fread(&c, sizeof(int), 1, f);
		total += c;
	}
	TotalSize = 2 * sizeof(int) + 2 * sizeof(short) + 16 + (2 + 3 * imgs) * sizeof(double) + imgs * sizeof(int) + total * (loadclusters ? sizeof(IntCluster) : 0);
	pData = malloc(TotalSize);
	fseek(f, 0, SEEK_SET);
	fread(pData, TotalSize, 1, f);
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