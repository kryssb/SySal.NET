#include "Stdafx.h"
#include "NVIDIAImgProc64.h"
#include <stdio.h>

using namespace SySal::Imaging;

#define READ_INT_INTO(p, var) { var = *(int *)p; p = (void *)((char *)p + sizeof(int)); }

static char *FromString(System::String ^s)
{
	const int len = 2048;
	static char str[len];
	int i, l;
	l = s->Length;
	if (l >= len) l = len - 1;
	for (i = 0; i < l; i++)
		str[i] = s[i];
	str[i] = 0;
	return str;
}

NVIDIAClusterSequenceContainer::NVIDIAClusterSequenceContainer(void *data, bool ownbuffer) : pData(data), m_OwnsBuffer(ownbuffer)
{
	void *p = pData;
	READ_INT_INTO(p, m_Images);
	if (m_Images == 0)
	{
		pImageClusterCounts = 0;
		pImageOffsets = 0;		
	}
	else
	{
		pImageClusterCounts = new int[m_Images];
		pImageOffsets = new IntCluster *[m_Images];		
	}
	READ_INT_INTO(p, m_Scale);
	m_RescalingFactor = 1.0 / m_Scale;
	int i;
	int totalclusters = 0;
	pImageInfo = (short *)p;
	p = (short *)p + 10;
	pPixelToMicron = (double *)p;
	p = (double *)p + 2;
	pImagePositions = (double *)p;
	p = (double *)p + 3 * m_Images;
	for (i = 0; i < m_Images; i++)
	{
		READ_INT_INTO(p, pImageClusterCounts[i]);
		totalclusters += pImageClusterCounts[i];
	}
	pImageOffsets[0] = (IntCluster *)p;
	for (i = 1; i < m_Images; i++)
		pImageOffsets[i] = pImageOffsets[i - 1] + pImageClusterCounts[i - 1];
	m_TotalSize = ((char *)p - (char *)pData) + sizeof(IntCluster) * totalclusters + sizeof(int);
	*((int *)(void *)((char *)pData + m_TotalSize - sizeof(int))) = m_TotalSize;
}

NVIDIAClusterSequenceContainer::~NVIDIAClusterSequenceContainer()
{
	if (m_OwnsBuffer && pData)
	{
		delete [] pData;
		pData = 0;
	}
	if (pImageClusterCounts)
	{
		delete [] pImageClusterCounts;
		pImageClusterCounts = 0;
	}
	if (pImageOffsets)
	{
		delete [] pImageOffsets;
		pImageOffsets = 0;
	}
}

NVIDIAClusterSequenceContainer::!NVIDIAClusterSequenceContainer()
{
	if (m_OwnsBuffer && pData)
	{
		delete [] pData;
		pData = 0;
	}
	if (pImageClusterCounts)
	{
		delete [] pImageClusterCounts;
		pImageClusterCounts = 0;
	}
	if (pImageOffsets)
	{
		delete [] pImageOffsets;
		pImageOffsets = 0;
	}
}

int NVIDIAClusterSequenceContainer::Images::get()
{
	return m_Images;
}

int NVIDIAClusterSequenceContainer::ClustersInImage(int img)
{
	if (img < 0 || img >= m_Images) 
		throw gcnew System::IndexOutOfRangeException();
	return pImageClusterCounts[img];
}

SySal::Imaging::Cluster NVIDIAClusterSequenceContainer::Cluster(int img, int number)
{
	if (img < 0 || img >= m_Images) throw gcnew System::IndexOutOfRangeException();
	if (number < 0 || number >= pImageClusterCounts[img]) throw gcnew System::IndexOutOfRangeException();
	SySal::Imaging::Cluster cl;
	IntCluster &rc = *(pImageOffsets[img] + number);
	cl.Area = rc.Area;
	cl.X = rc.X * m_RescalingFactor;
	cl.Y = rc.Y * m_RescalingFactor;
	cl.Inertia.IXX = Math::Max(rc.XX * m_RescalingFactor, 0.0);
	cl.Inertia.IYY = Math::Max(rc.YY * m_RescalingFactor, 0.0);
	cl.Inertia.IXY = rc.XY * m_RescalingFactor;
	double d = cl.Inertia.IXX * cl.Inertia.IYY;
	if (cl.Inertia.IXY * cl.Inertia.IXY > d)
		cl.Inertia.IXY = Math::Sign(cl.Inertia.IXY) * Math::Sqrt(d);
	return cl;
}

void NVIDIAClusterSequenceContainer::WriteToFile(System::String ^file)
{
	FILE *wf = fopen(FromString(file), "wb");
	if (wf == 0)
	{
		throw gcnew Exception("Can't open file.");
	}
	long wsize = fwrite(pData, 1, m_TotalSize, wf);
	fclose(wf);
	if (wsize != m_TotalSize && m_TotalSize != 0) throw gcnew Exception("Can't write file correctly.");
}

SySal::BasicTypes::Vector2 NVIDIAClusterSequenceContainer::PixelToMicron::get()
{
	SySal::BasicTypes::Vector2 v;
	v.X = pPixelToMicron[0];
	v.Y = pPixelToMicron[1];
	return v;
}

void NVIDIAClusterSequenceContainer::PixelToMicron::set(SySal::BasicTypes::Vector2 v)
{	
	pPixelToMicron[0] = v.X;
	pPixelToMicron[1] = v.Y;
}

void NVIDIAClusterSequenceContainer::SetImagePosition(int img, SySal::BasicTypes::Vector p)
{
	if (img < 0 || img >= m_Images) throw gcnew System::IndexOutOfRangeException();
	pImagePositions[3 * img] = p.X;
	pImagePositions[3 * img + 1] = p.Y;
	pImagePositions[3 * img + 2] = p.Z;
}

SySal::BasicTypes::Vector NVIDIAClusterSequenceContainer::GetImagePosition(int img)
{
	if (img < 0 || img >= m_Images) throw gcnew System::IndexOutOfRangeException();
	SySal::BasicTypes::Vector v;
	v.X = pImagePositions[3 * img];
	v.Y = pImagePositions[3 * img + 1];
	v.Z = pImagePositions[3 * img + 2];
	return v;
}

int NVIDIAClusterSequenceContainer::ImageWidth::get()
{
	return pImageInfo[0];
}

int NVIDIAClusterSequenceContainer::ImageHeight::get()
{
	return pImageInfo[1];
}