// This is the main DLL file.

#include "stdafx.h"

#include "NVIDIAImgProc64.h"
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

using namespace SySal::Imaging;

#define CHECKDUMP { FILE *ggf = fopen("c:\\temp\\nv.txt", "a+t"); fprintf(ggf, "\n%d", __LINE__); fclose(ggf); }
#define CHECKDUMPPTR(x) { FILE *ggf = fopen("c:\\temp\\nv.txt", "a+t"); fprintf(ggf, "\n%16X", x); fclose(ggf); }
#undef DEBUG_MONITOR_ALLOCATION

SySal::Imaging::ImageInfo NVIDIAImage::MakeInfo(int w, int h)
{
	SySal::Imaging::ImageInfo info;
	info.BitsPerPixel = 8;
	info.Height = h;
	info.Width = w;
	info.PixelFormat = SySal::Imaging::PixelFormatType::GrayScale8;
	return info;
}

NVIDIAImage::NVIDIAImage(int w, int h, unsigned subimgs) : LinearMemoryImage(MakeInfo(w, h * subimgs), System::IntPtr(CUDAManager::MallocHost((long)w * (long)h * subimgs)), subimgs, this)
{
	m_TotalSize = 0;	
	if ((void *)m_MemoryAddress == 0) throw gcnew System::Exception(gcnew System::String("Can't allocate memory for ") + w.ToString() + gcnew System::String("x") + h.ToString() + gcnew System::String(" image."));
	m_TotalSize = w * (long long)h * subimgs;	
	m_OwnsBuffer = true;
}

NVIDIAImage::NVIDIAImage(SySal::Imaging::Image ^im) : LinearMemoryImage(MakeInfo(im->Info.Width, im->Info.Height), System::IntPtr(CUDAManager::MallocHost((long)im->Info.Width * (long)im->Info.Height)), 1, this)
{
	m_OwnsBuffer = true;
	LinearMemoryImage ^lmi = nullptr;
	try
	{
		lmi = (LinearMemoryImage ^)im;
	}
	catch (System::Exception ^x) 
	{
		lmi = nullptr;
	}
	int w = this->Info.Width;
	int h = this->Info.Height;
	m_SubImages = 1;
	if (lmi != nullptr && lmi->Info.BitsPerPixel == this->Info.BitsPerPixel) 
	{
		m_SubImages = lmi->SubImages;
		memcpy((void *)m_MemoryAddress, (void *)NVIDIAImage::AccessMemoryAddress(lmi), (this->Info.BitsPerPixel / 8) * (long)w * (long)h * m_SubImages);
	}
	else
	{
		unsigned short ix, iy;
		unsigned char *pc = (unsigned char *)(void *)m_MemoryAddress;
		SySal::Imaging::IImagePixels ^ipix = im->Pixels;
		for (iy = 0; iy < h; iy++)
		{
			unsigned char *pd = pc + iy * w;
			for (ix = 0; ix < w; ix++)
			{
				pd[ix] = ipix[ix, iy, 0];
			}
		}
	}
}

NVIDIAImage::NVIDIAImage(int w, int h, unsigned subimgs, void *buffer) : LinearMemoryImage(MakeInfo(w, h), (System::IntPtr)buffer, subimgs, this) 
{
	m_OwnsBuffer = false;
}

SySal::Imaging::Image ^NVIDIAImage::SubImage(unsigned i)
{
	return gcnew NVIDIAImage(this->Info.Width, this->Info.Height / m_SubImages, 1, (void *)((unsigned char *)(void *)m_MemoryAddress + (((long long)this->Info.Width * (long long)this->Info.Height) / m_SubImages) * i));
}

System::Drawing::Image ^NVIDIAImage::DrawingImage::get()
{
	System::Drawing::Image ^im = gcnew System::Drawing::Bitmap((int)this->Info.Width, (int)this->Info.Height, (int)this->Info.Width, System::Drawing::Imaging::PixelFormat::Format8bppIndexed, m_MemoryAddress);
	System::Drawing::Imaging::ColorPalette ^palette = im->Palette;
	int i;
	cli::array<System::Drawing::Color> ^colors = im->Palette->Entries;
	for (i = 0; i < 256; i++)
	{
		colors[i] = System::Drawing::Color::FromArgb(i, i, i);
	}
	im->Palette = palette;
	return im;
}


NVIDIAImage::!NVIDIAImage()
{
	if (m_OwnsBuffer == false) return;
	if ((void *)m_MemoryAddress != 0) CUDAManager::FreeHost((void *)m_MemoryAddress);
	m_MemoryAddress = System::IntPtr(0);
}

NVIDIAImage::~NVIDIAImage()
{
	this->!NVIDIAImage();
}

unsigned char NVIDIAImage::default::get(unsigned p)
{
	if (p >= m_TotalSize) throw gcnew System::IndexOutOfRangeException();
	return ((unsigned char *)(void *)m_MemoryAddress)[p];
}

void NVIDIAImage::default::set(unsigned p, unsigned char ch)
{
	if (p >= m_TotalSize) throw gcnew System::IndexOutOfRangeException();
	((unsigned char *)(void *)m_MemoryAddress)[p] = ch;
}

unsigned char NVIDIAImage::default::get(unsigned short x, unsigned short y, unsigned short c)
{
	if (x >= this->Info.Width || y >= this->Info.Height || c > 0) throw gcnew System::IndexOutOfRangeException();
	return ((unsigned char *)(void *)m_MemoryAddress)[y * this->Info.Width + x];
}

void NVIDIAImage::default::set(unsigned short x, unsigned short y, unsigned short c, unsigned char ch)
{
	if (x >= this->Info.Width || y >= this->Info.Height || c > 0) throw gcnew System::IndexOutOfRangeException();
	((unsigned char *)(void *)m_MemoryAddress)[y * this->Info.Width + x] = ch;
}

unsigned short NVIDIAImage::Channels::get() { return 1; }

void *NVIDIAImage::_AccessMemoryAddress(SySal::Imaging::LinearMemoryImage ^im)
{
	return (void *)NVIDIAImage::AccessMemoryAddress(im);
}

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

NVIDIAImage ^NVIDIAImage::FromBMPFiles(cli::array<System::String ^> ^bmpfiles)
{
	BITMAPFILEHEADER FHdr;
	BITMAPINFO Hdr;	
	FILE *rf = fopen(FromString(bmpfiles[0]), "rb");
	if (rf == 0) throw gcnew Exception("Can't open file.");
	fread(&FHdr, sizeof(FHdr), 1, rf);
	fread(&Hdr, sizeof(Hdr), 1, rf);
	fclose(rf);
	int w = 0, h = 0;
	if (FHdr.bfType != (((unsigned)'M') << 8 | ((unsigned)'B')) || Hdr.bmiHeader.biBitCount != 8 || Hdr.bmiHeader.biCompression != BI_RGB) throw gcnew Exception("Wrong file format.");
	w = abs(Hdr.bmiHeader.biWidth);
	h = abs(Hdr.bmiHeader.biHeight);
	int h4 = (h / 4) * 4;
	NVIDIAImage ^img = gcnew NVIDIAImage(w, h4, bmpfiles->Length);
	int imageindex;
	for (imageindex = 0; imageindex < bmpfiles->Length; imageindex++)
	{
		rf = fopen(FromString(bmpfiles[imageindex]), "rb");
		if (rf == 0) 
		{
			img->!NVIDIAImage();
			throw gcnew Exception("Can't open file.");
		}
		fread(&FHdr, sizeof(FHdr), 1, rf);
		fread(&Hdr, sizeof(Hdr), 1, rf);
		if (FHdr.bfType != (((unsigned)'M') << 8 | ((unsigned)'B')) || Hdr.bmiHeader.biBitCount != 8 || Hdr.bmiHeader.biCompression != BI_RGB) 
		{
			fclose(rf);
			img->!NVIDIAImage();
			throw gcnew Exception("Wrong file format.");
		}
		if (abs(Hdr.bmiHeader.biWidth) != w || abs(Hdr.bmiHeader.biHeight) != h)
		{
			fclose(rf);
			img->!NVIDIAImage();
			throw gcnew Exception("All files must have the same format.");
		}		
		fseek(rf, FHdr.bfOffBits - sizeof(FHdr) - sizeof(Hdr), SEEK_CUR);
		if (Hdr.bmiHeader.biHeight < 0)
			fread((unsigned char *)(void *)img->m_MemoryAddress + w * h4 * imageindex, w * h4, 1, rf);
		else
		{
			int i;
			for (i = 0; i < h; i++)
				fread((unsigned char *)(void *)img->m_MemoryAddress + w * h4 * imageindex + w * (h4 - 1 - i), w, 1, rf);
		}
		fclose(rf);
	}
	return img;	
}

void NVIDIAImage::ToBMPFiles(cli::array<System::String ^> ^bmpfiles)
{
	if (bmpfiles->Length != m_SubImages) throw gcnew Exception("The number of files must match the number of sub-images.");
	int imageindex;
	BITMAPFILEHEADER FHdr;
	BITMAPINFOHEADER Hdr;	
	Hdr.biBitCount = 8;
	Hdr.biClrImportant = 256;
	Hdr.biClrUsed = 256;
	Hdr.biCompression = BI_RGB;
	Hdr.biHeight = this->Info.Height / m_SubImages;
	Hdr.biWidth = this->Info.Width;
	Hdr.biPlanes = 1;
	Hdr.biSize = sizeof(BITMAPINFOHEADER);
	Hdr.biSizeImage = 0;
	Hdr.biXPelsPerMeter = Hdr.biYPelsPerMeter = 10000;
	FHdr.bfType = (((unsigned)'M') << 8 | ((unsigned)'B'));
	FHdr.bfSize = sizeof(FHdr) + sizeof(Hdr) + sizeof(RGBQUAD) * 256 + Hdr.biWidth * Hdr.biHeight;
	FHdr.bfReserved1 = FHdr.bfReserved2 = 0;
	FHdr.bfOffBits = sizeof(FHdr) + sizeof(Hdr) + sizeof(RGBQUAD) * 256;
	int i;
	RGBQUAD palette[256];
	for (i = 0; i < 256; i++)
	{
		palette[i].rgbRed = palette[i].rgbGreen = palette[i].rgbBlue = i;
		palette[i].rgbReserved = 0;
	}
	for (imageindex = 0; imageindex < m_SubImages; imageindex++)
	{
		FILE *wf = fopen(FromString(bmpfiles[imageindex]), "wb");
		fwrite(&FHdr, sizeof(FHdr), 1, wf);
		fwrite(&Hdr, sizeof(Hdr), 1, wf);
		fwrite(palette, sizeof(RGBQUAD), 256, wf);
		for (i = 0; i < Hdr.biHeight; i++)
			fwrite((unsigned char *)(void *)m_MemoryAddress + imageindex * Hdr.biWidth * Hdr.biHeight + (Hdr.biHeight - 1 - i) * Hdr.biWidth, Hdr.biWidth, 1, wf);
		fclose(wf);
	}
}

int NVIDIAImageProcessor::NVIDIADevices::get()
{
	return CUDAManager::GetDeviceCount();
}

NVIDIAImageProcessor::NVIDIAImageProcessor() : pMgr(0)
{
	pMgr = new CUDAManager(0);
	if (pMgr->LastError[0]) throw gcnew Exception(gcnew System::String(pMgr->LastError));		
	m_ImgWnd.MinX = m_ImgWnd.MaxX = -1;
	m_ImgWnd.MinY = m_ImgWnd.MaxY = -1;
	o_Clusters = nullptr;
	o_Segments = nullptr;
	o_BinarizedImages = nullptr;
	m_ClusterContainer = nullptr;
	m_ClusterContainerMemBlockHalfSize = 512 * 1024 * 1024;
	while (m_ClusterContainerMemBlockHalfSize > 0 && (pClusterContainerMemBlockBase = (void *) GlobalAlloc(GMEM_FIXED, m_ClusterContainerMemBlockHalfSize)) == 0)	
		m_ClusterContainerMemBlockHalfSize /= 2;
	m_ClusterContainerMemBlockHalfSize /= 2;
	m_ClusterContainerWatermark0 = m_ClusterContainerWatermark1 = 0;
}

NVIDIAImageProcessor::NVIDIAImageProcessor(int board) : pMgr(0)
{
	pMgr = new CUDAManager(board);
	if (pMgr->LastError[0]) throw gcnew System::ArgumentException(gcnew System::String(pMgr->LastError));		
	m_ImgWnd.MinX = m_ImgWnd.MaxX = -1;
	m_ImgWnd.MinY = m_ImgWnd.MaxY = -1;
	o_Clusters = nullptr;
	o_Segments = nullptr;
	o_BinarizedImages = nullptr;
	m_ClusterContainer = nullptr;
	m_ClusterContainerMemBlockHalfSize = 512 * 1024 * 1024;
	while (m_ClusterContainerMemBlockHalfSize > 0 && (pClusterContainerMemBlockBase = (void *) GlobalAlloc(GMEM_FIXED, m_ClusterContainerMemBlockHalfSize)) == 0)	
		m_ClusterContainerMemBlockHalfSize /= 2;
	m_ClusterContainerMemBlockHalfSize /= 2;
	m_ClusterContainerWatermark0 = m_ClusterContainerWatermark1 = 0;
}

NVIDIAImageProcessor::!NVIDIAImageProcessor()
{
	if (pMgr) this->~NVIDIAImageProcessor();
}

NVIDIAImageProcessor::~NVIDIAImageProcessor()
{
	if (pClusterContainerMemBlockBase)
	{
		GlobalFree(pClusterContainerMemBlockBase);
		pClusterContainerMemBlockBase = 0;
	}
	if (pMgr)
	{
		delete pMgr;
		pMgr = 0;
	}
	if (m_ClusterContainer != nullptr)
	{

	}
	GC::SuppressFinalize(this);
}

ImageInfo NVIDIAImageProcessor::ImageFormat::get()
{
	ImageInfo info;
	info.BitsPerPixel = 8;
	info.PixelFormat = SySal::Imaging::PixelFormatType::GrayScale8;
	info.Width = pMgr->ImageWidth;
	info.Height = pMgr->ImageHeight;	
	return info;
}

void NVIDIAImageProcessor::ImageFormat::set(ImageInfo v)
{
	if (v.BitsPerPixel != 8 || v.PixelFormat != SySal::Imaging::PixelFormatType::GrayScale8) throw gcnew Exception("Only 8-bit grayscale images are supported.");
	if (v.Width <= 0 || v.Height <= 0 || (v.Width % 4) || (v.Height % 4) || (v.Width * v.Height * 4 > pMgr->AvailableMemory)) throw gcnew Exception("Invalid image size requested");
	m_ClusterContainerWatermark0 = m_ClusterContainerWatermark1 = 0;
	pMgr->ImageWidth = v.Width;
	pMgr->ImageHeight = v.Height;
	pMgr->Changed = true;	
	if (pMgr->ReconfigureMemory() == false)
		throw gcnew System::Exception(gcnew System::String(pMgr->LastError));
}

int NVIDIAImageProcessor::MaxImages::get()
{
	if (pMgr->ImageWidth <= 0 || pMgr->ImageHeight <= 0) throw gcnew Exception("Maximum number of images depends on image size.\r\nPlease set image size.");
	if (pMgr->Changed) 
	{
		if (pMgr->ReconfigureMemory() == false) throw gcnew Exception(gcnew System::String(pMgr->LastError));
	}
	return pMgr->MaxImages;
}

ImageProcessingFeatures NVIDIAImageProcessor::OutputFeatures::get()
{
	return (ImageProcessingFeatures)
		(
		(pMgr->WkCfgDumpEqImages ? (int)ImageProcessingFeatures::EqualizedImage : 0) |
		(pMgr->WkCfgDumpBinImages ? (int)ImageProcessingFeatures::BinarizedImage : 0) |
		(pMgr->WkCfgDumpSegments ? (int)ImageProcessingFeatures::Segments : 0) |
		(pMgr->WkCfgDumpClusters ? (int)ImageProcessingFeatures::Clusters : 0) |
		(pMgr->WkCfgDumpClusters2ndMomenta ? (int)ImageProcessingFeatures::Cluster2ndMomenta : 0)
		);
}

void NVIDIAImageProcessor::OutputFeatures::set(ImageProcessingFeatures v)
{
	pMgr->SetWorkingConfiguration(
		((int)v & (int)ImageProcessingFeatures::EqualizedImage) == (int)ImageProcessingFeatures::EqualizedImage, 
		((int)v & (int)ImageProcessingFeatures::BinarizedImage) == (int)ImageProcessingFeatures::BinarizedImage,
		((int)v & (int)ImageProcessingFeatures::Segments) == (int)ImageProcessingFeatures::Segments,
		((int)v & (int)ImageProcessingFeatures::Clusters) == (int)ImageProcessingFeatures::Clusters,
		((int)v & (int)ImageProcessingFeatures::Cluster2ndMomenta) == (int)ImageProcessingFeatures::Cluster2ndMomenta
		);
}
		
ImageWindow NVIDIAImageProcessor::ProcessingWindow::get() 
{
	return m_ImgWnd;
}

void NVIDIAImageProcessor::ProcessingWindow::set(ImageWindow v) 
{
	if (v.MinX < 0 || v.MinX >= pMgr->ImageWidth || v.MinY < 0 || v.MinY >= pMgr->ImageHeight || v.MaxX < v.MinX || v.MaxX >= pMgr->ImageWidth || v.MaxY < v.MinY || v.MaxY >= pMgr->ImageHeight) throw gcnew Exception("Wrong image window set.");
	m_ImgWnd = v;
}

bool NVIDIAImageProcessor::IsReady::get() 
{
	return !pMgr->Changed;
}

void NVIDIAImageProcessor::Input::set(SySal::Imaging::LinearMemoryImage ^inputimages)
{
	if (IsReady == false) throw gcnew System::Exception("GPU not ready to work.");
	const double r = 1.0 / 256.0;
	o_Clusters = nullptr;
	o_Segments = nullptr;
	o_BinarizedImages = nullptr;
	o_Warnings = nullptr;
	int n_img = inputimages->SubImages;
	if (m_ClusterContainerMemBlockHalfSize - m_ClusterContainerWatermark() < NVIDIAClusterSequenceContainer::MaxSize(n_img, pMgr->MaxClustersPerImage))
	{	
		char tempstr [512];
		if (NVIDIAClusterSequenceContainer::MaxSize(n_img, pMgr->MaxClustersPerImage) > m_ClusterContainerMemBlockHalfSize)
		{
			sprintf(tempstr, "Too big cluster container: %d available, %d required.", m_ClusterContainerMemBlockHalfSize, NVIDIAClusterSequenceContainer::MaxSize(n_img, pMgr->MaxClustersPerImage));
			throw gcnew SySal::Imaging::Fast::PermanentMemoryException(gcnew System::String(tempstr));
		}
		sprintf(tempstr, "Not enough space for cluster container: %d available, %d required.", m_ClusterContainerMemBlockHalfSize - m_ClusterContainerWatermark(), NVIDIAClusterSequenceContainer::MaxSize(n_img, pMgr->MaxClustersPerImage));
		throw gcnew SySal::Imaging::Fast::TemporaryMemoryException(gcnew System::String(tempstr));
	}	

	if (pMgr->ProcessImagesV1((unsigned char *)NVIDIAImage::_AccessMemoryAddress(inputimages), n_img, (char *)pClusterContainerMemBlock() + m_ClusterContainerWatermark()) == false)
		throw gcnew SySal::Imaging::Fast::AlgorithmException(gcnew System::String(pMgr->LastError));
	o_Median = pMgr->pHostErrorImage[pMgr->MaxImages];
	{
		int i, nw;
		for (i = nw = 0; i < n_img; i++)
			if (pMgr->pHostErrorImage[i])
			{
				if (pMgr->pHostErrorImage[i] & NVIDIAIMGPROC_ERR_SEGMENT_OVERFLOW) nw++;
				if (pMgr->pHostErrorImage[i] & NVIDIAIMGPROC_ERR_CLUSTER_OVERFLOW) nw++;
				if (pMgr->pHostErrorImage[i] & ~(NVIDIAIMGPROC_ERR_SEGMENT_OVERFLOW | NVIDIAIMGPROC_ERR_CLUSTER_OVERFLOW)) nw++;
			}
		o_Warnings = gcnew cli::array<SySal::Imaging::ImageProcessingException ^>(nw);
		for (i = nw = 0; i < n_img; i++)
			if (pMgr->pHostErrorImage[i])
			{
				if (pMgr->pHostErrorImage[i] & NVIDIAIMGPROC_ERR_SEGMENT_OVERFLOW) o_Warnings[nw++] = gcnew SySal::Imaging::ImageProcessingException((unsigned)i, gcnew System::String("Segment overflow during image processing. Solution: allow more segments."));
				if (pMgr->pHostErrorImage[i] & NVIDIAIMGPROC_ERR_CLUSTER_OVERFLOW) o_Warnings[nw++] = gcnew SySal::Imaging::ImageProcessingException((unsigned)i, gcnew System::String("Too many clusters during image processing. Solution: allow more clusters."));
				if (pMgr->pHostErrorImage[i] & ~(NVIDIAIMGPROC_ERR_SEGMENT_OVERFLOW | NVIDIAIMGPROC_ERR_CLUSTER_OVERFLOW)) o_Warnings[nw++] = gcnew SySal::Imaging::ImageProcessingException((unsigned)i, gcnew System::String("Unknown image processing exception (P=" + i + " C=" + pMgr->pHostErrorImage[i].ToString("X08") + ")."));
			}		
	}
	if (pMgr->WkCfgDumpBinImages)
	{
		o_BinarizedImages = gcnew NVIDIAImage(pMgr->ImageWidth, pMgr->ImageHeight * n_img, n_img, pMgr->pHostBinImage);
	}
	if (pMgr->WkCfgDumpSegments)
	{
		int i, j, k, n_seglines, n_s;		
		o_Segments = gcnew cli::array<cli::array<ClusterSegment> ^>(n_img);		
		SySal::Imaging::ClusterSegment seg;
		for (i = 0; i < n_img; i++)
		{
			n_s = 0;
			for (j = 0; j < pMgr->ImageHeight; j++) n_s += pMgr->pHostSegmentCountImage[i * pMgr->ImageHeight + j];
			o_Segments[i] = gcnew cli::array<ClusterSegment>(n_s);
			for (j = n_s = 0; j < pMgr->ImageHeight; j++)
				for (k = 0; k < pMgr->pHostSegmentCountImage[i * pMgr->ImageHeight + j]; k++)
				{
					IntSegment &rs = pMgr->pHostSegmentImage[(i * pMgr->ImageHeight + j) * pMgr->MaxSegmentsPerScanLine + k];
					seg.Left = rs.Left;
					seg.Right = rs.Right;
					seg.Line = j;
					o_Segments[i][n_s++] = seg;
				}			
		}
	}
	if (pMgr->WkCfgDumpClusters || pMgr->WkCfgDumpClusters2ndMomenta)
	{
		m_ClusterContainer = gcnew NVIDIAClusterSequenceContainer((char *)pClusterContainerMemBlock() + m_ClusterContainerWatermark(), false);
#if 0
		o_Clusters = gcnew cli::array<cli::array<Cluster> ^>(n_img);
		int i, j, n_c;
		SySal::Imaging::Cluster cl;
		IntCluster *pHC = (IntCluster *)(void *)((int *)pClusterContainerMemBlock() + (2 + n_img));//pMgr->pHostClusterImage;
		for (i = 0; i < n_img; i++)
		{
			o_Clusters[i] = gcnew cli::array<Cluster>(n_c = /*pMgr->pHostClusterCountImage[i]*/((int *)pClusterContainerMemBlock())[2 + i]);
			for (j = 0; j < n_c; j++)
			{
				IntCluster &rc = pHC[j];
				cl.Area = rc.Area;
				cl.X = rc.XSum * r;
				cl.Y = rc.YSum * r;
				cl.Inertia.IXX = Math::Max(rc.XXSum * r, 0.0);
				cl.Inertia.IYY = Math::Max(rc.YYSum * r, 0.0);
				cl.Inertia.IXY = rc.XYSum * r;
				double d = cl.Inertia.IXX * cl.Inertia.IYY;
				if (cl.Inertia.IXY * cl.Inertia.IXY > d)
					cl.Inertia.IXY = Math::Sign(cl.Inertia.IXY) * Math::Sqrt(d);
				o_Clusters[i][j] = cl;
			}
			pHC += n_c;
		}
#endif
	}
}

unsigned NVIDIAImageProcessor::GreyLevelMedian::get()
{
	return o_Median;
}

SySal::Imaging::LinearMemoryImage ^NVIDIAImageProcessor::EqualizedImages::get()
{
	throw gcnew System::Exception("EqualizedImages not implemented.");
}

SySal::Imaging::LinearMemoryImage ^NVIDIAImageProcessor::FilteredImages::get()
{
	throw gcnew System::Exception("FilteredImages not implemented.");
}

SySal::Imaging::LinearMemoryImage ^NVIDIAImageProcessor::BinarizedImages::get()
{
	return o_BinarizedImages;
}
cli::array<cli::array<ClusterSegment> ^> ^NVIDIAImageProcessor::ClusterSegments::get()
{
	return o_Segments;
}
cli::array<cli::array<Cluster> ^> ^NVIDIAImageProcessor::Clusters::get()
{
	int n_img;
	cli::array<cli::array<Cluster> ^> ^o_clusters = gcnew cli::array<cli::array<Cluster> ^>(n_img = m_ClusterContainer->Images);
	int i, j, n_c;
	SySal::Imaging::Cluster cl;	
	for (i = 0; i < n_img; i++)
	{
		o_clusters[i] = gcnew cli::array<Cluster>(n_c = m_ClusterContainer->ClustersInImage(i));
		for (j = 0; j < n_c; j++)
			o_clusters[i][j] = m_ClusterContainer->Cluster(i, j);			
	}
	return o_clusters;
}
cli::array<SySal::Imaging::ImageProcessingException ^> ^NVIDIAImageProcessor::Warnings::get()
{
	return o_Warnings;
}
void NVIDIAImageProcessor::EqGreyLevelTargetMedian::set(unsigned char level) 
{
	pMgr->GreyLevelTargetMedian = level;
}
unsigned char NVIDIAImageProcessor::EqGreyLevelTargetMedian::get() 
{
	return pMgr->GreyLevelTargetMedian;
}
void NVIDIAImageProcessor::MaxSegmentsPerScanLine::set(unsigned maxsegs)
{
	pMgr->MaxSegmentsPerScanLine = maxsegs;
	pMgr->Changed = true;
	m_ClusterContainerWatermark0 = m_ClusterContainerWatermark1 = 0;
}

unsigned NVIDIAImageProcessor::MaxSegmentsPerScanLine::get()
{
	return pMgr->MaxSegmentsPerScanLine;
}

void NVIDIAImageProcessor::MaxClustersPerImage::set(unsigned maxclusters)
{
	pMgr->MaxClustersPerImage = maxclusters;
	pMgr->Changed = true;
	m_ClusterContainerWatermark0 = m_ClusterContainerWatermark1 = 0;
}

unsigned NVIDIAImageProcessor::MaxClustersPerImage::get()
{
	return pMgr->MaxClustersPerImage;
}

void NVIDIAImageProcessor::EmptyImage::set(SySal::Imaging::Image ^im)
{
	if (im->Info.BitsPerPixel != 8 || 
		im->Info.Width != pMgr->ImageWidth || im->Info.Height != pMgr->ImageHeight)
		throw gcnew System::Exception("An empty image must have the same format as an input image.");	
	long size = im->Info.Width * im->Info.Height;
	SySal::Imaging::IImagePixels ^pixels = im->Pixels;
	unsigned short *pim = (unsigned short *)malloc(size * sizeof(short));
	int i;
	for (i = 0; i < size; i++)
		pim[i] = pixels[i];
	bool res = pMgr->SetEmptyImage(pim);
	free(pim);	
	if (res == false) throw gcnew System::Exception(gcnew System::String(pMgr->LastError));
}

void NVIDIAImageProcessor::ThresholdImage::set(SySal::Imaging::Image ^im)
{
	if (im->Info.BitsPerPixel != 16 ||
		im->Info.Width != pMgr->ImageWidth || im->Info.Height != pMgr->ImageHeight)
		throw gcnew System::Exception("A threshold image must have the same size as an input image and 16 bits per pixel.");
	long size = im->Info.Width * im->Info.Height;
	SySal::Imaging::IImagePixels ^pixels = im->Pixels;
	short *pim = (short *)malloc(size * sizeof(short));
	int iy, ix;
	for (iy = 0; iy < im->Info.Height; iy++)
		for (ix = 0; ix < im->Info.Width; ix++)
			pim[iy * im->Info.Width + ix] = (((unsigned short)pixels[ix, iy, 0]) << 8) | (unsigned short)pixels[ix, iy, 1];	
	bool res = pMgr->SetThresholdImage(pim);
	free(pim);
	if (res == false) throw gcnew System::Exception(gcnew System::String(pMgr->LastError));
}

SySal::Imaging::LinearMemoryImage ^NVIDIAImageProcessor::ImageFromFile(System::String ^filename)
{
	return NVIDIAImage::FromBMPFiles(gcnew cli::array<System::String ^>(1) { filename });
}

void NVIDIAImageProcessor::ImageToFile(SySal::Imaging::LinearMemoryImage ^im, System::String ^filename)
{
	SySal::Imaging::NVIDIAImage ^img = gcnew SySal::Imaging::NVIDIAImage(im->Info.Width, im->Info.Height, im->SubImages, NVIDIAImage::_AccessMemoryAddress(im));
	img->ToBMPFiles(gcnew cli::array<System::String ^>(1) { filename });
}


#define _STR_(x) gcnew System::String(x)
#define _MSTR_(x, _mult_) gcnew System::String("MemSize_" ## #x ## " = ") + (pMgr->MemSize_ ## x * _mult_).ToString() + gcnew System::String(" at ") + ((System::Int64)pMgr->x).ToString("X08") + newline

System::String ^NVIDIAImageProcessor::ToString()
{	
	System::String ^newline = gcnew System::String("\r\n");	
	return 
		_STR_("NVIDIAImageProcessor, Device = ") + pMgr->DeviceId.ToString() + _STR_(", ") + _STR_(pMgr->DeviceName) + newline + 
		_STR_("Total Memory = ") + pMgr->TotalMemory.ToString() + newline + 
		_STR_("Available Memory = ") + pMgr->AvailableMemory.ToString() + newline + 
		_STR_("Max Threads per Block = ") + pMgr->MaxThreadsPerBlock.ToString() + newline +
		_STR_("Max Segments per Scan Line = ") + pMgr->MaxSegmentsPerScanLine.ToString() + newline + 
		_STR_("Max Clusters per Image = ") + pMgr->MaxClustersPerImage.ToString() + newline +
		_STR_("Grey Level Target Median = ") + pMgr->GreyLevelTargetMedian.ToString() + newline +
		_STR_("Configured = ") + (!pMgr->Changed).ToString() + (pMgr->Changed ? newline : 
		(
			newline + _STR_("Max Images = ") + pMgr->MaxImages.ToString() + newline +
			_MSTR_(pDevImage, pMgr->MaxImages) + 
			_MSTR_(pHostEqImage, pMgr->MaxImages) + 
			_MSTR_(pHostBinImage, pMgr->MaxImages) + 
			_MSTR_(pDevHistoImage, 1) + 
			_MSTR_(pDevLookupTable, 1) + 
			_MSTR_(pDev16Image, pMgr->MaxImages) + 
			_MSTR_(pDevThresholdImage, 1) + 
			_MSTR_(pDevEmptyImage, 1) + 
			_MSTR_(pDevSegmentCountImage, pMgr->MaxImages) + 
			_MSTR_(pDevSegmentImage, pMgr->MaxImages) + 
			_MSTR_(pHostSegmentImage, pMgr->MaxImages) + 
			_MSTR_(pHostSegmentCountImage, pMgr->MaxImages) + 
			_MSTR_(pDevClusterWorkImage, pMgr->MaxImages) + 
			_MSTR_(pDevClusterWorkCountImage, pMgr->MaxImages) + 
			_MSTR_(pDevClusterBaseImage, pMgr->MaxImages) + 
			_MSTR_(pDevClusterImage, pMgr->MaxImages) + 
			_MSTR_(pDevClusterCountImage, pMgr->MaxImages) + 
			_MSTR_(pHostClusterImage, pMgr->MaxImages) + 
			_MSTR_(pHostClusterCountImage, pMgr->MaxImages) + 
			_MSTR_(pDevImage, pMgr->MaxImages) + 
			_MSTR_(pHostEqImage, pMgr->MaxImages) + 
			_MSTR_(pHostBinImage, pMgr->MaxImages) + 
			_MSTR_(pDevHistoImage, 1) + 
			_MSTR_(pDevLookupTable, 1) + 
			_MSTR_(pDev16Image, pMgr->MaxImages) + 
			_MSTR_(pDevThresholdImage, 1) + 
			_MSTR_(pDevEmptyImage, 1) + 
			_MSTR_(pDevSegmentCountImage, pMgr->MaxImages) + 
			_MSTR_(pDevSegmentImage, pMgr->MaxImages) + 
			_MSTR_(pHostSegmentImage, pMgr->MaxImages) + 
			_MSTR_(pHostSegmentCountImage, pMgr->MaxImages) + 
			_MSTR_(pDevClusterWorkImage, pMgr->MaxImages) + 
			_MSTR_(pDevClusterWorkCountImage, pMgr->MaxImages) + 
			_MSTR_(pDevClusterBaseImage, pMgr->MaxImages) + 
			_MSTR_(pDevClusterImage, pMgr->MaxImages) + 
			_MSTR_(pDevClusterCountImage, pMgr->MaxImages) + 
			_MSTR_(pHostClusterImage, pMgr->MaxImages) + 
			_MSTR_(pHostClusterCountImage, pMgr->MaxImages) +
			_MSTR_(pDevErrorImage, pMgr->MaxImages * 2) + 
			_MSTR_(pHostErrorImage, pMgr->MaxImages * 2)
		)
	);
}

SySal::Imaging::Fast::IClusterSequenceContainer ^NVIDIAImageProcessor::ClusterSequence::get()
{
	if (m_ClusterContainer != nullptr)
	{
		int oldwatermark = m_ClusterContainerWatermark();
		SetClusterContainerWatermark(((char *)m_ClusterContainer->pData - (char *)pClusterContainerMemBlock()) + m_ClusterContainer->m_TotalSize);
#ifdef DEBUG_MONITOR_ALLOCATION
		{
			FILE *f = fopen("c:\\sysal.net\\logs\\nvidiamem.txt", "a+t");
			fprintf(f, "\nGPU %d ALLOC BANK %d %08X -> %08X", pMgr->DeviceId, m_CurrentBank, oldwatermark, m_ClusterContainerWatermark());
			fclose(f);
		}
#endif
	}
	return m_ClusterContainer;
}

void NVIDIAImageProcessor::ReleaseClusterSequence(SySal::Imaging::Fast::IClusterSequenceContainer ^seq)
{
	SySal::Imaging::NVIDIAClusterSequenceContainer ^nvs = (SySal::Imaging::NVIDIAClusterSequenceContainer ^)seq;
	*(int *)nvs->pData = 0;
	void *pFreeMin = nvs->pData;
	void *pFreeMax = (char *)nvs->pData + nvs->m_TotalSize;	
	int bank = 0;
	char *bkbase = (char *)pClusterContainerMemBlockBase;
	if ((char *)pFreeMax - bkbase > m_ClusterContainerMemBlockHalfSize) 		
	{
		bank = 1;
		bkbase += m_ClusterContainerMemBlockHalfSize;
	}
#ifdef DEBUG_MONITOR_ALLOCATION
	{
		FILE *f = fopen("c:\\sysal.net\\logs\\nvidiamem.txt", "a+t");
		fprintf(f, "\nGPU %d DEALLOC BANK %d %08X %08X", pMgr->DeviceId, bank, (char *)pFreeMax - bkbase, bank ? m_ClusterContainerWatermark1 : m_ClusterContainerWatermark0);
		fclose(f);
	}
#endif
	if (bank)
	{
		if (pFreeMax == (bkbase + m_ClusterContainerWatermark1))
		{
			do
			{
				m_ClusterContainerWatermark1 = (char *)pFreeMin - bkbase;
				if (m_ClusterContainerWatermark1 == 0) break;
				pFreeMax = pFreeMin;
				pFreeMin = (char *)pFreeMax - ((int *)pFreeMax)[-1];
			}
			while (*(int *)pFreeMin == 0);
		}
	}
	else
	{
		if (pFreeMax == (bkbase + m_ClusterContainerWatermark0))
		{
			do
			{
				m_ClusterContainerWatermark0 = (char *)pFreeMin - bkbase;
				if (m_ClusterContainerWatermark0 == 0) break;
				pFreeMax = pFreeMin;
				pFreeMin = (char *)pFreeMax - ((int *)pFreeMax)[-1];
			}
			while (*(int *)pFreeMin == 0);
		}
	}
}

SySal::Imaging::Fast::IClusterSequenceContainer ^NVIDIAImageProcessor::FromFile(System::String ^filename)
{
		throw gcnew System::Exception("Not implemented!");
}

int NVIDIAImageProcessor::Banks::get() { return 2; }

int NVIDIAImageProcessor::CurrentBank::get() { return m_CurrentBank; }
				
void NVIDIAImageProcessor::CurrentBank::set(int v) { if (v < 0 || v > 1) throw gcnew System::Exception("Only banks 0 and 1 supported."); m_CurrentBank = v; }
				
bool NVIDIAImageProcessor::IsBankFree(int v) { if (v < 0 || v > 1) throw gcnew System::Exception("Only banks 0 and 1 supported."); return (v ? m_ClusterContainerWatermark1 : m_ClusterContainerWatermark0)  == 0; }

long CUDAManager::PreciseTimerMilliseconds()
{
	return NVIDIAImageProcessor::PreciseTimer->ElapsedMilliseconds;
}



