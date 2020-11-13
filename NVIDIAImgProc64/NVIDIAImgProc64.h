// NVIDIAImgProc.h

#pragma once

using namespace System;

#using "System.Drawing.dll"
#using "ImageGrabbing.dll"
#using "SySalCore.dll"
#using "Imaging.dll"
#using "FastImaging.dll"

#include "CUDAMgr.h"

namespace SySal 
{
	namespace Imaging
	{
		public ref class NVIDIAClusterSequenceContainer : public SySal::Imaging::Fast::IClusterSequenceContainer
		{
			public protected:
				void *pData;
				int m_TotalSize;				
			protected:
				bool m_OwnsBuffer;				
				int m_Images;
				int m_Scale;
				double *pPixelToMicron;
				double *pImagePositions;
				double m_RescalingFactor;
				int *pImageClusterCounts;
				IntCluster  **pImageOffsets;
				short *pImageInfo;
				
			public:

				NVIDIAClusterSequenceContainer(void *data, bool ownbuffer);
				virtual ~NVIDIAClusterSequenceContainer();
				!NVIDIAClusterSequenceContainer();
				
				virtual property int Images
				{
					int get();
				}

				virtual property int ImageWidth
				{
					int get();
				}

				virtual property int ImageHeight
				{
					int get();
				}

				virtual int ClustersInImage(int img);

				virtual SySal::Imaging::Cluster Cluster(int img, int number);

				virtual property SySal::BasicTypes::Vector2 PixelToMicron
				{
					SySal::BasicTypes::Vector2 get();
					void set(SySal::BasicTypes::Vector2 v);
				}

				virtual void SetImagePosition(int img, SySal::BasicTypes::Vector p);

				virtual SySal::BasicTypes::Vector GetImagePosition(int img);

				virtual void WriteToFile(System::String ^file);

				inline static int MaxSize(int images, int maxclusters) { return (2 + images) * sizeof(int) + 10 * sizeof(short) + (2 + 3 * images) * sizeof(double) + sizeof(IntCluster) * (maxclusters * images); }
		};

		public ref class NVIDIAImageProcessor : public SySal::Imaging::IImageProcessor, SySal::Imaging::Fast::IImageProcessorFast
		{
			private:
				CUDAManager *pMgr;				
				ImageWindow m_ImgWnd;
				NVIDIAClusterSequenceContainer ^m_ClusterContainer;
				cli::array<cli::array<Cluster> ^> ^o_Clusters;
				cli::array<cli::array<ClusterSegment> ^> ^o_Segments;
				SySal::Imaging::LinearMemoryImage ^o_BinarizedImages;
				cli::array<SySal::Imaging::ImageProcessingException ^> ^o_Warnings;
				unsigned o_Median;		
				void *pClusterContainerMemBlockBase;
				int m_ClusterContainerMemBlockHalfSize;
				int m_ClusterContainerWatermark0;
				int m_ClusterContainerWatermark1;
				inline int m_ClusterContainerWatermark() { if (m_CurrentBank == 0) return m_ClusterContainerWatermark0; return m_ClusterContainerWatermark1; }
				inline void SetClusterContainerWatermark(int v) { if (m_CurrentBank == 0) m_ClusterContainerWatermark0 = v; else m_ClusterContainerWatermark1 = v; }
				inline void *pClusterContainerMemBlock() { return (char *)pClusterContainerMemBlockBase + ((m_CurrentBank == 0) ? 0 : m_ClusterContainerMemBlockHalfSize); }
				int m_CurrentBank;				

			public:

				static System::Diagnostics::Stopwatch ^PreciseTimer = System::Diagnostics::Stopwatch::StartNew();				
				static property int NVIDIADevices { int get(); }

				NVIDIAImageProcessor();
				NVIDIAImageProcessor(int board);
				virtual ~NVIDIAImageProcessor();
				!NVIDIAImageProcessor();
				virtual System::String ^ ToString() override;
				virtual property ImageInfo ImageFormat 
				{
					ImageInfo get();
					void set(ImageInfo v);
				}
				virtual property int MaxImages
				{
					int get();
				}
				virtual property bool IsReady
				{
					bool get();
				}
				virtual property SySal::Imaging::Image ^EmptyImage
				{
					void set(SySal::Imaging::Image ^);
				}
				virtual property SySal::Imaging::Image ^ThresholdImage
				{
					void set(SySal::Imaging::Image ^);					
				}
				virtual property unsigned MaxSegmentsPerScanLine
				{
					void set(unsigned);
					unsigned get();
				}
				virtual property unsigned MaxClustersPerImage
				{
					void set(unsigned);
					unsigned get();
				}
				virtual property unsigned char EqGreyLevelTargetMedian
				{
					void set(unsigned char);
					unsigned char get();
				}
				virtual property ImageProcessingFeatures OutputFeatures
				{
					ImageProcessingFeatures get();
					void set(ImageProcessingFeatures v);
				}
				virtual property ImageWindow ProcessingWindow
				{
					ImageWindow get();
					void set(ImageWindow v);
				}
				virtual property SySal::Imaging::LinearMemoryImage ^Input
				{
					void set(SySal::Imaging::LinearMemoryImage ^);
				}
				virtual property unsigned GreyLevelMedian
				{
					unsigned get();
				}
				virtual property SySal::Imaging::LinearMemoryImage ^EqualizedImages
				{
					SySal::Imaging::LinearMemoryImage ^get();
				}
				virtual property SySal::Imaging::LinearMemoryImage ^FilteredImages
				{
					SySal::Imaging::LinearMemoryImage ^get();
				}
				virtual property SySal::Imaging::LinearMemoryImage ^BinarizedImages
				{
					SySal::Imaging::LinearMemoryImage ^get();
				}
				virtual property cli::array<cli::array<ClusterSegment> ^> ^ClusterSegments
				{
					cli::array<cli::array<ClusterSegment> ^> ^get();
				}
				virtual property cli::array<cli::array<Cluster> ^> ^Clusters
				{
					cli::array<cli::array<Cluster> ^> ^get();
				}
				virtual property cli::array<SySal::Imaging::ImageProcessingException ^> ^Warnings
				{
					cli::array<SySal::Imaging::ImageProcessingException ^> ^get();
				}
				virtual SySal::Imaging::LinearMemoryImage ^ImageFromFile(System::String ^filename);
				virtual void ImageToFile(SySal::Imaging::LinearMemoryImage ^im, System::String ^filename);
				
				virtual property SySal::Imaging::Fast::IClusterSequenceContainer ^ClusterSequence
				{
					SySal::Imaging::Fast::IClusterSequenceContainer ^get();
				}
				virtual void ReleaseClusterSequence(SySal::Imaging::Fast::IClusterSequenceContainer ^seq);
				virtual SySal::Imaging::Fast::IClusterSequenceContainer ^FromFile(System::String ^filename);
				virtual property int Banks
				{
					int get();
				}
				virtual property int CurrentBank
				{
					int get();
					void set(int v);
				}
				virtual bool IsBankFree(int b);
		};

		public ref class NVIDIAImage : public LinearMemoryImage, IDisposable, IImagePixels
		{
			protected:
				static SySal::Imaging::ImageInfo MakeInfo(int w, int h);
				long long m_TotalSize;				

			public protected:			
				bool m_OwnsBuffer;
				NVIDIAImage(int w, int h, unsigned subimgs, void *buffer);
				static void *_AccessMemoryAddress(SySal::Imaging::LinearMemoryImage ^im);

			public:
				NVIDIAImage(int w, int h, unsigned subimgs);
				NVIDIAImage(SySal::Imaging::Image ^im);
				!NVIDIAImage();
				~NVIDIAImage();

				virtual public Image ^SubImage(unsigned i) override;

				virtual property unsigned short Channels
				{
					unsigned short get();
				}

				virtual property unsigned char default[unsigned]
				{ 
					unsigned char get(unsigned p); 
					void set(unsigned int p, unsigned char ch);
				}

				virtual property unsigned char default[unsigned short, unsigned short, unsigned short]
				{ 
					unsigned char get(unsigned short x, unsigned short y, unsigned short c); 
					void set(unsigned short x, unsigned short y, unsigned short c, unsigned char ch);
				}

				property System::Drawing::Image ^DrawingImage
				{
					System::Drawing::Image ^get();
				}				

				static NVIDIAImage ^FromBMPFiles(cli::array<System::String ^> ^bmpfiles);
				void ToBMPFiles(cli::array<System::String ^> ^bmpfiles);
		};

	}
}
