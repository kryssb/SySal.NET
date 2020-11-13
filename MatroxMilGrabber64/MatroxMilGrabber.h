// MatroxMilGrabber.h

#pragma once

using namespace System;

#using "SySalCore.dll"
#using "Imaging.dll"
#using "ImageGrabbing.dll"
#include "mil.h"
#include "Configs.h"

namespace SySal
{
	namespace Imaging
	{
		static MIL_INT MFTYPE GrabHookHandler(MIL_INT HookType, MIL_ID EventId, void MPTYPE *UserDataPtr);

		ref class IntGrabSequence
		{
		public:
			int Id;
			int Images;
			bool Free;
			double Timebase;
			double *pTimesMS;			
			MIL_ID *pBuffers;
			MIL_ID SequenceBuffer;
		};

		ref class IntHostMap
		{
		public:
			int Id;
			int Images;
			bool Free;
			void *pHostMap;
			MIL_ID SequenceBuffer;
			MIL_ID *pBuffers;
		};

		ref class IntLinearMemoryImage : public SySal::Imaging::LinearMemoryImage
		{					
		public:
			IntHostMap ^HostMap;

			IntLinearMemoryImage(IntHostMap ^hs, SySal::Imaging::ImageInfo imgfmt, void *pAddress, int subimages) : LinearMemoryImage(imgfmt, (System::IntPtr)pAddress, subimages, nullptr) 
			{
				HostMap = hs;
			}


			~IntLinearMemoryImage() {}

			void *Address() { return (void *)m_MemoryAddress; }

			virtual SySal::Imaging::Image ^SubImage(unsigned i) override
			{
				return gcnew IntLinearMemoryImage(HostMap, this->Info, ((unsigned char *)(void *)m_MemoryAddress) + i * (this->Info.Width * this->Info.Height), 1);
			}
		};

		public ref class MatroxMilGrabber : SySal::Management::IManageable, SySal::Management::IMachineSettingsEditor, SySal::Imaging::IImageGrabber, SySal::Imaging::IImageGrabberWithTimer
		{

		private:

			MIL_ID MilApplication, MilSystem, MilDigitizer;
			MIL_ID GrabBuffer;
			MIL_ID HostBuffer;
			int XSize, YSize, BitSize;
			int Stride;
			Int64 m_HostMemoryAvailable;
			Int64 m_GrabMemoryAvailable;
			int m_SequenceSize;
			cli::array<IntGrabSequence ^> ^IntGrabSequences;
			cli::array<IntHostMap ^> ^IntHostMaps;
			System::Diagnostics::Stopwatch ^Timer;
			double TimebaseFrequency;
			Int64 TimebaseMS;
			System::String ^m_Name;
			SySal::Imaging::MatroxMilGrabberSettings ^m_S;
			SySal::Imaging::Configuration ^m_C;

		public:

			virtual property System::Diagnostics::Stopwatch ^TimeSource
			{
				void set(System::Diagnostics::Stopwatch ^w);
			}			

			virtual property bool IsReady
			{
				bool get();
			}

			virtual property Int64 HostMemoryAvailable
			{
				Int64 get();
			}

			virtual property Int64 GrabMemoryAvailable
			{
				Int64 get();
			}
			
			virtual property int SequenceSize
			{
				int get();
				void set(int seqsize);
			}

			virtual property int Sequences
			{
				int get();
			}

			virtual property int MappedSequences
			{
				int get();
			}

			MatroxMilGrabber();
			!MatroxMilGrabber();

			virtual ~MatroxMilGrabber();

			virtual property SySal::Imaging::ImageInfo ImageFormat
			{
				SySal::Imaging::ImageInfo get();
			}		

			virtual property int MappedBufferStride
			{
				int get();
			}
			
			virtual System::Object ^ GrabSequence();

			virtual void ClearGrabSequence(System::Object ^gbseq);

			virtual cli::array<double> ^GetImageTimesMS(System::Object ^gbseq);

			virtual SySal::Imaging::Image ^MapSequenceToSingleImage(System::Object ^gbseq);

			virtual void ClearMappedImage(SySal::Imaging::Image ^img);

			virtual System::String ^ToString() override ;


			virtual property System::String ^Name
			{
				System::String ^get();
				void set(System::String ^name);
			}

			virtual property SySal::Management::Configuration ^Config
			{
				SySal::Management::Configuration ^get();
				void set(SySal::Management::Configuration ^cfg);
			}

			virtual bool EditConfiguration(SySal::Management::Configuration ^%c);

			virtual property SySal::Management::IConnectionList ^Connections
			{
				SySal::Management::IConnectionList ^get();
			}

			virtual property bool MonitorEnabled
			{
				bool get();
				void set(bool);
			}

			virtual bool EditMachineSettings(System::Type ^);	

			virtual void Idle();

			virtual System::Object ^GrabSequenceAtTime(long long timems);
		};
	}
}
