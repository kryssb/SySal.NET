// This is the main DLL file.

#include "stdafx.h"

#include "MatroxMilGrabber.h"
#include "MachineSettingsForm.h"
#include <stdio.h>

#undef DUMPINFO

MIL_INT MFTYPE SySal::Imaging::GrabHookHandler(MIL_INT HookType, MIL_ID EventId, void MPTYPE *UserDataPtr)
{
	MIL_DOUBLE timestamp;
	MIL_INT id;
	MdigGetHookInfo(EventId, M_MODIFIED_BUFFER + M_BUFFER_INDEX, &id);
	MdigGetHookInfo(EventId, M_MODIFIED_BUFFER + M_TIME_STAMP, &timestamp);
	bool corrupt = (MdigGetHookInfo(EventId, M_MODIFIED_BUFFER + M_CORRUPTED_FRAME, M_NULL) == M_YES);
	double *pTimesMS = (double *)UserDataPtr;
	pTimesMS[id] = corrupt ? -1.0 : timestamp;
	return M_NULL;
}

void SySal::Imaging::MatroxMilGrabber::TimeSource::set(System::Diagnostics::Stopwatch ^w)
{
	Timer = w;
}

Int64 SySal::Imaging::MatroxMilGrabber::HostMemoryAvailable::get() 
{
	return m_HostMemoryAvailable; 
}

Int64 SySal::Imaging::MatroxMilGrabber::GrabMemoryAvailable::get()
{
	return m_GrabMemoryAvailable;
}	

int SySal::Imaging::MatroxMilGrabber::SequenceSize::get()
{
	return m_SequenceSize;
}

void SySal::Imaging::MatroxMilGrabber::SequenceSize::set(int seqsize)
{
	if (seqsize <= 0) throw gcnew System::Exception("Invalid value for sequence size.");
	int i, j;
	for (i = 0; i < IntGrabSequences->Length; i++)
	{
		delete [] IntGrabSequences[i]->pTimesMS;
		for (j = 0; j < IntGrabSequences[i]->Images; j++)
			MbufFree(IntGrabSequences[i]->pBuffers[j]);
		delete [] IntGrabSequences[i]->pBuffers;		
		MbufFree(IntGrabSequences[i]->SequenceBuffer);
	}	
	IntGrabSequences = gcnew cli::array<IntGrabSequence ^>(0);
	for (i = 0; i < IntHostMaps->Length; i++)
	{
		for (j = 0; j < IntHostMaps[i]->Images; j++)
			MbufFree(IntHostMaps[i]->pBuffers[j]);
		delete [] IntHostMaps[i]->pBuffers;		
		MbufFree(IntHostMaps[i]->SequenceBuffer);
		delete [] IntHostMaps[i]->pHostMap;
	}
	IntHostMaps = gcnew cli::array<IntHostMap ^>(0);
	
	MIL_ID temp_id = M_NULL;
	
	System::Collections::ArrayList ^arr = gcnew System::Collections::ArrayList();
	i = 0;
	while (i < 128 && (temp_id = MbufAlloc2d(MilSystem, XSize, YSize * seqsize, BitSize, M_IMAGE | M_PROC | M_GRAB | M_ON_BOARD, M_NULL)) != M_NULL)
	{		
		IntGrabSequence ^gs = gcnew IntGrabSequence();
		gs->Id = ++i;
		gs->Images = seqsize;
		gs->Free = false;
		gs->SequenceBuffer = temp_id;
		gs->pBuffers = new MIL_ID[seqsize];
		gs->pTimesMS = new double[seqsize];						
		for (j = 0; j < seqsize; j++)
		{							
			gs->pTimesMS[j] = -1.0;
			gs->pBuffers[j] = MbufChild2d(gs->SequenceBuffer, 0, j * YSize, XSize, YSize, M_NULL);							
		}
		arr->Add(gs);
	}	
	IntGrabSequences = (cli::array<IntGrabSequence ^> ^)arr->ToArray(IntGrabSequence::typeid);	
	for (i = 0; i < IntGrabSequences->Length; i++) IntGrabSequences[i]->Free = true;

	arr->Clear();	
	i = 0;
	void *pbuff = 0;
	while (i < 128 && i < IntGrabSequences->Length && 
		(temp_id = MbufCreate2d(
			M_DEFAULT_HOST, 
			XSize, 
			YSize * seqsize, 
			BitSize, 
			M_IMAGE | M_PAGED | M_HOST_MEMORY, 
			M_HOST_ADDRESS | M_PITCH_BYTE, 
			XSize, 
			pbuff = new unsigned char[XSize * YSize * seqsize], M_NULL)
			) != M_NULL)
	{
		IntHostMap ^hs = gcnew IntHostMap();
		hs->Id = ++i;
		hs->Images = seqsize;
		hs->Free = false;		
		hs->SequenceBuffer = temp_id;
		hs->pBuffers = new MIL_ID[seqsize];
		for (j = 0; j < seqsize; j++)
		{							
			hs->pBuffers[j] = MbufChild2d(hs->SequenceBuffer, 0, j * YSize, XSize, YSize, M_NULL);							
		}
		hs->pHostMap = pbuff;
		arr->Add(hs);
	}
	IntHostMaps = (cli::array<IntHostMap ^> ^)arr->ToArray(IntHostMap::typeid);	
	for (i = 0; i < IntHostMaps->Length; i++) IntHostMaps[i]->Free = true;
	
	m_SequenceSize = seqsize;

	if (IntHostMaps->Length > 0) Stride = MbufInquire(IntHostMaps[0]->SequenceBuffer, M_PITCH_BYTE, M_NULL);
	else Stride = 0;
#ifdef DUMPINFO
	{
		FILE *f = fopen("c:\\temp\\milgrabbererrors.txt","a+t");
		int w, h, s;
		fprintf(f, "\nSeqsize %d", seqsize);
		for (i = 0; i < IntGrabSequences->Length; i++) 
		{
			w = MbufInquire(IntGrabSequences[i]->SequenceBuffer, M_SIZE_X, M_NULL);
			h = MbufInquire(IntGrabSequences[i]->SequenceBuffer, M_SIZE_Y, M_NULL);
			s = MbufInquire(IntGrabSequences[i]->SequenceBuffer, M_PITCH_BYTE, M_NULL);
			fprintf(f, "\nGrab %d Buff -> %d %d %d %d", i, IntGrabSequences[i]->SequenceBuffer, w, h, s);
			for (j = 0; j < IntGrabSequences[i]->Images; j++)
			{
				w = MbufInquire(IntGrabSequences[i]->pBuffers[j], M_SIZE_X, M_NULL);
				h = MbufInquire(IntGrabSequences[i]->pBuffers[j], M_SIZE_Y, M_NULL);
				s = MbufInquire(IntGrabSequences[i]->pBuffers[j], M_PITCH_BYTE, M_NULL);
				fprintf(f, "\nGrab %d Buff %d -> %d %d %d %d", i, j, IntGrabSequences[i]->pBuffers[j], w, h, s);
			}
		}
		for (i = 0; i < IntHostMaps->Length; i++) 
		{
			w = MbufInquire(IntHostMaps[i]->SequenceBuffer, M_SIZE_X, M_NULL);
			h = MbufInquire(IntHostMaps[i]->SequenceBuffer, M_SIZE_Y, M_NULL);
			s = MbufInquire(IntHostMaps[i]->SequenceBuffer, M_PITCH_BYTE, M_NULL);
			fprintf(f, "\nHMap %d Buff -> %d %d %d %d", i, IntHostMaps[i]->SequenceBuffer, w, h, s);
		}
		fclose(f);
	}
#endif
}

int SySal::Imaging::MatroxMilGrabber::Sequences::get()
{			
	return IntGrabSequences->Length;
}

int SySal::Imaging::MatroxMilGrabber::MappedSequences::get()
{
	return IntHostMaps->Length;
}

int SySal::Imaging::MatroxMilGrabber::MappedBufferStride::get()
{
	return Stride;
}

SySal::Imaging::MatroxMilGrabber::MatroxMilGrabber()
{
	MilApplication = MilSystem = MilDigitizer = M_NULL;
	m_SequenceSize = 0;
	XSize = YSize = BitSize = 0;
	Stride = 0;
	m_HostMemoryAvailable = 0;
	m_GrabMemoryAvailable = 0;
	MIL_ID app = M_NULL;
	MIL_ID sys = M_NULL;
	MIL_ID dig = M_NULL;				
	m_C = gcnew SySal::Imaging::Configuration();
	m_S = (SySal::Imaging::MatroxMilGrabberSettings ^)SySal::Management::MachineSettings::GetSettings(SySal::Imaging::MatroxMilGrabberSettings::typeid);
	if (m_S == nullptr) m_S = gcnew SySal::Imaging::MatroxMilGrabberSettings();
	MappAllocDefault(M_COMPLETE, &app, &sys, M_NULL, &dig, M_NULL);
	MappControl(M_ERROR, M_PRINT_DISABLE);
	MilApplication = app;
	MilSystem = sys;
	MilDigitizer = dig;
	TimebaseFrequency = (double)System::Diagnostics::Stopwatch::Frequency;
	TimebaseMS = TimebaseFrequency / 1000;
	if (MilApplication == M_NULL || MilSystem == M_NULL || MilDigitizer == M_NULL)
		throw gcnew System::Exception("Could not initialize board.\r\n\r\nPlease notice that because of a limitation in Matrox MIL, if you had previously started a Matrox grabber, you cannot create any more Matrox grabbers until you restart the program.\r\n");
	XSize = MdigInquire(MilDigitizer, M_SIZE_X, M_NULL);
	YSize = MdigInquire(MilDigitizer, M_SIZE_Y, M_NULL);
	BitSize = MdigInquire(MilDigitizer, M_SIZE_BIT, M_NULL);				
	if (BitSize != 8) throw gcnew System::Exception("Invalid size/depth of camera image.");
	m_HostMemoryAvailable = MappInquire(M_NON_PAGED_MEMORY_FREE, M_NULL);
	m_GrabMemoryAvailable = MsysInquire(MilSystem, M_MEMORY_FREE, M_NULL);
	Timer = gcnew System::Diagnostics::Stopwatch();
	IntGrabSequences = gcnew cli::array<IntGrabSequence ^>(0);
	IntHostMaps = gcnew cli::array<IntHostMap ^>(0);
}

SySal::Imaging::MatroxMilGrabber::!MatroxMilGrabber()
{
	if (MilSystem != M_NULL) this->~MatroxMilGrabber();
}

SySal::Imaging::MatroxMilGrabber::~MatroxMilGrabber()
{		
	int i;
	for (i = 0; i < IntGrabSequences->Length; i++)
	{
		delete [] IntGrabSequences[i]->pTimesMS;
		delete [] IntGrabSequences[i]->pBuffers;
		MbufFree(IntGrabSequences[i]->SequenceBuffer);
	}	
	IntGrabSequences = nullptr;
	for (i = 0; i < IntHostMaps->Length; i++)
	{
		delete [] IntHostMaps[i]->pBuffers;
		MbufFree(IntHostMaps[i]->SequenceBuffer);
	}
	IntHostMaps = nullptr;
	MappFreeDefault(MilApplication, MilSystem, M_NULL, MilDigitizer, M_NULL);
	MilDigitizer = M_NULL;
	MilSystem = M_NULL;
	MilApplication = M_NULL;
	GC::SuppressFinalize(this);	
}

SySal::Imaging::ImageInfo SySal::Imaging::MatroxMilGrabber::ImageFormat::get()
{
	SySal::Imaging::ImageInfo info;
	info.Width = XSize;
	info.Height = YSize;
	info.BitsPerPixel = BitSize;
	info.PixelFormat = SySal::Imaging::PixelFormatType::GrayScale8;
	return info;
}		
			
System::Object ^SySal::Imaging::MatroxMilGrabber::GrabSequence()
{
	int i;
	for (i = 0; i < IntGrabSequences->Length && IntGrabSequences[i]->Free == false; i++);				
	if (i == IntGrabSequences->Length) throw gcnew System::Exception("All available sequences are locked. Free some sequence.");
	IntGrabSequence ^gs = IntGrabSequences[i];
	gs->Free = false;
	Int64 e1, e2, e3;
	do
	{ 
		e1 = System::Diagnostics::Stopwatch::GetTimestamp();
		e2 = Timer->ElapsedTicks;
		e3 = System::Diagnostics::Stopwatch::GetTimestamp();
	}
	while (e3 - e1 > TimebaseMS);
	gs->Timebase = (double)(e2 - (e1 + e3) / 2) / (double)TimebaseFrequency;
	MdigProcess(MilDigitizer, gs->pBuffers, gs->Images, M_SEQUENCE, M_SYNCHRONOUS, GrabHookHandler, gs->pTimesMS);
	return gs;
}

void SySal::Imaging::MatroxMilGrabber::ClearGrabSequence(System::Object ^gbseq)
{
	IntGrabSequence ^gs = (IntGrabSequence ^)gbseq;
	gs->Free = true;
}

const double c_CameraLinkDataRate = 1024 * 1024 * 850.0;

cli::array<double> ^SySal::Imaging::MatroxMilGrabber::GetImageTimesMS(System::Object ^gbseq)
{
	IntGrabSequence ^gs = (IntGrabSequence ^)gbseq;
	if (gs->Free) throw gcnew System::Exception("Sequence is not locked.");
	cli::array<double> ^ret = gcnew cli::array<double>(gs->Images);
	double c_TimeCorrection = /*(XSize * YSize) / c_CameraLinkDataRate*/m_S->FrameDelayMS * 0.001;
	int i;
	for (i = 0; i < gs->Images; i++) ret[i] = 1000.0 * (gs->pTimesMS[i] + gs->Timebase - c_TimeCorrection);
	return ret;
}

SySal::Imaging::Image ^SySal::Imaging::MatroxMilGrabber::MapSequenceToSingleImage(System::Object ^gbseq)
{
	IntGrabSequence ^gs = (IntGrabSequence ^)gbseq;
	if (gs->Free) throw gcnew System::Exception("Sequence is not locked.");
	int i;
	for (i = 0; i < IntHostMaps->Length && IntHostMaps[i]->Free == false; i++);
	if (i == IntHostMaps->Length) throw gcnew System::Exception("Cannot map more image sequences to memory. Please unmap some sequence.");
	IntHostMap ^hs = IntHostMaps[i];
	hs->Free = false;
	hs->Images = gs->Images;
	for (i = 0; i < hs->Images; i++)
	{
		MbufCopy(gs->pBuffers[i], hs->pBuffers[i]);
#ifdef DUMPINFO
		MIL_TEXT_CHAR errstr[2048];
		MIL_ID err = MappGetError(M_MESSAGE | M_CURRENT , errstr);
		if (err)
		{
			FILE *f = fopen("c:\\sysal.net\\logs\\milgrabbererrors.txt", "a+t");
			fprintf(f, "\nERROR COPYING %d to %d, img %d", gs->SequenceBuffer, hs->SequenceBuffer, i);
			fclose(f);
			f = fopen("c:\\sysal.net\\logs\\milgrabbererrors.txt", "a+t");
			fprintf(f, "\nSubbuff gs %d", gs->pBuffers[i]);
			fclose(f);
			f = fopen("c:\\sysal.net\\logs\\milgrabbererrors.txt", "a+t");
			fprintf(f, "\nSubbuff hs %d", hs->pBuffers[i]);
			fclose(f);
			f = fopen("c:\\sysal.net\\logs\\milgrabbererrors.txt", "a+t");
			fprintf(f, "\n");
			int jj;
			for (jj = 0; jj < M_ERROR_MESSAGE_SIZE && errstr[jj]; jj++)
				fputc(errstr[jj], f);
			fclose(f);
		}
#endif
	}
	//MbufCopy(gs->SequenceBuffer, hs->SequenceBuffer);
#ifdef DUMPINFO
	MIL_TEXT_CHAR errstr[2048];
	{
		FILE *f = fopen("c:\\sysal.net\\logs\\milgrabbererrors.txt", "a+t");
		fprintf(f, "\nCopy %d -> %d", gs->SequenceBuffer, hs->SequenceBuffer);
		fclose(f);
	}
	MIL_ID err = MappGetError(M_MESSAGE | M_CURRENT , errstr);
	if (err)
	{
		FILE *f = fopen("c:\\sysal.net\\logs\\milgrabbererrors.txt", "a+t");
		fprintf(f, "\n");
		int i;
		for (i = 0; i < M_ERROR_MESSAGE_SIZE && errstr[i]; i++)
			fputc(errstr[i], f);
		fclose(f);
	}
#endif
	SySal::Imaging::ImageInfo info = ImageFormat;
	//info.Height *= hs->Images;
	return gcnew IntLinearMemoryImage(hs, info, hs->pHostMap, hs->Images);
}

void SySal::Imaging::MatroxMilGrabber::ClearMappedImage(SySal::Imaging::Image ^img)
{
	((IntLinearMemoryImage ^)img)->HostMap->Free = true;
}

bool SySal::Imaging::MatroxMilGrabber::IsReady::get()
{
	return XSize > 0 && YSize > 0;
}

System::String ^SySal::Imaging::MatroxMilGrabber::ToString()
{
	System::String ^ret = gcnew System::String("MatroxMilGrabber\r\nImage Format:");
	ret += "\r\nWidth: ";
	ret += XSize.ToString();
	ret += "\r\nHeight: ";
	ret += YSize.ToString();
	ret += "\r\nBits/Pixel: ";
	ret += BitSize.ToString();
	ret += "\r\nPixelFormat ";
	ret += SySal::Imaging::PixelFormatType::GrayScale8.ToString();
	ret += "\r\nFrameDelay (ms): ";
	ret += m_S->FrameDelayMS;
	return ret;
}



SySal::Management::Configuration ^SySal::Imaging::MatroxMilGrabber::Config::get()
{
	return (SySal::Management::Configuration ^)(m_C->Clone());
}

void SySal::Imaging::MatroxMilGrabber::Config::set(SySal::Management::Configuration ^cfg)
{
	m_C = (SySal::Imaging::Configuration ^)(cfg->Clone());
}

System::String ^SySal::Imaging::MatroxMilGrabber::Name::get()
{
	return (System::String ^)(m_Name->Clone());
}

void SySal::Imaging::MatroxMilGrabber::Name::set(System::String ^name)
{
	m_Name = (System::String ^)(name->Clone());
}

bool SySal::Imaging::MatroxMilGrabber::EditConfiguration(SySal::Management::Configuration ^%c)
{
	SySal::Imaging::MatroxMilGrabberSettings ^cfg = (SySal::Imaging::MatroxMilGrabberSettings ^)c;
	return false;
}

SySal::Management::IConnectionList ^SySal::Imaging::MatroxMilGrabber::Connections::get()
{
	return gcnew SySal::Management::FixedConnectionList(gcnew cli::array<SySal::Management::FixedTypeConnection::ConnectionDescriptor>(0));
}

bool SySal::Imaging::MatroxMilGrabber::MonitorEnabled::get()
{	
	return false;
}

void SySal::Imaging::MatroxMilGrabber::MonitorEnabled::set(bool monitorenabled)
{
	/* no monitor */
}

bool SySal::Imaging::MatroxMilGrabber::EditMachineSettings(System::Type ^t)
{
	SySal::Imaging::MatroxMilGrabberSettings ^C = (SySal::Imaging::MatroxMilGrabberSettings ^)SySal::Management::MachineSettings::GetSettings(SySal::Imaging::MatroxMilGrabberSettings::typeid);	
	if (C == nullptr)
	{
		MessageBox::Show("No valid configuration found, switching to default", "Configuration warning", MessageBoxButtons::OK, MessageBoxIcon::Warning);
		C = gcnew SySal::Imaging::MatroxMilGrabberSettings();
		C->Name = gcnew System::String("Default MatroxMilGrabber configuration");
		C->FrameDelayMS = 0.0;
	}
	MachineSettingsForm ^ef = gcnew MachineSettingsForm();
	ef->MC = C;
	if (ef->ShowDialog() == DialogResult::OK)
	{
		try
		{			
			SySal::Management::MachineSettings::SetSettings(SySal::Imaging::MatroxMilGrabberSettings::typeid, ef->MC);
			MessageBox::Show("Configuration saved", "Success", MessageBoxButtons::OK, MessageBoxIcon::Information);
			m_S = ef->MC;
			return true;
		}
		catch (Exception ^x)
		{
			MessageBox::Show("Error saving configuration\r\n\r\n" + x->ToString(), "Error", MessageBoxButtons::OK, MessageBoxIcon::Error);
			return false;
		}
	}
	return false;
}

void SySal::Imaging::MatroxMilGrabber::Idle() {}

System::Object ^SySal::Imaging::MatroxMilGrabber::GrabSequenceAtTime(long long timems)
{
	int i;
	for (i = 0; i < IntGrabSequences->Length && IntGrabSequences[i]->Free == false; i++);				
	if (i == IntGrabSequences->Length) throw gcnew System::Exception("All available sequences are locked. Free some sequence.");
	IntGrabSequence ^gs = IntGrabSequences[i];
	gs->Free = false;
	Int64 e1, e2, e3;
	do
	{ 
		e1 = System::Diagnostics::Stopwatch::GetTimestamp();
		e2 = Timer->ElapsedTicks;
		e3 = System::Diagnostics::Stopwatch::GetTimestamp();
	}
	while (e3 - e1 > TimebaseMS);
	gs->Timebase = (double)(e2 - (e1 + e3) / 2) / (double)TimebaseFrequency;
	while (Timer->ElapsedMilliseconds < timems);
	MdigProcess(MilDigitizer, gs->pBuffers, gs->Images, M_SEQUENCE, M_SYNCHRONOUS, GrabHookHandler, gs->pTimesMS);
	return gs;
}

