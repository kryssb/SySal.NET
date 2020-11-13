// This is the main DLL file.

#include "stdafx.h"

#include "MatroxMilGrabber.h"
#include "MachineSettingsForm.h"
#include <stdio.h>
#include <malloc.h>
//#include "MemoryManagement.h"

#undef DUMPINFO

MIL_INT MFTYPE SySal::Imaging::LeftSyncGrabHookHandler(MIL_INT HookType, MIL_ID EventId, void MPTYPE *UserDataPtr)
{
	IntGrabData *pGD = (IntGrabData *)UserDataPtr;
	if (pGD->RequestAcquireFrames == 0) return M_NULL;
	MIL_DOUBLE timestamp;
	MIL_INT id = -1;
	MdigGetHookInfo(EventId, M_MODIFIED_BUFFER + M_BUFFER_INDEX, &id);
	MdigGetHookInfo(EventId, M_MODIFIED_BUFFER + M_TIME_STAMP, &timestamp);	
	bool corrupt = (MdigGetHookInfo(EventId, M_MODIFIED_BUFFER + M_CORRUPTED_FRAME, M_NULL) == M_YES);	
	//pGD->pLeftTimesMS[id] = corrupt ? 0.0 : timestamp;	
	pGD->pLeftTimesMS[id] = timestamp;	
	if (id == 0 && pGD->LeftFirstFieldReady == false) pGD->LeftFirstFieldReady = true;	
	pGD->LastLeftFrame = id;
	if (pGD->LeftAcquireFrameCount < pGD->RequestAcquireFrames + 2)
	{
		//pGD->pLeftTimesMS[id] = timestamp;	
		if (pGD->LeftAcquireFrameCount == 0)
			pGD->LeftAcquireFrameStart = id;
		pGD->pLeftBufferMap[pGD->LeftAcquireFrameCount] = id;
		pGD->LeftAcquireFrameCount++;
	}	
	return M_NULL;
}

MIL_INT MFTYPE SySal::Imaging::RightSyncGrabHookHandler(MIL_INT HookType, MIL_ID EventId, void MPTYPE *UserDataPtr)
{
	IntGrabData *pGD = (IntGrabData *)UserDataPtr;
	if (pGD->RequestAcquireFrames == 0) return M_NULL;
	MIL_DOUBLE timestamp;
	MIL_INT id = -1;
	MdigGetHookInfo(EventId, M_MODIFIED_BUFFER + M_BUFFER_INDEX, &id);
	MdigGetHookInfo(EventId, M_MODIFIED_BUFFER + M_TIME_STAMP, &timestamp);	
	bool corrupt = (MdigGetHookInfo(EventId, M_MODIFIED_BUFFER + M_CORRUPTED_FRAME, M_NULL) == M_YES);
	//pGD->pRightTimesMS[id] = corrupt ? 0.0 : timestamp;	
	pGD->pRightTimesMS[id] = timestamp;	
	if (id == 0 && pGD->RightFirstFieldReady == false) pGD->RightFirstFieldReady = true;
	pGD->LastRightFrame = id;
	if (pGD->RightAcquireFrameCount < pGD->RequestAcquireFrames + 2)
	{
		//pGD->pRightTimesMS[id] = timestamp;
		if (pGD->RightAcquireFrameCount == 0)
			pGD->RightAcquireFrameStart = id;
		pGD->pRightBufferMap[pGD->RightAcquireFrameCount] = id;
		pGD->RightAcquireFrameCount++;
	}	
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
		{
			MbufFree(IntGrabSequences[i]->pRightBuffers[j]);
			MbufFree(IntGrabSequences[i]->pLeftBuffers[j]);
			MbufFree(IntGrabSequences[i]->pBuffers[j]);
		}
		delete [] IntGrabSequences[i]->pBuffers;
		delete [] IntGrabSequences[i]->pLeftBuffers;
		delete [] IntGrabSequences[i]->pRightBuffers;
		MbufFree(IntGrabSequences[i]->SequenceBuffer);
	}	
	IntGrabSequences = gcnew cli::array<IntGrabSequence ^>(0);
	
	for (i = 0; i < IntHostMaps->Length; i++)
	{
		for (j = 0; j < IntHostMaps[i]->Images; j++)
		{
			MbufFree(IntHostMaps[i]->pRightBuffers[j]);
			MbufFree(IntHostMaps[i]->pLeftBuffers[j]);
			MbufFree(IntHostMaps[i]->pBuffers[j]);
		}
		delete [] IntHostMaps[i]->pTimesMS;		
		delete [] IntHostMaps[i]->pRightBuffers;
		delete [] IntHostMaps[i]->pLeftBuffers;
		delete [] IntHostMaps[i]->pBuffers;
		MbufFree(IntHostMaps[i]->SequenceBuffer);		
		//MemMgrHostFree/*free*/((void *)IntHostMaps[i]->pHostMap);
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
		gs->pLeftBuffers = new MIL_ID[seqsize];
		gs->pRightBuffers = new MIL_ID[seqsize];
		gs->pTimesMS = new double[seqsize];
		for (j = 0; j < seqsize; j++)
		{							
			gs->pTimesMS[j] = -1.0;
			gs->pBuffers[j] = MbufChild2d(gs->SequenceBuffer, 0, j * YSize, XSize, YSize, M_NULL);
			gs->pLeftBuffers[j] = MbufChild2d(gs->SequenceBuffer, 0, j * YSize, HalfXSize, YSize, M_NULL);
			gs->pRightBuffers[j] = MbufChild2d(gs->SequenceBuffer, HalfXSize, j * YSize, HalfXSize, YSize, M_NULL);
		}
		arr->Add(gs);
	}	
	IntGrabSequences = (cli::array<IntGrabSequence ^> ^)arr->ToArray(IntGrabSequence::typeid);	
	for (i = 0; i < IntGrabSequences->Length; i++) IntGrabSequences[i]->Free = true;
	arr->Clear();	
	
	i = 0;
	void *pbuff = 0;

#if 0
	while (i < 128 && i < IntGrabSequences->Length && (pbuff = /*malloc*/MemMgrHostAlloc(XSize * YSize * seqsize)) &&
		(temp_id = MbufCreate2d(
			M_DEFAULT_HOST, 
			XSize, 
			YSize * seqsize, 
			BitSize, 
			M_IMAGE | M_PAGED | M_HOST_MEMORY, 
			M_HOST_ADDRESS | M_PITCH_BYTE, 
			XSize, 
			pbuff, M_NULL)
			) != M_NULL)
#else
	while (i < 128 && i < IntGrabSequences->Length && 
		(temp_id = MbufAlloc2d(
			MilSystem, 
			XSize, 
			YSize * seqsize, 
			BitSize, 
			M_IMAGE | M_HOST_MEMORY | M_GRAB | M_NON_PAGED,						
			M_NULL)
			) != M_NULL && MbufInquire(temp_id, M_HOST_ADDRESS, &pbuff) != M_ERROR)
#endif
	{
		IntHostMap ^hs = gcnew IntHostMap();
		hs->Id = ++i;
		hs->Images = seqsize;
		hs->FreeAsGrabSeq = false;
		hs->FreeAsHostMap = false;		
		hs->SequenceBuffer = temp_id;
		hs->pTimesMS = new double[seqsize];
		hs->pBuffers = new MIL_ID[seqsize];
		hs->pLeftBuffers = new MIL_ID[seqsize];
		hs->pRightBuffers = new MIL_ID[seqsize];		
		for (j = 0; j < seqsize; j++)
		{							
			hs->pTimesMS[j] = -1.0;
			hs->pBuffers[j] = MbufChild2d(hs->SequenceBuffer, 0, j * YSize, XSize, YSize, M_NULL);
			hs->pLeftBuffers[j] = MbufChild2d(hs->SequenceBuffer, 0, j * YSize, HalfXSize, YSize, M_NULL);
			hs->pRightBuffers[j] = MbufChild2d(hs->SequenceBuffer, HalfXSize, j * YSize, HalfXSize, YSize, M_NULL);
		}
		hs->pHostMap = pbuff;
		arr->Add(hs);
	}
	IntHostMaps = (cli::array<IntHostMap ^> ^)arr->ToArray(IntHostMap::typeid);	
	for (i = 0; i < IntHostMaps->Length; i++) 	
		IntHostMaps[i]->FreeAsHostMap = IntHostMaps[i]->FreeAsGrabSeq = true;	

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
	//System::Threading::Thread::Sleep(500);
}

int SySal::Imaging::MatroxMilGrabber::Sequences::get()
{			
	//return IntHostMaps->Length;
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
	m_Name = gcnew System::String("Matrox Dual Div ") + this->GetHashCode().ToString();
	pIntGrabData = 0;	
	MilApplication = MilSystem = MilDigitizers0 = MilDigitizers1 = M_NULL;
	m_SequenceSize = 0;
	XSize = YSize = BitSize = 0;
	Stride = 0;
	HostBuffer = 0;
	m_HostMemoryAvailable = 0;
	TotalDMAImages = 0;
	m_GrabMemoryAvailable = 0;
	MIL_ID app = M_NULL;
	MIL_ID sys = M_NULL;	
	m_C = gcnew SySal::Imaging::Configuration();
	m_S = (SySal::Imaging::MatroxMilGrabberSettings ^)SySal::Management::MachineSettings::GetSettings(SySal::Imaging::MatroxMilGrabberSettings::typeid);
	if (m_S == nullptr) m_S = gcnew SySal::Imaging::MatroxMilGrabberSettings();
	MappAllocDefault(M_COMPLETE, &app, &sys, M_NULL, NULL, M_NULL);
	MappControl(M_ERROR, M_PRINT_DISABLE);
	MilApplication = app;
	MilSystem = sys;
	//MsysControl(MilSystem, M_MODIFIED_BUFFER_HOOK_MODE, M_MULTI_THREAD + 2);
	MilDigitizers0 = MdigAlloc(sys, M_DEV0, MIL_TEXT("M_DEFAULT"), M_DEFAULT, M_NULL);
	MilDigitizers1 = MdigAlloc(sys, M_DEV1, MIL_TEXT("M_DEFAULT"), M_DEFAULT, M_NULL);
	TimebaseFrequency = (double)System::Diagnostics::Stopwatch::Frequency;
	TimebaseMS = TimebaseFrequency / 1000;
	if (MilApplication == M_NULL || MilSystem == M_NULL || MilDigitizers0 == M_NULL || MilDigitizers1 == M_NULL)
		throw gcnew System::Exception("Could not initialize board.\r\n\r\nPlease notice that because of a limitation in Matrox MIL, if you had previously started a Matrox grabber, you cannot create any more Matrox grabbers until you restart the program.\r\n");
	XSize = 2 * (HalfXSize = MdigInquire(MilDigitizers0, M_SIZE_X, M_NULL));
	YSize = MdigInquire(MilDigitizers0, M_SIZE_Y, M_NULL);
	BitSize = MdigInquire(MilDigitizers0, M_SIZE_BIT, M_NULL);				
	if (BitSize != 8) throw gcnew System::Exception("Invalid size/depth of camera image.");
	m_HostMemoryAvailable = MappInquire(M_NON_PAGED_MEMORY_FREE, M_NULL);
	TotalDMAImages = (m_HostMemoryAvailable / (YSize * XSize));
	//HostBuffer = MbufAlloc2d(sys, XSize, TotalDMAImages * YSize, BitSize, M_IMAGE | M_GRAB | M_OFF_BOARD | M_NON_PAGED, M_NULL);
	m_GrabMemoryAvailable = MsysInquire(MilSystem, M_MEMORY_FREE, M_NULL);
	Timer = gcnew System::Diagnostics::Stopwatch();
	IntGrabSequences = gcnew cli::array<IntGrabSequence ^>(0);
	IntHostMaps = gcnew cli::array<IntHostMap ^>(0);

	TotalGrabImages = (m_GrabMemoryAvailable - 32 * (1 << 20)) / (XSize * YSize) / 2; 
//	if (TotalGrabImages > 128) TotalGrabImages = 128;

	GrabBuffer = MbufAlloc2d(MilSystem, XSize, YSize * TotalGrabImages, BitSize, M_IMAGE | M_PROC | M_GRAB | M_ON_BOARD, M_NULL);

	pIntGrabData = new IntGrabData(GrabBuffer, TotalGrabImages, XSize, HalfXSize, YSize, BitSize);

	InternalStartGrabAndCheck();
	/*
	int i;
	for (i = 0; i < 8; i++)
	{
		SequenceSize = 31;
		IntGrabSequence ^gs = (IntGrabSequence ^)GrabSequence();
		SySal::Imaging::Image ^im = (SySal::Imaging::Image ^)MapSequenceToSingleImage(gs);
		ClearGrabSequence(gs);
		ClearMappedImage(im);

		SequenceSize = 1;
		gs = (IntGrabSequence ^)GrabSequence();
		im = (SySal::Imaging::Image ^)MapSequenceToSingleImage(gs);
		ClearGrabSequence(gs);
		ClearMappedImage(im);
	}*/
}

SySal::Imaging::IntGrabData::IntGrabData(MIL_ID grabbuffid, int maxframes, int xsize, int halfxsize, int ysize, int bitsize) :
	TotalGrabFrames(0), LeftGrabBuffer(M_NULL), RightGrabBuffer(M_NULL), LeftGrabBufferIdBuffer(M_NULL), RightGrabBufferIdBuffer(M_NULL),
	pLeftBuffers(0), pRightBuffers(0), pLeftTimesMS(0), pRightTimesMS(0), LeftFirstFieldReady(false), RightFirstFieldReady(false),
	LeftQWORDBuffer(M_NULL), RightQWORDBuffer(M_NULL), RequestAcquireFrames(0), LeftAcquireFrameCount(0), RightAcquireFrameCount(0), 
	LeftAcquireFrameStart(false), RightAcquireFrameStart(false), pLeftBufferMap(0), pRightBufferMap(0)
{
	TotalGrabFrames = maxframes;
	LeftGrabBuffer = MbufChild2d(grabbuffid, 0, 0, halfxsize, ysize * TotalGrabFrames, NULL);
	RightGrabBuffer = MbufChild2d(grabbuffid, halfxsize, 0, halfxsize, ysize * TotalGrabFrames, NULL);
	LeftGrabBufferIdBuffer = MbufChild2d(grabbuffid, 0, 0, 8, 1, NULL);
	RightGrabBufferIdBuffer = MbufChild2d(grabbuffid, halfxsize, 0, 8, 1, NULL);
	pLeftBuffers = new MIL_ID[TotalGrabFrames];
	pRightBuffers = new MIL_ID[TotalGrabFrames];
	pLeftTimesMS = new double[TotalGrabFrames];
	pRightTimesMS = new double[TotalGrabFrames];
	pLeftBufferMap = new int[TotalGrabFrames];
	pRightBufferMap = new int[TotalGrabFrames];

	int i;
	for (i = 0; i < TotalGrabFrames; i++)
	{
		pLeftBuffers[i] = MbufChild2d(grabbuffid, 0, i * ysize, halfxsize, ysize, NULL);
		pRightBuffers[i] = MbufChild2d(grabbuffid, halfxsize, i * ysize, halfxsize, ysize, NULL);
	}
	LeftFirstFieldReady = false;
	RightFirstFieldReady = false;

	LeftQWORDBuffer = MbufCreate2d(M_DEFAULT_HOST, sizeof(QWORD), 1, bitsize,  M_IMAGE | M_PAGED | M_HOST_MEMORY, M_HOST_ADDRESS | M_PITCH_BYTE, sizeof(QWORD), &LeftGrabData, M_NULL);
	RightQWORDBuffer = MbufCreate2d(M_DEFAULT_HOST, sizeof(QWORD), 1, bitsize,  M_IMAGE | M_PAGED | M_HOST_MEMORY, M_HOST_ADDRESS | M_PITCH_BYTE, sizeof(QWORD), &RightGrabData, M_NULL);
}

SySal::Imaging::IntGrabData::~IntGrabData()
{
	if (pRightBufferMap)
	{
		delete [] pRightBufferMap;
		pRightBufferMap = 0;
	}
	if (pLeftBufferMap) 
	{
		delete [] pLeftBufferMap;
		pLeftBufferMap = 0;
	}
	if (RightQWORDBuffer != M_NULL) 
	{
		MbufFree(RightQWORDBuffer);
		RightQWORDBuffer = M_NULL;
	}
	if (LeftQWORDBuffer != M_NULL)
	{	
		MbufFree(LeftQWORDBuffer);
		LeftQWORDBuffer = M_NULL;
	}
	int i;
	for (i = 0; i < TotalGrabFrames; i++)
	{
		if (pLeftBuffers != 0 && pLeftBuffers[i] != M_NULL)
			MbufFree(pLeftBuffers[i]);
		if (pRightBuffers != 0 && pRightBuffers[i] != M_NULL)
			MbufFree(pRightBuffers[i]);
	}
	if (pRightTimesMS) 
	{
		delete [] pRightTimesMS;
		pRightTimesMS = 0;
	}
	if (pLeftTimesMS) 
	{
		delete [] pLeftTimesMS;
		pLeftTimesMS = 0;
	}
	if (pRightBuffers) 
	{
		delete [] pRightBuffers;
		pRightBuffers = 0;
	}
	if (pLeftBuffers) 
	{
		delete [] pLeftBuffers;
		pLeftBuffers = 0;
	}
}


bool SySal::Imaging::IntGrabData::RequestAcquire(int frames)
{
	if (RequestAcquireFrames > 0) return false;

	int i;
	LeftAcquireFrameStart = RightAcquireFrameStart = -1;
	LeftAcquireFrameCount = RightAcquireFrameCount = 0;
	RequestAcquireFrames = frames;
	while (LeftAcquireFrameCount == 0);	
	while (RightAcquireFrameCount == 0);
	int rstart = LastRightFrame;
	int lstart = LastRightFrame + (RightFirstFieldNum - LeftFirstFieldNum);
	while (lstart < 0) lstart += TotalGrabFrames;
	while (lstart >= TotalGrabFrames) lstart -= TotalGrabFrames;
	SyncLeftFirstField = lstart;
	SyncRightFirstField = rstart;
	return true;
}

bool SySal::Imaging::IntGrabData::ResetStopAcquire()
{
	if (RequestAcquireFrames == 0) return false;

	RequestAcquireFrames = 0;
	return true;
}

bool SySal::Imaging::IntGrabData::WaitAndGetSync(int frame, int &leftframe, int &rightframe)
{	
	int a;
	while (LeftAcquireFrameCount <= frame);
	while (RightAcquireFrameCount <= frame);
	leftframe = pLeftBufferMap[frame];
	rightframe = pRightBufferMap[frame];
	return true;
/*
	int a;
	while (LeftAcquireFrameCount < frame);
	while (RightAcquireFrameCount < frame);
	//a = SyncLeftFirstField + frame;
	a = LeftAcquireFrameStart + frame;
	if (a >= TotalGrabFrames) a -= TotalGrabFrames;
	leftframe = a;
	//a = SyncRightFirstField + frame;
	a = RightAcquireFrameStart + frame;
	if (a >= TotalGrabFrames) a -= TotalGrabFrames;
	rightframe = a;
	return true;
*/
/*
	int a;	
	while (LeftAcquireFrameCount < frame);
	a = SyncLeftFirstField + frame;
	if (a >= TotalGrabFrames) a -= TotalGrabFrames;
	leftframe = a;
	while (RightAcquireFrameCount < frame);
	a = SyncRightFirstField + frame;
	if (a >= TotalGrabFrames) a -= TotalGrabFrames;
	rightframe = a;	
	return true;
*/
}

bool SySal::Imaging::IntGrabData::Wait(int frame)
{
	int a;
	while (LeftAcquireFrameCount < frame);
	while (RightAcquireFrameCount < frame);
	return true;
}

bool SySal::Imaging::MatroxMilGrabber::InternalStartGrabAndCheck()
{
	pIntGrabData->RequestAcquireFrames = 0;	
	if (IsGrabbing)
	{
		MdigProcess(MilDigitizers0, M_NULL, 1, M_STOP | M_WAIT, M_ASYNCHRONOUS, M_NULL, M_NULL);
		MdigProcess(MilDigitizers1, M_NULL, 1, M_STOP | M_WAIT, M_ASYNCHRONOUS, M_NULL, M_NULL);
		MdigHalt(MilDigitizers0);
		MdigHalt(MilDigitizers1);
		IsGrabbing = false;
	}
	pIntGrabData->LeftFirstFieldReady = false;
	pIntGrabData->RightFirstFieldReady = false;
	pIntGrabData->RequestAcquireFrames = 1;	
	{
		MdigProcess(MilDigitizers0, pIntGrabData->pLeftBuffers, TotalGrabImages, M_START, M_ASYNCHRONOUS, LeftSyncGrabHookHandler, pIntGrabData);
		MdigProcess(MilDigitizers1, pIntGrabData->pRightBuffers, TotalGrabImages, M_START, M_ASYNCHRONOUS, RightSyncGrabHookHandler, pIntGrabData);
		MIL_TEXT_CHAR  errtext[1024];
		MappGetError (M_CURRENT | M_MESSAGE , errtext);
		IsGrabbing = true;
	}
	while (pIntGrabData->LeftFirstFieldReady == false || pIntGrabData->RightFirstFieldReady == false);
	MbufCopy(pIntGrabData->LeftGrabBufferIdBuffer, pIntGrabData->LeftQWORDBuffer);
	MbufCopy(pIntGrabData->RightGrabBufferIdBuffer, pIntGrabData->RightQWORDBuffer);
	pIntGrabData->LeftFirstFieldNum = pIntGrabData->LeftGrabData.Tag.Frame;
	pIntGrabData->RightFirstFieldNum = pIntGrabData->RightGrabData.Tag.Frame;
	pIntGrabData->RequestAcquireFrames = 0;	
	return true;
}

SySal::Imaging::MatroxMilGrabber::!MatroxMilGrabber()
{
	if (MilSystem != M_NULL) this->~MatroxMilGrabber();
}

SySal::Imaging::MatroxMilGrabber::~MatroxMilGrabber()
{		
	if (pIntGrabData)
	{
		delete pIntGrabData;
		pIntGrabData = 0;
	}

	int i;
	for (i = 0; i < IntGrabSequences->Length; i++)
	{
		delete [] IntGrabSequences[i]->pTimesMS;
		delete [] IntGrabSequences[i]->pBuffers;
		delete [] IntGrabSequences[i]->pLeftBuffers;
		delete [] IntGrabSequences[i]->pRightBuffers;
		MbufFree(IntGrabSequences[i]->SequenceBuffer);
	}	
	IntGrabSequences = nullptr;
	for (i = 0; i < IntHostMaps->Length; i++)
	{
		delete [] IntHostMaps[i]->pBuffers;
		MbufFree(IntHostMaps[i]->SequenceBuffer);
	}
	IntHostMaps = nullptr;
	if (HostBuffer)
	{
		MbufFree(HostBuffer);
	}
	MdigFree(MilDigitizers1);
	MdigFree(MilDigitizers0);
	MappFreeDefault(MilApplication, MilSystem, M_NULL, M_NULL, M_NULL);
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

#if 0
#define DUMPGRABSTR(x) { FILE *fg = fopen("c:\\grab.txt", "a+t"); fprintf(fg, "%s ", x); fclose(fg); }
#define DUMPGRABINT(x) { FILE *fg = fopen("c:\\grab.txt", "a+t"); fprintf(fg, "%d ", x); fclose(fg); }
#define DUMPGRABDOUBLE(x) { FILE *fg = fopen("c:\\grab.txt", "a+t"); fprintf(fg, "%f ", x); fclose(fg); }
#else
#define DUMPGRABSTR(x) { }
#define DUMPGRABINT(x) { }
#define DUMPGRABDOUBLE(x) { }
#endif

			
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

	pIntGrabData->RequestAcquire(gs->Images + 1);
	int leftf = -1, rightf = -1;
	int finesyncleft = 0, finesyncright = 0, imexcess = 0;
	pIntGrabData->WaitAndGetSync(0, leftf, rightf);
	pIntGrabData->Wait(1);
	{
		int lqn = -1;
		int rqn = -1;
		MbufGet2d(pIntGrabData->pLeftBuffers[leftf], 4, 0, 4, 1, &lqn);
		MbufGet2d(pIntGrabData->pRightBuffers[rightf], 4, 0, 4, 1, &rqn);
		if (gs->Images > 1)
		{
			DUMPGRABSTR("\nLEFTF ");
			DUMPGRABINT(leftf);
			DUMPGRABSTR("\nRIGHTF ");
			DUMPGRABINT(rightf);
			DUMPGRABSTR("\nLQN ");
			DUMPGRABINT(lqn);
			DUMPGRABSTR("\nRQN ");
			DUMPGRABINT(rqn);
		}
		if (lqn < rqn)
		{
			finesyncleft = -1;
			imexcess = 1;
		}
		else if (rqn > lqn)
		{
			finesyncright = -1;
			imexcess = 1;			
		}
	}
	for (i = 0; i < gs->Images + imexcess; i++)
	{		
		pIntGrabData->WaitAndGetSync(i, leftf, rightf);
		pIntGrabData->Wait(i + 1);
		if (i + finesyncleft >= 0 && i + finesyncleft < gs->Images)
		{
			gs->pTimesMS[i + finesyncleft] = pIntGrabData->pLeftTimesMS[leftf];			
			MbufCopy(pIntGrabData->pLeftBuffers[leftf], gs->pLeftBuffers[i + finesyncleft]);
		}
		if (i + finesyncright >= 0 && i + finesyncright < gs->Images)
		{			
			MbufCopy(pIntGrabData->pRightBuffers[rightf], gs->pRightBuffers[i + finesyncright]);
		}
	}
	pIntGrabData->ResetStopAcquire();
	return gs;
/*
	int i;
	for (i = 0; i < IntHostMaps->Length && IntHostMaps[i]->FreeAsGrabSeq == false; i++);				
	if (i == IntHostMaps->Length) throw gcnew System::Exception("All available sequences are locked. Free some sequence.");
	IntHostMap ^hs = IntHostMaps[i];
	hs->FreeAsGrabSeq = false;
	Int64 e1, e2, e3;
	do
	{ 
		e1 = System::Diagnostics::Stopwatch::GetTimestamp();
		e2 = Timer->ElapsedTicks;
		e3 = System::Diagnostics::Stopwatch::GetTimestamp();
	}
	while (e3 - e1 > TimebaseMS);
	hs->Timebase = (double)(e2 - (e1 + e3) / 2) / (double)TimebaseFrequency;

	pIntGrabData->RequestAcquire(hs->Images + 1);
	DUMPGRABSTR("\nREQ\n");
	int leftf = -1, rightf = -1;
	int finesyncleft = 0, finesyncright = 0, imexcess = 0;
	pIntGrabData->WaitAndGetSync(0, leftf, rightf);
	pIntGrabData->Wait(1);
	DUMPGRABINT(leftf);
	DUMPGRABINT(rightf);
	{
		int lqn = -1;
		int rqn = -1;
		MbufGet2d(pIntGrabData->pLeftBuffers[leftf], 4, 0, 4, 1, &lqn);
		MbufGet2d(pIntGrabData->pRightBuffers[rightf], 4, 0, 4, 1, &rqn);
		if (lqn < rqn)
		{
			finesyncleft = -1;
			imexcess = 1;
		}
		else if (rqn > lqn)
		{
			finesyncright = -1;
			imexcess = 1;			
		}
	}
	for (i = 0; i < hs->Images + imexcess; i++)
	{
		pIntGrabData->WaitAndGetSync(i, leftf, rightf);
		pIntGrabData->Wait(i + 1);
		if (i + finesyncleft >= 0 && i + finesyncleft < hs->Images)
		{
			hs->pTimesMS[i + finesyncleft] = pIntGrabData->pLeftTimesMS[leftf];
			MbufCopy(pIntGrabData->pLeftBuffers[leftf], hs->pLeftBuffers[i + finesyncleft]);
		}
		if (i + finesyncright >= 0 && i + finesyncright < hs->Images)
		{
			MbufCopy(pIntGrabData->pRightBuffers[rightf], hs->pRightBuffers[i + finesyncright]);
		}
	}
	pIntGrabData->ResetStopAcquire();
	{
		DUMPGRABINT(*(int *)(void *)((char *)hs->pHostMap + 4));
		DUMPGRABINT(*(int *)(void *)((char *)hs->pHostMap + 4 + 1160));
	}
	return hs;
*/
/*
	for (i = 0; i < hs->Images; i++)
	{
		int leftf = -1, rightf = -1;
		pIntGrabData->WaitAndGetSync(i, leftf, rightf);
		pIntGrabData->Wait(i + 1);		
		DUMPGRABINT(leftf);
		DUMPGRABINT(rightf);
		hs->pTimesMS[i] = pIntGrabData->pLeftTimesMS[leftf];
		DUMPGRABDOUBLE(hs->pTimesMS[i]);
		MbufCopy(pIntGrabData->pLeftBuffers[leftf], hs->pLeftBuffers[i]);
		MbufCopy(pIntGrabData->pRightBuffers[rightf], hs->pRightBuffers[i]);
		if (i == 0)
		{
			int lqn = -1;		
			int rqn = -1;
			MbufGet2d(pIntGrabData->pLeftBuffers[leftf], 4, 0, 4, 1, &lqn);
			MbufGet2d(pIntGrabData->pRightBuffers[rightf], 4, 0, 4, 1, &rqn);			
		}
		DUMPGRABINT(lqn);
		DUMPGRABINT(rqn);
		DUMPGRABINT(rightf - leftf);
		DUMPGRABINT(rqn - lqn);
		DUMPGRABSTR("\n");
	}
	pIntGrabData->ResetStopAcquire();
	return hs;
*/
/*
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
*/	
}

void SySal::Imaging::MatroxMilGrabber::ClearGrabSequence(System::Object ^gbseq)
{
/*
	IntHostMap ^hs = (IntHostMap ^)gbseq;
	hs->FreeAsGrabSeq = hs->FreeAsHostMap;
*/
	if (gbseq == nullptr) throw gcnew System::Exception("Null sequence specified in ClearGrabSequence.");
	IntGrabSequence ^gs = (IntGrabSequence ^)gbseq;
	gs->Free = true;
}

const double c_CameraLinkDataRate = 2 * 1024 * 1024 * 850.0;
#define MSG(x) { FILE *f = fopen("c:\\sysal.net\\logs\\nvlog.txt", "a+t"); fprintf(f, "%s", x); fclose(f); }
cli::array<double> ^SySal::Imaging::MatroxMilGrabber::GetImageTimesMS(System::Object ^gbseq)
{
	if (gbseq == nullptr) throw gcnew System::Exception("Null sequence specified in GetImageTimesMS.");
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
/*
	IntHostMap ^hs = (IntHostMap ^)gbseq;
	if (hs->FreeAsHostMap == false) throw gcnew System::Exception("Sequence already mapped.");
	hs->FreeAsHostMap = false;
	SySal::Imaging::ImageInfo info = ImageFormat;
	return gcnew IntLinearMemoryImage(hs, info, hs->pHostMap, hs->Images);
*/
	if (gbseq == nullptr) throw gcnew System::Exception("Null sequence specified in MapSequenceToSingleImage.");
	IntGrabSequence ^gs = (IntGrabSequence ^)gbseq;
	if (gs->Free) throw gcnew System::Exception("Sequence is not locked.");
	int i;
	for (i = 0; i < IntHostMaps->Length && IntHostMaps[i]->FreeAsHostMap == false; i++);
	if (i == IntHostMaps->Length) throw gcnew System::Exception("Cannot map more image sequences to memory. Please unmap some sequence.");
	IntHostMap ^hs = IntHostMaps[i];
	hs->FreeAsHostMap = false;
	hs->Images = gs->Images;
	System::TimeSpan start = Timer->Elapsed;
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
	System::TimeSpan end = Timer->Elapsed;
#ifdef DUMPINFO
	{
		FILE *f = fopen("c:\\temp\\radient.txt", "a+t");
		fprintf(f, "\nImages %d time %f", hs->Images, (end - start).TotalMilliseconds);
		fclose(f);
	}
#endif
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
	if (hs->Images > 1)
	{
		DUMPGRABSTR("\nFRAMES\n ");
		for (i = 0; i < gs->Images; i++)
		{
			DUMPGRABINT(i);
			DUMPGRABINT(*(int *)(void *)((char *)hs->pHostMap + 4 + i * XSize * YSize));
			DUMPGRABINT(*(int *)(void *)((char *)hs->pHostMap + 4 + i * XSize * YSize + XSize / 2));
			DUMPGRABSTR("\n");
		}
	}
	return gcnew IntLinearMemoryImage(hs, info, (void *)hs->pHostMap, hs->Images);
}

void SySal::Imaging::MatroxMilGrabber::ClearMappedImage(SySal::Imaging::Image ^img)
{
	if (img == nullptr) throw gcnew System::Exception("Null sequence specified in ClearMappedImage.");
	((IntLinearMemoryImage ^)img)->HostMap->FreeAsGrabSeq = ((IntLinearMemoryImage ^)img)->HostMap->FreeAsHostMap = true;
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
	ret += "\r\nGrabMemoryAvailable (MB) ";
	ret += (m_GrabMemoryAvailable >> 20).ToString();
	ret += "\r\nTotalGrabImages ";
	ret += TotalGrabImages.ToString();
	ret += "\r\nHostMemoryAvailable (MB) ";
	ret += (m_HostMemoryAvailable >> 20).ToString();
	ret += "\r\nTotalDMAImages ";
	ret += TotalDMAImages.ToString();
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
	//timems -= (long)(4 * m_S->FrameDelayMS);
	while (Timer->ElapsedMilliseconds < timems);
/*	do
	{ */
		e1 = System::Diagnostics::Stopwatch::GetTimestamp();
		e2 = Timer->ElapsedTicks;
		e3 = System::Diagnostics::Stopwatch::GetTimestamp();
/*	}
	while (e3 - e1 > TimebaseMS);*/
	gs->Timebase = (double)(e2 - (e1 + e3) / 2) / (double)TimebaseFrequency;

	pIntGrabData->RequestAcquire(gs->Images + 1);
	int leftf = -1, rightf = -1;
	int finesyncleft = 0, finesyncright = 0, imexcess = 0;
	pIntGrabData->WaitAndGetSync(0, leftf, rightf);
	pIntGrabData->Wait(1);
	{
		int lqn = -1;
		int rqn = -1;
		MbufGet2d(pIntGrabData->pLeftBuffers[leftf], 4, 0, 4, 1, &lqn);
		MbufGet2d(pIntGrabData->pRightBuffers[rightf], 4, 0, 4, 1, &rqn);
		if (gs->Images > 1)
		{
			DUMPGRABSTR("\nLEFTF ");
			DUMPGRABINT(leftf);
			DUMPGRABSTR("\nRIGHTF ");
			DUMPGRABINT(rightf);
			DUMPGRABSTR("\nLQN ");
			DUMPGRABINT(lqn);
			DUMPGRABSTR("\nRQN ");
			DUMPGRABINT(rqn);
		}
		if (lqn < rqn)
		{
			finesyncleft = -1;
			imexcess = 1;
		}
		else if (rqn > lqn)
		{
			finesyncright = -1;
			imexcess = 1;			
		}
	}
	for (i = 0; i < gs->Images + imexcess; i++)
	{		
		pIntGrabData->WaitAndGetSync(i, leftf, rightf);
		pIntGrabData->Wait(i + 1);
		if (i + finesyncleft >= 0 && i + finesyncleft < gs->Images)
		{
			gs->pTimesMS[i + finesyncleft] = pIntGrabData->pLeftTimesMS[leftf];			
			MbufCopy(pIntGrabData->pLeftBuffers[leftf], gs->pLeftBuffers[i + finesyncleft]);
		}
		if (i + finesyncright >= 0 && i + finesyncright < gs->Images)
		{			
			MbufCopy(pIntGrabData->pRightBuffers[rightf], gs->pRightBuffers[i + finesyncright]);
		}
	}
	pIntGrabData->ResetStopAcquire();
	return gs;
}

