// OmsMAXStage.h

#pragma once

#using "StageControl.dll"
#using "SySalCore.dll"
#define WIN32_LEAN_AND_MEAN
#include "Configs.h"
#include <windows.h>
#include <stdio.h>

using namespace System;
using namespace System::Xml;
using namespace System::Xml::Serialization;
using namespace SySal;
using namespace SySal::Management;


namespace SySal 
{
	namespace StageControl
	{
		volatile struct SyncData
		{
			unsigned int m_Counter;
			unsigned int StartSignature;
			unsigned int EndSignature;
			unsigned int LastSignatureRead;
			void Reset()
			{
				StartSignature = EndSignature = LastSignatureRead = 0;
				m_Counter = 0;
			}
			inline unsigned Counter() { if (++m_Counter == 0) ++m_Counter; return m_Counter; }			
		};

		volatile struct SyncLightLevel
		{
			private:
				SyncData SD;
				unsigned short LightLevel;				
			public:
				void Reset() { SD.Reset(); }
				inline void Write(unsigned short lightlev/*, unsigned int signature*/)
				{				
					unsigned int signature = SD.Counter();
					do
					{
						SD.StartSignature = signature;
						LightLevel = lightlev;						
						SD.EndSignature = signature;
					}
					while (SD.StartSignature != signature || SD.EndSignature != signature);
				}
				inline bool Read(unsigned short &value)
				{
					unsigned start = SD.StartSignature;
					value = LightLevel;
					if (SD.EndSignature == start && SD.LastSignatureRead != start)
					{
						SD.LastSignatureRead = start;
						return true;
					}
					return false;
				}
		};

		enum AxisOpCode { Null, Stop, Reset, PosMove, SpeedMove, MultiPosMove, ProfileMove };
		enum AxisCond { End, LessThan, EqualTo, GreaterThan };

#define AXIS_MAX_STATES 4

		struct AxisCommand
		{			
			AxisOpCode OpCode;
			int StatusNumber;
			int TotalStates;
			double AtTime;
			AxisCond Condition[AXIS_MAX_STATES];
			union
			{
				double CheckPosition[AXIS_MAX_STATES];
				bool IsPosition[AXIS_MAX_STATES];
			};
			double Position[AXIS_MAX_STATES];
			double Speed[AXIS_MAX_STATES];
			double Acceleration[AXIS_MAX_STATES];
			double Deceleration[AXIS_MAX_STATES];
			long long Wait[AXIS_MAX_STATES];
			double DeltaTarget;
			
			bool operator==(AxisCommand &a)
			{
				if (OpCode != a.OpCode) return false;
				if (OpCode == AxisOpCode::Null) return true;
				if (AtTime != a.AtTime) return false;
				if (TotalStates != a.TotalStates) return false;				
				int i;
				for (i = 0; i < TotalStates; i++)
				{
					if (Condition[i] != a.Condition[i]) return false;
					if (CheckPosition[i] != a.CheckPosition[i]) return false;
					if (Position[i] != a.Position[i]) return false;
					if (Speed[i] != a.Speed[i]) return false;
					if (Acceleration[i] != a.Acceleration[i]) return false;
					if (Deceleration[i] != a.Deceleration[i]) return false;
				}
				return true;
			}

			bool operator!=(AxisCommand &a)
			{
				return !(*this == a);
			}
		};

		volatile struct SyncAxisOp
		{
			public:
				SyncData SD;
				AxisCommand Cmd;				
			public:
				void Reset() 
				{ 
					SD.Reset(); 
				}
				inline void Write(AxisCommand cmd/*, unsigned int signature*/)
				{				
					volatile unsigned int signature = SD.Counter();
					SD.StartSignature = signature;						
					Cmd = cmd;						
					SD.EndSignature = signature;
					Sleep(10);
				}
				inline bool Read(AxisCommand &cmd)
				{
/*
					unsigned start = SD.EndSignature;
					cmd = Cmd;					
					if (SD.StartSignature == start && SD.LastSignatureRead != start)
					{
						SD.LastSignatureRead = start;
						return true;
					}
*/
					volatile unsigned start = SD.EndSignature;
					cmd = Cmd;					
					if (SD.StartSignature == start && SD.LastSignatureRead != start)
					{
						SD.LastSignatureRead = start;
						return true;
					}

					return false;
				}				
		};

		public interface class StageMonitorBase
		{
			System::Void SetIDLabel(System::String ^txt);
			System::Void SetHomingText(System::String ^str);			
			System::Void SetHomingButton(bool ishoming);
		};

		public ref class OmsMAXStage : public IStageWithTimer, public IManageable, public IMachineSettingsEditor, public IStageWithDirectTrajectoryData
		{
			private:

				HANDLE hStageManagerThread;
				System::Diagnostics::Stopwatch ^m_CmdTimer;
				System::Diagnostics::Stopwatch ^m_TimeSource;
				volatile bool *pTerminate;
				volatile bool *pRecord;
				volatile int *pLastSample;
				volatile double *pAxisPos;
				SySal::StageControl::AxisStatus *pAxisStatus;
				SyncAxisOp *pAxisMove;
				SyncLightLevel *pWriteLight;
				SySal::StageControl::TrajectorySample *pTrajSamples;
				volatile int StatusFlags;
								
				double XYMicronsToSteps, ZMicronsToSteps;
				System::String ^m_Name;
				SySal::StageControl::Configuration ^m_C;
				SySal::StageControl::OmsMAXStageSettings ^m_S;
				void Initialize();
				unsigned short m_LightLevel;
				HANDLE hOmsDev;
				volatile bool m_Homed;

				inline int ComputeXYMicronToSteps(double micron) { return (int)(XYMicronsToSteps * micron); }
				inline int ComputeZMicronToSteps(double micron) { return (int)(ZMicronsToSteps * micron); }
				inline double ComputeXYLinesToMicron(int lines) { return m_S->XYEncoderToMicrons * lines; }
				inline double ComputeZLinesToMicron(int lines) { return m_S->ZEncoderToMicrons * lines; }

				void StageManagerThreadProc();

				System::Threading::Thread ^m_StageManagerThread;

				System::Windows::Forms::Form ^m_StageMonitor;				

				static void ExecAxisMove(HANDLE homsdev, int axis, AxisCommand &cmd, double microntosteps);

				bool m_CancelHoming;
				void SetHomingText(char *text, bool state);	

				System::String ^IDStr;

				double m_MS_Offset;
				double m_MS_to_UpdateCounter;
				bool m_Synchronized;

			public:
				virtual property unsigned short LightLevel
				{
					unsigned short get();
					void set(unsigned short);
				}
				virtual double GetPos(SySal::StageControl::Axis ax);
				virtual void PosMove(SySal::StageControl::Axis ax, double pos, double speed, double acc, double dec);
				virtual void SpeedMove(SySal::StageControl::Axis ax, double speed, double acc);
				virtual void Reset(SySal::StageControl::Axis ax);
				virtual void Stop(SySal::StageControl::Axis ax);
				virtual void SawToothPosMove(SySal::StageControl::Axis ax, double pos1, double speed1, double acc1, double dec1, double checkpos, double pos2, double speed2, double acc2, double dec2);
				virtual SySal::StageControl::AxisStatus GetStatus(SySal::StageControl::Axis ax);
				virtual property System::Diagnostics::Stopwatch ^TimeSource
				{
					void set(System::Diagnostics::Stopwatch ^);
				}
				virtual void StartRecording(double mindeltams, double totaltimems);
				virtual void CancelRecording();
				virtual property cli::array<TrajectorySample> ^Trajectory
				{
					cli::array<TrajectorySample> ^get();
				}
				OmsMAXStage();
				~OmsMAXStage();
				!OmsMAXStage();

				void ManualMove(SySal::StageControl::Axis ax, bool highspeed, bool forward);

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

				virtual double GetNamedReferencePosition(System::String ^);

				bool Home();
				void vHome();
				void vStopHome();
				void vForceHome();
				SyncAxisOp *vLastMoveCmd();

				void SendStr(System::String ^s);
				System::String ^SendRecvStr(System::String ^s);

				virtual void Idle();

				virtual void AtTimePosMove(long long timems, SySal::StageControl::Axis ax, double pos, double speed, double acc, double dec);
				virtual void AtTimeMoveProfile(long long timems, SySal::StageControl::Axis ax, cli::array<bool>^ ispos, cli::array<double>^ pos, cli::array<double>^ speed, cli::array<long long>^ waitms, double acc, double dec);

				System::Diagnostics::Stopwatch ^GetTimeSource();

				virtual bool GetTrajectoryData(unsigned int s, SySal::StageControl::TrajectorySample %ts);

#define STATUS_INIT 1
#define STATUS_INITERROR 2
#define STATUS_WARNINGCC 4
#define STATUS_OVERFLOWCC 8
				int GetGeneralStatusFlags();
		};		
	}
}
