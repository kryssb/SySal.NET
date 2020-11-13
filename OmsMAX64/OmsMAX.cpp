// This is the main DLL file.

#include "stdafx.h"
#define _WIN32_WINNT 0x0501

#include "OmsMAXp.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "MachineSettingsForm.h"
#include "StageMonitor.h"
#include <gcroot.h>
using namespace System::Windows::Forms;

#define _THROWSTR_(str) throw gcnew System::Exception(gcnew System::String(str))

void SySal::StageControl::OmsMAXStage::Initialize()
{	
	m_Homed = false;
	if (m_StageManagerThread != nullptr)
	{		
		*pTerminate = true;
		m_StageManagerThread->Join();
		m_StageManagerThread = nullptr;
		*pTerminate = false;		
	}
	*pRecord = false;
/*
	XYMicronsToSteps = (double)m_S->XYStepsRev/((double)m_S->XYLinesRev * m_S->XYEncoderToMicrons);
	ZMicronsToSteps = (double)m_S->ZStepsRev/((double)m_S->ZLinesRev * m_S->ZEncoderToMicrons);
*/
	XYMicronsToSteps = 1.0/m_S->XYEncoderToMicrons;
	ZMicronsToSteps = 1.0/m_S->ZEncoderToMicrons;

	if (pTrajSamples)
	{
		GlobalFree(pTrajSamples);
		pTrajSamples = 0;
	}
	pTrajSamples = (SySal::StageControl::TrajectorySample *)(void *)GlobalAlloc(GMEM_FIXED, m_C->MaxTrajectorySamples * sizeof(TrajectorySample));

	if (m_S->BoardId < 0) 
	{
		MessageBox::Show("Board not set.\r\nPlease define the machine settings.", "Setup needed", MessageBoxButtons::OK, MessageBoxIcon::Warning);
		return;
	};

	char board_str[32];
	sprintf(board_str, "OmsMAXp%c", '0' + m_S->BoardId);
	hOmsDev = GetOmsHandle(board_str);
	if (hOmsDev == INVALID_HANDLE_VALUE) _THROWSTR_("Can't connect to board");

	char wy_str[1024];
	SendAndGetString(hOmsDev, "WY;", wy_str);
	if (strstr(wy_str, "MAXp-") == 0) _THROWSTR_("Can't ensure the board is an OMS MAXp one.");

	if (SendString(hOmsDev, "WY;") != SUCCESS) _THROWSTR_("Can't configure X axis.");

	char axis_cfg_str[256];
	const char axis_template_str[] = ";A%c;PSE;ECI;ER%d,%d;RTL;LT%c;LMH;CL1;SME;HV10000;HD%d;HG100;BC0;";
	sprintf(axis_cfg_str, axis_template_str, 'X', m_S->XYLinesRev, m_S->XYStepsRev, m_S->InvertLimiterPolarity ? 'H' : 'L', 10/*1,((int)XYMicronsToSteps) + 1*/ );
	if (SendString(hOmsDev, axis_cfg_str) != SUCCESS) _THROWSTR_("Can't configure X axis.");
	sprintf(axis_cfg_str, axis_template_str, 'Y', m_S->XYLinesRev, m_S->XYStepsRev, m_S->InvertLimiterPolarity ? 'H' : 'L',  10/*1,((int)XYMicronsToSteps) + 1*/ );
	if (SendString(hOmsDev, axis_cfg_str) != SUCCESS) _THROWSTR_("Can't configure Y axis.");
	sprintf(axis_cfg_str, axis_template_str, 'Z', m_S->ZLinesRev, m_S->ZStepsRev, m_S->InvertLimiterPolarity ? 'H' : 'L',  10/*1,((int)XYMicronsToSteps) + 1*/ );
	if (SendString(hOmsDev, axis_cfg_str) != SUCCESS) _THROWSTR_("Can't configure Z axis.");
//	if (flex_config_axis(m_S->BoardId, 4, 0, 0, 0, 0) != NIMC_noError)  _THROWSTR_("Can't configure DAC.");

//	if (flex_load_dac(m_S->BoardId, 0x34, m_LightLevel = 0, 0xFF) != NIMC_noError) _THROWSTR_("Can't set light level.");	

	if (SendString(hOmsDev, ";AT;SVB1;SVN;PSM;KO0;") != SUCCESS) _THROWSTR_("Can't configure lamp control.");
	LightLevel = m_S->WorkingLight;
	m_StageManagerThread = gcnew System::Threading::Thread(gcnew System::Threading::ThreadStart(this, &SySal::StageControl::OmsMAXStage::StageManagerThreadProc));
	//m_StageManagerThread->Priority = System::Threading::ThreadPriority::Highest;
	m_StageManagerThread->Start();
}

SySal::StageControl::OmsMAXStage::OmsMAXStage() : m_StageMonitor(nullptr), m_TimeSource(nullptr), m_StageManagerThread(nullptr)
{	
	hOmsDev = INVALID_HANDLE_VALUE;
	pTerminate = 0;
	pRecord = 0;
	pAxisPos = 0;
	pAxisStatus = 0;
	pAxisMove = 0;
	pWriteLight = 0;	
	pTrajSamples = 0;	
	m_CmdTimer = gcnew System::Diagnostics::Stopwatch();
	m_CmdTimer->Reset();
	m_CmdTimer->Start();
	char *mBlock = (char *)(void *)GlobalAlloc(GMEM_FIXED, sizeof(bool) + sizeof(bool) + sizeof(double) * 3 + sizeof(SySal::StageControl::AxisStatus) * 3 + sizeof(SySal::StageControl::SyncAxisOp) * 3 + sizeof(SySal::StageControl::SyncLightLevel));
	pTerminate = (bool *)(void *)mBlock; mBlock += sizeof(bool);
	pRecord = (bool *)(void *)mBlock; mBlock += sizeof(bool);
	pAxisPos = (double *)(void *)mBlock; mBlock += sizeof(double) * 3;
	pAxisStatus = (SySal::StageControl::AxisStatus *)(void *)mBlock; mBlock += sizeof(SySal::StageControl::AxisStatus) * 3;
	pAxisMove = (SySal::StageControl::SyncAxisOp *)(void *)mBlock; mBlock += sizeof(SySal::StageControl::SyncAxisOp) * 3;
	pAxisMove[0].Reset();
	pAxisMove[1].Reset();
	pAxisMove[2].Reset();
	pWriteLight = (SySal::StageControl::SyncLightLevel *)(void *)mBlock; mBlock += sizeof(SySal::StageControl::SyncLightLevel); 	
	pWriteLight->Reset();
	*pTerminate = false;
	*pRecord = false;
	m_C = gcnew SySal::StageControl::Configuration();
	m_S = (SySal::StageControl::OmsMAXStageSettings ^)SySal::Management::MachineSettings::GetSettings(SySal::StageControl::OmsMAXStageSettings::typeid);
	if (m_S == nullptr) m_S = gcnew SySal::StageControl::OmsMAXStageSettings();
	try
	{
		Initialize();
	}
	catch (System::Exception ^x)
	{
		MessageBox::Show(x->ToString(), "Initialization Error", MessageBoxButtons::OK, MessageBoxIcon::Error);		
	}
	m_StageMonitor = gcnew ::OmsMAXStage::StageMonitor(this, m_S);
	if (m_StageManagerThread != nullptr) ((::OmsMAXStage::StageMonitor ^)m_StageMonitor)->RunGUI();
	//m_StageMonitor->Visible = (m_StageManagerThread != nullptr);
}

SySal::StageControl::OmsMAXStage::~OmsMAXStage()
{			
	if (m_StageManagerThread != nullptr && m_StageMonitor != nullptr) ((::OmsMAXStage::StageMonitor ^)m_StageMonitor)->CloseGUI();
	m_StageMonitor = nullptr;
	if (pTerminate)
	{		
		*pTerminate = true;
		if (m_StageManagerThread != nullptr)
		{
			m_StageManagerThread->Join();
			m_StageManagerThread = nullptr;
		}
		GlobalFree((HGLOBAL)pTerminate);
		pTerminate = 0;
	}
	if (pTrajSamples)
	{
		GlobalFree(pTrajSamples);
		pTrajSamples = 0;
	}
	if (hOmsDev != INVALID_HANDLE_VALUE)
	{
		SendString(hOmsDev, ";AT;KO0;");
		CloseHandle(hOmsDev);
		hOmsDev = INVALID_HANDLE_VALUE;
	}
}

SySal::StageControl::OmsMAXStage::!OmsMAXStage()
{
	if (pTerminate) this->~OmsMAXStage();	
}



SySal::Management::Configuration ^SySal::StageControl::OmsMAXStage::Config::get()
{
	return (SySal::Management::Configuration ^)(m_C->Clone());
}

void SySal::StageControl::OmsMAXStage::Config::set(SySal::Management::Configuration ^cfg)
{
	m_C = (SySal::StageControl::Configuration ^)(cfg->Clone());
}

System::String ^SySal::StageControl::OmsMAXStage::Name::get()
{
	return (System::String ^)(m_Name->Clone());
}

void SySal::StageControl::OmsMAXStage::Name::set(System::String ^name)
{
	m_Name = (System::String ^)(name->Clone());
}

bool SySal::StageControl::OmsMAXStage::EditConfiguration(SySal::Management::Configuration ^%c)
{
	SySal::StageControl::Configuration ^cfg = (SySal::StageControl::Configuration ^)c;
	return false;
}

SySal::Management::IConnectionList ^SySal::StageControl::OmsMAXStage::Connections::get()
{
	return gcnew SySal::Management::FixedConnectionList(gcnew cli::array<SySal::Management::FixedTypeConnection::ConnectionDescriptor>(0));
}

bool SySal::StageControl::OmsMAXStage::MonitorEnabled::get()
{	
	return m_StageMonitor->Visible;
}

void SySal::StageControl::OmsMAXStage::MonitorEnabled::set(bool monitorenabled)
{
	m_StageMonitor->Visible = monitorenabled;
}

bool SySal::StageControl::OmsMAXStage::EditMachineSettings(System::Type ^t)
{
	OmsMAXStageSettings ^C = (OmsMAXStageSettings ^)SySal::Management::MachineSettings::GetSettings(OmsMAXStageSettings::typeid);	
	if (C == nullptr)
	{
		MessageBox::Show("No valid configuration found, switching to default", "Configuration warning", MessageBoxButtons::OK, MessageBoxIcon::Warning);
		C = gcnew OmsMAXStageSettings();
		C->Name = gcnew System::String("Default OmsMAXStage configuration");
		C->BoardId = 0;
		C->CtlModeIsCWCCW = false;
		C->InvertLimiterPolarity = true;
		C->InvertX = true;
		C->InvertY = false;
		C->InvertZ = false;
		C->LightLimit = 32767;
		C->TurnOffLightTimeSeconds = 60;
		C->XYEncoderToMicrons = 0.1;
		C->XYLinesRev = 10000;
		C->XYStepsRev = 10000;
		C->ZEncoderToMicrons = 0.05;
		C->ZLinesRev = 10000;
		C->ZStepsRev = 10000;
	}
	MachineSettingsForm ^ef = gcnew MachineSettingsForm();
	ef->MC = C;
	if (ef->ShowDialog() == DialogResult::OK)
	{
		try
		{			
			SySal::Management::MachineSettings::SetSettings(OmsMAXStageSettings::typeid, ef->MC);
			MessageBox::Show("Configuration saved", "Success", MessageBoxButtons::OK, MessageBoxIcon::Information);
			m_S = ef->MC;
			Initialize();
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

unsigned short SySal::StageControl::OmsMAXStage::LightLevel::get()
{
	return m_LightLevel;
}

void SySal::StageControl::OmsMAXStage::LightLevel::set(unsigned short lightlev)
{
	if (lightlev > m_S->LightLimit) lightlev = m_S->LightLimit;
	m_LightLevel = lightlev;
	this->pWriteLight->Write(lightlev);
}

double SySal::StageControl::OmsMAXStage::GetPos(SySal::StageControl::Axis ax)
{
	switch (ax)
	{
		case SySal::StageControl::Axis::X: return pAxisPos[0];
		case SySal::StageControl::Axis::Y: return pAxisPos[1];
		case SySal::StageControl::Axis::Z: return pAxisPos[2];
		default: _THROWSTR_("Unknown axis.");
	}
}

SySal::StageControl::AxisStatus SySal::StageControl::OmsMAXStage::GetStatus(SySal::StageControl::Axis ax)
{
	switch (ax)
	{
		case SySal::StageControl::Axis::X: return pAxisStatus[0];
		case SySal::StageControl::Axis::Y: return pAxisStatus[1];
		case SySal::StageControl::Axis::Z: return pAxisStatus[2];
		default: _THROWSTR_("Unknown axis.");
	}	
}

void SySal::StageControl::OmsMAXStage::Stop(SySal::StageControl::Axis ax)
{
	int axis = (int)ax - (int)SySal::StageControl::Axis::X;
	AxisCommand cmd;
	cmd.OpCode = AxisOpCode::Stop;	
	cmd.TotalStates = 1;
	cmd.StatusNumber = 0;
	cmd.Condition[0] = AxisCond::End;
	cmd.CheckPosition[0] = 0.0;
	cmd.Position[0] = 0.0;
	cmd.Speed[0] = 0.0;
	cmd.Acceleration[0] = 0.0;
	cmd.Deceleration[0] = 0.0;
/*
				FILE *f = fopen("c:\\temp\\omscmd.txt", "a+t");
				if (f) 
				{
					fprintf(f, "\nstop %d %d", GetTickCount(), axis);
					fclose(f);
				}
*/
	this->pAxisMove[axis].Write(cmd/*, m_CmdTimer->ElapsedTicks*/);
/*				
				f = fopen("c:\\temp\\omscmd.txt", "a+t");
				if (f) 
				{
					fprintf(f, " %d %d %d", this->pAxisMove[axis].SD.StartSignature, this->pAxisMove[axis].SD.LastSignatureRead, this->pAxisMove[axis].SD.EndSignature);
					fclose(f);
				}
*/
}

void SySal::StageControl::OmsMAXStage::Reset(SySal::StageControl::Axis ax)
{
	int axis = (int)ax - (int)SySal::StageControl::Axis::X;
	AxisCommand cmd;
	cmd.OpCode = AxisOpCode::Reset;	
	cmd.TotalStates = 1;
	cmd.StatusNumber = 0;
	cmd.Condition[0] = AxisCond::End;
	cmd.CheckPosition[0] = 0.0;
	cmd.Position[0] = 0.0;
	cmd.Speed[0] = 0.0;
	cmd.Acceleration[0] = 0.0;
	cmd.Deceleration[0] = 0.0;
	this->pAxisMove[axis].Write(cmd/*, m_CmdTimer->ElapsedTicks*/);	
}

void SySal::StageControl::OmsMAXStage::PosMove(SySal::StageControl::Axis ax, double pos, double speed, double acc, double dec)
{
	int axis = (int)ax - (int)SySal::StageControl::Axis::X;
	AxisCommand cmd;
	cmd.OpCode = AxisOpCode::PosMove;
	cmd.TotalStates = 1;
	cmd.StatusNumber = 0;
	cmd.Condition[0] = AxisCond::End;
	cmd.CheckPosition[0] = 0.0;
	cmd.Position[0] = pos;
	cmd.Speed[0] = speed;
	cmd.Acceleration[0] = acc;
	cmd.Deceleration[0] = dec;
	this->pAxisMove[axis].Write(cmd/*, m_CmdTimer->ElapsedTicks*/);	
}

void SySal::StageControl::OmsMAXStage::SawToothPosMove(SySal::StageControl::Axis ax, double pos1, double speed1, double acc1, double dec1, double checkpos, double pos2, double speed2, double acc2, double dec2)
{
	int axis = (int)ax - (int)SySal::StageControl::Axis::X;
	AxisCommand cmd;
	cmd.OpCode = AxisOpCode::MultiPosMove;
	cmd.TotalStates = 2;
	cmd.StatusNumber = 0;
	cmd.Condition[0] = (checkpos < pos2) ? AxisCond::LessThan : AxisCond::GreaterThan;
	cmd.CheckPosition[0] = checkpos;
	cmd.Position[0] = pos1;
	cmd.Speed[0] = speed1;
	cmd.Acceleration[0] = acc1;
	cmd.Deceleration[0] = dec1;
	cmd.Condition[1] = AxisCond::End;
	cmd.CheckPosition[1] = 0.0;
	cmd.Position[1] = pos2;
	cmd.Speed[1] = speed2;
	cmd.Acceleration[1] = acc2;
	cmd.Deceleration[1] = dec2;
	this->pAxisMove[axis].Write(cmd/*, m_CmdTimer->ElapsedTicks*/);	
}

void SySal::StageControl::OmsMAXStage::ManualMove(SySal::StageControl::Axis ax, bool highspeed, bool forward)
{
	double speed, acc;
	if (ax == SySal::StageControl::Axis::Z)
	{
		speed = (highspeed ? m_S->ZHighSpeed : m_S->ZLowSpeed) * (forward ? 1.0 : -1.0);
		acc = m_S->ZAcceleration;
	}
	else
	{
		speed = (highspeed ? m_S->XYHighSpeed : m_S->XYLowSpeed) * (forward ? 1.0 : -1.0);
		acc = m_S->XYAcceleration;
	}
	SpeedMove(ax, speed, acc);
}

void SySal::StageControl::OmsMAXStage::SpeedMove(SySal::StageControl::Axis ax, double speed, double acc)
{
	int axis = (int)ax - (int)SySal::StageControl::Axis::X;
	AxisCommand cmd;
	cmd.OpCode = AxisOpCode::SpeedMove;	
	cmd.TotalStates = 1;
	cmd.StatusNumber = 0;
	cmd.Condition[0] = AxisCond::End;
	cmd.CheckPosition[0] = 0.0;
	cmd.Position[0] = 0.0;
	cmd.Speed[0] = speed;
	cmd.Acceleration[0] = acc;
	this->pAxisMove[axis].Write(cmd/*, m_CmdTimer->ElapsedTicks*/);	
}

void SySal::StageControl::OmsMAXStage::StageManagerThreadProc()
{		
	System::Threading::Thread::CurrentThread->Priority = System::Threading::ThreadPriority::Highest;
	volatile bool *pQTerminate = pTerminate;
	volatile bool *pQRecord = pRecord;
	volatile double *pQAxisPos = pAxisPos;
	SySal::StageControl::AxisStatus *pQAxisStatus = pAxisStatus;
	SyncAxisOp *pQAxisMove = pAxisMove;
	SyncLightLevel *pQWriteLight = pWriteLight;
	TrajectorySample *pQTrajSamples = pTrajSamples;
	AxisCommand cmd;
	AxisCommand CurrentAxisCommands[3];
	CurrentAxisCommands[0].OpCode = CurrentAxisCommands[1].OpCode = CurrentAxisCommands[2].OpCode = AxisOpCode::Null;
	int maxsamples = m_C->MaxTrajectorySamples;
	int currentsample = -1;
	double timebracketingtolerance = m_S->TimeBracketingTolerance;
	double currentms = 0.0, checkcurrentms = 0.0, lastcmdms = 0.0;
	double lamptimeoutms = m_S->TurnOffLightTimeSeconds * 1000.0;
	bool WasRecording = false;

	long retvalx;
	long retvaly;
	long retvalz;
	int BoardId = m_S->BoardId;
	bool InvX = m_S->InvertX;
	bool InvY = m_S->InvertY;
	bool InvZ = m_S->InvertZ;
	double microntosteps[3];
	microntosteps[0] = this->XYMicronsToSteps;
	if (InvX) microntosteps[0] *= -1.0;
	microntosteps[1] = this->XYMicronsToSteps;
	if (InvY) microntosteps[1] *= -1.0;
	microntosteps[2] = this->ZMicronsToSteps;
	if (InvZ) microntosteps[2] *= -1.0;
	int ax;	
	char pos_ret_str[256];
	char light_set_str[256];
	ENCODER_POSITIONS enc_pos;
	int axstat;
	unsigned int i_lightlev = 0;
	bool lightoff = false;
	if (m_TimeSource != nullptr) lastcmdms = m_TimeSource->Elapsed.TotalMilliseconds;
	int resynccount = 0;
	while (*pQTerminate == false)
	{
		if (m_TimeSource != nullptr) 
		{
			checkcurrentms = m_TimeSource->Elapsed.TotalMilliseconds;
			if (checkcurrentms - lastcmdms > lamptimeoutms)
			{
				SendString(hOmsDev, ";AT;KO0;");
				lightoff = true;
			}
		}
		if (++resynccount == 1000) 
		{
			SendString(hOmsDev, ";AX;SME;AY;SME;AZ;SME;");
			resynccount = 0;
		}
		/*
			SendAndGetString(hOmsDev, "PE;", pos_ret_str);
			if (sscanf(pos_ret_str, "%d,%d,%d", &retvalx, &retvaly, &retvalz) != 3) { retvalx = retvaly = retvalz = 0; }
		*/
		if (GetEncoderPositions(hOmsDev, &enc_pos) == SUCCESS)
		{
			retvalx = enc_pos.X;
			retvaly = enc_pos.Y;
			retvalz = enc_pos.Z;
		}
		pQAxisPos[0] = ComputeXYLinesToMicron(retvalx) * (InvX ? -1.0 : 1.0);		
		pQAxisPos[1] = ComputeXYLinesToMicron(retvaly) * (InvY ? -1.0 : 1.0);		
		pQAxisPos[2] = ComputeZLinesToMicron(retvalz) * (InvZ ? -1.0 : 1.0);	
		if (*pQRecord && m_TimeSource != nullptr)
		{
			if (pQTrajSamples == 0)			
				*pQRecord = WasRecording = false;
			else
			{
				currentms = m_TimeSource->Elapsed.TotalMilliseconds;
				if (currentms - checkcurrentms < timebracketingtolerance)
				{
					if (WasRecording == false) currentsample = 0;			
					if (currentsample >= maxsamples || pTrajSamples[currentsample].TimeMS < 0.0) *pQRecord = false;
					else
					{				
						if (currentms >= pTrajSamples[currentsample].TimeMS)
						{
							pQTrajSamples[currentsample].Position.X = pQAxisPos[0];
							pQTrajSamples[currentsample].Position.Y = pQAxisPos[1];
							pQTrajSamples[currentsample].Position.Z = pQAxisPos[2];
							pQTrajSamples[currentsample].TimeMS = currentms;
							currentsample++;
							WasRecording = true;
						}
					}
				}
			}
		}
		else WasRecording = false;
		//unsigned limiters = 0;
		long limiters = 0;
		char lim_str[16];
		int statusn;
		/*
			SendAndGetString(hOmsDev, "QL;", lim_str);
			if (sscanf(lim_str, "%0X", &limiters) != 1) limiters = 0xffff;
		*/
		GetOmsLimitSensors(hOmsDev, &limiters);		

		axstat = 0;
		if (limiters & 0x0001) { axstat |= InvX ? (int)SySal::StageControl::AxisStatus::ForwardLimitActive : (int)SySal::StageControl::AxisStatus::ReverseLimitActive; }
		if (limiters & 0x0100) { axstat |= InvX ? (int)SySal::StageControl::AxisStatus::ReverseLimitActive : (int)SySal::StageControl::AxisStatus::ForwardLimitActive; }
		pQAxisStatus[0] = (SySal::StageControl::AxisStatus)axstat;
		axstat = 0;
		if (limiters & 0x0002) { axstat |= InvY ? (int)SySal::StageControl::AxisStatus::ForwardLimitActive : (int)SySal::StageControl::AxisStatus::ReverseLimitActive; }
		if (limiters & 0x0200) { axstat |= InvY ? (int)SySal::StageControl::AxisStatus::ReverseLimitActive : (int)SySal::StageControl::AxisStatus::ForwardLimitActive; }
		pQAxisStatus[1] = (SySal::StageControl::AxisStatus)axstat;
		axstat = 0;
		if (limiters & 0x0004) { axstat |= InvZ ? (int)SySal::StageControl::AxisStatus::ForwardLimitActive : (int)SySal::StageControl::AxisStatus::ReverseLimitActive; }
		if (limiters & 0x0400) { axstat |= InvZ ? (int)SySal::StageControl::AxisStatus::ReverseLimitActive : (int)SySal::StageControl::AxisStatus::ForwardLimitActive; }
		pQAxisStatus[2] = (SySal::StageControl::AxisStatus)axstat;
		
		for (ax = 0; ax < 3; ax++)
			if (pQAxisMove[ax].Read(cmd) && cmd != CurrentAxisCommands[ax]) 
			{
				lastcmdms = checkcurrentms;
				CurrentAxisCommands[ax] = cmd;				
				SySal::StageControl::OmsMAXStage::ExecAxisMove(hOmsDev, ax, CurrentAxisCommands[ax], microntosteps[ax]);
			}
			else if (CurrentAxisCommands[ax].OpCode == AxisOpCode::MultiPosMove && CurrentAxisCommands[ax].StatusNumber < CurrentAxisCommands[ax].TotalStates)
			{
				statusn = CurrentAxisCommands[ax].StatusNumber;
				switch (CurrentAxisCommands[ax].Condition[statusn])
				{

				case AxisCond::LessThan: if (pQAxisPos[ax] < CurrentAxisCommands[ax].CheckPosition[statusn])										 
											 if (++CurrentAxisCommands[ax].StatusNumber < CurrentAxisCommands[ax].TotalStates)
												 SySal::StageControl::OmsMAXStage::ExecAxisMove(hOmsDev, ax, CurrentAxisCommands[ax], microntosteps[ax]);
										break;

				case AxisCond::EqualTo: if (pQAxisPos[ax] == CurrentAxisCommands[ax].CheckPosition[statusn])										 
											 if (++CurrentAxisCommands[ax].StatusNumber < CurrentAxisCommands[ax].TotalStates)
												 SySal::StageControl::OmsMAXStage::ExecAxisMove(hOmsDev, ax, CurrentAxisCommands[ax], microntosteps[ax]);
										break;

				case AxisCond::GreaterThan: if (pQAxisPos[ax] > CurrentAxisCommands[ax].CheckPosition[statusn])										 
											 if (++CurrentAxisCommands[ax].StatusNumber < CurrentAxisCommands[ax].TotalStates)
												 SySal::StageControl::OmsMAXStage::ExecAxisMove(hOmsDev, ax, CurrentAxisCommands[ax], microntosteps[ax]);
										break;

				case AxisCond::End:
				default:
									CurrentAxisCommands[ax].StatusNumber = CurrentAxisCommands[ax].TotalStates;
									break;
				}
			}

		unsigned short lightlev;
		bool newlight = false;
		if ((newlight = pQWriteLight->Read(lightlev)) || (lightoff && (checkcurrentms - lastcmdms) <= lamptimeoutms))
		{
			lastcmdms = checkcurrentms;
			if (newlight) i_lightlev = lightlev;
			sprintf(light_set_str, ";AT;KO%d;", i_lightlev);
			SendString(hOmsDev, light_set_str);
			lightoff = false;
		}
	}	
	*pQTerminate = false;
}

void SySal::StageControl::OmsMAXStage::ExecAxisMove(HANDLE homsdev, int axis, AxisCommand &cmd, double microntosteps)
{	
	switch (cmd.OpCode)
	{
		case AxisOpCode::Null: return;
		case AxisOpCode::Stop: 
			{
				static char str_stop[3][13] = { "AX;ST;ST;ST;", "AY;ST;ST;ST;", "AZ;ST;ST;ST;" };
				SendString(homsdev, str_stop[axis]);
				/*
				FILE *f = fopen("c:\\temp\\omsexe.txt", "a+t");
				if (f) 
				{
					fprintf(f, "\n%d %s", GetTickCount(), str_stop[axis]);
					fclose(f);
				}
				*/
			}
			return;
		case AxisOpCode::Reset:
			{
				static char str_reset[3][12] = { "AX;LP0;SME;", "AY;LP0;SME;", "AZ;LP0;SME;" };
				SendString(homsdev, str_reset[axis]);
			}
			return;
		case AxisOpCode::PosMove:
			{
				char str_move[256];
				int ActPos;
				int ActSpeed;
			    int ActAccel;
				int ActDecel;

				ActPos = cmd.Position[0] * microntosteps;
				ActSpeed = abs((int)(cmd.Speed[0] * microntosteps));
				ActAccel = abs((int)(cmd.Acceleration[0] * microntosteps));
				ActDecel = abs((int)(cmd.Deceleration[0] * microntosteps));
				/* set limiters on speed/accel */

				sprintf(str_move, "A%c;AC%d;DC%d;VL%d;MA%d;GO;", 'X' + axis, ActAccel, ActDecel, ActSpeed, ActPos);
				SendString(homsdev, str_move);
/*				
				{
					FILE *f = fopen("c:\\sysal.net\\logs\\omslog.txt", "a+t");
					fprintf(f, "\n%s", str_move);
					fclose(f);
				}
*/				
			}
		case AxisOpCode::MultiPosMove:
			{
				char str_move[256];
				int ActPos;
				int ActSpeed;
			    int ActAccel;
				int ActDecel;
				int statusn = cmd.StatusNumber;

				ActPos = cmd.Position[statusn] * microntosteps;
				ActSpeed = abs((int)(cmd.Speed[statusn] * microntosteps));
				ActAccel = abs((int)(cmd.Acceleration[statusn] * microntosteps));
				ActDecel = abs((int)(cmd.Deceleration[statusn] * microntosteps));
				/* set limiters on speed/accel */

				sprintf(str_move, "A%c;AC%d;DC%d;VL%d;MA%d;GO;", 'X' + axis, ActAccel, ActDecel, ActSpeed, ActPos);
				SendString(homsdev, str_move);
				/*
				{
					FILE *f = fopen("c:\\sysal.net\\logs\\omslog.txt", "a+t");
					fprintf(f, "\n%s", str_move);
					fclose(f);
				}
				*/
			}
			return;
		case AxisOpCode::SpeedMove:
			{
				char str_move[256];
				int ActSpeed;
			    int ActAccel;

				ActSpeed = (int)(cmd.Speed[0] * microntosteps);
				ActAccel = abs((int)(cmd.Acceleration[0] * microntosteps));
				/* set limiters on speed/accel */

				sprintf(str_move, "A%c;AC%d;DC%d;JG%d;", 'X' + axis, ActAccel, ActAccel, ActSpeed);
				SendString(homsdev, str_move);
			}
			return;
		default: break;
	}
}

void SySal::StageControl::OmsMAXStage::TimeSource::set(System::Diagnostics::Stopwatch ^w)
{
	m_TimeSource = w;
}

void SySal::StageControl::OmsMAXStage::StartRecording(double mindeltams, double totaltimems)
{
	if (m_TimeSource == nullptr) _THROWSTR_("A time source must be set before recording.");
	int samples = 1 + (int)floor(totaltimems / mindeltams);
	if (m_C->MaxTrajectorySamples < samples) _THROWSTR_("Too many samples requested.");
	int i;
	pTrajSamples[0].TimeMS = m_TimeSource->Elapsed.TotalMilliseconds;
	for (i = 1; i < samples; i++)
		pTrajSamples[i].TimeMS = mindeltams + pTrajSamples[i - 1].TimeMS;
	for (; i < m_C->MaxTrajectorySamples; i++)
		pTrajSamples[i].TimeMS = -1.0;
	*pRecord = true;
}

void SySal::StageControl::OmsMAXStage::CancelRecording()
{
	*pRecord = false;
}

cli::array<SySal::StageControl::TrajectorySample> ^SySal::StageControl::OmsMAXStage::Trajectory::get()
{
	while (*pRecord) ;
	int samples;
	for (samples = 0; samples < m_C->MaxTrajectorySamples && pTrajSamples[samples].TimeMS >= 0.0; samples++);
	cli::array<SySal::StageControl::TrajectorySample> ^tj = gcnew cli::array<SySal::StageControl::TrajectorySample>(samples);
	int i;
	for (i = 0; i < samples; i++)
		tj[i] = pTrajSamples[i];
	*pRecord = false;
	return tj;
}

double SySal::StageControl::OmsMAXStage::GetNamedReferencePosition(System::String ^name)
{
	if (String::Compare(name, "LowestZ", true) == 0) return m_S->LowestZ;
	if (String::Compare(name, "ReferenceX", true) == 0)
	{
		if (m_Homed == false && Home() == false) throw gcnew System::Exception("Need valid homing.");
		return m_S->ReferenceX;
	}
	if (String::Compare(name, "ReferenceY", true) == 0)
	{
		if (m_Homed == false && Home() == false) throw gcnew System::Exception("Need valid homing.");
		return m_S->ReferenceY;
	}
	throw gcnew System::Exception("Unknown reference position.");
}

void SySal::StageControl::OmsMAXStage::SetHomingText(char *text, bool state)
{
	if (m_StageMonitor != nullptr)
	{
		((StageMonitorBase ^)m_StageMonitor)->SetHomingText(gcnew System::String(text));
		((StageMonitorBase ^)m_StageMonitor)->SetHomingButton(state);
	}
}

enum HomingState { HS_Idle, HS_Starting, HS_ParkingZ, HS_HomingXY, HS_ParkingXY, HS_HomingZ, HS_DoneParking, HS_Done, HS_FailParking, HS_Fail };

bool SySal::StageControl::OmsMAXStage::Home()
{
	m_CancelHoming = false;
	HomingState HS = HS_Starting;
	SetHomingText("Starting", true);	
	while (HS != HS_Idle)
	{
		switch (HS)
		{
		case HS_Starting:			
			Stop(SySal::StageControl::Axis::X);
			Stop(SySal::StageControl::Axis::Y);
			Stop(SySal::StageControl::Axis::Z);
			PosMove(SySal::StageControl::Axis::Z, m_S->HomingZ, m_S->ZHighSpeed, m_S->ZAcceleration, m_S->ZAcceleration);
			HS = HS_ParkingZ;
			SetHomingText("Parking Z to a safe position", true);
			break;

		case HS_ParkingZ:
			if (m_CancelHoming) HS = HS_Fail;
			if (fabs(GetPos(SySal::StageControl::Axis::Z) - m_S->HomingZ) < 50.0)
			{
				Stop(SySal::StageControl::Axis::Z);
				PosMove(SySal::StageControl::Axis::X, (m_S->HomingX > 0.0) ? -1e8 : 1e8, m_S->XYHighSpeed, m_S->XYAcceleration, m_S->XYAcceleration);
				PosMove(SySal::StageControl::Axis::Y, (m_S->HomingY > 0.0) ? -1e8 : 1e8, m_S->XYHighSpeed, m_S->XYAcceleration, m_S->XYAcceleration);
				HS = HS_HomingXY;
				SetHomingText("Finding limiters for X and Y", true);
			}
			break;

		case HS_HomingXY:
			if (m_CancelHoming) HS = HS_Fail;
			if (
				((int)(GetStatus(SySal::StageControl::Axis::X) & ((m_S->HomingX > 0.0) ? SySal::StageControl::AxisStatus::ReverseLimitActive : SySal::StageControl::AxisStatus::ForwardLimitActive))) &&
				((int)(GetStatus(SySal::StageControl::Axis::Y) & ((m_S->HomingY > 0.0) ? SySal::StageControl::AxisStatus::ReverseLimitActive : SySal::StageControl::AxisStatus::ForwardLimitActive)))
				)
			{
				Stop(SySal::StageControl::Axis::X);
				Stop(SySal::StageControl::Axis::Y);
				Sleep(500);
				Reset(SySal::StageControl::Axis::X);
				Reset(SySal::StageControl::Axis::Y);
				PosMove(SySal::StageControl::Axis::X, m_S->HomingX, m_S->XYHighSpeed, m_S->XYAcceleration, m_S->XYAcceleration);
				PosMove(SySal::StageControl::Axis::Y, m_S->HomingY, m_S->XYHighSpeed, m_S->XYAcceleration, m_S->XYAcceleration);
				HS = HS_ParkingXY;
				SetHomingText("Parking X and Y to a safe position", true);
			}
			break;

		case HS_ParkingXY:
			if (m_CancelHoming) HS = HS_Fail;
			if (
				(fabs(GetPos(SySal::StageControl::Axis::X) - m_S->HomingX)) < 50.0 &&
				(fabs(GetPos(SySal::StageControl::Axis::Y) - m_S->HomingY)) < 50.0
				)
			{
				Stop(SySal::StageControl::Axis::X);
				Stop(SySal::StageControl::Axis::Y);
				PosMove(SySal::StageControl::Axis::Z, -1e8, m_S->ZHighSpeed, m_S->ZAcceleration, m_S->ZAcceleration);
				HS = HS_HomingZ;
				SetHomingText("Finding reverse Z limiter", true);
			}
			break;

		case HS_HomingZ:
			if (m_CancelHoming) HS = HS_Fail;
			if (
				(int)(GetStatus(SySal::StageControl::Axis::Z) & SySal::StageControl::AxisStatus::ReverseLimitActive)
				)
			{
				Stop(SySal::StageControl::Axis::Z);
				Sleep(500);
				Reset(SySal::StageControl::Axis::Z);
				HS = HS_DoneParking;
				Stop(SySal::StageControl::Axis::X);
				Stop(SySal::StageControl::Axis::Y);
				Stop(SySal::StageControl::Axis::Z);
				PosMove(SySal::StageControl::Axis::Z, m_S->LowestZ, m_S->ZHighSpeed, m_S->ZAcceleration, m_S->ZAcceleration);				
				SetHomingText("Homing done, parking Z.", false);
			}
			break;

		case HS_Done:
			SetHomingText("Homing done.", false);
			Stop(SySal::StageControl::Axis::Z);
			m_Homed = true;
			HS = HS_Idle;
			return true;
			break;

		case HS_DoneParking:
			if (m_CancelHoming) HS = HS_Done;
			if (fabs(GetPos(SySal::StageControl::Axis::Z) - m_S->LowestZ) < 20.0)
			{
				Stop(SySal::StageControl::Axis::Z);
				m_Homed = true;
				HS = HS_Done;
			}
			break;

		case HS_Fail:
			SetHomingText("Homing failed.", false);
			Stop(SySal::StageControl::Axis::Z);			
			m_Homed = false;
			HS = HS_Idle;
			return false;			
			break;

		case HS_FailParking:
			SetHomingText("Homing failed, parking Z.", false);
			Stop(SySal::StageControl::Axis::X);
			Stop(SySal::StageControl::Axis::Y);
			Stop(SySal::StageControl::Axis::Z);			
			m_Homed = false;
			HS = HS_Fail;
			break;

		}
		Sleep(100);
	}
	return m_Homed;
}

void SySal::StageControl::OmsMAXStage::vHome()
{
	Home();
}

void SySal::StageControl::OmsMAXStage::vStopHome()
{
	m_CancelHoming = true;
}