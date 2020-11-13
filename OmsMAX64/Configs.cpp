#include "stdafx.h"

#include "Configs.h"

SySal::StageControl::Configuration::Configuration() : SySal::Management::Configuration(gcnew System::String(""))
{
	MaxTrajectorySamples = 100000;
}

SySal::StageControl::Configuration::Configuration(System::String ^name) : SySal::Management::Configuration(name)
{
	MaxTrajectorySamples = 100000;
}

System::Object ^SySal::StageControl::Configuration::Clone()
{
	Configuration ^cfg = gcnew Configuration(this->Name);
	cfg->MaxTrajectorySamples = this->MaxTrajectorySamples;
	return cfg;
}

SySal::StageControl::OmsMAXStageSettings::OmsMAXStageSettings() : SySal::Management::Configuration(gcnew System::String(""))
{
	BoardId = -1;
	XYStepsRev = ZStepsRev = 10000;
	XYLinesRev = ZLinesRev = 10000;
	XYEncoderToMicrons = ZEncoderToMicrons = 0.1;
	CtlModeIsCWCCW = false;
	TurnOffLightTimeSeconds = 60;
	InvertLimiterPolarity = true;
	LightLimit = 32767;
	InvertX = InvertY = InvertZ = false;
	XYLowSpeed = 10.0;
	XYHighSpeed = 5000.0;
	XYAcceleration = 10000.0;
	ZLowSpeed = 10.0;
	ZHighSpeed = 100.0;
	ZAcceleration = 1000.0;
	TimeBracketingTolerance = 2.0;
	LowestZ = 0.0;
	WorkingLight = 0;
	ReferenceX = ReferenceY = 0.0;
	HomingX = 1e5;
	HomingY = 1e5;
	HomingZ = 3e3;
}

SySal::StageControl::OmsMAXStageSettings::OmsMAXStageSettings(System::String ^name) : SySal::Management::Configuration(name)
{
	BoardId = -1;
	XYStepsRev = ZStepsRev = 10000;
	XYLinesRev = ZLinesRev = 10000;
	XYEncoderToMicrons = ZEncoderToMicrons = 0.1;	
	CtlModeIsCWCCW = false;
	TurnOffLightTimeSeconds = 60;
	InvertLimiterPolarity = true;
	LightLimit = 32767;
	InvertX = InvertY = InvertZ = false;
	XYLowSpeed = 10.0;
	XYHighSpeed = 5000.0;
	XYAcceleration = 10000.0;
	ZLowSpeed = 10.0;
	ZHighSpeed = 100.0;
	ZAcceleration = 1000.0;
	TimeBracketingTolerance = 2.0;
	LowestZ = 0.0;
	WorkingLight = 0;
	ReferenceX = ReferenceY = 0.0;
	HomingX = 1e5;
	HomingY = 1e5;
	HomingZ = 3e3;
}

System::Object ^SySal::StageControl::OmsMAXStageSettings::Clone()
{
	SySal::StageControl::OmsMAXStageSettings ^mcfg = gcnew OmsMAXStageSettings(this->Name);
	mcfg->BoardId = this->BoardId;
	mcfg->CtlModeIsCWCCW = this->CtlModeIsCWCCW;
	mcfg->InvertLimiterPolarity = this->InvertLimiterPolarity;
	mcfg->InvertX = this->InvertX;
	mcfg->InvertY = this->InvertY;
	mcfg->InvertZ = this->InvertZ;
	mcfg->LightLimit = this->LightLimit;
	mcfg->TurnOffLightTimeSeconds = this->TurnOffLightTimeSeconds;
	mcfg->XYEncoderToMicrons = this->XYEncoderToMicrons;
	mcfg->ZEncoderToMicrons = this->ZEncoderToMicrons;
	mcfg->XYLinesRev = this->XYLinesRev;
	mcfg->ZLinesRev = this->ZLinesRev;
	mcfg->XYStepsRev = this->XYStepsRev;
	mcfg->ZStepsRev = this->ZStepsRev;	
	mcfg->XYAcceleration = this->XYAcceleration;
	mcfg->XYLowSpeed = this->XYLowSpeed;
	mcfg->XYHighSpeed = this->XYHighSpeed;
	mcfg->ZAcceleration = this->ZAcceleration;
	mcfg->ZLowSpeed = this->ZLowSpeed;
	mcfg->ZHighSpeed = this->ZHighSpeed;
	mcfg->TimeBracketingTolerance = this->TimeBracketingTolerance;
	mcfg->LowestZ = this->LowestZ;
	mcfg->WorkingLight = this->WorkingLight;
	mcfg->ReferenceX = this->ReferenceX;
	mcfg->ReferenceY = this->ReferenceY;
	mcfg->HomingX = this->HomingX;
	mcfg->HomingY = this->HomingY;
	mcfg->HomingZ = this->HomingZ;
	return mcfg;
}