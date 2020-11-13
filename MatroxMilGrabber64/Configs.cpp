#include "stdafx.h"

#include "Configs.h"

SySal::Imaging::Configuration::Configuration() : SySal::Management::Configuration(gcnew System::String(""))
{	
}

SySal::Imaging::Configuration::Configuration(System::String ^name) : SySal::Management::Configuration(name)
{	
}

System::Object ^SySal::Imaging::Configuration::Clone()
{
	Configuration ^cfg = gcnew Configuration(this->Name);
	return cfg;
}

SySal::Imaging::MatroxMilGrabberSettings::MatroxMilGrabberSettings() : Configuration("")
{
	FrameDelayMS = 0.0;
}

SySal::Imaging::MatroxMilGrabberSettings::MatroxMilGrabberSettings(System::String ^name) : Configuration(name)
{
	FrameDelayMS = 0.0;
}

System::Object ^SySal::Imaging::MatroxMilGrabberSettings::Clone()
{
	MatroxMilGrabberSettings ^cfg = gcnew MatroxMilGrabberSettings(this->Name);
	cfg->FrameDelayMS = this->FrameDelayMS;
	return cfg;
}
