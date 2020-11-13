#pragma once

#using "SySalCore.dll"
#using "System.Xml.dll"

using namespace System;
using namespace System::Xml;
using namespace System::Xml::Serialization;
using namespace SySal;
using namespace SySal::Management;

namespace SySal 
{
	namespace Imaging
	{
		[Serializable]
		public ref class Configuration : public SySal::Management::Configuration, public System::ICloneable
		{
			public:
				Configuration();
				Configuration(System::String ^name);
				virtual System::Object ^Clone() override;
		};

		[Serializable]
		public ref class MatroxMilGrabberSettings : public SySal::Management::Configuration, public System::ICloneable
		{			
			public:
				double FrameDelayMS;
				MatroxMilGrabberSettings();
				MatroxMilGrabberSettings(System::String ^name);
				virtual System::Object ^Clone() override;
		};
	}	
}