#pragma once

#using "StageControl.dll"
#using "SySalCore.dll"
#using "System.Xml.dll"

using namespace System;
using namespace System::Xml;
using namespace System::Xml::Serialization;
using namespace SySal;
using namespace SySal::Management;

namespace SySal 
{
	namespace StageControl
	{
		[Serializable]
		public ref class Configuration : public SySal::Management::Configuration, public System::ICloneable
		{
			public:
				int MaxTrajectorySamples;
				Configuration();
				Configuration(System::String ^name);
				virtual System::Object ^Clone() override;
		};

		[Serializable]
		[XmlType("SySal.OmsMAXStage.OmsMAXStageSettings")]
		public ref class OmsMAXStageSettings : public SySal::Management::Configuration, public System::ICloneable
		{			
			public:
				int BoardId;
				unsigned XYStepsRev, ZStepsRev;
				unsigned XYLinesRev, ZLinesRev;
				double XYEncoderToMicrons, ZEncoderToMicrons;				
				bool CtlModeIsCWCCW;
				unsigned TurnOffLightTimeSeconds;
				bool InvertLimiterPolarity;
				unsigned LightLimit;
				bool InvertX, InvertY, InvertZ;
				double XYLowSpeed, XYHighSpeed, XYAcceleration;
				double ZLowSpeed, ZHighSpeed, ZAcceleration;
				double TimeBracketingTolerance;
				double LowestZ;
				unsigned WorkingLight;
				double ReferenceX, ReferenceY;
				double HomingX, HomingY, HomingZ;
				OmsMAXStageSettings();
				OmsMAXStageSettings(System::String ^name);
				virtual System::Object ^Clone() override;
		};
	}	
}