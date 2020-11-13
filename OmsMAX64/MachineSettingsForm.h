#pragma once

using namespace System;
using namespace System::ComponentModel;
using namespace System::Collections;
using namespace System::Windows::Forms;
using namespace System::Data;
using namespace System::Drawing;

#include "Configs.h"

namespace SySal 
{
	namespace StageControl 
	{

		/// <summary>
		/// Summary for MachineSettingsForm
		///
		/// WARNING: If you change the name of this class, you will need to change the
		///          'Resource File Name' property for the managed resource compiler tool
		///          associated with all .resx files this class depends on.  Otherwise,
		///          the designers will not be able to interact properly with localized
		///          resources associated with this form.
		/// </summary>
		public ref class MachineSettingsForm : public System::Windows::Forms::Form
		{
		public:

			OmsMAXStageSettings ^MC;

			MachineSettingsForm(void)
			{
				InitializeComponent();
				//
				//TODO: Add the constructor code here
				//
				MC = gcnew OmsMAXStageSettings();
			}

		protected:
			/// <summary>
			/// Clean up any resources being used.
			/// </summary>
			~MachineSettingsForm()
			{
				if (components)
				{
					delete components;
				}
			}
		private: System::Windows::Forms::Label^  label1;
		protected: 
		private: System::Windows::Forms::TextBox^  txtBoardId;
		private: System::Windows::Forms::RadioButton^  rdCtlIsCWCCW;
		private: System::Windows::Forms::RadioButton^  rdCtlIsStepDir;

		private: System::Windows::Forms::CheckBox^  chkInvertLimiters;
		private: System::Windows::Forms::TextBox^  txtLightLimit;

		private: System::Windows::Forms::Label^  label2;
		private: System::Windows::Forms::TextBox^  txtTurnOffLightLimitSeconds;

		private: System::Windows::Forms::Label^  label3;
		private: System::Windows::Forms::GroupBox^  groupBox1;
		private: System::Windows::Forms::TextBox^  txtXYMotorStepsRev;

		private: System::Windows::Forms::Label^  label4;
		private: System::Windows::Forms::TextBox^  txtXYEncoderLinesRev;

		private: System::Windows::Forms::Label^  label5;
		private: System::Windows::Forms::TextBox^  txtXYMicronLine;


		private: System::Windows::Forms::Label^  label6;
		private: System::Windows::Forms::CheckBox^  chkInvertX;
		private: System::Windows::Forms::CheckBox^  chkInvertY;
		private: System::Windows::Forms::GroupBox^  groupBox2;
		private: System::Windows::Forms::CheckBox^  chkInvertZ;

		private: System::Windows::Forms::TextBox^  txtZMicronLine;

		private: System::Windows::Forms::Label^  label7;
		private: System::Windows::Forms::TextBox^  txtZMotorStepsRev;

		private: System::Windows::Forms::Label^  label8;
		private: System::Windows::Forms::TextBox^  txtZEncoderLinesRev;

		private: System::Windows::Forms::Label^  label9;
		private: System::Windows::Forms::Button^  btnOK;
		private: System::Windows::Forms::Button^  btnCancel;
		private: System::Windows::Forms::GroupBox^  groupBox3;
		private: System::Windows::Forms::TextBox^  txtXYHighSpeed;

		private: System::Windows::Forms::Label^  label10;
		private: System::Windows::Forms::TextBox^  txtXYLowSpeed;

		private: System::Windows::Forms::Label^  label11;
		private: System::Windows::Forms::TextBox^  txtXYAccel;

		private: System::Windows::Forms::Label^  label12;
		private: System::Windows::Forms::TextBox^  txtZAccel;

		private: System::Windows::Forms::Label^  label13;
		private: System::Windows::Forms::TextBox^  txtZHighSpeed;

		private: System::Windows::Forms::Label^  label14;
		private: System::Windows::Forms::TextBox^  txtZLowSpeed;

		private: System::Windows::Forms::Label^  label15;
		private: System::Windows::Forms::TextBox^  txtTimeBracketTol;
		private: System::Windows::Forms::Label^  label16;
		private: System::Windows::Forms::GroupBox^  groupBox4;
		private: System::Windows::Forms::TextBox^  txtLowestZ;
		private: System::Windows::Forms::Label^  label17;
		private: System::Windows::Forms::TextBox^  txtWorkingLight;
		private: System::Windows::Forms::Label^  label18;
		private: System::Windows::Forms::TextBox^  txtReferenceY;
		private: System::Windows::Forms::Label^  label19;
		private: System::Windows::Forms::TextBox^  txtReferenceX;
		private: System::Windows::Forms::Label^  label20;
		private: System::Windows::Forms::TextBox^  txtHomingX;
		private: System::Windows::Forms::Label^  label21;
		private: System::Windows::Forms::TextBox^  txtHomingY;
		private: System::Windows::Forms::TextBox^  txtHomingZ;


		private:
			/// <summary>
			/// Required designer variable.
			/// </summary>
			System::ComponentModel::Container ^components;

#pragma region Windows Form Designer generated code
			/// <summary>
			/// Required method for Designer support - do not modify
			/// the contents of this method with the code editor.
			/// </summary>
			void InitializeComponent(void)
			{
				this->label1 = (gcnew System::Windows::Forms::Label());
				this->txtBoardId = (gcnew System::Windows::Forms::TextBox());
				this->rdCtlIsCWCCW = (gcnew System::Windows::Forms::RadioButton());
				this->rdCtlIsStepDir = (gcnew System::Windows::Forms::RadioButton());
				this->chkInvertLimiters = (gcnew System::Windows::Forms::CheckBox());
				this->txtLightLimit = (gcnew System::Windows::Forms::TextBox());
				this->label2 = (gcnew System::Windows::Forms::Label());
				this->txtTurnOffLightLimitSeconds = (gcnew System::Windows::Forms::TextBox());
				this->label3 = (gcnew System::Windows::Forms::Label());
				this->groupBox1 = (gcnew System::Windows::Forms::GroupBox());
				this->txtXYMotorStepsRev = (gcnew System::Windows::Forms::TextBox());
				this->label4 = (gcnew System::Windows::Forms::Label());
				this->txtXYEncoderLinesRev = (gcnew System::Windows::Forms::TextBox());
				this->label5 = (gcnew System::Windows::Forms::Label());
				this->txtXYMicronLine = (gcnew System::Windows::Forms::TextBox());
				this->label6 = (gcnew System::Windows::Forms::Label());
				this->chkInvertX = (gcnew System::Windows::Forms::CheckBox());
				this->chkInvertY = (gcnew System::Windows::Forms::CheckBox());
				this->groupBox2 = (gcnew System::Windows::Forms::GroupBox());
				this->chkInvertZ = (gcnew System::Windows::Forms::CheckBox());
				this->txtZMicronLine = (gcnew System::Windows::Forms::TextBox());
				this->label7 = (gcnew System::Windows::Forms::Label());
				this->txtZMotorStepsRev = (gcnew System::Windows::Forms::TextBox());
				this->label8 = (gcnew System::Windows::Forms::Label());
				this->txtZEncoderLinesRev = (gcnew System::Windows::Forms::TextBox());
				this->label9 = (gcnew System::Windows::Forms::Label());
				this->btnOK = (gcnew System::Windows::Forms::Button());
				this->btnCancel = (gcnew System::Windows::Forms::Button());
				this->groupBox3 = (gcnew System::Windows::Forms::GroupBox());
				this->txtXYHighSpeed = (gcnew System::Windows::Forms::TextBox());
				this->label10 = (gcnew System::Windows::Forms::Label());
				this->txtXYLowSpeed = (gcnew System::Windows::Forms::TextBox());
				this->label11 = (gcnew System::Windows::Forms::Label());
				this->txtXYAccel = (gcnew System::Windows::Forms::TextBox());
				this->label12 = (gcnew System::Windows::Forms::Label());
				this->txtZAccel = (gcnew System::Windows::Forms::TextBox());
				this->label13 = (gcnew System::Windows::Forms::Label());
				this->txtZHighSpeed = (gcnew System::Windows::Forms::TextBox());
				this->label14 = (gcnew System::Windows::Forms::Label());
				this->txtZLowSpeed = (gcnew System::Windows::Forms::TextBox());
				this->label15 = (gcnew System::Windows::Forms::Label());
				this->txtTimeBracketTol = (gcnew System::Windows::Forms::TextBox());
				this->label16 = (gcnew System::Windows::Forms::Label());
				this->groupBox4 = (gcnew System::Windows::Forms::GroupBox());
				this->txtLowestZ = (gcnew System::Windows::Forms::TextBox());
				this->label17 = (gcnew System::Windows::Forms::Label());
				this->txtWorkingLight = (gcnew System::Windows::Forms::TextBox());
				this->label18 = (gcnew System::Windows::Forms::Label());
				this->txtReferenceY = (gcnew System::Windows::Forms::TextBox());
				this->label19 = (gcnew System::Windows::Forms::Label());
				this->txtReferenceX = (gcnew System::Windows::Forms::TextBox());
				this->label20 = (gcnew System::Windows::Forms::Label());
				this->txtHomingX = (gcnew System::Windows::Forms::TextBox());
				this->label21 = (gcnew System::Windows::Forms::Label());
				this->txtHomingY = (gcnew System::Windows::Forms::TextBox());
				this->txtHomingZ = (gcnew System::Windows::Forms::TextBox());
				this->SuspendLayout();
				// 
				// label1
				// 
				this->label1->AutoSize = true;
				this->label1->Location = System::Drawing::Point(12, 9);
				this->label1->Name = L"label1";
				this->label1->Size = System::Drawing::Size(47, 13);
				this->label1->TabIndex = 0;
				this->label1->Text = L"Board Id";
				// 
				// txtBoardId
				// 
				this->txtBoardId->Location = System::Drawing::Point(73, 5);
				this->txtBoardId->Name = L"txtBoardId";
				this->txtBoardId->Size = System::Drawing::Size(32, 20);
				this->txtBoardId->TabIndex = 1;
				this->txtBoardId->Leave += gcnew System::EventHandler(this, &MachineSettingsForm::OnLeaveBoardId);
				// 
				// rdCtlIsCWCCW
				// 
				this->rdCtlIsCWCCW->AutoSize = true;
				this->rdCtlIsCWCCW->Location = System::Drawing::Point(15, 46);
				this->rdCtlIsCWCCW->Name = L"rdCtlIsCWCCW";
				this->rdCtlIsCWCCW->Size = System::Drawing::Size(73, 17);
				this->rdCtlIsCWCCW->TabIndex = 2;
				this->rdCtlIsCWCCW->TabStop = true;
				this->rdCtlIsCWCCW->Text = L"CW/CCW";
				this->rdCtlIsCWCCW->UseVisualStyleBackColor = true;
				// 
				// rdCtlIsStepDir
				// 
				this->rdCtlIsStepDir->AutoSize = true;
				this->rdCtlIsStepDir->Location = System::Drawing::Point(15, 69);
				this->rdCtlIsStepDir->Name = L"rdCtlIsStepDir";
				this->rdCtlIsStepDir->Size = System::Drawing::Size(94, 17);
				this->rdCtlIsStepDir->TabIndex = 3;
				this->rdCtlIsStepDir->TabStop = true;
				this->rdCtlIsStepDir->Text = L"Step/Direction";
				this->rdCtlIsStepDir->UseVisualStyleBackColor = true;
				// 
				// chkInvertLimiters
				// 
				this->chkInvertLimiters->AutoSize = true;
				this->chkInvertLimiters->Location = System::Drawing::Point(14, 98);
				this->chkInvertLimiters->Name = L"chkInvertLimiters";
				this->chkInvertLimiters->Size = System::Drawing::Size(91, 17);
				this->chkInvertLimiters->TabIndex = 4;
				this->chkInvertLimiters->Text = L"Invert Limiters";
				this->chkInvertLimiters->UseVisualStyleBackColor = true;
				// 
				// txtLightLimit
				// 
				this->txtLightLimit->Location = System::Drawing::Point(338, 42);
				this->txtLightLimit->Name = L"txtLightLimit";
				this->txtLightLimit->Size = System::Drawing::Size(59, 20);
				this->txtLightLimit->TabIndex = 6;
				this->txtLightLimit->Leave += gcnew System::EventHandler(this, &MachineSettingsForm::OnLeaveLightLimit);
				// 
				// label2
				// 
				this->label2->AutoSize = true;
				this->label2->Location = System::Drawing::Point(151, 47);
				this->label2->Name = L"label2";
				this->label2->Size = System::Drawing::Size(54, 13);
				this->label2->TabIndex = 5;
				this->label2->Text = L"Light Limit";
				// 
				// txtTurnOffLightLimitSeconds
				// 
				this->txtTurnOffLightLimitSeconds->Location = System::Drawing::Point(338, 67);
				this->txtTurnOffLightLimitSeconds->Name = L"txtTurnOffLightLimitSeconds";
				this->txtTurnOffLightLimitSeconds->Size = System::Drawing::Size(59, 20);
				this->txtTurnOffLightLimitSeconds->TabIndex = 8;
				this->txtTurnOffLightLimitSeconds->Leave += gcnew System::EventHandler(this, &MachineSettingsForm::OnLeaveTurnOffLightSeconds);
				// 
				// label3
				// 
				this->label3->AutoSize = true;
				this->label3->Location = System::Drawing::Point(151, 72);
				this->label3->Name = L"label3";
				this->label3->Size = System::Drawing::Size(104, 13);
				this->label3->TabIndex = 7;
				this->label3->Text = L"Turn off light after (s)";
				// 
				// groupBox1
				// 
				this->groupBox1->Location = System::Drawing::Point(15, 117);
				this->groupBox1->Name = L"groupBox1";
				this->groupBox1->Size = System::Drawing::Size(380, 10);
				this->groupBox1->TabIndex = 9;
				this->groupBox1->TabStop = false;
				// 
				// txtXYMotorStepsRev
				// 
				this->txtXYMotorStepsRev->Location = System::Drawing::Point(130, 161);
				this->txtXYMotorStepsRev->Name = L"txtXYMotorStepsRev";
				this->txtXYMotorStepsRev->Size = System::Drawing::Size(59, 20);
				this->txtXYMotorStepsRev->TabIndex = 13;
				this->txtXYMotorStepsRev->Leave += gcnew System::EventHandler(this, &MachineSettingsForm::OnLeaveXYMotorStepsRev);
				// 
				// label4
				// 
				this->label4->AutoSize = true;
				this->label4->Location = System::Drawing::Point(12, 165);
				this->label4->Name = L"label4";
				this->label4->Size = System::Drawing::Size(99, 13);
				this->label4->TabIndex = 12;
				this->label4->Text = L"XY Motor steps/rev";
				// 
				// txtXYEncoderLinesRev
				// 
				this->txtXYEncoderLinesRev->Location = System::Drawing::Point(130, 136);
				this->txtXYEncoderLinesRev->Name = L"txtXYEncoderLinesRev";
				this->txtXYEncoderLinesRev->Size = System::Drawing::Size(59, 20);
				this->txtXYEncoderLinesRev->TabIndex = 11;
				this->txtXYEncoderLinesRev->Leave += gcnew System::EventHandler(this, &MachineSettingsForm::OnLeaveXYEncoderLinesRev);
				// 
				// label5
				// 
				this->label5->AutoSize = true;
				this->label5->Location = System::Drawing::Point(12, 140);
				this->label5->Name = L"label5";
				this->label5->Size = System::Drawing::Size(108, 13);
				this->label5->TabIndex = 10;
				this->label5->Text = L"XY Encoder lines/rev";
				// 
				// txtXYMicronLine
				// 
				this->txtXYMicronLine->Location = System::Drawing::Point(130, 186);
				this->txtXYMicronLine->Name = L"txtXYMicronLine";
				this->txtXYMicronLine->Size = System::Drawing::Size(59, 20);
				this->txtXYMicronLine->TabIndex = 15;
				this->txtXYMicronLine->Leave += gcnew System::EventHandler(this, &MachineSettingsForm::OnLeaveXYMicronLine);
				// 
				// label6
				// 
				this->label6->AutoSize = true;
				this->label6->Location = System::Drawing::Point(12, 190);
				this->label6->Name = L"label6";
				this->label6->Size = System::Drawing::Size(119, 13);
				this->label6->TabIndex = 14;
				this->label6->Text = L"XY Micron/encoder line";
				// 
				// chkInvertX
				// 
				this->chkInvertX->AutoSize = true;
				this->chkInvertX->Location = System::Drawing::Point(14, 217);
				this->chkInvertX->Name = L"chkInvertX";
				this->chkInvertX->Size = System::Drawing::Size(63, 17);
				this->chkInvertX->TabIndex = 16;
				this->chkInvertX->Text = L"Invert X";
				this->chkInvertX->UseVisualStyleBackColor = true;
				// 
				// chkInvertY
				// 
				this->chkInvertY->AutoSize = true;
				this->chkInvertY->Location = System::Drawing::Point(83, 217);
				this->chkInvertY->Name = L"chkInvertY";
				this->chkInvertY->Size = System::Drawing::Size(63, 17);
				this->chkInvertY->TabIndex = 17;
				this->chkInvertY->Text = L"Invert Y";
				this->chkInvertY->UseVisualStyleBackColor = true;
				// 
				// groupBox2
				// 
				this->groupBox2->Location = System::Drawing::Point(201, 131);
				this->groupBox2->Name = L"groupBox2";
				this->groupBox2->Size = System::Drawing::Size(4, 180);
				this->groupBox2->TabIndex = 18;
				this->groupBox2->TabStop = false;
				// 
				// chkInvertZ
				// 
				this->chkInvertZ->AutoSize = true;
				this->chkInvertZ->Location = System::Drawing::Point(222, 217);
				this->chkInvertZ->Name = L"chkInvertZ";
				this->chkInvertZ->Size = System::Drawing::Size(63, 17);
				this->chkInvertZ->TabIndex = 25;
				this->chkInvertZ->Text = L"Invert Z";
				this->chkInvertZ->UseVisualStyleBackColor = true;
				// 
				// txtZMicronLine
				// 
				this->txtZMicronLine->Location = System::Drawing::Point(338, 186);
				this->txtZMicronLine->Name = L"txtZMicronLine";
				this->txtZMicronLine->Size = System::Drawing::Size(59, 20);
				this->txtZMicronLine->TabIndex = 24;
				this->txtZMicronLine->Leave += gcnew System::EventHandler(this, &MachineSettingsForm::OnLeaveZMicronLine);
				// 
				// label7
				// 
				this->label7->AutoSize = true;
				this->label7->Location = System::Drawing::Point(220, 190);
				this->label7->Name = L"label7";
				this->label7->Size = System::Drawing::Size(112, 13);
				this->label7->TabIndex = 23;
				this->label7->Text = L"Z Micron/encoder line";
				// 
				// txtZMotorStepsRev
				// 
				this->txtZMotorStepsRev->Location = System::Drawing::Point(338, 161);
				this->txtZMotorStepsRev->Name = L"txtZMotorStepsRev";
				this->txtZMotorStepsRev->Size = System::Drawing::Size(59, 20);
				this->txtZMotorStepsRev->TabIndex = 22;
				this->txtZMotorStepsRev->Leave += gcnew System::EventHandler(this, &MachineSettingsForm::OnLeaveZMotorStepsRev);
				// 
				// label8
				// 
				this->label8->AutoSize = true;
				this->label8->Location = System::Drawing::Point(220, 165);
				this->label8->Name = L"label8";
				this->label8->Size = System::Drawing::Size(92, 13);
				this->label8->TabIndex = 21;
				this->label8->Text = L"Z Motor steps/rev";
				// 
				// txtZEncoderLinesRev
				// 
				this->txtZEncoderLinesRev->Location = System::Drawing::Point(338, 136);
				this->txtZEncoderLinesRev->Name = L"txtZEncoderLinesRev";
				this->txtZEncoderLinesRev->Size = System::Drawing::Size(59, 20);
				this->txtZEncoderLinesRev->TabIndex = 20;
				this->txtZEncoderLinesRev->Leave += gcnew System::EventHandler(this, &MachineSettingsForm::OnLeaveZEncoderLinesRev);
				// 
				// label9
				// 
				this->label9->AutoSize = true;
				this->label9->Location = System::Drawing::Point(220, 140);
				this->label9->Name = L"label9";
				this->label9->Size = System::Drawing::Size(101, 13);
				this->label9->TabIndex = 19;
				this->label9->Text = L"Z Encoder lines/rev";
				// 
				// btnOK
				// 
				this->btnOK->Location = System::Drawing::Point(12, 434);
				this->btnOK->Name = L"btnOK";
				this->btnOK->Size = System::Drawing::Size(76, 26);
				this->btnOK->TabIndex = 26;
				this->btnOK->Text = L"OK";
				this->btnOK->UseVisualStyleBackColor = true;
				this->btnOK->Click += gcnew System::EventHandler(this, &MachineSettingsForm::btnOK_Click);
				// 
				// btnCancel
				// 
				this->btnCancel->Location = System::Drawing::Point(321, 434);
				this->btnCancel->Name = L"btnCancel";
				this->btnCancel->Size = System::Drawing::Size(76, 26);
				this->btnCancel->TabIndex = 27;
				this->btnCancel->Text = L"Cancel";
				this->btnCancel->UseVisualStyleBackColor = true;
				this->btnCancel->Click += gcnew System::EventHandler(this, &MachineSettingsForm::btnCancel_Click);
				// 
				// groupBox3
				// 
				this->groupBox3->Location = System::Drawing::Point(15, 26);
				this->groupBox3->Name = L"groupBox3";
				this->groupBox3->Size = System::Drawing::Size(380, 10);
				this->groupBox3->TabIndex = 28;
				this->groupBox3->TabStop = false;
				// 
				// txtXYHighSpeed
				// 
				this->txtXYHighSpeed->Location = System::Drawing::Point(130, 268);
				this->txtXYHighSpeed->Name = L"txtXYHighSpeed";
				this->txtXYHighSpeed->Size = System::Drawing::Size(59, 20);
				this->txtXYHighSpeed->TabIndex = 32;
				this->txtXYHighSpeed->Leave += gcnew System::EventHandler(this, &MachineSettingsForm::OnLeaveXYHighSpeed);
				// 
				// label10
				// 
				this->label10->AutoSize = true;
				this->label10->Location = System::Drawing::Point(12, 272);
				this->label10->Name = L"label10";
				this->label10->Size = System::Drawing::Size(80, 13);
				this->label10->TabIndex = 31;
				this->label10->Text = L"XY High Speed";
				// 
				// txtXYLowSpeed
				// 
				this->txtXYLowSpeed->Location = System::Drawing::Point(130, 243);
				this->txtXYLowSpeed->Name = L"txtXYLowSpeed";
				this->txtXYLowSpeed->Size = System::Drawing::Size(59, 20);
				this->txtXYLowSpeed->TabIndex = 30;
				this->txtXYLowSpeed->Leave += gcnew System::EventHandler(this, &MachineSettingsForm::OnLeaveXYLowSpeed);
				// 
				// label11
				// 
				this->label11->AutoSize = true;
				this->label11->Location = System::Drawing::Point(12, 247);
				this->label11->Name = L"label11";
				this->label11->Size = System::Drawing::Size(78, 13);
				this->label11->TabIndex = 29;
				this->label11->Text = L"XY Low Speed";
				// 
				// txtXYAccel
				// 
				this->txtXYAccel->Location = System::Drawing::Point(130, 293);
				this->txtXYAccel->Name = L"txtXYAccel";
				this->txtXYAccel->Size = System::Drawing::Size(59, 20);
				this->txtXYAccel->TabIndex = 34;
				this->txtXYAccel->Leave += gcnew System::EventHandler(this, &MachineSettingsForm::OnLeaveXYAccel);
				// 
				// label12
				// 
				this->label12->AutoSize = true;
				this->label12->Location = System::Drawing::Point(12, 297);
				this->label12->Name = L"label12";
				this->label12->Size = System::Drawing::Size(83, 13);
				this->label12->TabIndex = 33;
				this->label12->Text = L"XY Acceleration";
				// 
				// txtZAccel
				// 
				this->txtZAccel->Location = System::Drawing::Point(338, 293);
				this->txtZAccel->Name = L"txtZAccel";
				this->txtZAccel->Size = System::Drawing::Size(59, 20);
				this->txtZAccel->TabIndex = 40;
				this->txtZAccel->Leave += gcnew System::EventHandler(this, &MachineSettingsForm::OnLeaveZAccel);
				// 
				// label13
				// 
				this->label13->AutoSize = true;
				this->label13->Location = System::Drawing::Point(220, 297);
				this->label13->Name = L"label13";
				this->label13->Size = System::Drawing::Size(76, 13);
				this->label13->TabIndex = 39;
				this->label13->Text = L"Z Acceleration";
				// 
				// txtZHighSpeed
				// 
				this->txtZHighSpeed->Location = System::Drawing::Point(338, 268);
				this->txtZHighSpeed->Name = L"txtZHighSpeed";
				this->txtZHighSpeed->Size = System::Drawing::Size(59, 20);
				this->txtZHighSpeed->TabIndex = 38;
				this->txtZHighSpeed->Leave += gcnew System::EventHandler(this, &MachineSettingsForm::OnLeaveZHighSpeed);
				// 
				// label14
				// 
				this->label14->AutoSize = true;
				this->label14->Location = System::Drawing::Point(220, 272);
				this->label14->Name = L"label14";
				this->label14->Size = System::Drawing::Size(73, 13);
				this->label14->TabIndex = 37;
				this->label14->Text = L"Z High Speed";
				// 
				// txtZLowSpeed
				// 
				this->txtZLowSpeed->Location = System::Drawing::Point(338, 243);
				this->txtZLowSpeed->Name = L"txtZLowSpeed";
				this->txtZLowSpeed->Size = System::Drawing::Size(59, 20);
				this->txtZLowSpeed->TabIndex = 36;
				this->txtZLowSpeed->Leave += gcnew System::EventHandler(this, &MachineSettingsForm::OnLeaveZLowSpeed);
				// 
				// label15
				// 
				this->label15->AutoSize = true;
				this->label15->Location = System::Drawing::Point(220, 247);
				this->label15->Name = L"label15";
				this->label15->Size = System::Drawing::Size(71, 13);
				this->label15->TabIndex = 35;
				this->label15->Text = L"Z Low Speed";
				// 
				// txtTimeBracketTol
				// 
				this->txtTimeBracketTol->Location = System::Drawing::Point(338, 93);
				this->txtTimeBracketTol->Name = L"txtTimeBracketTol";
				this->txtTimeBracketTol->Size = System::Drawing::Size(59, 20);
				this->txtTimeBracketTol->TabIndex = 10;
				this->txtTimeBracketTol->Leave += gcnew System::EventHandler(this, &MachineSettingsForm::OnLeaveTimeBracketTol);
				// 
				// label16
				// 
				this->label16->AutoSize = true;
				this->label16->Location = System::Drawing::Point(151, 98);
				this->label16->Name = L"label16";
				this->label16->Size = System::Drawing::Size(152, 13);
				this->label16->TabIndex = 9;
				this->label16->Text = L"Time bracketing tolerance (ms)";
				// 
				// groupBox4
				// 
				this->groupBox4->Location = System::Drawing::Point(15, 319);
				this->groupBox4->Name = L"groupBox4";
				this->groupBox4->Size = System::Drawing::Size(380, 10);
				this->groupBox4->TabIndex = 41;
				this->groupBox4->TabStop = false;
				// 
				// txtLowestZ
				// 
				this->txtLowestZ->Location = System::Drawing::Point(130, 338);
				this->txtLowestZ->Name = L"txtLowestZ";
				this->txtLowestZ->Size = System::Drawing::Size(59, 20);
				this->txtLowestZ->TabIndex = 43;
				this->txtLowestZ->Leave += gcnew System::EventHandler(this, &MachineSettingsForm::OnLeaveLowestZ);
				// 
				// label17
				// 
				this->label17->AutoSize = true;
				this->label17->Location = System::Drawing::Point(12, 342);
				this->label17->Name = L"label17";
				this->label17->Size = System::Drawing::Size(115, 13);
				this->label17->TabIndex = 42;
				this->label17->Text = L"Lowest plate surface Z";
				// 
				// txtWorkingLight
				// 
				this->txtWorkingLight->Location = System::Drawing::Point(336, 339);
				this->txtWorkingLight->Name = L"txtWorkingLight";
				this->txtWorkingLight->Size = System::Drawing::Size(59, 20);
				this->txtWorkingLight->TabIndex = 45;
				this->txtWorkingLight->Leave += gcnew System::EventHandler(this, &MachineSettingsForm::OnLeaveWorkingLight);
				// 
				// label18
				// 
				this->label18->AutoSize = true;
				this->label18->Location = System::Drawing::Point(218, 343);
				this->label18->Name = L"label18";
				this->label18->Size = System::Drawing::Size(69, 13);
				this->label18->TabIndex = 44;
				this->label18->Text = L"Working light";
				// 
				// txtReferenceY
				// 
				this->txtReferenceY->Location = System::Drawing::Point(336, 364);
				this->txtReferenceY->Name = L"txtReferenceY";
				this->txtReferenceY->Size = System::Drawing::Size(59, 20);
				this->txtReferenceY->TabIndex = 49;
				this->txtReferenceY->Leave += gcnew System::EventHandler(this, &MachineSettingsForm::OnLeaveReferenceY);
				// 
				// label19
				// 
				this->label19->AutoSize = true;
				this->label19->Location = System::Drawing::Point(218, 368);
				this->label19->Name = L"label19";
				this->label19->Size = System::Drawing::Size(67, 13);
				this->label19->TabIndex = 48;
				this->label19->Text = L"Reference Y";
				// 
				// txtReferenceX
				// 
				this->txtReferenceX->Location = System::Drawing::Point(130, 363);
				this->txtReferenceX->Name = L"txtReferenceX";
				this->txtReferenceX->Size = System::Drawing::Size(59, 20);
				this->txtReferenceX->TabIndex = 47;
				this->txtReferenceX->Leave += gcnew System::EventHandler(this, &MachineSettingsForm::OnLeaveReferenceX);
				// 
				// label20
				// 
				this->label20->AutoSize = true;
				this->label20->Location = System::Drawing::Point(12, 367);
				this->label20->Name = L"label20";
				this->label20->Size = System::Drawing::Size(67, 13);
				this->label20->TabIndex = 46;
				this->label20->Text = L"Reference X";
				// 
				// txtHomingX
				// 
				this->txtHomingX->Location = System::Drawing::Point(130, 389);
				this->txtHomingX->Name = L"txtHomingX";
				this->txtHomingX->Size = System::Drawing::Size(59, 20);
				this->txtHomingX->TabIndex = 51;
				this->txtHomingX->Leave += gcnew System::EventHandler(this, &MachineSettingsForm::OnLeaveHomingX);
				// 
				// label21
				// 
				this->label21->AutoSize = true;
				this->label21->Location = System::Drawing::Point(12, 393);
				this->label21->Name = L"label21";
				this->label21->Size = System::Drawing::Size(79, 13);
				this->label21->TabIndex = 50;
				this->label21->Text = L"Homing X, Y, Z";
				// 
				// txtHomingY
				// 
				this->txtHomingY->Location = System::Drawing::Point(195, 389);
				this->txtHomingY->Name = L"txtHomingY";
				this->txtHomingY->Size = System::Drawing::Size(59, 20);
				this->txtHomingY->TabIndex = 52;
				this->txtHomingY->Leave += gcnew System::EventHandler(this, &MachineSettingsForm::OnLeaveHomingY);
				// 
				// txtHomingZ
				// 
				this->txtHomingZ->Location = System::Drawing::Point(260, 389);
				this->txtHomingZ->Name = L"txtHomingZ";
				this->txtHomingZ->Size = System::Drawing::Size(59, 20);
				this->txtHomingZ->TabIndex = 53;
				this->txtHomingZ->Leave += gcnew System::EventHandler(this, &MachineSettingsForm::OnLeaveHomingZ);
				// 
				// MachineSettingsForm
				// 
				this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
				this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
				this->ClientSize = System::Drawing::Size(412, 481);
				this->Controls->Add(this->txtHomingZ);
				this->Controls->Add(this->txtHomingY);
				this->Controls->Add(this->txtHomingX);
				this->Controls->Add(this->label21);
				this->Controls->Add(this->txtReferenceY);
				this->Controls->Add(this->label19);
				this->Controls->Add(this->txtReferenceX);
				this->Controls->Add(this->label20);
				this->Controls->Add(this->txtWorkingLight);
				this->Controls->Add(this->label18);
				this->Controls->Add(this->txtLowestZ);
				this->Controls->Add(this->label17);
				this->Controls->Add(this->groupBox4);
				this->Controls->Add(this->txtTimeBracketTol);
				this->Controls->Add(this->label16);
				this->Controls->Add(this->txtZAccel);
				this->Controls->Add(this->label13);
				this->Controls->Add(this->txtZHighSpeed);
				this->Controls->Add(this->label14);
				this->Controls->Add(this->txtZLowSpeed);
				this->Controls->Add(this->label15);
				this->Controls->Add(this->txtXYAccel);
				this->Controls->Add(this->label12);
				this->Controls->Add(this->txtXYHighSpeed);
				this->Controls->Add(this->label10);
				this->Controls->Add(this->txtXYLowSpeed);
				this->Controls->Add(this->label11);
				this->Controls->Add(this->groupBox3);
				this->Controls->Add(this->btnCancel);
				this->Controls->Add(this->btnOK);
				this->Controls->Add(this->chkInvertZ);
				this->Controls->Add(this->txtZMicronLine);
				this->Controls->Add(this->label7);
				this->Controls->Add(this->txtZMotorStepsRev);
				this->Controls->Add(this->label8);
				this->Controls->Add(this->txtZEncoderLinesRev);
				this->Controls->Add(this->label9);
				this->Controls->Add(this->groupBox2);
				this->Controls->Add(this->chkInvertY);
				this->Controls->Add(this->chkInvertX);
				this->Controls->Add(this->txtXYMicronLine);
				this->Controls->Add(this->label6);
				this->Controls->Add(this->txtXYMotorStepsRev);
				this->Controls->Add(this->label4);
				this->Controls->Add(this->txtXYEncoderLinesRev);
				this->Controls->Add(this->label5);
				this->Controls->Add(this->groupBox1);
				this->Controls->Add(this->txtTurnOffLightLimitSeconds);
				this->Controls->Add(this->label3);
				this->Controls->Add(this->txtLightLimit);
				this->Controls->Add(this->label2);
				this->Controls->Add(this->chkInvertLimiters);
				this->Controls->Add(this->rdCtlIsStepDir);
				this->Controls->Add(this->rdCtlIsCWCCW);
				this->Controls->Add(this->txtBoardId);
				this->Controls->Add(this->label1);
				this->FormBorderStyle = System::Windows::Forms::FormBorderStyle::FixedToolWindow;
				this->Name = L"MachineSettingsForm";
				this->Text = L"OmsMAXStage Machine Settings";
				this->Load += gcnew System::EventHandler(this, &MachineSettingsForm::OnLoad);
				this->ResumeLayout(false);
				this->PerformLayout();

			}
#pragma endregion
private: System::Void OnLoad(System::Object^  sender, System::EventArgs^  e) {
			 this->txtBoardId->Text = MC->BoardId.ToString();
			 this->txtLightLimit->Text = MC->LightLimit.ToString();
			 this->txtTurnOffLightLimitSeconds->Text = MC->TurnOffLightTimeSeconds.ToString();
			 this->txtXYEncoderLinesRev->Text = MC->XYLinesRev.ToString();
			 this->txtZEncoderLinesRev->Text = MC->ZLinesRev.ToString();
			 this->txtXYMotorStepsRev->Text = MC->XYStepsRev.ToString();
			 this->txtZMotorStepsRev->Text = MC->ZStepsRev.ToString();
			 this->txtXYMicronLine->Text = MC->XYEncoderToMicrons.ToString("F4", System::Globalization::CultureInfo::InvariantCulture);
			 this->txtZMicronLine->Text = MC->ZEncoderToMicrons.ToString("F4", System::Globalization::CultureInfo::InvariantCulture);
			 this->rdCtlIsStepDir->Checked = !MC->CtlModeIsCWCCW;
			 this->rdCtlIsCWCCW->Checked = MC->CtlModeIsCWCCW;
			 this->chkInvertLimiters->Checked = MC->InvertLimiterPolarity;
			 this->chkInvertX->Checked = MC->InvertX;
			 this->chkInvertY->Checked = MC->InvertY;
			 this->chkInvertZ->Checked = MC->InvertZ;
			 this->txtXYAccel->Text = MC->XYAcceleration.ToString("F1", System::Globalization::CultureInfo::InvariantCulture);
			 this->txtXYLowSpeed->Text = MC->XYLowSpeed.ToString("F1", System::Globalization::CultureInfo::InvariantCulture);
			 this->txtXYHighSpeed->Text = MC->XYHighSpeed.ToString("F1", System::Globalization::CultureInfo::InvariantCulture);
			 this->txtZAccel->Text = MC->ZAcceleration.ToString("F1", System::Globalization::CultureInfo::InvariantCulture);
			 this->txtZLowSpeed->Text = MC->ZLowSpeed.ToString("F1", System::Globalization::CultureInfo::InvariantCulture);
			 this->txtZHighSpeed->Text = MC->ZHighSpeed.ToString("F1", System::Globalization::CultureInfo::InvariantCulture);
			 this->txtTimeBracketTol->Text = MC->TimeBracketingTolerance.ToString("F2", System::Globalization::CultureInfo::InvariantCulture);
			 this->txtLowestZ->Text = MC->LowestZ.ToString("F2", System::Globalization::CultureInfo::InvariantCulture);
			 this->txtWorkingLight->Text = MC->WorkingLight.ToString();
			 this->txtReferenceX->Text = MC->ReferenceX.ToString("F1", System::Globalization::CultureInfo::InvariantCulture);
			 this->txtReferenceY->Text = MC->ReferenceY.ToString("F1", System::Globalization::CultureInfo::InvariantCulture);
			 this->txtHomingX->Text = MC->HomingX.ToString("F1", System::Globalization::CultureInfo::InvariantCulture);
			 this->txtHomingY->Text = MC->HomingY.ToString("F1", System::Globalization::CultureInfo::InvariantCulture);
			 this->txtHomingZ->Text = MC->HomingZ.ToString("F1", System::Globalization::CultureInfo::InvariantCulture);
		 }

#define VALIDATEINT(ctl,var) try { var = Convert::ToInt32(ctl->Text); } catch (Exception ^) { ctl->Text = var.ToString(); ctl->Focus(); }
#define VALIDATEUINT(ctl,var) try { var = Convert::ToUInt32(ctl->Text); } catch (Exception ^) { ctl->Text = var.ToString(); ctl->Focus(); }
#define VALIDATEDOUBLE(ctl,var,fmt) try { var = Convert::ToDouble(ctl->Text, System::Globalization::CultureInfo::InvariantCulture); } catch (Exception ^) { ctl->Text = var.ToString(fmt, System::Globalization::CultureInfo::InvariantCulture); ctl->Focus(); }
#define VALIDATEPOSITIVEDOUBLE(ctl,var,fmt) try { double v = Convert::ToDouble(ctl->Text, System::Globalization::CultureInfo::InvariantCulture); if (v <= 0.0) throw gcnew System::Exception("A positive number is required."); var = v; } catch (Exception ^) { ctl->Text = var.ToString(fmt, System::Globalization::CultureInfo::InvariantCulture); ctl->Focus(); }

private: System::Void OnLeaveBoardId(System::Object^  sender, System::EventArgs^  e) {
			VALIDATEINT(txtBoardId, MC->BoardId)
		 }
private: System::Void OnLeaveLightLimit(System::Object^  sender, System::EventArgs^  e) {
			VALIDATEUINT(txtLightLimit, MC->LightLimit)
		 }
private: System::Void OnLeaveTurnOffLightSeconds(System::Object^  sender, System::EventArgs^  e) {
			VALIDATEUINT(txtTurnOffLightLimitSeconds, MC->TurnOffLightTimeSeconds)
		 }
private: System::Void OnLeaveXYEncoderLinesRev(System::Object^  sender, System::EventArgs^  e) {
			VALIDATEUINT(txtXYEncoderLinesRev, MC->XYLinesRev)			
		 }
private: System::Void OnLeaveXYMotorStepsRev(System::Object^  sender, System::EventArgs^  e) {
			VALIDATEUINT(txtXYMotorStepsRev, MC->XYStepsRev)
		 }
private: System::Void OnLeaveXYMicronLine(System::Object^  sender, System::EventArgs^  e) {
			VALIDATEDOUBLE(txtXYMicronLine, MC->XYEncoderToMicrons, "F4")
		 }
private: System::Void OnLeaveZEncoderLinesRev(System::Object^  sender, System::EventArgs^  e) {
			VALIDATEUINT(txtZEncoderLinesRev, MC->ZLinesRev)
		 }
private: System::Void OnLeaveZMotorStepsRev(System::Object^  sender, System::EventArgs^  e) {
			VALIDATEUINT(txtZMotorStepsRev, MC->ZStepsRev)
		 }
private: System::Void OnLeaveZMicronLine(System::Object^  sender, System::EventArgs^  e) {
			VALIDATEDOUBLE(txtZMicronLine, MC->ZEncoderToMicrons, "F4")
		 }
private: System::Void btnOK_Click(System::Object^  sender, System::EventArgs^  e) {
			 MC->CtlModeIsCWCCW = rdCtlIsCWCCW->Checked;
			 MC->InvertLimiterPolarity = chkInvertLimiters->Checked;
			 MC->InvertX = chkInvertX->Checked;
			 MC->InvertY = chkInvertY->Checked;
			 MC->InvertZ = chkInvertZ->Checked;
			 this->DialogResult = System::Windows::Forms::DialogResult::OK;
			 this->Close();
		 }
private: System::Void btnCancel_Click(System::Object^  sender, System::EventArgs^  e) {
			 this->DialogResult = System::Windows::Forms::DialogResult::Cancel;
			 this->Close();
		 }
private: System::Void OnLeaveXYLowSpeed(System::Object^  sender, System::EventArgs^  e) {
			 VALIDATEPOSITIVEDOUBLE(txtXYLowSpeed, MC->XYLowSpeed, "F1")
		 }
private: System::Void OnLeaveXYHighSpeed(System::Object^  sender, System::EventArgs^  e) {
			 VALIDATEPOSITIVEDOUBLE(txtXYHighSpeed, MC->XYHighSpeed, "F1")
		 }
private: System::Void OnLeaveXYAccel(System::Object^  sender, System::EventArgs^  e) {
			 VALIDATEPOSITIVEDOUBLE(txtXYAccel, MC->XYAcceleration, "F1")
		 }
private: System::Void OnLeaveZLowSpeed(System::Object^  sender, System::EventArgs^  e) {
			 VALIDATEPOSITIVEDOUBLE(txtZLowSpeed, MC->ZLowSpeed, "F1")
		 }
private: System::Void OnLeaveZHighSpeed(System::Object^  sender, System::EventArgs^  e) {
			 VALIDATEPOSITIVEDOUBLE(txtZHighSpeed, MC->ZHighSpeed, "F1")
		 }
private: System::Void OnLeaveZAccel(System::Object^  sender, System::EventArgs^  e) {
			 VALIDATEPOSITIVEDOUBLE(txtZAccel, MC->ZAcceleration, "F1")
		 }
private: System::Void OnLeaveTimeBracketTol(System::Object^  sender, System::EventArgs^  e) {
			 VALIDATEPOSITIVEDOUBLE(txtTimeBracketTol, MC->TimeBracketingTolerance, "F2")
		 }
private: System::Void OnLeaveLowestZ(System::Object^  sender, System::EventArgs^  e) {
			 VALIDATEDOUBLE(txtLowestZ, MC->LowestZ, "F2")
		 }
private: System::Void OnLeaveWorkingLight(System::Object^  sender, System::EventArgs^  e) {
			 VALIDATEUINT(txtWorkingLight, MC->WorkingLight)
		 }
private: System::Void OnLeaveReferenceX(System::Object^  sender, System::EventArgs^  e) {
			 VALIDATEDOUBLE(txtReferenceX, MC->ReferenceX, "F1")
		 }
private: System::Void OnLeaveReferenceY(System::Object^  sender, System::EventArgs^  e) {
			 VALIDATEDOUBLE(txtReferenceY, MC->ReferenceY, "F1")
		 }
private: System::Void OnLeaveHomingX(System::Object^  sender, System::EventArgs^  e) {
			 VALIDATEDOUBLE(txtHomingX, MC->HomingX, "F1")
		 }
private: System::Void OnLeaveHomingY(System::Object^  sender, System::EventArgs^  e) {
			 VALIDATEDOUBLE(txtHomingY, MC->HomingY, "F1")
		 }
private: System::Void OnLeaveHomingZ(System::Object^  sender, System::EventArgs^  e) {
			 VALIDATEDOUBLE(txtHomingZ, MC->HomingZ, "F1")
		 }
};
	}
}