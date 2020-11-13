#pragma once

using namespace System;
using namespace System::ComponentModel;
using namespace System::Collections;
using namespace System::Windows::Forms;
using namespace System::Data;
using namespace System::Drawing;

#include "OmsMAX.h"

namespace OmsMAXStage {

	/// <summary>
	/// Summary for StageMonitor
	///
	/// WARNING: If you change the name of this class, you will need to change the
	///          'Resource File Name' property for the managed resource compiler tool
	///          associated with all .resx files this class depends on.  Otherwise,
	///          the designers will not be able to interact properly with localized
	///          resources associated with this form.
	/// </summary>
	public ref class StageMonitor : public System::Windows::Forms::Form, SySal::StageControl::StageMonitorBase
	{			
	private:
		SySal::StageControl::OmsMAXStage ^m_Stage;
		SySal::StageControl::OmsMAXStageSettings ^m_StageSettings;
	private: System::Windows::Forms::TextBox^  txtLight;
	private: System::Windows::Forms::Label^  label4;
	private: System::Windows::Forms::Button^  btnExtras;
	private: System::Windows::Forms::Button^  btnReset;
	private: System::Windows::Forms::TextBox^  txtGoPos;

	private: System::Windows::Forms::Button^  btnGo;
	private: System::Windows::Forms::ComboBox^  cmbAxis;
	private: System::Windows::Forms::Button^  btnHome;
	private: System::Windows::Forms::TextBox^  txtHoming;
	private: System::Windows::Forms::Button^  btnSENDSTR;
	private: System::Windows::Forms::TextBox^  txtSENDSTR;
	private: System::Windows::Forms::Label^  IDlabel;
	private: System::Windows::Forms::Button^  btnForceHomed;
	private: System::Windows::Forms::TextBox^  txtLastCmdX;
	private: System::Windows::Forms::TextBox^  txtLastCmdY;
	private: System::Windows::Forms::TextBox^  txtLastCmdZ;
	private: System::Windows::Forms::Button^  btnDEBUGCHECK;
	private: System::Windows::Forms::Button^  btnSendReceive;
	private: System::Windows::Forms::Button^  btnTest;
	private: System::Windows::Forms::TextBox^  txtGeneral;

			 bool HighSpeed;
	public:
		StageMonitor(SySal::StageControl::OmsMAXStage ^s, SySal::StageControl::OmsMAXStageSettings ^ss)
		{			
			InitializeComponent();
			//
			//TODO: Add the constructor code here
			//			
			m_Stage = s;
			m_StageSettings = ss;
			GUIThread = gcnew System::Threading::Thread(gcnew System::Threading::ThreadStart(this, &OmsMAXStage::StageMonitor::GUIProc));
		}

	protected:
		delegate void dSetIDlabel(System::String ^txt);
		delegate void dClose();
		void GUIProc()
		{
			ShowDialog();
		}

	public:		
		void RunGUI() { GUIThread->Start(); }
		void CloseGUI() { this->Invoke(gcnew dClose(this, &System::Windows::Forms::Form::Close)); }
		virtual System::Void SetIDLabel(System::String ^txt) 
		{ 
			if (this->InvokeRequired) 
			{
				cli::array<System::Object ^> ^paramarr = gcnew cli::array<System::Object ^>(1);
				paramarr[0] = txt;
				this->Invoke(gcnew dSetIDlabel(this, &StageMonitor::SetIDLabel), paramarr); 
			}
			else IDlabel->Text = txt;		
		}

		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~StageMonitor()
		{
			if (components)
			{
				delete components;
			}
		}
	private: System::Windows::Forms::Label^  label1;
	protected: 
	private: System::Windows::Forms::TextBox^  txtX;
	private: System::Windows::Forms::TextBox^  txtY;

	private: System::Windows::Forms::Label^  label2;
	private: System::Windows::Forms::TextBox^  txtZ;

	private: System::Windows::Forms::Label^  label3;
	private: System::Windows::Forms::TextBox^  txtXRev;
	private: System::Windows::Forms::TextBox^  txtXFwd;
	private: System::Windows::Forms::TextBox^  txtXMotor;
	private: System::Windows::Forms::TextBox^  txtYMotor;
	private: System::Windows::Forms::TextBox^  txtYFwd;
	private: System::Windows::Forms::TextBox^  txtYRev;
	private: System::Windows::Forms::TextBox^  txtZMotor;
	private: System::Windows::Forms::TextBox^  txtZFwd;
	private: System::Windows::Forms::TextBox^  txtZRev;
	private: System::Windows::Forms::Timer^  m_RefreshTimer;
	private: System::Windows::Forms::TextBox^  txtKeys;


	private: System::Threading::Thread ^GUIThread;




	private: System::ComponentModel::IContainer^  components;



	private:
		/// <summary>
		/// Required designer variable.
		/// </summary>


#pragma region Windows Form Designer generated code
		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		void InitializeComponent(void)
		{
			this->components = (gcnew System::ComponentModel::Container());
			this->label1 = (gcnew System::Windows::Forms::Label());
			this->txtX = (gcnew System::Windows::Forms::TextBox());
			this->txtY = (gcnew System::Windows::Forms::TextBox());
			this->label2 = (gcnew System::Windows::Forms::Label());
			this->txtZ = (gcnew System::Windows::Forms::TextBox());
			this->label3 = (gcnew System::Windows::Forms::Label());
			this->txtXRev = (gcnew System::Windows::Forms::TextBox());
			this->txtXFwd = (gcnew System::Windows::Forms::TextBox());
			this->txtXMotor = (gcnew System::Windows::Forms::TextBox());
			this->txtYMotor = (gcnew System::Windows::Forms::TextBox());
			this->txtYFwd = (gcnew System::Windows::Forms::TextBox());
			this->txtYRev = (gcnew System::Windows::Forms::TextBox());
			this->txtZMotor = (gcnew System::Windows::Forms::TextBox());
			this->txtZFwd = (gcnew System::Windows::Forms::TextBox());
			this->txtZRev = (gcnew System::Windows::Forms::TextBox());
			this->m_RefreshTimer = (gcnew System::Windows::Forms::Timer(this->components));
			this->txtKeys = (gcnew System::Windows::Forms::TextBox());
			this->txtLight = (gcnew System::Windows::Forms::TextBox());
			this->label4 = (gcnew System::Windows::Forms::Label());
			this->btnExtras = (gcnew System::Windows::Forms::Button());
			this->btnReset = (gcnew System::Windows::Forms::Button());
			this->txtGoPos = (gcnew System::Windows::Forms::TextBox());
			this->btnGo = (gcnew System::Windows::Forms::Button());
			this->cmbAxis = (gcnew System::Windows::Forms::ComboBox());
			this->btnHome = (gcnew System::Windows::Forms::Button());
			this->txtHoming = (gcnew System::Windows::Forms::TextBox());
			this->btnSENDSTR = (gcnew System::Windows::Forms::Button());
			this->txtSENDSTR = (gcnew System::Windows::Forms::TextBox());
			this->IDlabel = (gcnew System::Windows::Forms::Label());
			this->btnForceHomed = (gcnew System::Windows::Forms::Button());
			this->txtLastCmdX = (gcnew System::Windows::Forms::TextBox());
			this->txtLastCmdY = (gcnew System::Windows::Forms::TextBox());
			this->txtLastCmdZ = (gcnew System::Windows::Forms::TextBox());
			this->btnDEBUGCHECK = (gcnew System::Windows::Forms::Button());
			this->btnSendReceive = (gcnew System::Windows::Forms::Button());
			this->btnTest = (gcnew System::Windows::Forms::Button());
			this->txtGeneral = (gcnew System::Windows::Forms::TextBox());
			this->SuspendLayout();
			// 
			// label1
			// 
			this->label1->AutoSize = true;
			this->label1->Location = System::Drawing::Point(12, 31);
			this->label1->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label1->Name = L"label1";
			this->label1->Size = System::Drawing::Size(14, 13);
			this->label1->TabIndex = 0;
			this->label1->Text = L"X";
			// 
			// txtX
			// 
			this->txtX->Location = System::Drawing::Point(52, 29);
			this->txtX->Margin = System::Windows::Forms::Padding(4, 3, 4, 3);
			this->txtX->Name = L"txtX";
			this->txtX->ReadOnly = true;
			this->txtX->Size = System::Drawing::Size(56, 20);
			this->txtX->TabIndex = 1;
			// 
			// txtY
			// 
			this->txtY->Location = System::Drawing::Point(52, 55);
			this->txtY->Margin = System::Windows::Forms::Padding(4, 3, 4, 3);
			this->txtY->Name = L"txtY";
			this->txtY->ReadOnly = true;
			this->txtY->Size = System::Drawing::Size(56, 20);
			this->txtY->TabIndex = 3;
			// 
			// label2
			// 
			this->label2->AutoSize = true;
			this->label2->Location = System::Drawing::Point(12, 57);
			this->label2->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label2->Name = L"label2";
			this->label2->Size = System::Drawing::Size(14, 13);
			this->label2->TabIndex = 2;
			this->label2->Text = L"Y";
			// 
			// txtZ
			// 
			this->txtZ->Location = System::Drawing::Point(52, 81);
			this->txtZ->Margin = System::Windows::Forms::Padding(4, 3, 4, 3);
			this->txtZ->Name = L"txtZ";
			this->txtZ->ReadOnly = true;
			this->txtZ->Size = System::Drawing::Size(56, 20);
			this->txtZ->TabIndex = 5;
			// 
			// label3
			// 
			this->label3->AutoSize = true;
			this->label3->Location = System::Drawing::Point(12, 81);
			this->label3->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label3->Name = L"label3";
			this->label3->Size = System::Drawing::Size(14, 13);
			this->label3->TabIndex = 4;
			this->label3->Text = L"Z";
			// 
			// txtXRev
			// 
			this->txtXRev->BackColor = System::Drawing::Color::Teal;
			this->txtXRev->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
			this->txtXRev->Location = System::Drawing::Point(116, 31);
			this->txtXRev->Margin = System::Windows::Forms::Padding(4, 3, 4, 3);
			this->txtXRev->Name = L"txtXRev";
			this->txtXRev->ReadOnly = true;
			this->txtXRev->Size = System::Drawing::Size(20, 20);
			this->txtXRev->TabIndex = 6;
			this->txtXRev->Text = L"<";
			this->txtXRev->TextAlign = System::Windows::Forms::HorizontalAlignment::Center;
			// 
			// txtXFwd
			// 
			this->txtXFwd->BackColor = System::Drawing::Color::Teal;
			this->txtXFwd->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
			this->txtXFwd->Location = System::Drawing::Point(138, 31);
			this->txtXFwd->Margin = System::Windows::Forms::Padding(4, 3, 4, 3);
			this->txtXFwd->Name = L"txtXFwd";
			this->txtXFwd->ReadOnly = true;
			this->txtXFwd->Size = System::Drawing::Size(20, 20);
			this->txtXFwd->TabIndex = 7;
			this->txtXFwd->Text = L">";
			this->txtXFwd->TextAlign = System::Windows::Forms::HorizontalAlignment::Center;
			// 
			// txtXMotor
			// 
			this->txtXMotor->BackColor = System::Drawing::Color::Teal;
			this->txtXMotor->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
			this->txtXMotor->Location = System::Drawing::Point(161, 31);
			this->txtXMotor->Margin = System::Windows::Forms::Padding(4, 3, 4, 3);
			this->txtXMotor->Name = L"txtXMotor";
			this->txtXMotor->ReadOnly = true;
			this->txtXMotor->Size = System::Drawing::Size(20, 20);
			this->txtXMotor->TabIndex = 8;
			this->txtXMotor->Text = L"M";
			this->txtXMotor->TextAlign = System::Windows::Forms::HorizontalAlignment::Center;
			// 
			// txtYMotor
			// 
			this->txtYMotor->BackColor = System::Drawing::Color::Teal;
			this->txtYMotor->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
			this->txtYMotor->Location = System::Drawing::Point(161, 55);
			this->txtYMotor->Margin = System::Windows::Forms::Padding(4, 3, 4, 3);
			this->txtYMotor->Name = L"txtYMotor";
			this->txtYMotor->ReadOnly = true;
			this->txtYMotor->Size = System::Drawing::Size(20, 20);
			this->txtYMotor->TabIndex = 11;
			this->txtYMotor->Text = L"M";
			this->txtYMotor->TextAlign = System::Windows::Forms::HorizontalAlignment::Center;
			// 
			// txtYFwd
			// 
			this->txtYFwd->BackColor = System::Drawing::Color::Teal;
			this->txtYFwd->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
			this->txtYFwd->Location = System::Drawing::Point(138, 55);
			this->txtYFwd->Margin = System::Windows::Forms::Padding(4, 3, 4, 3);
			this->txtYFwd->Name = L"txtYFwd";
			this->txtYFwd->ReadOnly = true;
			this->txtYFwd->Size = System::Drawing::Size(20, 20);
			this->txtYFwd->TabIndex = 10;
			this->txtYFwd->Text = L">";
			this->txtYFwd->TextAlign = System::Windows::Forms::HorizontalAlignment::Center;
			// 
			// txtYRev
			// 
			this->txtYRev->BackColor = System::Drawing::Color::Teal;
			this->txtYRev->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
			this->txtYRev->Location = System::Drawing::Point(116, 55);
			this->txtYRev->Margin = System::Windows::Forms::Padding(4, 3, 4, 3);
			this->txtYRev->Name = L"txtYRev";
			this->txtYRev->ReadOnly = true;
			this->txtYRev->Size = System::Drawing::Size(20, 20);
			this->txtYRev->TabIndex = 9;
			this->txtYRev->Text = L"<";
			this->txtYRev->TextAlign = System::Windows::Forms::HorizontalAlignment::Center;
			// 
			// txtZMotor
			// 
			this->txtZMotor->BackColor = System::Drawing::Color::Teal;
			this->txtZMotor->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
			this->txtZMotor->Location = System::Drawing::Point(161, 80);
			this->txtZMotor->Margin = System::Windows::Forms::Padding(4, 3, 4, 3);
			this->txtZMotor->Name = L"txtZMotor";
			this->txtZMotor->ReadOnly = true;
			this->txtZMotor->Size = System::Drawing::Size(20, 20);
			this->txtZMotor->TabIndex = 14;
			this->txtZMotor->Text = L"M";
			this->txtZMotor->TextAlign = System::Windows::Forms::HorizontalAlignment::Center;
			// 
			// txtZFwd
			// 
			this->txtZFwd->BackColor = System::Drawing::Color::Teal;
			this->txtZFwd->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
			this->txtZFwd->Location = System::Drawing::Point(138, 80);
			this->txtZFwd->Margin = System::Windows::Forms::Padding(4, 3, 4, 3);
			this->txtZFwd->Name = L"txtZFwd";
			this->txtZFwd->ReadOnly = true;
			this->txtZFwd->Size = System::Drawing::Size(20, 20);
			this->txtZFwd->TabIndex = 13;
			this->txtZFwd->Text = L">";
			this->txtZFwd->TextAlign = System::Windows::Forms::HorizontalAlignment::Center;
			// 
			// txtZRev
			// 
			this->txtZRev->BackColor = System::Drawing::Color::Teal;
			this->txtZRev->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
			this->txtZRev->Location = System::Drawing::Point(116, 80);
			this->txtZRev->Margin = System::Windows::Forms::Padding(4, 3, 4, 3);
			this->txtZRev->Name = L"txtZRev";
			this->txtZRev->ReadOnly = true;
			this->txtZRev->Size = System::Drawing::Size(20, 20);
			this->txtZRev->TabIndex = 12;
			this->txtZRev->Text = L"<";
			this->txtZRev->TextAlign = System::Windows::Forms::HorizontalAlignment::Center;
			// 
			// m_RefreshTimer
			// 
			this->m_RefreshTimer->Enabled = true;
			this->m_RefreshTimer->Interval = 50;
			this->m_RefreshTimer->Tick += gcnew System::EventHandler(this, &StageMonitor::OnRefreshTick);
			// 
			// txtKeys
			// 
			this->txtKeys->BackColor = System::Drawing::Color::Gray;
			this->txtKeys->Cursor = System::Windows::Forms::Cursors::Default;
			this->txtKeys->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 8.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point, 
				static_cast<System::Byte>(0)));
			this->txtKeys->Location = System::Drawing::Point(192, 29);
			this->txtKeys->Margin = System::Windows::Forms::Padding(4, 3, 4, 3);
			this->txtKeys->Multiline = true;
			this->txtKeys->Name = L"txtKeys";
			this->txtKeys->ReadOnly = true;
			this->txtKeys->Size = System::Drawing::Size(96, 96);
			this->txtKeys->TabIndex = 15;
			this->txtKeys->Text = L"X:Left/Right\r\nY:Up/Down\r\nZ:PgUp/Dwn\r\nLight:+/-\r\nSPACE:SpeedUp";
			this->txtKeys->TextAlign = System::Windows::Forms::HorizontalAlignment::Center;
			this->txtKeys->Enter += gcnew System::EventHandler(this, &StageMonitor::OnEnterKeys);
			this->txtKeys->KeyDown += gcnew System::Windows::Forms::KeyEventHandler(this, &StageMonitor::OnKeysKeyDown);
			this->txtKeys->KeyUp += gcnew System::Windows::Forms::KeyEventHandler(this, &StageMonitor::OnKeysKeyUp);
			this->txtKeys->Leave += gcnew System::EventHandler(this, &StageMonitor::OnLeaveKeys);
			// 
			// txtLight
			// 
			this->txtLight->Location = System::Drawing::Point(52, 107);
			this->txtLight->Margin = System::Windows::Forms::Padding(4, 3, 4, 3);
			this->txtLight->Name = L"txtLight";
			this->txtLight->ReadOnly = true;
			this->txtLight->Size = System::Drawing::Size(56, 20);
			this->txtLight->TabIndex = 17;
			// 
			// label4
			// 
			this->label4->AutoSize = true;
			this->label4->Location = System::Drawing::Point(12, 107);
			this->label4->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label4->Name = L"label4";
			this->label4->Size = System::Drawing::Size(30, 13);
			this->label4->TabIndex = 16;
			this->label4->Text = L"Light";
			// 
			// btnExtras
			// 
			this->btnExtras->Location = System::Drawing::Point(115, 106);
			this->btnExtras->Name = L"btnExtras";
			this->btnExtras->Size = System::Drawing::Size(66, 21);
			this->btnExtras->TabIndex = 18;
			this->btnExtras->Text = L"Extras";
			this->btnExtras->UseVisualStyleBackColor = true;
			this->btnExtras->Click += gcnew System::EventHandler(this, &StageMonitor::btnExtras_Click);
			// 
			// btnReset
			// 
			this->btnReset->Location = System::Drawing::Point(15, 143);
			this->btnReset->Name = L"btnReset";
			this->btnReset->Size = System::Drawing::Size(93, 21);
			this->btnReset->TabIndex = 19;
			this->btnReset->Text = L"Reset";
			this->btnReset->UseVisualStyleBackColor = true;
			this->btnReset->Click += gcnew System::EventHandler(this, &StageMonitor::btnReset_Click);
			// 
			// txtGoPos
			// 
			this->txtGoPos->Location = System::Drawing::Point(231, 144);
			this->txtGoPos->Margin = System::Windows::Forms::Padding(4, 3, 4, 3);
			this->txtGoPos->Name = L"txtGoPos";
			this->txtGoPos->Size = System::Drawing::Size(56, 20);
			this->txtGoPos->TabIndex = 20;
			// 
			// btnGo
			// 
			this->btnGo->Location = System::Drawing::Point(192, 144);
			this->btnGo->Name = L"btnGo";
			this->btnGo->Size = System::Drawing::Size(32, 21);
			this->btnGo->TabIndex = 21;
			this->btnGo->Text = L"Go";
			this->btnGo->UseVisualStyleBackColor = true;
			this->btnGo->Click += gcnew System::EventHandler(this, &StageMonitor::btnGo_Click);
			// 
			// cmbAxis
			// 
			this->cmbAxis->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
			this->cmbAxis->FormattingEnabled = true;
			this->cmbAxis->Items->AddRange(gcnew cli::array< System::Object^  >(3) {L"Axis X", L"Axis Y", L"Axis Z"});
			this->cmbAxis->Location = System::Drawing::Point(116, 143);
			this->cmbAxis->Name = L"cmbAxis";
			this->cmbAxis->Size = System::Drawing::Size(65, 21);
			this->cmbAxis->TabIndex = 22;
			// 
			// btnHome
			// 
			this->btnHome->Location = System::Drawing::Point(15, 170);
			this->btnHome->Name = L"btnHome";
			this->btnHome->Size = System::Drawing::Size(93, 21);
			this->btnHome->TabIndex = 23;
			this->btnHome->Text = L"Home";
			this->btnHome->UseVisualStyleBackColor = true;
			this->btnHome->Click += gcnew System::EventHandler(this, &StageMonitor::btnHome_Click);
			// 
			// txtHoming
			// 
			this->txtHoming->Location = System::Drawing::Point(115, 170);
			this->txtHoming->Margin = System::Windows::Forms::Padding(4, 3, 4, 3);
			this->txtHoming->Name = L"txtHoming";
			this->txtHoming->ReadOnly = true;
			this->txtHoming->Size = System::Drawing::Size(172, 20);
			this->txtHoming->TabIndex = 24;
			this->txtHoming->Text = L"Home undefined.";
			// 
			// btnSENDSTR
			// 
			this->btnSENDSTR->Location = System::Drawing::Point(295, 28);
			this->btnSENDSTR->Name = L"btnSENDSTR";
			this->btnSENDSTR->Size = System::Drawing::Size(59, 21);
			this->btnSENDSTR->TabIndex = 25;
			this->btnSENDSTR->Text = L"SEND";
			this->btnSENDSTR->UseVisualStyleBackColor = true;
			this->btnSENDSTR->Click += gcnew System::EventHandler(this, &StageMonitor::btnSENDSTR_Click);
			// 
			// txtSENDSTR
			// 
			this->txtSENDSTR->Location = System::Drawing::Point(295, 55);
			this->txtSENDSTR->Margin = System::Windows::Forms::Padding(4, 3, 4, 3);
			this->txtSENDSTR->Name = L"txtSENDSTR";
			this->txtSENDSTR->Size = System::Drawing::Size(128, 20);
			this->txtSENDSTR->TabIndex = 26;
			this->txtSENDSTR->Leave += gcnew System::EventHandler(this, &StageMonitor::OnSENDSTRChanged);
			// 
			// IDlabel
			// 
			this->IDlabel->Location = System::Drawing::Point(39, 3);
			this->IDlabel->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->IDlabel->Name = L"IDlabel";
			this->IDlabel->Size = System::Drawing::Size(384, 21);
			this->IDlabel->TabIndex = 27;
			this->IDlabel->Text = L"Unknown board";
			this->IDlabel->TextAlign = System::Drawing::ContentAlignment::MiddleLeft;
			// 
			// btnForceHomed
			// 
			this->btnForceHomed->Location = System::Drawing::Point(294, 170);
			this->btnForceHomed->Name = L"btnForceHomed";
			this->btnForceHomed->Size = System::Drawing::Size(129, 21);
			this->btnForceHomed->TabIndex = 28;
			this->btnForceHomed->Text = L"Force homed";
			this->btnForceHomed->UseVisualStyleBackColor = true;
			this->btnForceHomed->Click += gcnew System::EventHandler(this, &StageMonitor::btnForceHomed_Click);
			// 
			// txtLastCmdX
			// 
			this->txtLastCmdX->Location = System::Drawing::Point(294, 101);
			this->txtLastCmdX->Margin = System::Windows::Forms::Padding(4, 3, 4, 3);
			this->txtLastCmdX->Name = L"txtLastCmdX";
			this->txtLastCmdX->ReadOnly = true;
			this->txtLastCmdX->Size = System::Drawing::Size(130, 20);
			this->txtLastCmdX->TabIndex = 29;
			this->txtLastCmdX->Text = L"-";
			// 
			// txtLastCmdY
			// 
			this->txtLastCmdY->Location = System::Drawing::Point(294, 123);
			this->txtLastCmdY->Margin = System::Windows::Forms::Padding(4, 3, 4, 3);
			this->txtLastCmdY->Name = L"txtLastCmdY";
			this->txtLastCmdY->ReadOnly = true;
			this->txtLastCmdY->Size = System::Drawing::Size(130, 20);
			this->txtLastCmdY->TabIndex = 30;
			this->txtLastCmdY->Text = L"-";
			// 
			// txtLastCmdZ
			// 
			this->txtLastCmdZ->Location = System::Drawing::Point(294, 146);
			this->txtLastCmdZ->Margin = System::Windows::Forms::Padding(4, 3, 4, 3);
			this->txtLastCmdZ->Name = L"txtLastCmdZ";
			this->txtLastCmdZ->ReadOnly = true;
			this->txtLastCmdZ->Size = System::Drawing::Size(130, 20);
			this->txtLastCmdZ->TabIndex = 31;
			this->txtLastCmdZ->Text = L"-";
			// 
			// btnDEBUGCHECK
			// 
			this->btnDEBUGCHECK->Location = System::Drawing::Point(295, 79);
			this->btnDEBUGCHECK->Name = L"btnDEBUGCHECK";
			this->btnDEBUGCHECK->Size = System::Drawing::Size(129, 21);
			this->btnDEBUGCHECK->TabIndex = 32;
			this->btnDEBUGCHECK->Text = L"DEBUG-CHECK";
			this->btnDEBUGCHECK->UseVisualStyleBackColor = true;
			this->btnDEBUGCHECK->Click += gcnew System::EventHandler(this, &StageMonitor::btnDEBUGCHECK_Click);
			// 
			// btnSendReceive
			// 
			this->btnSendReceive->Location = System::Drawing::Point(360, 28);
			this->btnSendReceive->Name = L"btnSendReceive";
			this->btnSendReceive->Size = System::Drawing::Size(64, 21);
			this->btnSendReceive->TabIndex = 33;
			this->btnSendReceive->Text = L"SNDREC";
			this->btnSendReceive->UseVisualStyleBackColor = true;
			this->btnSendReceive->Click += gcnew System::EventHandler(this, &StageMonitor::btnSendReceive_Click);
			// 
			// btnTest
			// 
			this->btnTest->Location = System::Drawing::Point(294, 1);
			this->btnTest->Name = L"btnTest";
			this->btnTest->Size = System::Drawing::Size(129, 21);
			this->btnTest->TabIndex = 34;
			this->btnTest->Text = L"TEST";
			this->btnTest->UseVisualStyleBackColor = true;
			this->btnTest->Visible = false;
			this->btnTest->Click += gcnew System::EventHandler(this, &StageMonitor::btnTest_Click);
			// 
			// txtGeneral
			// 
			this->txtGeneral->BackColor = System::Drawing::Color::Teal;
			this->txtGeneral->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
			this->txtGeneral->Location = System::Drawing::Point(13, 3);
			this->txtGeneral->Margin = System::Windows::Forms::Padding(4, 3, 4, 3);
			this->txtGeneral->Name = L"txtGeneral";
			this->txtGeneral->ReadOnly = true;
			this->txtGeneral->Size = System::Drawing::Size(20, 20);
			this->txtGeneral->TabIndex = 35;
			this->txtGeneral->TextAlign = System::Windows::Forms::HorizontalAlignment::Center;
			// 
			// StageMonitor
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(436, 196);
			this->ControlBox = false;
			this->Controls->Add(this->txtGeneral);
			this->Controls->Add(this->btnTest);
			this->Controls->Add(this->btnSendReceive);
			this->Controls->Add(this->btnDEBUGCHECK);
			this->Controls->Add(this->txtLastCmdZ);
			this->Controls->Add(this->txtLastCmdY);
			this->Controls->Add(this->txtLastCmdX);
			this->Controls->Add(this->btnForceHomed);
			this->Controls->Add(this->IDlabel);
			this->Controls->Add(this->txtSENDSTR);
			this->Controls->Add(this->btnSENDSTR);
			this->Controls->Add(this->txtHoming);
			this->Controls->Add(this->btnHome);
			this->Controls->Add(this->cmbAxis);
			this->Controls->Add(this->btnGo);
			this->Controls->Add(this->txtGoPos);
			this->Controls->Add(this->btnReset);
			this->Controls->Add(this->btnExtras);
			this->Controls->Add(this->txtLight);
			this->Controls->Add(this->label4);
			this->Controls->Add(this->txtKeys);
			this->Controls->Add(this->txtZMotor);
			this->Controls->Add(this->txtZFwd);
			this->Controls->Add(this->txtZRev);
			this->Controls->Add(this->txtYMotor);
			this->Controls->Add(this->txtYFwd);
			this->Controls->Add(this->txtYRev);
			this->Controls->Add(this->txtXMotor);
			this->Controls->Add(this->txtXFwd);
			this->Controls->Add(this->txtXRev);
			this->Controls->Add(this->txtZ);
			this->Controls->Add(this->label3);
			this->Controls->Add(this->txtY);
			this->Controls->Add(this->label2);
			this->Controls->Add(this->txtX);
			this->Controls->Add(this->label1);
			this->FormBorderStyle = System::Windows::Forms::FormBorderStyle::FixedToolWindow;
			this->Margin = System::Windows::Forms::Padding(4, 3, 4, 3);
			this->MaximizeBox = false;
			this->MinimizeBox = false;
			this->Name = L"StageMonitor";
			this->ShowIcon = false;
			this->ShowInTaskbar = false;
			this->StartPosition = System::Windows::Forms::FormStartPosition::CenterScreen;
			this->Text = L"OmsMAXStage Monitor";
			this->Load += gcnew System::EventHandler(this, &StageMonitor::OnLoad);
			this->ResumeLayout(false);
			this->PerformLayout();

		}
#pragma endregion
		
	private:
			delegate void dRefresh();
			void RefreshView()
			 {				
				if (m_Stage != nullptr && this->Visible)
				{					
					txtX->Text = m_Stage->GetPos(SySal::StageControl::Axis::X).ToString("F1");
					txtY->Text = m_Stage->GetPos(SySal::StageControl::Axis::Y).ToString("F1");
					txtZ->Text = m_Stage->GetPos(SySal::StageControl::Axis::Z).ToString("F1");
					txtLight->Text = m_Stage->LightLevel.ToString();
					SySal::StageControl::AxisStatus as;
					as = m_Stage->GetStatus(SySal::StageControl::Axis::X);					
					txtXRev->BackColor = (((int)as & (int)SySal::StageControl::AxisStatus::ReverseLimitActive) != 0) ? System::Drawing::Color::Red : System::Drawing::Color::Lime;
					txtXFwd->BackColor = (((int)as & (int)SySal::StageControl::AxisStatus::ForwardLimitActive) != 0) ? System::Drawing::Color::Red : System::Drawing::Color::Lime;
					txtXMotor->BackColor = (((int)as & (int)SySal::StageControl::AxisStatus::MotorOff) != 0) ? System::Drawing::Color::Red : System::Drawing::Color::Lime;
					as = m_Stage->GetStatus(SySal::StageControl::Axis::Y);
					txtYRev->BackColor = (((int)as & (int)SySal::StageControl::AxisStatus::ReverseLimitActive) != 0) ? System::Drawing::Color::Red : System::Drawing::Color::Lime;
					txtYFwd->BackColor = (((int)as & (int)SySal::StageControl::AxisStatus::ForwardLimitActive) != 0) ? System::Drawing::Color::Red : System::Drawing::Color::Lime;
					txtYMotor->BackColor = (((int)as & (int)SySal::StageControl::AxisStatus::MotorOff) != 0) ? System::Drawing::Color::Red : System::Drawing::Color::Lime;
					as = m_Stage->GetStatus(SySal::StageControl::Axis::Z);
					txtZRev->BackColor = (((int)as & (int)SySal::StageControl::AxisStatus::ReverseLimitActive) != 0) ? System::Drawing::Color::Red : System::Drawing::Color::Lime;
					txtZFwd->BackColor = (((int)as & (int)SySal::StageControl::AxisStatus::ForwardLimitActive) != 0) ? System::Drawing::Color::Red : System::Drawing::Color::Lime;
					txtZMotor->BackColor = (((int)as & (int)SySal::StageControl::AxisStatus::MotorOff) != 0) ? System::Drawing::Color::Red : System::Drawing::Color::Lime;
					int genstat = m_Stage->GetGeneralStatusFlags();
					if (genstat & STATUS_OVERFLOWCC)
						txtGeneral->BackColor = System::Drawing::Color::Red;
					else if (genstat & STATUS_WARNINGCC)
						txtGeneral->BackColor = System::Drawing::Color::Yellow;
					else if (genstat & STATUS_INIT)
						txtGeneral->BackColor = System::Drawing::Color::Purple;
					else if (genstat & STATUS_INITERROR)
						txtGeneral->BackColor = System::Drawing::Color::Black;
					else
						txtGeneral->BackColor = System::Drawing::Color::Lime;
				}
			 }
		System::Void OnRefreshTick(System::Object^  sender, System::EventArgs^  e)
		{
				this->Invoke(gcnew dRefresh(this, &OmsMAXStage::StageMonitor::RefreshView));
		}

private: System::Void OnKeysKeyDown(System::Object^  sender, System::Windows::Forms::KeyEventArgs^  e) 
		 {
			 SySal::StageControl::Axis ax;
			 bool forward;
			 switch (e->KeyCode)
			 {
			 case System::Windows::Forms::Keys::Space:
				 HighSpeed = true;				 
				 txtKeys->BackColor = System::Drawing::Color::Yellow;
				 return;

			 case System::Windows::Forms::Keys::OemMinus:
				 if (m_Stage != nullptr) 
				 {
					 int ll = ((int)m_Stage->LightLevel) - (HighSpeed ? 128 : 4);
					 if (ll < 0) ll = 0;
					 m_Stage->LightLevel = ll;
				 }
				 return;

			 case System::Windows::Forms::Keys::Oemplus:
				 if (m_Stage != nullptr) 
				 {
					 int ll = ((int)m_Stage->LightLevel) + (HighSpeed ? 128 : 4);
					 if (ll < 0) ll = 0;
					 m_Stage->LightLevel = ll;
				 }
				 return;

			 case System::Windows::Forms::Keys::Left: 
				 ax = SySal::StageControl::Axis::X;
				 forward = false;
				 break;

			 case System::Windows::Forms::Keys::Right:
				 ax = SySal::StageControl::Axis::X;
				 forward = true;
				 break;

			 case System::Windows::Forms::Keys::Up: 
				 ax = SySal::StageControl::Axis::Y;
				 forward = true;
				 break;

			 case System::Windows::Forms::Keys::Down:
				 ax = SySal::StageControl::Axis::Y;
				 forward = false;
				 break;

			 case System::Windows::Forms::Keys::PageUp: 
				 ax = SySal::StageControl::Axis::Z;
				 forward = true;
				 break;

			 case System::Windows::Forms::Keys::PageDown:
				 ax = SySal::StageControl::Axis::Z;
				 forward = false;
				 break;

			 default: return;
			 }
			 if (m_Stage != nullptr)
				 m_Stage->ManualMove(ax, HighSpeed, forward);
		 }

private: System::Void OnKeysKeyUp(System::Object^  sender, System::Windows::Forms::KeyEventArgs^  e) 
		 {
			 switch (e->KeyCode)
			 {
			 case System::Windows::Forms::Keys::Space:
				 HighSpeed = false;				 
				 txtKeys->BackColor = System::Drawing::Color::PaleGreen;
				 return;

			 case System::Windows::Forms::Keys::Left: 
			 case System::Windows::Forms::Keys::Right:
				 if (m_Stage != nullptr) m_Stage->Stop(SySal::StageControl::Axis::X);
				 return;

			 case System::Windows::Forms::Keys::Up: 
			 case System::Windows::Forms::Keys::Down:
				 if (m_Stage != nullptr) m_Stage->Stop(SySal::StageControl::Axis::Y);
				 break;

			 case System::Windows::Forms::Keys::PageUp: 
			 case System::Windows::Forms::Keys::PageDown:
				 if (m_Stage != nullptr) m_Stage->Stop(SySal::StageControl::Axis::Z);
				 break;
			 }
		 }
private: System::Void OnLeaveKeys(System::Object^  sender, System::EventArgs^  e) 
		 {
			 if (m_Stage != nullptr)
			 {
				 m_Stage->Stop(SySal::StageControl::Axis::X);
				 m_Stage->Stop(SySal::StageControl::Axis::Y);
				 m_Stage->Stop(SySal::StageControl::Axis::Z);
			 }
			 txtKeys->BackColor = System::Drawing::Color::Gray;
		 }
private: System::Void OnEnterKeys(System::Object^  sender, System::EventArgs^  e) 
		 {
			 txtKeys->BackColor = System::Drawing::Color::PaleGreen;			 
			 HighSpeed = false;
		 }
private: System::Void btnExtras_Click(System::Object^  sender, System::EventArgs^  e) {
			 this->Height = (this->Height == HeightPartial) ? HeightFull : HeightPartial;
		 }
private: System::Void btnReset_Click(System::Object^  sender, System::EventArgs^  e) {
			 if (m_Stage == nullptr) return;
			 if (MessageBox::Show("Are you sure you want to reset the coordinates?", "Confirmation required", MessageBoxButtons::YesNo, MessageBoxIcon::Warning) == ::DialogResult::Yes)
			 {
				 m_Stage->Reset(SySal::StageControl::Axis::X);
				 m_Stage->Reset(SySal::StageControl::Axis::Y);
				 m_Stage->Reset(SySal::StageControl::Axis::Z);
			 }
		 }
private: int HeightFull;
private: int HeightPartial;
private: System::Void OnLoad(System::Object^  sender, System::EventArgs^  e) {
			 int HeightDelta = this->btnHome->Bottom - this->btnExtras->Bottom;
			 HeightFull = this->Height;
			 HeightPartial = HeightFull - HeightDelta;
			 this->Height = HeightPartial;
		 }
private: System::Void btnGo_Click(System::Object^  sender, System::EventArgs^  e) {
			 if (m_Stage == nullptr) return;
			 SySal::StageControl::Axis ax;
			 double speed = 0.0;
			 double acc = 0.0;
			 switch (cmbAxis->SelectedIndex)
			 {
				case 0: ax = SySal::StageControl::Axis::X; speed = m_StageSettings->XYHighSpeed; acc = m_StageSettings->XYAcceleration; break;
				case 1: ax = SySal::StageControl::Axis::Y; speed = m_StageSettings->XYHighSpeed; acc = m_StageSettings->XYAcceleration; break;
				case 2: ax = SySal::StageControl::Axis::Z; speed = m_StageSettings->ZHighSpeed; acc = m_StageSettings->ZAcceleration; break;
				return;
			 }
			 try
			 {
				 double p = double::Parse(txtGoPos->Text, System::Globalization::CultureInfo::InvariantCulture);
				 m_Stage->PosMove(ax, p, speed, acc, acc);
			 }
			 catch (System::Exception ^exc)
			 {
				 return;
			 }
		 }
private: System::Void btnHome_Click(System::Object^  sender, System::EventArgs^  e) {
			 if (m_Stage == nullptr) return;
			 if (String::Compare(btnHome->Text, gcnew System::String("Home")) != 0) m_Stage->vStopHome();
			 else if (MessageBox::Show("Are you sure you want to home the stage?", "Confirmation required", MessageBoxButtons::YesNo, MessageBoxIcon::Warning) == ::DialogResult::Yes)
			 {
				 System::Threading::Thread ^m_home = gcnew System::Threading::Thread(gcnew System::Threading::ThreadStart(m_Stage, &SySal::StageControl::OmsMAXStage::vHome));
				 m_home->Start();
			 }
		 }

public: delegate void dSetText(System::String ^str);

public: delegate void dSetBool(bool b);

public: virtual System::Void SetHomingText(System::String ^str)  {
			 if (txtHoming->InvokeRequired) txtHoming->Invoke(gcnew dSetText(this, &OmsMAXStage::StageMonitor::SetHomingText), str);
			 else txtHoming->Text = str;
		 }

public: virtual System::Void SetHomingButton(bool ishoming) {
			 if (btnHome->InvokeRequired) btnHome->Invoke(gcnew dSetBool(this, &OmsMAXStage::StageMonitor::SetHomingButton), ishoming);
			 else btnHome->Text = gcnew System::String(ishoming ? "Abort home" : "Home");
		}

private: System::String ^_sendstr;

private: System::Void btnSENDSTR_Click(System::Object^  sender, System::EventArgs^  e) {			 
			 if (_sendstr != nullptr) m_Stage->SendStr(_sendstr);
		 }
private: System::Void OnSENDSTRChanged(System::Object^  sender, System::EventArgs^  e) {
			 _sendstr = txtSENDSTR->Text;
		 }

private: delegate void dVoid();

private: System::Void ForceHomed()
{
		if (MessageBox::Show("Are you sure you want to force the stage to homed?", "Confirmation required", MessageBoxButtons::OKCancel, MessageBoxIcon::Warning) == System::Windows::Forms::DialogResult::OK)
		{
			m_Stage->vForceHome();
		}
}

private: System::Void btnForceHomed_Click(System::Object^  sender, System::EventArgs^  e) {
			 btnForceHomed->Invoke(gcnew dVoid(this, &OmsMAXStage::StageMonitor::ForceHomed));
		 }

private: System::String ^AxisOpToText(SySal::StageControl::SyncAxisOp *paxop)
		 {
			 switch (paxop->Cmd.OpCode)
			 {
				case SySal::StageControl::AxisOpCode::Reset: return gcnew System::String("R"); break;
				case SySal::StageControl::AxisOpCode::Stop: return gcnew System::String("S"); break;
				case SySal::StageControl::AxisOpCode::PosMove: return gcnew System::String("P ") + paxop->Cmd.Position[0].ToString("F1", System::Globalization::CultureInfo::InvariantCulture); break;
				case SySal::StageControl::AxisOpCode::SpeedMove: return gcnew System::String("V ") + paxop->Cmd.Speed[0].ToString("F1", System::Globalization::CultureInfo::InvariantCulture); break;
				case SySal::StageControl::AxisOpCode::MultiPosMove: return gcnew System::String("M") + paxop->Cmd.StatusNumber.ToString() + gcnew System::String(" ") + paxop->Cmd.Position[0].ToString("F1", System::Globalization::CultureInfo::InvariantCulture) + gcnew System::String(" ") + paxop->Cmd.Position[1].ToString("F1", System::Globalization::CultureInfo::InvariantCulture); break;
				case SySal::StageControl::AxisOpCode::Null: return gcnew System::String("N"); break;
				default: return gcnew System::String("?"); break;
			 }
		 }
private: System::Void btnDEBUGCHECK_Click(System::Object^  sender, System::EventArgs^  e) {
			SySal::StageControl::SyncAxisOp *axop = m_Stage->vLastMoveCmd();
			txtLastCmdX->Text = AxisOpToText(&axop[0]);
			txtLastCmdY->Text = AxisOpToText(&axop[1]);
			txtLastCmdZ->Text = AxisOpToText(&axop[2]);
		 }
private: System::Void btnSendReceive_Click(System::Object^  sender, System::EventArgs^  e) {
			 if (_sendstr != nullptr) txtSENDSTR->Text = m_Stage->SendRecvStr(_sendstr);
		 }
private: System::Void btnTest_Click(System::Object^  sender, System::EventArgs^  e) {
			 long timesourcebase = m_Stage->GetTimeSource()->ElapsedMilliseconds;
			 m_Stage->StartRecording(1.0,10000.0);
			 cli::array<SySal::StageControl::TrajectorySample> ^samples = m_Stage->Trajectory;
			 FILE *f = fopen("c:\\sysal.net\\logs\\t.txt", "wt");
			 fprintf(f, "ID\tT\tX\tY\tZ");
			 if (f == 0) return;
			 int i;
			 for (i = 0; i < samples->Length; i++)
				 fprintf(f, "\n%d\t%f\t%f\t%f\t%f", i, samples[i].TimeMS - timesourcebase, samples[i].Position.X, samples[i].Position.Y, samples[i].Position.Z);
			 fclose(f);
		 }
};
}
