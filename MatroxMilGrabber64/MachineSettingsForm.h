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
	namespace Imaging
	{	

	/// <summary>
	/// Summary for MachineSettingsForm
	/// </summary>
	public ref class MachineSettingsForm : public System::Windows::Forms::Form
	{
	public:

		MatroxMilGrabberSettings ^MC;
		MachineSettingsForm(void)
		{
			InitializeComponent();
			//
			//TODO: Add the constructor code here
			//
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
	private: System::Windows::Forms::TextBox^  txtFrameDelayMS;
	private: System::Windows::Forms::Button^  btnOK;
	private: System::Windows::Forms::Button^  btnCancel;

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
			this->txtFrameDelayMS = (gcnew System::Windows::Forms::TextBox());
			this->btnOK = (gcnew System::Windows::Forms::Button());
			this->btnCancel = (gcnew System::Windows::Forms::Button());
			this->SuspendLayout();
			// 
			// label1
			// 
			this->label1->AutoSize = true;
			this->label1->Location = System::Drawing::Point(12, 9);
			this->label1->Name = L"label1";
			this->label1->Size = System::Drawing::Size(93, 13);
			this->label1->TabIndex = 0;
			this->label1->Text = L"Frame Delay in ms";
			// 
			// txtFrameDelayMS
			// 
			this->txtFrameDelayMS->Location = System::Drawing::Point(138, 6);
			this->txtFrameDelayMS->Name = L"txtFrameDelayMS";
			this->txtFrameDelayMS->Size = System::Drawing::Size(50, 20);
			this->txtFrameDelayMS->TabIndex = 1;
			this->txtFrameDelayMS->Leave += gcnew System::EventHandler(this, &MachineSettingsForm::OnFrameDelayMSLeave);
			// 
			// btnOK
			// 
			this->btnOK->Location = System::Drawing::Point(15, 38);
			this->btnOK->Name = L"btnOK";
			this->btnOK->Size = System::Drawing::Size(74, 28);
			this->btnOK->TabIndex = 2;
			this->btnOK->Text = L"OK";
			this->btnOK->UseVisualStyleBackColor = true;
			this->btnOK->Click += gcnew System::EventHandler(this, &MachineSettingsForm::btnOK_Click);
			// 
			// btnCancel
			// 
			this->btnCancel->Location = System::Drawing::Point(114, 38);
			this->btnCancel->Name = L"btnCancel";
			this->btnCancel->Size = System::Drawing::Size(74, 28);
			this->btnCancel->TabIndex = 3;
			this->btnCancel->Text = L"Cancel";
			this->btnCancel->UseVisualStyleBackColor = true;
			this->btnCancel->Click += gcnew System::EventHandler(this, &MachineSettingsForm::btnCancel_Click);
			// 
			// MachineSettingsForm
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(212, 76);
			this->Controls->Add(this->btnCancel);
			this->Controls->Add(this->btnOK);
			this->Controls->Add(this->txtFrameDelayMS);
			this->Controls->Add(this->label1);
			this->FormBorderStyle = System::Windows::Forms::FormBorderStyle::FixedToolWindow;
			this->Name = L"MachineSettingsForm";
			this->Text = L"Matrox Mil Grabber Settings";
			this->Load += gcnew System::EventHandler(this, &MachineSettingsForm::OnLoad);
			this->ResumeLayout(false);
			this->PerformLayout();

		}
#pragma endregion
	private: System::Void OnLoad(System::Object^  sender, System::EventArgs^  e) {
				 this->txtFrameDelayMS->Text = MC->FrameDelayMS.ToString("F4", System::Globalization::CultureInfo::InvariantCulture);
			 }
#define VALIDATEPOSITIVEDOUBLE(ctl,var,fmt) try { double v = Convert::ToDouble(ctl->Text, System::Globalization::CultureInfo::InvariantCulture); if (v <= 0.0) throw gcnew System::Exception("A positive number is required."); var = v; } catch (Exception ^) { ctl->Text = var.ToString(fmt, System::Globalization::CultureInfo::InvariantCulture); ctl->Focus(); }
	private: System::Void OnFrameDelayMSLeave(System::Object^  sender, System::EventArgs^  e) {
				 VALIDATEPOSITIVEDOUBLE(txtFrameDelayMS, MC->FrameDelayMS, "F4")
			 }
private: System::Void btnOK_Click(System::Object^  sender, System::EventArgs^  e) {
			 this->DialogResult = System::Windows::Forms::DialogResult::OK;
			 this->Close();
		 }
private: System::Void btnCancel_Click(System::Object^  sender, System::EventArgs^  e) {
			 this->DialogResult = System::Windows::Forms::DialogResult::Cancel;
			 this->Close();
		 }
	};
}
}
