using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;
using System.Runtime.Remoting;
using System.Runtime.Remoting.Channels;
using System.Runtime.Remoting.Channels.Tcp;
using SySal;
using SySal.DAQSystem.Drivers;

namespace SySal.DAQSystem.Drivers.SimpleScanback3Driver
{
	/// <summary>
	/// Interrupt Form.
	/// </summary>
	internal class frmEasyInterrupt : System.Windows.Forms.Form
	{
		private System.Windows.Forms.Label label1;
		private System.Windows.Forms.ComboBox comboBatchManager;
		private System.Windows.Forms.Label label2;
		private System.Windows.Forms.ComboBox comboProcOpId;
		private System.Windows.Forms.Label label3;
		private System.Windows.Forms.TextBox textOPERAUsername;
		private System.Windows.Forms.TextBox textOPERAPwd;
		private System.Windows.Forms.Label label4;
        private System.Windows.Forms.Button buttonCancel;
		private System.Windows.Forms.RadioButton radioMarkPlateDamagedN;
		private System.Windows.Forms.TextBox textMarkPlateDamagedN;
		private System.Windows.Forms.Label label5;
		private System.Windows.Forms.TextBox textMarkPlateDamagedCode;
		private System.Windows.Forms.Button buttonMarkPlateDamagedSend;
		private System.Windows.Forms.Label label6;
		private System.Windows.Forms.TextBox textGoBackToPlateN;
		private System.Windows.Forms.CheckBox checkGoBackToPlateCalibrations;
		private System.Windows.Forms.Button buttonGoBackToPlateSend;
        private System.Windows.Forms.RadioButton radioMarkPlateDamagedCurrent;
		private System.Windows.Forms.Button buttonPredictionsSend;
		private System.Windows.Forms.Label label7;
		private System.Windows.Forms.ComboBox comboPredSeparator;
		private System.Windows.Forms.Button buttonPredFileLoad;
		private System.Windows.Forms.TextBox textPredFileCount;
        private System.Windows.Forms.OpenFileDialog openPredFileDialog;
		private System.Windows.Forms.Button buttonIgnoreCalibrationsSend;
		private System.Windows.Forms.TextBox textIgnoreCalibrations;
        private System.Windows.Forms.Label label8;
		private System.Windows.Forms.Button buttonForceCalibrationsSend;
		private System.Windows.Forms.TextBox textForceCalibrations;
		private System.Windows.Forms.Label label9;
        private RadioButton radioClose;
        private Button buttonExecCommandSend;
        private RadioButton radioContinue;
        private TextBox textSetPathCount;
        private Button buttonPathFileLoad;
        private ComboBox comboPathSeparator;
        private Label label10;
        private Button buttonSetPathSend;
        private TabControl tabControl1;
        private TabPage tabPage1;
        private TabPage tabPage2;
        private TabPage tabPage3;
        private TabPage tabPage4;
        private TabPage tabPage5;
        private TabPage tabPage6;
        private TabPage tabPage7;
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;

		public frmEasyInterrupt()
		{
			//
			// Required for Windows Form Designer support
			//
			InitializeComponent();

			//
			// TODO: Add any constructor code after InitializeComponent call
			//
		}

		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		protected override void Dispose( bool disposing )
		{
			if( disposing )
			{
				if(components != null)
				{
					components.Dispose();
				}
			}
			base.Dispose( disposing );
		}

		#region Windows Form Designer generated code
		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		private void InitializeComponent()
		{
            this.label1 = new System.Windows.Forms.Label();
            this.comboBatchManager = new System.Windows.Forms.ComboBox();
            this.label2 = new System.Windows.Forms.Label();
            this.comboProcOpId = new System.Windows.Forms.ComboBox();
            this.label3 = new System.Windows.Forms.Label();
            this.textOPERAUsername = new System.Windows.Forms.TextBox();
            this.textOPERAPwd = new System.Windows.Forms.TextBox();
            this.label4 = new System.Windows.Forms.Label();
            this.buttonCancel = new System.Windows.Forms.Button();
            this.buttonMarkPlateDamagedSend = new System.Windows.Forms.Button();
            this.buttonGoBackToPlateSend = new System.Windows.Forms.Button();
            this.checkGoBackToPlateCalibrations = new System.Windows.Forms.CheckBox();
            this.textGoBackToPlateN = new System.Windows.Forms.TextBox();
            this.label6 = new System.Windows.Forms.Label();
            this.textMarkPlateDamagedCode = new System.Windows.Forms.TextBox();
            this.label5 = new System.Windows.Forms.Label();
            this.textMarkPlateDamagedN = new System.Windows.Forms.TextBox();
            this.radioMarkPlateDamagedN = new System.Windows.Forms.RadioButton();
            this.radioMarkPlateDamagedCurrent = new System.Windows.Forms.RadioButton();
            this.textPredFileCount = new System.Windows.Forms.TextBox();
            this.buttonPredFileLoad = new System.Windows.Forms.Button();
            this.comboPredSeparator = new System.Windows.Forms.ComboBox();
            this.label7 = new System.Windows.Forms.Label();
            this.buttonPredictionsSend = new System.Windows.Forms.Button();
            this.openPredFileDialog = new System.Windows.Forms.OpenFileDialog();
            this.buttonIgnoreCalibrationsSend = new System.Windows.Forms.Button();
            this.textIgnoreCalibrations = new System.Windows.Forms.TextBox();
            this.label8 = new System.Windows.Forms.Label();
            this.buttonForceCalibrationsSend = new System.Windows.Forms.Button();
            this.textForceCalibrations = new System.Windows.Forms.TextBox();
            this.label9 = new System.Windows.Forms.Label();
            this.radioClose = new System.Windows.Forms.RadioButton();
            this.buttonExecCommandSend = new System.Windows.Forms.Button();
            this.radioContinue = new System.Windows.Forms.RadioButton();
            this.textSetPathCount = new System.Windows.Forms.TextBox();
            this.buttonPathFileLoad = new System.Windows.Forms.Button();
            this.comboPathSeparator = new System.Windows.Forms.ComboBox();
            this.label10 = new System.Windows.Forms.Label();
            this.buttonSetPathSend = new System.Windows.Forms.Button();
            this.tabControl1 = new System.Windows.Forms.TabControl();
            this.tabPage1 = new System.Windows.Forms.TabPage();
            this.tabPage2 = new System.Windows.Forms.TabPage();
            this.tabPage3 = new System.Windows.Forms.TabPage();
            this.tabPage4 = new System.Windows.Forms.TabPage();
            this.tabPage5 = new System.Windows.Forms.TabPage();
            this.tabPage6 = new System.Windows.Forms.TabPage();
            this.tabPage7 = new System.Windows.Forms.TabPage();
            this.tabControl1.SuspendLayout();
            this.tabPage1.SuspendLayout();
            this.tabPage2.SuspendLayout();
            this.tabPage3.SuspendLayout();
            this.tabPage4.SuspendLayout();
            this.tabPage5.SuspendLayout();
            this.tabPage6.SuspendLayout();
            this.tabPage7.SuspendLayout();
            this.SuspendLayout();
            // 
            // label1
            // 
            this.label1.Location = new System.Drawing.Point(8, 8);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(88, 24);
            this.label1.TabIndex = 0;
            this.label1.Text = "Batch Manager";
            this.label1.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // comboBatchManager
            // 
            this.comboBatchManager.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.comboBatchManager.Location = new System.Drawing.Point(136, 8);
            this.comboBatchManager.Name = "comboBatchManager";
            this.comboBatchManager.Size = new System.Drawing.Size(264, 21);
            this.comboBatchManager.TabIndex = 1;
            this.comboBatchManager.SelectionChangeCommitted += new System.EventHandler(this.OnBatchManagerSelected);
            this.comboBatchManager.SelectedIndexChanged += new System.EventHandler(this.OnBatchManagerSelected);
            // 
            // label2
            // 
            this.label2.Location = new System.Drawing.Point(8, 32);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(120, 24);
            this.label2.TabIndex = 2;
            this.label2.Text = "Process Operation";
            this.label2.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // comboProcOpId
            // 
            this.comboProcOpId.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.comboProcOpId.Location = new System.Drawing.Point(136, 32);
            this.comboProcOpId.Name = "comboProcOpId";
            this.comboProcOpId.Size = new System.Drawing.Size(264, 21);
            this.comboProcOpId.TabIndex = 3;
            // 
            // label3
            // 
            this.label3.Location = new System.Drawing.Point(5, 285);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(104, 24);
            this.label3.TabIndex = 5;
            this.label3.Text = "OPERA Username";
            this.label3.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // textOPERAUsername
            // 
            this.textOPERAUsername.Location = new System.Drawing.Point(109, 285);
            this.textOPERAUsername.Name = "textOPERAUsername";
            this.textOPERAUsername.Size = new System.Drawing.Size(186, 20);
            this.textOPERAUsername.TabIndex = 6;
            // 
            // textOPERAPwd
            // 
            this.textOPERAPwd.Location = new System.Drawing.Point(109, 309);
            this.textOPERAPwd.Name = "textOPERAPwd";
            this.textOPERAPwd.PasswordChar = '*';
            this.textOPERAPwd.Size = new System.Drawing.Size(186, 20);
            this.textOPERAPwd.TabIndex = 8;
            // 
            // label4
            // 
            this.label4.Location = new System.Drawing.Point(5, 309);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(104, 24);
            this.label4.TabIndex = 7;
            this.label4.Text = "OPERA Password";
            this.label4.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // buttonCancel
            // 
            this.buttonCancel.Location = new System.Drawing.Point(5, 333);
            this.buttonCancel.Name = "buttonCancel";
            this.buttonCancel.Size = new System.Drawing.Size(64, 24);
            this.buttonCancel.TabIndex = 9;
            this.buttonCancel.Text = "Cancel";
            this.buttonCancel.Click += new System.EventHandler(this.buttonCancel_Click);
            // 
            // buttonMarkPlateDamagedSend
            // 
            this.buttonMarkPlateDamagedSend.Location = new System.Drawing.Point(124, 153);
            this.buttonMarkPlateDamagedSend.Name = "buttonMarkPlateDamagedSend";
            this.buttonMarkPlateDamagedSend.Size = new System.Drawing.Size(64, 24);
            this.buttonMarkPlateDamagedSend.TabIndex = 10;
            this.buttonMarkPlateDamagedSend.Text = "Send";
            this.buttonMarkPlateDamagedSend.Click += new System.EventHandler(this.buttonMarkPlateDamagedSend_Click);
            // 
            // buttonGoBackToPlateSend
            // 
            this.buttonGoBackToPlateSend.Location = new System.Drawing.Point(124, 153);
            this.buttonGoBackToPlateSend.Name = "buttonGoBackToPlateSend";
            this.buttonGoBackToPlateSend.Size = new System.Drawing.Size(64, 24);
            this.buttonGoBackToPlateSend.TabIndex = 11;
            this.buttonGoBackToPlateSend.Text = "Send";
            this.buttonGoBackToPlateSend.Click += new System.EventHandler(this.buttonGoBackToPlateSend_Click);
            // 
            // checkGoBackToPlateCalibrations
            // 
            this.checkGoBackToPlateCalibrations.Checked = true;
            this.checkGoBackToPlateCalibrations.CheckState = System.Windows.Forms.CheckState.Checked;
            this.checkGoBackToPlateCalibrations.Location = new System.Drawing.Point(12, 105);
            this.checkGoBackToPlateCalibrations.Name = "checkGoBackToPlateCalibrations";
            this.checkGoBackToPlateCalibrations.Size = new System.Drawing.Size(176, 40);
            this.checkGoBackToPlateCalibrations.TabIndex = 4;
            this.checkGoBackToPlateCalibrations.Text = "Cancel calibrations done within this operation";
            // 
            // textGoBackToPlateN
            // 
            this.textGoBackToPlateN.Location = new System.Drawing.Point(80, 81);
            this.textGoBackToPlateN.Name = "textGoBackToPlateN";
            this.textGoBackToPlateN.Size = new System.Drawing.Size(40, 20);
            this.textGoBackToPlateN.TabIndex = 3;
            this.textGoBackToPlateN.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label6
            // 
            this.label6.Location = new System.Drawing.Point(12, 81);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(64, 24);
            this.label6.TabIndex = 0;
            this.label6.Text = "Plate #";
            this.label6.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // textMarkPlateDamagedCode
            // 
            this.textMarkPlateDamagedCode.Location = new System.Drawing.Point(148, 129);
            this.textMarkPlateDamagedCode.Name = "textMarkPlateDamagedCode";
            this.textMarkPlateDamagedCode.Size = new System.Drawing.Size(40, 20);
            this.textMarkPlateDamagedCode.TabIndex = 4;
            this.textMarkPlateDamagedCode.Text = "N";
            this.textMarkPlateDamagedCode.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label5
            // 
            this.label5.Location = new System.Drawing.Point(12, 129);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(112, 24);
            this.label5.TabIndex = 3;
            this.label5.Text = "Plate damage code";
            this.label5.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // textMarkPlateDamagedN
            // 
            this.textMarkPlateDamagedN.Location = new System.Drawing.Point(92, 105);
            this.textMarkPlateDamagedN.Name = "textMarkPlateDamagedN";
            this.textMarkPlateDamagedN.Size = new System.Drawing.Size(40, 20);
            this.textMarkPlateDamagedN.TabIndex = 2;
            this.textMarkPlateDamagedN.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // radioMarkPlateDamagedN
            // 
            this.radioMarkPlateDamagedN.Location = new System.Drawing.Point(12, 105);
            this.radioMarkPlateDamagedN.Name = "radioMarkPlateDamagedN";
            this.radioMarkPlateDamagedN.Size = new System.Drawing.Size(80, 24);
            this.radioMarkPlateDamagedN.TabIndex = 1;
            this.radioMarkPlateDamagedN.Text = "Plate #";
            // 
            // radioMarkPlateDamagedCurrent
            // 
            this.radioMarkPlateDamagedCurrent.Checked = true;
            this.radioMarkPlateDamagedCurrent.Location = new System.Drawing.Point(12, 81);
            this.radioMarkPlateDamagedCurrent.Name = "radioMarkPlateDamagedCurrent";
            this.radioMarkPlateDamagedCurrent.Size = new System.Drawing.Size(80, 24);
            this.radioMarkPlateDamagedCurrent.TabIndex = 0;
            this.radioMarkPlateDamagedCurrent.TabStop = true;
            this.radioMarkPlateDamagedCurrent.Text = "Current";
            // 
            // textPredFileCount
            // 
            this.textPredFileCount.Location = new System.Drawing.Point(111, 105);
            this.textPredFileCount.Name = "textPredFileCount";
            this.textPredFileCount.ReadOnly = true;
            this.textPredFileCount.Size = new System.Drawing.Size(64, 20);
            this.textPredFileCount.TabIndex = 14;
            this.textPredFileCount.Text = "0";
            this.textPredFileCount.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // buttonPredFileLoad
            // 
            this.buttonPredFileLoad.Location = new System.Drawing.Point(7, 105);
            this.buttonPredFileLoad.Name = "buttonPredFileLoad";
            this.buttonPredFileLoad.Size = new System.Drawing.Size(96, 24);
            this.buttonPredFileLoad.TabIndex = 13;
            this.buttonPredFileLoad.Text = "Load and count";
            this.buttonPredFileLoad.Click += new System.EventHandler(this.buttonPredFileLoad_Click);
            // 
            // comboPredSeparator
            // 
            this.comboPredSeparator.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.comboPredSeparator.Items.AddRange(new object[] {
            "newline",
            ";",
            ",",
            "."});
            this.comboPredSeparator.Location = new System.Drawing.Point(111, 81);
            this.comboPredSeparator.Name = "comboPredSeparator";
            this.comboPredSeparator.Size = new System.Drawing.Size(64, 21);
            this.comboPredSeparator.TabIndex = 12;
            // 
            // label7
            // 
            this.label7.Location = new System.Drawing.Point(7, 81);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(96, 24);
            this.label7.TabIndex = 11;
            this.label7.Text = "6-tuple separator";
            this.label7.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // buttonPredictionsSend
            // 
            this.buttonPredictionsSend.Location = new System.Drawing.Point(111, 153);
            this.buttonPredictionsSend.Name = "buttonPredictionsSend";
            this.buttonPredictionsSend.Size = new System.Drawing.Size(64, 24);
            this.buttonPredictionsSend.TabIndex = 10;
            this.buttonPredictionsSend.Text = "Send";
            this.buttonPredictionsSend.Click += new System.EventHandler(this.buttonPredictionsSend_Click);
            // 
            // openPredFileDialog
            // 
            this.openPredFileDialog.Filter = "Text files (*.txt)|*.txt|All files (*.*)|*.*";
            this.openPredFileDialog.Title = "Select prediction file";
            // 
            // buttonIgnoreCalibrationsSend
            // 
            this.buttonIgnoreCalibrationsSend.Location = new System.Drawing.Point(206, 153);
            this.buttonIgnoreCalibrationsSend.Name = "buttonIgnoreCalibrationsSend";
            this.buttonIgnoreCalibrationsSend.Size = new System.Drawing.Size(64, 24);
            this.buttonIgnoreCalibrationsSend.TabIndex = 11;
            this.buttonIgnoreCalibrationsSend.Text = "Send";
            this.buttonIgnoreCalibrationsSend.Click += new System.EventHandler(this.buttonIgnoreCalibrationsSend_Click);
            // 
            // textIgnoreCalibrations
            // 
            this.textIgnoreCalibrations.Location = new System.Drawing.Point(74, 81);
            this.textIgnoreCalibrations.Multiline = true;
            this.textIgnoreCalibrations.Name = "textIgnoreCalibrations";
            this.textIgnoreCalibrations.Size = new System.Drawing.Size(196, 64);
            this.textIgnoreCalibrations.TabIndex = 3;
            this.textIgnoreCalibrations.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label8
            // 
            this.label8.Location = new System.Drawing.Point(6, 81);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(64, 48);
            this.label8.TabIndex = 0;
            this.label8.Text = "Plate # (comma separated)";
            this.label8.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // buttonForceCalibrationsSend
            // 
            this.buttonForceCalibrationsSend.Location = new System.Drawing.Point(206, 153);
            this.buttonForceCalibrationsSend.Name = "buttonForceCalibrationsSend";
            this.buttonForceCalibrationsSend.Size = new System.Drawing.Size(64, 24);
            this.buttonForceCalibrationsSend.TabIndex = 11;
            this.buttonForceCalibrationsSend.Text = "Send";
            this.buttonForceCalibrationsSend.Click += new System.EventHandler(this.buttonForceCalibrationsSend_Click);
            // 
            // textForceCalibrations
            // 
            this.textForceCalibrations.Location = new System.Drawing.Point(74, 81);
            this.textForceCalibrations.Multiline = true;
            this.textForceCalibrations.Name = "textForceCalibrations";
            this.textForceCalibrations.Size = new System.Drawing.Size(196, 64);
            this.textForceCalibrations.TabIndex = 3;
            this.textForceCalibrations.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label9
            // 
            this.label9.Location = new System.Drawing.Point(6, 81);
            this.label9.Name = "label9";
            this.label9.Size = new System.Drawing.Size(64, 48);
            this.label9.TabIndex = 0;
            this.label9.Text = "Plate # (comma separated)";
            this.label9.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // radioClose
            // 
            this.radioClose.Location = new System.Drawing.Point(7, 78);
            this.radioClose.Name = "radioClose";
            this.radioClose.Size = new System.Drawing.Size(80, 24);
            this.radioClose.TabIndex = 1;
            this.radioClose.Text = "Close";
            // 
            // buttonExecCommandSend
            // 
            this.buttonExecCommandSend.Location = new System.Drawing.Point(82, 54);
            this.buttonExecCommandSend.Name = "buttonExecCommandSend";
            this.buttonExecCommandSend.Size = new System.Drawing.Size(64, 24);
            this.buttonExecCommandSend.TabIndex = 10;
            this.buttonExecCommandSend.Text = "Send";
            this.buttonExecCommandSend.Click += new System.EventHandler(this.buttonExecCommandSend_Click);
            // 
            // radioContinue
            // 
            this.radioContinue.Checked = true;
            this.radioContinue.Location = new System.Drawing.Point(7, 54);
            this.radioContinue.Name = "radioContinue";
            this.radioContinue.Size = new System.Drawing.Size(80, 24);
            this.radioContinue.TabIndex = 0;
            this.radioContinue.TabStop = true;
            this.radioContinue.Text = "Continue";
            // 
            // textSetPathCount
            // 
            this.textSetPathCount.Location = new System.Drawing.Point(111, 105);
            this.textSetPathCount.Name = "textSetPathCount";
            this.textSetPathCount.ReadOnly = true;
            this.textSetPathCount.Size = new System.Drawing.Size(64, 20);
            this.textSetPathCount.TabIndex = 14;
            this.textSetPathCount.Text = "0";
            this.textSetPathCount.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // buttonPathFileLoad
            // 
            this.buttonPathFileLoad.Location = new System.Drawing.Point(7, 105);
            this.buttonPathFileLoad.Name = "buttonPathFileLoad";
            this.buttonPathFileLoad.Size = new System.Drawing.Size(96, 24);
            this.buttonPathFileLoad.TabIndex = 13;
            this.buttonPathFileLoad.Text = "Load and count";
            this.buttonPathFileLoad.Click += new System.EventHandler(this.buttonPathFileLoad_Click);
            // 
            // comboPathSeparator
            // 
            this.comboPathSeparator.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.comboPathSeparator.Items.AddRange(new object[] {
            "newline",
            ";",
            ",",
            "."});
            this.comboPathSeparator.Location = new System.Drawing.Point(111, 81);
            this.comboPathSeparator.Name = "comboPathSeparator";
            this.comboPathSeparator.Size = new System.Drawing.Size(64, 21);
            this.comboPathSeparator.TabIndex = 12;
            // 
            // label10
            // 
            this.label10.Location = new System.Drawing.Point(7, 81);
            this.label10.Name = "label10";
            this.label10.Size = new System.Drawing.Size(96, 24);
            this.label10.TabIndex = 11;
            this.label10.Text = "5-tuple separator";
            this.label10.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // buttonSetPathSend
            // 
            this.buttonSetPathSend.Location = new System.Drawing.Point(111, 153);
            this.buttonSetPathSend.Name = "buttonSetPathSend";
            this.buttonSetPathSend.Size = new System.Drawing.Size(64, 24);
            this.buttonSetPathSend.TabIndex = 10;
            this.buttonSetPathSend.Text = "Send";
            this.buttonSetPathSend.Click += new System.EventHandler(this.buttonSetPathSend_Click);
            // 
            // tabControl1
            // 
            this.tabControl1.Controls.Add(this.tabPage1);
            this.tabControl1.Controls.Add(this.tabPage2);
            this.tabControl1.Controls.Add(this.tabPage3);
            this.tabControl1.Controls.Add(this.tabPage4);
            this.tabControl1.Controls.Add(this.tabPage5);
            this.tabControl1.Controls.Add(this.tabPage6);
            this.tabControl1.Controls.Add(this.tabPage7);
            this.tabControl1.Location = new System.Drawing.Point(8, 59);
            this.tabControl1.Name = "tabControl1";
            this.tabControl1.SelectedIndex = 0;
            this.tabControl1.Size = new System.Drawing.Size(392, 213);
            this.tabControl1.TabIndex = 18;
            // 
            // tabPage1
            // 
            this.tabPage1.BackColor = System.Drawing.SystemColors.Control;
            this.tabPage1.Controls.Add(this.textMarkPlateDamagedCode);
            this.tabPage1.Controls.Add(this.radioMarkPlateDamagedCurrent);
            this.tabPage1.Controls.Add(this.label5);
            this.tabPage1.Controls.Add(this.buttonMarkPlateDamagedSend);
            this.tabPage1.Controls.Add(this.textMarkPlateDamagedN);
            this.tabPage1.Controls.Add(this.radioMarkPlateDamagedN);
            this.tabPage1.Location = new System.Drawing.Point(4, 22);
            this.tabPage1.Name = "tabPage1";
            this.tabPage1.Padding = new System.Windows.Forms.Padding(3);
            this.tabPage1.Size = new System.Drawing.Size(384, 187);
            this.tabPage1.TabIndex = 0;
            this.tabPage1.Text = "Plate Damaged";
            // 
            // tabPage2
            // 
            this.tabPage2.BackColor = System.Drawing.SystemColors.Control;
            this.tabPage2.Controls.Add(this.buttonGoBackToPlateSend);
            this.tabPage2.Controls.Add(this.label6);
            this.tabPage2.Controls.Add(this.checkGoBackToPlateCalibrations);
            this.tabPage2.Controls.Add(this.textGoBackToPlateN);
            this.tabPage2.Location = new System.Drawing.Point(4, 22);
            this.tabPage2.Name = "tabPage2";
            this.tabPage2.Padding = new System.Windows.Forms.Padding(3);
            this.tabPage2.Size = new System.Drawing.Size(384, 187);
            this.tabPage2.TabIndex = 1;
            this.tabPage2.Text = "Go back to plate";
            // 
            // tabPage3
            // 
            this.tabPage3.BackColor = System.Drawing.SystemColors.Control;
            this.tabPage3.Controls.Add(this.textPredFileCount);
            this.tabPage3.Controls.Add(this.label7);
            this.tabPage3.Controls.Add(this.buttonPredFileLoad);
            this.tabPage3.Controls.Add(this.buttonPredictionsSend);
            this.tabPage3.Controls.Add(this.comboPredSeparator);
            this.tabPage3.Location = new System.Drawing.Point(4, 22);
            this.tabPage3.Name = "tabPage3";
            this.tabPage3.Size = new System.Drawing.Size(384, 187);
            this.tabPage3.TabIndex = 2;
            this.tabPage3.Text = "Predictions";
            // 
            // tabPage4
            // 
            this.tabPage4.BackColor = System.Drawing.SystemColors.Control;
            this.tabPage4.Controls.Add(this.textSetPathCount);
            this.tabPage4.Controls.Add(this.label10);
            this.tabPage4.Controls.Add(this.buttonPathFileLoad);
            this.tabPage4.Controls.Add(this.buttonSetPathSend);
            this.tabPage4.Controls.Add(this.comboPathSeparator);
            this.tabPage4.Location = new System.Drawing.Point(4, 22);
            this.tabPage4.Name = "tabPage4";
            this.tabPage4.Size = new System.Drawing.Size(384, 187);
            this.tabPage4.TabIndex = 3;
            this.tabPage4.Text = "Set Paths";
            // 
            // tabPage5
            // 
            this.tabPage5.BackColor = System.Drawing.SystemColors.Control;
            this.tabPage5.Controls.Add(this.buttonIgnoreCalibrationsSend);
            this.tabPage5.Controls.Add(this.label8);
            this.tabPage5.Controls.Add(this.textIgnoreCalibrations);
            this.tabPage5.Location = new System.Drawing.Point(4, 22);
            this.tabPage5.Name = "tabPage5";
            this.tabPage5.Size = new System.Drawing.Size(384, 187);
            this.tabPage5.TabIndex = 4;
            this.tabPage5.Text = "Ignore Calibrations";
            // 
            // tabPage6
            // 
            this.tabPage6.BackColor = System.Drawing.SystemColors.Control;
            this.tabPage6.Controls.Add(this.buttonForceCalibrationsSend);
            this.tabPage6.Controls.Add(this.label9);
            this.tabPage6.Controls.Add(this.textForceCalibrations);
            this.tabPage6.Location = new System.Drawing.Point(4, 22);
            this.tabPage6.Name = "tabPage6";
            this.tabPage6.Size = new System.Drawing.Size(384, 187);
            this.tabPage6.TabIndex = 5;
            this.tabPage6.Text = "Force Calibrations";
            // 
            // tabPage7
            // 
            this.tabPage7.BackColor = System.Drawing.SystemColors.Control;
            this.tabPage7.Controls.Add(this.radioClose);
            this.tabPage7.Controls.Add(this.buttonExecCommandSend);
            this.tabPage7.Controls.Add(this.radioContinue);
            this.tabPage7.Location = new System.Drawing.Point(4, 22);
            this.tabPage7.Name = "tabPage7";
            this.tabPage7.Size = new System.Drawing.Size(384, 187);
            this.tabPage7.TabIndex = 6;
            this.tabPage7.Text = "Execution";
            // 
            // frmEasyInterrupt
            // 
            this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
            this.ClientSize = new System.Drawing.Size(407, 363);
            this.Controls.Add(this.tabControl1);
            this.Controls.Add(this.buttonCancel);
            this.Controls.Add(this.textOPERAPwd);
            this.Controls.Add(this.textOPERAUsername);
            this.Controls.Add(this.label4);
            this.Controls.Add(this.label3);
            this.Controls.Add(this.comboProcOpId);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.comboBatchManager);
            this.Controls.Add(this.label1);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
            this.Name = "frmEasyInterrupt";
            this.Text = "Easy Interrupt for SimpleScanback3Driver";
            this.Load += new System.EventHandler(this.OnLoad);
            this.tabControl1.ResumeLayout(false);
            this.tabPage1.ResumeLayout(false);
            this.tabPage1.PerformLayout();
            this.tabPage2.ResumeLayout(false);
            this.tabPage2.PerformLayout();
            this.tabPage3.ResumeLayout(false);
            this.tabPage3.PerformLayout();
            this.tabPage4.ResumeLayout(false);
            this.tabPage4.PerformLayout();
            this.tabPage5.ResumeLayout(false);
            this.tabPage5.PerformLayout();
            this.tabPage6.ResumeLayout(false);
            this.tabPage6.PerformLayout();
            this.tabPage7.ResumeLayout(false);
            this.ResumeLayout(false);
            this.PerformLayout();

		}
		#endregion

		SySal.OperaDb.OperaDbConnection Conn = null;
		SySal.DAQSystem.BatchManager BM = null;

		private void OnLoad(object sender, System.EventArgs e)
		{
			SySal.OperaDb.OperaDbCredentials cred = SySal.OperaDb.OperaDbCredentials.CreateFromRecord();
			System.Runtime.Remoting.Channels.ChannelServices.RegisterChannel(new TcpChannel());
			Conn = new SySal.OperaDb.OperaDbConnection(cred.DBServer, cred.DBUserName, cred.DBPassword);
			Conn.Open();
			System.Data.DataSet ds = new System.Data.DataSet();
			new SySal.OperaDb.OperaDbDataAdapter("select name from tb_machines where id_site = (select to_number(value) from OPERA.LZ_SITEVARS where name = 'ID_SITE') and isbatchserver = 1", Conn, null).Fill(ds);
			foreach (System.Data.DataRow dr in ds.Tables[0].Rows)
				comboBatchManager.Items.Add(dr[0].ToString());
			textOPERAUsername.Text = cred.OPERAUserName;
			textOPERAPwd.Text = cred.OPERAPassword;
			comboPredSeparator.SelectedIndex = 0;
            new SySal.OperaDb.OperaDbCommand("alter session set nls_comp='LINGUISTIC'", Conn).ExecuteNonQuery();
            new SySal.OperaDb.OperaDbCommand("alter session set NLS_SORT='BINARY_CI'", Conn).ExecuteNonQuery();
		}

		private void OnBatchManagerSelected(object sender, System.EventArgs e)
		{
			comboProcOpId.Items.Clear();
			string addr = new SySal.OperaDb.OperaDbCommand("SELECT ADDRESS FROM TB_MACHINES WHERE NAME = '" + comboBatchManager.Text + "'", Conn, null).ExecuteScalar().ToString();
			BM = (SySal.DAQSystem.BatchManager)System.Runtime.Remoting.RemotingServices.Connect(typeof(SySal.DAQSystem.BatchManager), "tcp://" + addr + ":" + ((int)SySal.DAQSystem.OperaPort.BatchServer).ToString() + "/BatchManager.rem");
			long [] ids = BM.Operations;
			if (ids.Length == 0) return;
			string wherestr = ids[0].ToString();
			int i;
			for (i = 1; i < ids.Length; i++)
				wherestr += ", " + ids[i].ToString();
			System.Data.DataSet ds = new System.Data.DataSet();
			new SySal.OperaDb.OperaDbDataAdapter("SELECT TB_PROC_OPERATIONS.ID FROM TB_PROC_OPERATIONS INNER JOIN TB_PROGRAMSETTINGS ON (TB_PROC_OPERATIONS.ID_PROGRAMSETTINGS = TB_PROGRAMSETTINGS.ID AND TB_PROGRAMSETTINGS.EXECUTABLE = 'SimpleScanback3Driver.exe') WHERE TB_PROC_OPERATIONS.ID IN (" + wherestr + ")", Conn, null).Fill(ds);
			foreach (System.Data.DataRow dr in ds.Tables[0].Rows)
				comboProcOpId.Items.Add(dr[0].ToString());
		}

		private void buttonCancel_Click(object sender, System.EventArgs e)
		{
			Close();
		}

		private void buttonMarkPlateDamagedSend_Click(object sender, System.EventArgs e)
		{
			char damagecode = 'N';
			int damagedplate = 0;
			try
			{
				if (textMarkPlateDamagedCode.Text.Length != 1) throw new Exception();
				damagecode = textMarkPlateDamagedCode.Text[0];
			}
			catch (Exception)
			{
				MessageBox.Show("The damage code must be a char", "Input error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				textMarkPlateDamagedCode.Text = "N";
				return;
			}
			if (radioMarkPlateDamagedCurrent.Checked)
			{
				SendInterrupt("PlateDamagedCode " + damagecode);
			}
			else
			{
				try
				{
					damagedplate = SySal.OperaDb.Convert.ToInt32(textMarkPlateDamagedN.Text);
				}
				catch (Exception)
				{
					MessageBox.Show("The damage plate must be represented by an integer", "Input error", MessageBoxButtons.OK, MessageBoxIcon.Error);
					textMarkPlateDamagedN.Text = "";
					return;
				}		
				SendInterrupt("PlateDamaged " + damagedplate + ", PlateDamagedCode " + damagecode);
			}
		}

		private void buttonGoBackToPlateSend_Click(object sender, System.EventArgs e)
		{
			int plate = 0;
			try
			{
				plate = SySal.OperaDb.Convert.ToInt32(textGoBackToPlateN.Text);
			}
			catch (Exception)
			{
				MessageBox.Show("The plate to go back to must be represented by an integer", "Input error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				textGoBackToPlateN.Text = "";
				return;
			}		
			if (checkGoBackToPlateCalibrations.Checked)
			{
				SendInterrupt("GoBackToPlateNCancelCalibrations " + plate);
			}	
			else
			{
				SendInterrupt("GoBackToPlateN " + plate);
			}
		}

		private void SendInterrupt(string interruptdata)
		{
			try
			{
				BM.Interrupt(SySal.OperaDb.Convert.ToInt64(comboProcOpId.Text), textOPERAUsername.Text, textOPERAPwd.Text, interruptdata);
				MessageBox.Show("Interrupt message:\r\n" + interruptdata, "Interrupt sent", MessageBoxButtons.OK, MessageBoxIcon.Information);
			}
			catch(Exception x)
			{
				MessageBox.Show(x.Message, "Error sending interrupt", MessageBoxButtons.OK, MessageBoxIcon.Error);
			}
		}

		static System.Text.RegularExpressions.Regex PredEx = new System.Text.RegularExpressions.Regex(@"\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*");

		private string Pred = "";

		private void buttonPredFileLoad_Click(object sender, System.EventArgs e)
		{
			if (openPredFileDialog.ShowDialog() == DialogResult.OK)
			{
				System.IO.StreamReader r = null;
				int correctcount = 0;
				Pred = "";
				try
				{
					r = new System.IO.StreamReader(openPredFileDialog.FileName);
					string preds = r.ReadToEnd();
					char separator = (comboPredSeparator.Text == "newline") ? '\n' : comboPredSeparator.Text[0];
					string [] lines = preds.Split(separator);					
					foreach (string s in lines)
					{
						System.Text.RegularExpressions.Match m = PredEx.Match(s);
						if (m.Success == true && m.Length == s.Length)
						{
							Convert.ToInt32(m.Groups[1].Value);
							Convert.ToDouble(m.Groups[2].Value, System.Globalization.CultureInfo.InvariantCulture);
							Convert.ToDouble(m.Groups[3].Value, System.Globalization.CultureInfo.InvariantCulture);
							Convert.ToDouble(m.Groups[4].Value, System.Globalization.CultureInfo.InvariantCulture);
							Convert.ToDouble(m.Groups[5].Value, System.Globalization.CultureInfo.InvariantCulture);
							Convert.ToInt32(m.Groups[6].Value);
							correctcount++;
							Pred += "; " + m.Groups[1].Value + " " + m.Groups[2].Value + " " + m.Groups[3].Value + " " + m.Groups[4].Value + " " + m.Groups[5].Value + " " + m.Groups[6].Value + " ";
						}
						else throw new Exception("Incorrect prediction syntax found.");						
					}
					r.Close();
					r = null;
					Pred = "Paths " + correctcount + " " + Pred;
				}
				catch (Exception x)
				{					
					MessageBox.Show(x.Message + "\r\nPredictions read: " + correctcount, "Error trying to build predictions", MessageBoxButtons.OK, MessageBoxIcon.Error);
					Pred = "";
				}
				if (r != null) r.Close();
				textPredFileCount.Text = correctcount.ToString();
			}
		}

		private void buttonPredictionsSend_Click(object sender, System.EventArgs e)
		{
			if (Pred != null && Pred.Length > 0) SendInterrupt(Pred);		
		}

		private void buttonIgnoreCalibrationsSend_Click(object sender, System.EventArgs e)
		{
			string interruptdata = "";
			try
			{
				string [] idstr = textIgnoreCalibrations.Text.Split(',');				
				foreach (string s in idstr)
					if (interruptdata.Length == 0) interruptdata = Convert.ToInt64(s).ToString();
					else interruptdata += " " + Convert.ToInt64(s);
				interruptdata = "IgnoreCalibrations " + idstr.Length + "; " + interruptdata;
			}
			catch (Exception x)
			{
				MessageBox.Show(x.Message, "Error reading calibration list", MessageBoxButtons.OK, MessageBoxIcon.Error);
			}
			SendInterrupt(interruptdata);
		}

		private void buttonForceCalibrationsSend_Click(object sender, System.EventArgs e)
		{
			string interruptdata = "";
			try
			{
				string [] idstr = textForceCalibrations.Text.Split(',');				
				foreach (string s in idstr)
					if (interruptdata.Length == 0) interruptdata = Convert.ToInt64(s).ToString();
					else interruptdata += " " + Convert.ToInt64(s);
				interruptdata = "ForceCalibrations " + idstr.Length + "; " + interruptdata;
			}
			catch (Exception x)
			{
				MessageBox.Show(x.Message, "Error reading calibration list", MessageBoxButtons.OK, MessageBoxIcon.Error);
			}
			SendInterrupt(interruptdata);		
		}

        private void buttonExecCommandSend_Click(object sender, EventArgs e)
        {
            SendInterrupt(radioContinue.Checked ? "Continue" : "Close");
        }

        static System.Text.RegularExpressions.Regex SetPathEx = new System.Text.RegularExpressions.Regex(@"\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*");

        private string SetPath = "";

        private void buttonPathFileLoad_Click(object sender, EventArgs e)
        {
            if (openPredFileDialog.ShowDialog() == DialogResult.OK)
            {
                System.IO.StreamReader r = null;
                int correctcount = 0;
                SetPath = "";
                try
                {
                    r = new System.IO.StreamReader(openPredFileDialog.FileName);
                    string setpaths = r.ReadToEnd();
                    char separator = (comboPredSeparator.Text == "newline") ? '\n' : comboPredSeparator.Text[0];
                    string[] lines = setpaths.Split(separator);
                    foreach (string s in lines)
                    {
                        System.Text.RegularExpressions.Match m = SetPathEx.Match(s);
                        if (m.Success == true && m.Length == s.Length)
                        {
                            Convert.ToInt32(m.Groups[1].Value);
                            Convert.ToInt64(m.Groups[3].Value);
                            Convert.ToInt32(m.Groups[4].Value);
                            Convert.ToInt32(m.Groups[5].Value);
                            Convert.ToInt32(m.Groups[6].Value);
                            correctcount++;
                            SetPath += "; " + m.Groups[1].Value + " " + m.Groups[3].Value + " " + m.Groups[4].Value + " " + m.Groups[5].Value + " " + m.Groups[6].Value + " ";
                        }
                        else throw new Exception("Incorrect prediction syntax found.");
                    }
                    r.Close();
                    r = null;
                    SetPath = "SetPaths " + correctcount + " " + SetPath;
                }
                catch (Exception x)
                {
                    MessageBox.Show(x.Message + "\r\nSetPath rows read: " + correctcount, "Error trying to build predictions", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    SetPath = "";
                }
                if (r != null) r.Close();
                textSetPathCount.Text = correctcount.ToString();
            }
        }

        private void buttonSetPathSend_Click(object sender, EventArgs e)
        {
            if (SetPath != null && SetPath.Length > 0) SendInterrupt(SetPath);		
        }
	}
}
