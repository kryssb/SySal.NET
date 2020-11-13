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

namespace SySal.DAQSystem.Drivers.TotalScanDriver
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
		private System.Windows.Forms.GroupBox groupBox1;
		private System.Windows.Forms.GroupBox groupBox2;
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
		private System.Windows.Forms.GroupBox groupBox3;
		private System.Windows.Forms.TextBox textVolFileCount;
		private System.Windows.Forms.Label label7;
		private System.Windows.Forms.OpenFileDialog openVolFileDialog;
		private System.Windows.Forms.Button buttonVolFileLoad;
		private System.Windows.Forms.ComboBox comboVolSeparator;
		private System.Windows.Forms.Button buttonVolumesSend;
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
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.buttonGoBackToPlateSend = new System.Windows.Forms.Button();
            this.checkGoBackToPlateCalibrations = new System.Windows.Forms.CheckBox();
            this.textGoBackToPlateN = new System.Windows.Forms.TextBox();
            this.label6 = new System.Windows.Forms.Label();
            this.groupBox2 = new System.Windows.Forms.GroupBox();
            this.textMarkPlateDamagedCode = new System.Windows.Forms.TextBox();
            this.label5 = new System.Windows.Forms.Label();
            this.textMarkPlateDamagedN = new System.Windows.Forms.TextBox();
            this.radioMarkPlateDamagedN = new System.Windows.Forms.RadioButton();
            this.radioMarkPlateDamagedCurrent = new System.Windows.Forms.RadioButton();
            this.groupBox3 = new System.Windows.Forms.GroupBox();
            this.textVolFileCount = new System.Windows.Forms.TextBox();
            this.buttonVolFileLoad = new System.Windows.Forms.Button();
            this.comboVolSeparator = new System.Windows.Forms.ComboBox();
            this.label7 = new System.Windows.Forms.Label();
            this.buttonVolumesSend = new System.Windows.Forms.Button();
            this.openVolFileDialog = new System.Windows.Forms.OpenFileDialog();
            this.groupBox1.SuspendLayout();
            this.groupBox2.SuspendLayout();
            this.groupBox3.SuspendLayout();
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
            this.label3.Location = new System.Drawing.Point(8, 184);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(104, 24);
            this.label3.TabIndex = 5;
            this.label3.Text = "OPERA Username";
            this.label3.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // textOPERAUsername
            // 
            this.textOPERAUsername.Location = new System.Drawing.Point(112, 184);
            this.textOPERAUsername.Name = "textOPERAUsername";
            this.textOPERAUsername.Size = new System.Drawing.Size(200, 20);
            this.textOPERAUsername.TabIndex = 6;
            // 
            // textOPERAPwd
            // 
            this.textOPERAPwd.Location = new System.Drawing.Point(112, 208);
            this.textOPERAPwd.Name = "textOPERAPwd";
            this.textOPERAPwd.PasswordChar = '*';
            this.textOPERAPwd.Size = new System.Drawing.Size(200, 20);
            this.textOPERAPwd.TabIndex = 8;
            // 
            // label4
            // 
            this.label4.Location = new System.Drawing.Point(8, 208);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(104, 24);
            this.label4.TabIndex = 7;
            this.label4.Text = "OPERA Password";
            this.label4.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // buttonCancel
            // 
            this.buttonCancel.Location = new System.Drawing.Point(8, 232);
            this.buttonCancel.Name = "buttonCancel";
            this.buttonCancel.Size = new System.Drawing.Size(64, 24);
            this.buttonCancel.TabIndex = 9;
            this.buttonCancel.Text = "Cancel";
            this.buttonCancel.Click += new System.EventHandler(this.buttonCancel_Click);
            // 
            // buttonMarkPlateDamagedSend
            // 
            this.buttonMarkPlateDamagedSend.Location = new System.Drawing.Point(120, 88);
            this.buttonMarkPlateDamagedSend.Name = "buttonMarkPlateDamagedSend";
            this.buttonMarkPlateDamagedSend.Size = new System.Drawing.Size(64, 24);
            this.buttonMarkPlateDamagedSend.TabIndex = 10;
            this.buttonMarkPlateDamagedSend.Text = "Send";
            this.buttonMarkPlateDamagedSend.Click += new System.EventHandler(this.buttonMarkPlateDamagedSend_Click);
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.buttonGoBackToPlateSend);
            this.groupBox1.Controls.Add(this.checkGoBackToPlateCalibrations);
            this.groupBox1.Controls.Add(this.textGoBackToPlateN);
            this.groupBox1.Controls.Add(this.label6);
            this.groupBox1.Location = new System.Drawing.Point(8, 56);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(192, 120);
            this.groupBox1.TabIndex = 11;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "send Go back to plate";
            // 
            // buttonGoBackToPlateSend
            // 
            this.buttonGoBackToPlateSend.Location = new System.Drawing.Point(120, 88);
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
            this.checkGoBackToPlateCalibrations.Location = new System.Drawing.Point(8, 40);
            this.checkGoBackToPlateCalibrations.Name = "checkGoBackToPlateCalibrations";
            this.checkGoBackToPlateCalibrations.Size = new System.Drawing.Size(176, 40);
            this.checkGoBackToPlateCalibrations.TabIndex = 4;
            this.checkGoBackToPlateCalibrations.Text = "Cancel calibrations done within this operation";
            // 
            // textGoBackToPlateN
            // 
            this.textGoBackToPlateN.Location = new System.Drawing.Point(76, 16);
            this.textGoBackToPlateN.Name = "textGoBackToPlateN";
            this.textGoBackToPlateN.Size = new System.Drawing.Size(40, 20);
            this.textGoBackToPlateN.TabIndex = 3;
            this.textGoBackToPlateN.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label6
            // 
            this.label6.Location = new System.Drawing.Point(8, 16);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(64, 24);
            this.label6.TabIndex = 0;
            this.label6.Text = "Plate #";
            this.label6.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // groupBox2
            // 
            this.groupBox2.Controls.Add(this.textMarkPlateDamagedCode);
            this.groupBox2.Controls.Add(this.label5);
            this.groupBox2.Controls.Add(this.textMarkPlateDamagedN);
            this.groupBox2.Controls.Add(this.radioMarkPlateDamagedN);
            this.groupBox2.Controls.Add(this.radioMarkPlateDamagedCurrent);
            this.groupBox2.Controls.Add(this.buttonMarkPlateDamagedSend);
            this.groupBox2.Location = new System.Drawing.Point(208, 56);
            this.groupBox2.Name = "groupBox2";
            this.groupBox2.Size = new System.Drawing.Size(192, 120);
            this.groupBox2.TabIndex = 12;
            this.groupBox2.TabStop = false;
            this.groupBox2.Text = "send Mark plate damaged";
            // 
            // textMarkPlateDamagedCode
            // 
            this.textMarkPlateDamagedCode.Location = new System.Drawing.Point(144, 64);
            this.textMarkPlateDamagedCode.Name = "textMarkPlateDamagedCode";
            this.textMarkPlateDamagedCode.Size = new System.Drawing.Size(40, 20);
            this.textMarkPlateDamagedCode.TabIndex = 4;
            this.textMarkPlateDamagedCode.Text = "N";
            this.textMarkPlateDamagedCode.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label5
            // 
            this.label5.Location = new System.Drawing.Point(8, 64);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(112, 24);
            this.label5.TabIndex = 3;
            this.label5.Text = "Plate damage code";
            this.label5.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // textMarkPlateDamagedN
            // 
            this.textMarkPlateDamagedN.Location = new System.Drawing.Point(88, 40);
            this.textMarkPlateDamagedN.Name = "textMarkPlateDamagedN";
            this.textMarkPlateDamagedN.Size = new System.Drawing.Size(40, 20);
            this.textMarkPlateDamagedN.TabIndex = 2;
            this.textMarkPlateDamagedN.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // radioMarkPlateDamagedN
            // 
            this.radioMarkPlateDamagedN.Location = new System.Drawing.Point(8, 40);
            this.radioMarkPlateDamagedN.Name = "radioMarkPlateDamagedN";
            this.radioMarkPlateDamagedN.Size = new System.Drawing.Size(80, 24);
            this.radioMarkPlateDamagedN.TabIndex = 1;
            this.radioMarkPlateDamagedN.Text = "Plate #";
            // 
            // radioMarkPlateDamagedCurrent
            // 
            this.radioMarkPlateDamagedCurrent.Checked = true;
            this.radioMarkPlateDamagedCurrent.Location = new System.Drawing.Point(8, 16);
            this.radioMarkPlateDamagedCurrent.Name = "radioMarkPlateDamagedCurrent";
            this.radioMarkPlateDamagedCurrent.Size = new System.Drawing.Size(80, 24);
            this.radioMarkPlateDamagedCurrent.TabIndex = 0;
            this.radioMarkPlateDamagedCurrent.TabStop = true;
            this.radioMarkPlateDamagedCurrent.Text = "Current";
            // 
            // groupBox3
            // 
            this.groupBox3.Controls.Add(this.textVolFileCount);
            this.groupBox3.Controls.Add(this.buttonVolFileLoad);
            this.groupBox3.Controls.Add(this.comboVolSeparator);
            this.groupBox3.Controls.Add(this.label7);
            this.groupBox3.Controls.Add(this.buttonVolumesSend);
            this.groupBox3.Location = new System.Drawing.Point(408, 56);
            this.groupBox3.Name = "groupBox3";
            this.groupBox3.Size = new System.Drawing.Size(184, 120);
            this.groupBox3.TabIndex = 14;
            this.groupBox3.TabStop = false;
            this.groupBox3.Text = "send predictions";
            // 
            // textVolFileCount
            // 
            this.textVolFileCount.Location = new System.Drawing.Point(112, 40);
            this.textVolFileCount.Name = "textVolFileCount";
            this.textVolFileCount.ReadOnly = true;
            this.textVolFileCount.Size = new System.Drawing.Size(64, 20);
            this.textVolFileCount.TabIndex = 14;
            this.textVolFileCount.Text = "0";
            this.textVolFileCount.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // buttonVolFileLoad
            // 
            this.buttonVolFileLoad.Location = new System.Drawing.Point(8, 40);
            this.buttonVolFileLoad.Name = "buttonVolFileLoad";
            this.buttonVolFileLoad.Size = new System.Drawing.Size(96, 24);
            this.buttonVolFileLoad.TabIndex = 13;
            this.buttonVolFileLoad.Text = "Load and count";
            this.buttonVolFileLoad.Click += new System.EventHandler(this.buttonVolFileLoad_Click);
            // 
            // comboVolSeparator
            // 
            this.comboVolSeparator.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.comboVolSeparator.Items.AddRange(new object[] {
            "newline",
            ";",
            ",",
            "."});
            this.comboVolSeparator.Location = new System.Drawing.Point(112, 16);
            this.comboVolSeparator.Name = "comboVolSeparator";
            this.comboVolSeparator.Size = new System.Drawing.Size(64, 21);
            this.comboVolSeparator.TabIndex = 12;
            // 
            // label7
            // 
            this.label7.Location = new System.Drawing.Point(8, 16);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(96, 24);
            this.label7.TabIndex = 11;
            this.label7.Text = "n-tuple separator";
            this.label7.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // buttonVolumesSend
            // 
            this.buttonVolumesSend.Location = new System.Drawing.Point(112, 88);
            this.buttonVolumesSend.Name = "buttonVolumesSend";
            this.buttonVolumesSend.Size = new System.Drawing.Size(64, 24);
            this.buttonVolumesSend.TabIndex = 10;
            this.buttonVolumesSend.Text = "Send";
            this.buttonVolumesSend.Click += new System.EventHandler(this.buttonVolumesSend_Click);
            // 
            // openVolFileDialog
            // 
            this.openVolFileDialog.Filter = "Text files (*.txt)|*.txt|All files (*.*)|*.*";
            this.openVolFileDialog.Title = "Select volume file";
            // 
            // frmEasyInterrupt
            // 
            this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
            this.ClientSize = new System.Drawing.Size(602, 264);
            this.Controls.Add(this.groupBox3);
            this.Controls.Add(this.groupBox2);
            this.Controls.Add(this.groupBox1);
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
            this.Text = "Easy Interrupt for TotalScanDriver";
            this.Load += new System.EventHandler(this.OnLoad);
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            this.groupBox2.ResumeLayout(false);
            this.groupBox2.PerformLayout();
            this.groupBox3.ResumeLayout(false);
            this.groupBox3.PerformLayout();
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
			new SySal.OperaDb.OperaDbDataAdapter("select name from tb_machines where id_site = (select to_number(value) from lZ_sitevars where name = 'ID_SITE') and isbatchserver = 1", Conn, null).Fill(ds);
			foreach (System.Data.DataRow dr in ds.Tables[0].Rows)
				comboBatchManager.Items.Add(dr[0].ToString());
			textOPERAUsername.Text = cred.OPERAUserName;
			textOPERAPwd.Text = cred.OPERAPassword;
            comboVolSeparator.SelectedIndex = 0;
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
			new SySal.OperaDb.OperaDbDataAdapter("SELECT TB_PROC_OPERATIONS.ID FROM TB_PROC_OPERATIONS INNER JOIN TB_PROGRAMSETTINGS ON (TB_PROC_OPERATIONS.ID_PROGRAMSETTINGS = TB_PROGRAMSETTINGS.ID AND TB_PROGRAMSETTINGS.EXECUTABLE = 'TotalScanDriver.exe') WHERE TB_PROC_OPERATIONS.ID IN (" + wherestr + ")", Conn, null).Fill(ds);
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

		static System.Text.RegularExpressions.Regex VolEx = new System.Text.RegularExpressions.Regex(@"\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\d+)\s+(\d+)\s*");

        static System.Text.RegularExpressions.Regex VolEx2 = new System.Text.RegularExpressions.Regex(@"\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s*");

        static System.Text.RegularExpressions.Regex VolEx3 = new System.Text.RegularExpressions.Regex(@"\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\d+)\s*");

        static System.Text.RegularExpressions.Regex VolEx4 = new System.Text.RegularExpressions.Regex(@"\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\d+\s*\(\s*\d+[\s+\d+]*\s*\))\s");

		private string Vol = "";

		private void buttonVolFileLoad_Click(object sender, System.EventArgs e)
		{
			if (openVolFileDialog.ShowDialog() == DialogResult.OK)
			{
				System.IO.StreamReader r = null;
				int correctcount = 0;
				Vol = "";
				try
				{
					r = new System.IO.StreamReader(openVolFileDialog.FileName);
					string preds = r.ReadToEnd();
					char separator = (comboVolSeparator.Text == "newline") ? '\n' : comboVolSeparator.Text[0];
					string [] lines = preds.Split(separator);					
					foreach (string s in lines)
					{
						System.Text.RegularExpressions.Match m = VolEx.Match(s);
                        System.Text.RegularExpressions.Match m2 = VolEx2.Match(s);
                        System.Text.RegularExpressions.Match m3 = VolEx3.Match(s);
                        System.Text.RegularExpressions.Match m4 = VolEx4.Match(s);
                        if (m4.Success == true && m4.Length == s.Length)
                        {                            
                            Convert.ToInt32(m4.Groups[1].Value);
                            Convert.ToDouble(m4.Groups[2].Value, System.Globalization.CultureInfo.InvariantCulture);
                            Convert.ToDouble(m4.Groups[3].Value, System.Globalization.CultureInfo.InvariantCulture);
                            Convert.ToDouble(m4.Groups[4].Value, System.Globalization.CultureInfo.InvariantCulture);
                            Convert.ToDouble(m4.Groups[5].Value, System.Globalization.CultureInfo.InvariantCulture);
                            Convert.ToInt32(m4.Groups[6].Value);
                            Convert.ToInt32(m4.Groups[7].Value);
                            Convert.ToInt32(m4.Groups[8].Value);
                            Convert.ToDouble(m4.Groups[9].Value, System.Globalization.CultureInfo.InvariantCulture);
                            Convert.ToDouble(m4.Groups[10].Value, System.Globalization.CultureInfo.InvariantCulture);
                            //Convert.ToInt64(m4.Groups[11].Value, System.Globalization.CultureInfo.InvariantCulture);
                            correctcount++;
                            Vol += "; " + m4.Groups[1].Value + " " + m4.Groups[2].Value + " " + m4.Groups[3].Value + " " + m4.Groups[4].Value + " " + m4.Groups[5].Value + " " + m4.Groups[6].Value + " " + m4.Groups[7].Value + " " + m4.Groups[8].Value + " " + m4.Groups[9].Value + " " + m4.Groups[10].Value + " " + m4.Groups[11].Value;
                        }
						if (m3.Success == true && m3.Length == s.Length)
                        {
                            Convert.ToInt32(m3.Groups[1].Value);
                            Convert.ToDouble(m3.Groups[2].Value, System.Globalization.CultureInfo.InvariantCulture);
                            Convert.ToDouble(m3.Groups[3].Value, System.Globalization.CultureInfo.InvariantCulture);
                            Convert.ToDouble(m3.Groups[4].Value, System.Globalization.CultureInfo.InvariantCulture);
                            Convert.ToDouble(m3.Groups[5].Value, System.Globalization.CultureInfo.InvariantCulture);
                            Convert.ToInt32(m3.Groups[6].Value);
                            Convert.ToInt32(m3.Groups[7].Value);
                            Convert.ToInt32(m3.Groups[8].Value);
                            Convert.ToDouble(m3.Groups[9].Value, System.Globalization.CultureInfo.InvariantCulture);
                            Convert.ToDouble(m3.Groups[10].Value, System.Globalization.CultureInfo.InvariantCulture);
                            Convert.ToInt64(m3.Groups[11].Value, System.Globalization.CultureInfo.InvariantCulture);
                            correctcount++;
                            Vol += "; " + m3.Groups[1].Value + " " + m3.Groups[2].Value + " " + m3.Groups[3].Value + " " + m3.Groups[4].Value + " " + m3.Groups[5].Value + " " + m3.Groups[6].Value + " " + m3.Groups[7].Value + " " + m3.Groups[8].Value + " " + m3.Groups[9].Value + " " + m3.Groups[10].Value + " " + m3.Groups[11].Value;
                        }
                        else if (m2.Success == true && m2.Length == s.Length)
                        {
                            Convert.ToInt32(m2.Groups[1].Value);
                            Convert.ToDouble(m2.Groups[2].Value, System.Globalization.CultureInfo.InvariantCulture);
                            Convert.ToDouble(m2.Groups[3].Value, System.Globalization.CultureInfo.InvariantCulture);
                            Convert.ToDouble(m2.Groups[4].Value, System.Globalization.CultureInfo.InvariantCulture);
                            Convert.ToDouble(m2.Groups[5].Value, System.Globalization.CultureInfo.InvariantCulture);
                            Convert.ToInt32(m2.Groups[6].Value);
                            Convert.ToInt32(m2.Groups[7].Value);
                            Convert.ToInt32(m2.Groups[8].Value);
                            Convert.ToDouble(m2.Groups[9].Value, System.Globalization.CultureInfo.InvariantCulture);
                            Convert.ToDouble(m2.Groups[10].Value, System.Globalization.CultureInfo.InvariantCulture);
                            correctcount++;
                            Vol += "; " + m2.Groups[1].Value + " " + m2.Groups[2].Value + " " + m2.Groups[3].Value + " " + m2.Groups[4].Value + " " + m2.Groups[5].Value + " " + m2.Groups[6].Value + " " + m2.Groups[7].Value + " " + m2.Groups[8].Value + " " + m2.Groups[9].Value + " " + m2.Groups[10].Value;
                        }
                        else if (m.Success == true && m.Length == s.Length)
						{
							Convert.ToInt32(m.Groups[1].Value);
							Convert.ToDouble(m.Groups[2].Value, System.Globalization.CultureInfo.InvariantCulture);
							Convert.ToDouble(m.Groups[3].Value, System.Globalization.CultureInfo.InvariantCulture);
							Convert.ToDouble(m.Groups[4].Value, System.Globalization.CultureInfo.InvariantCulture);
							Convert.ToDouble(m.Groups[5].Value, System.Globalization.CultureInfo.InvariantCulture);
							Convert.ToInt32(m.Groups[6].Value);
							Convert.ToInt32(m.Groups[7].Value);
							correctcount++;
							Vol += "; " + m.Groups[1].Value + " " + m.Groups[2].Value + " " + m.Groups[3].Value + " " + m.Groups[4].Value + " " + m.Groups[5].Value + " " + m.Groups[6].Value + " " + m.Groups[7].Value + " ";
						}                        
						else throw new Exception("Incorrect volume syntax found.");
					}
					r.Close();
					r = null;
					Vol = "Volumes " + correctcount + " " + Vol;
				}
				catch (Exception x)
				{					
					MessageBox.Show(x.Message + "\r\nVolumes read: " + correctcount, "Error trying to build volume interrupt", MessageBoxButtons.OK, MessageBoxIcon.Error);
					Vol = "";
				}
				if (r != null) r.Close();
				textVolFileCount.Text = correctcount.ToString();
			}		
		}

		private void buttonVolumesSend_Click(object sender, System.EventArgs e)
		{		
			if (Vol != null && Vol.Length > 0) SendInterrupt(Vol);		
		}
	}
}
