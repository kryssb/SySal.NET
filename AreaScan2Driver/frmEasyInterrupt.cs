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

namespace SySal.DAQSystem.Drivers.AreaScan2Driver
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
		private System.Windows.Forms.GroupBox groupBox1;
		private System.Windows.Forms.CheckBox checkIgnoreScanFailure;
		private System.Windows.Forms.RadioButton radioIgnoreScanFailureTrue;
		private System.Windows.Forms.RadioButton radioIgnoreScanFailureFalse;
		private System.Windows.Forms.Label label3;
		private System.Windows.Forms.TextBox textOPERAUsername;
		private System.Windows.Forms.TextBox textOPERAPwd;
		private System.Windows.Forms.Label label4;
		private System.Windows.Forms.Button buttonCancel;
		private System.Windows.Forms.Button buttonSend;
		private System.Windows.Forms.GroupBox groupBox2;
		private System.Windows.Forms.RadioButton radioIgnoreRecalFailureFalse;
		private System.Windows.Forms.RadioButton radioIgnoreRecalFailureTrue;
		private System.Windows.Forms.CheckBox checkIgnoreRecalFailure;
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
			this.groupBox1 = new System.Windows.Forms.GroupBox();
			this.radioIgnoreScanFailureFalse = new System.Windows.Forms.RadioButton();
			this.radioIgnoreScanFailureTrue = new System.Windows.Forms.RadioButton();
			this.checkIgnoreScanFailure = new System.Windows.Forms.CheckBox();
			this.label3 = new System.Windows.Forms.Label();
			this.textOPERAUsername = new System.Windows.Forms.TextBox();
			this.textOPERAPwd = new System.Windows.Forms.TextBox();
			this.label4 = new System.Windows.Forms.Label();
			this.buttonCancel = new System.Windows.Forms.Button();
			this.buttonSend = new System.Windows.Forms.Button();
			this.groupBox2 = new System.Windows.Forms.GroupBox();
			this.radioIgnoreRecalFailureFalse = new System.Windows.Forms.RadioButton();
			this.radioIgnoreRecalFailureTrue = new System.Windows.Forms.RadioButton();
			this.checkIgnoreRecalFailure = new System.Windows.Forms.CheckBox();
			this.groupBox1.SuspendLayout();
			this.groupBox2.SuspendLayout();
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
			this.comboBatchManager.SelectedIndexChanged += new System.EventHandler(this.OnBatchManagerSelected);
			this.comboBatchManager.SelectionChangeCommitted += new System.EventHandler(this.OnBatchManagerSelected);
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
			// groupBox1
			// 
			this.groupBox1.Controls.Add(this.radioIgnoreScanFailureFalse);
			this.groupBox1.Controls.Add(this.radioIgnoreScanFailureTrue);
			this.groupBox1.Controls.Add(this.checkIgnoreScanFailure);
			this.groupBox1.Location = new System.Drawing.Point(8, 64);
			this.groupBox1.Name = "groupBox1";
			this.groupBox1.Size = new System.Drawing.Size(160, 72);
			this.groupBox1.TabIndex = 4;
			this.groupBox1.TabStop = false;
			this.groupBox1.Text = "send IgnoreScanFailure";
			// 
			// radioIgnoreScanFailureFalse
			// 
			this.radioIgnoreScanFailureFalse.Location = new System.Drawing.Point(80, 48);
			this.radioIgnoreScanFailureFalse.Name = "radioIgnoreScanFailureFalse";
			this.radioIgnoreScanFailureFalse.Size = new System.Drawing.Size(72, 16);
			this.radioIgnoreScanFailureFalse.TabIndex = 2;
			this.radioIgnoreScanFailureFalse.Text = "False";
			// 
			// radioIgnoreScanFailureTrue
			// 
			this.radioIgnoreScanFailureTrue.Checked = true;
			this.radioIgnoreScanFailureTrue.Location = new System.Drawing.Point(80, 24);
			this.radioIgnoreScanFailureTrue.Name = "radioIgnoreScanFailureTrue";
			this.radioIgnoreScanFailureTrue.Size = new System.Drawing.Size(72, 16);
			this.radioIgnoreScanFailureTrue.TabIndex = 1;
			this.radioIgnoreScanFailureTrue.TabStop = true;
			this.radioIgnoreScanFailureTrue.Text = "True";
			// 
			// checkIgnoreScanFailure
			// 
			this.checkIgnoreScanFailure.Location = new System.Drawing.Point(8, 16);
			this.checkIgnoreScanFailure.Name = "checkIgnoreScanFailure";
			this.checkIgnoreScanFailure.Size = new System.Drawing.Size(64, 32);
			this.checkIgnoreScanFailure.TabIndex = 0;
			this.checkIgnoreScanFailure.Text = "Enable";
			// 
			// label3
			// 
			this.label3.Location = new System.Drawing.Point(8, 144);
			this.label3.Name = "label3";
			this.label3.Size = new System.Drawing.Size(104, 24);
			this.label3.TabIndex = 5;
			this.label3.Text = "OPERA Username";
			this.label3.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
			// 
			// textOPERAUsername
			// 
			this.textOPERAUsername.Location = new System.Drawing.Point(112, 144);
			this.textOPERAUsername.Name = "textOPERAUsername";
			this.textOPERAUsername.Size = new System.Drawing.Size(200, 20);
			this.textOPERAUsername.TabIndex = 6;
			this.textOPERAUsername.Text = "";
			// 
			// textOPERAPwd
			// 
			this.textOPERAPwd.Location = new System.Drawing.Point(112, 168);
			this.textOPERAPwd.Name = "textOPERAPwd";
			this.textOPERAPwd.PasswordChar = '*';
			this.textOPERAPwd.Size = new System.Drawing.Size(200, 20);
			this.textOPERAPwd.TabIndex = 8;
			this.textOPERAPwd.Text = "";
			// 
			// label4
			// 
			this.label4.Location = new System.Drawing.Point(8, 168);
			this.label4.Name = "label4";
			this.label4.Size = new System.Drawing.Size(104, 24);
			this.label4.TabIndex = 7;
			this.label4.Text = "OPERA Password";
			this.label4.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
			// 
			// buttonCancel
			// 
			this.buttonCancel.Location = new System.Drawing.Point(8, 192);
			this.buttonCancel.Name = "buttonCancel";
			this.buttonCancel.Size = new System.Drawing.Size(64, 24);
			this.buttonCancel.TabIndex = 9;
			this.buttonCancel.Text = "Cancel";
			this.buttonCancel.Click += new System.EventHandler(this.buttonCancel_Click);
			// 
			// buttonSend
			// 
			this.buttonSend.Location = new System.Drawing.Point(336, 192);
			this.buttonSend.Name = "buttonSend";
			this.buttonSend.Size = new System.Drawing.Size(64, 24);
			this.buttonSend.TabIndex = 10;
			this.buttonSend.Text = "Send";
			this.buttonSend.Click += new System.EventHandler(this.buttonSend_Click);
			// 
			// groupBox2
			// 
			this.groupBox2.Controls.Add(this.radioIgnoreRecalFailureFalse);
			this.groupBox2.Controls.Add(this.radioIgnoreRecalFailureTrue);
			this.groupBox2.Controls.Add(this.checkIgnoreRecalFailure);
			this.groupBox2.Location = new System.Drawing.Point(176, 64);
			this.groupBox2.Name = "groupBox2";
			this.groupBox2.Size = new System.Drawing.Size(160, 72);
			this.groupBox2.TabIndex = 11;
			this.groupBox2.TabStop = false;
			this.groupBox2.Text = "send IgnoreRecalFailure";
			// 
			// radioIgnoreRecalFailureFalse
			// 
			this.radioIgnoreRecalFailureFalse.Location = new System.Drawing.Point(80, 48);
			this.radioIgnoreRecalFailureFalse.Name = "radioIgnoreRecalFailureFalse";
			this.radioIgnoreRecalFailureFalse.Size = new System.Drawing.Size(72, 16);
			this.radioIgnoreRecalFailureFalse.TabIndex = 2;
			this.radioIgnoreRecalFailureFalse.Text = "False";
			// 
			// radioIgnoreRecalFailureTrue
			// 
			this.radioIgnoreRecalFailureTrue.Checked = true;
			this.radioIgnoreRecalFailureTrue.Location = new System.Drawing.Point(80, 24);
			this.radioIgnoreRecalFailureTrue.Name = "radioIgnoreRecalFailureTrue";
			this.radioIgnoreRecalFailureTrue.Size = new System.Drawing.Size(72, 16);
			this.radioIgnoreRecalFailureTrue.TabIndex = 1;
			this.radioIgnoreRecalFailureTrue.TabStop = true;
			this.radioIgnoreRecalFailureTrue.Text = "True";
			// 
			// checkIgnoreRecalFailure
			// 
			this.checkIgnoreRecalFailure.Location = new System.Drawing.Point(8, 16);
			this.checkIgnoreRecalFailure.Name = "checkIgnoreRecalFailure";
			this.checkIgnoreRecalFailure.Size = new System.Drawing.Size(64, 32);
			this.checkIgnoreRecalFailure.TabIndex = 0;
			this.checkIgnoreRecalFailure.Text = "Enable";
			// 
			// frmEasyInterrupt
			// 
			this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
			this.ClientSize = new System.Drawing.Size(408, 222);
			this.Controls.Add(this.groupBox2);
			this.Controls.Add(this.buttonSend);
			this.Controls.Add(this.buttonCancel);
			this.Controls.Add(this.textOPERAPwd);
			this.Controls.Add(this.textOPERAUsername);
			this.Controls.Add(this.label4);
			this.Controls.Add(this.label3);
			this.Controls.Add(this.groupBox1);
			this.Controls.Add(this.comboProcOpId);
			this.Controls.Add(this.label2);
			this.Controls.Add(this.comboBatchManager);
			this.Controls.Add(this.label1);
			this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
			this.Name = "frmEasyInterrupt";
			this.Text = "Easy Interrupt for AreaScan2Driver";
			this.Load += new System.EventHandler(this.OnLoad);
			this.groupBox1.ResumeLayout(false);
			this.groupBox2.ResumeLayout(false);
			this.ResumeLayout(false);

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
			new SySal.OperaDb.OperaDbDataAdapter("select name from tb_machines where id_site = (select to_number(value) from opera.lz_sitevars where name = 'ID_SITE') and isbatchserver = 1", Conn, null).Fill(ds);
			foreach (System.Data.DataRow dr in ds.Tables[0].Rows)
				comboBatchManager.Items.Add(dr[0].ToString());
			textOPERAUsername.Text = cred.OPERAUserName;
			textOPERAPwd.Text = cred.OPERAPassword;
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
			new SySal.OperaDb.OperaDbDataAdapter("SELECT TB_PROC_OPERATIONS.ID FROM TB_PROC_OPERATIONS INNER JOIN TB_PROGRAMSETTINGS ON (TB_PROC_OPERATIONS.ID_PROGRAMSETTINGS = TB_PROGRAMSETTINGS.ID AND TB_PROGRAMSETTINGS.EXECUTABLE = 'AreaScan2Driver.exe') WHERE TB_PROC_OPERATIONS.ID IN (" + wherestr + ")", Conn, null).Fill(ds);
			foreach (System.Data.DataRow dr in ds.Tables[0].Rows)
				comboProcOpId.Items.Add(dr[0].ToString());
		}

		private void buttonCancel_Click(object sender, System.EventArgs e)
		{
			Close();
		}

		private void buttonSend_Click(object sender, System.EventArgs e)
		{
			string interruptdata = "";
			if (checkIgnoreScanFailure.Checked)
			{
				if (interruptdata.Length > 0) interruptdata += ",";
				interruptdata += "IgnoreScanFailure " + (radioIgnoreScanFailureTrue.Checked ? "True" : "False");
			}
			if (checkIgnoreRecalFailure.Checked)
			{
				if (interruptdata.Length > 0) interruptdata += ",";
				interruptdata += "IgnoreRecalFailure " + (radioIgnoreRecalFailureTrue.Checked ? "True" : "False");
			}
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
	}
}
