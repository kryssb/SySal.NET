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
using ZoneStatus;

namespace SySal.DAQSystem.Drivers.WideAreaScanDriver
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
        private System.Windows.Forms.GroupBox groupScanZones;
		private System.Windows.Forms.Label label3;
		private System.Windows.Forms.TextBox textOPERAUsername;
		private System.Windows.Forms.TextBox textOPERAPwd;
		private System.Windows.Forms.Label label4;
		private System.Windows.Forms.Button buttonCancel;
        private System.Windows.Forms.Button buttonSend;
        private DataGridView gridScanningAreas;
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
            this.groupScanZones = new System.Windows.Forms.GroupBox();
            this.gridScanningAreas = new System.Windows.Forms.DataGridView();
            this.label3 = new System.Windows.Forms.Label();
            this.textOPERAUsername = new System.Windows.Forms.TextBox();
            this.textOPERAPwd = new System.Windows.Forms.TextBox();
            this.label4 = new System.Windows.Forms.Label();
            this.buttonCancel = new System.Windows.Forms.Button();
            this.buttonSend = new System.Windows.Forms.Button();
            this.groupScanZones.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.gridScanningAreas)).BeginInit();
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
            // groupScanZones
            // 
            this.groupScanZones.Controls.Add(this.gridScanningAreas);
            this.groupScanZones.Location = new System.Drawing.Point(8, 64);
            this.groupScanZones.Name = "groupScanZones";
            this.groupScanZones.Size = new System.Drawing.Size(405, 187);
            this.groupScanZones.TabIndex = 4;
            this.groupScanZones.TabStop = false;
            this.groupScanZones.Text = "send ScanZoneFailure";
            // 
            // gridScanningAreas
            // 
            this.gridScanningAreas.ColumnHeadersHeightSizeMode = System.Windows.Forms.DataGridViewColumnHeadersHeightSizeMode.AutoSize;
            this.gridScanningAreas.Location = new System.Drawing.Point(61, 19);
            this.gridScanningAreas.Name = "gridScanningAreas";
            this.gridScanningAreas.Size = new System.Drawing.Size(240, 150);
            this.gridScanningAreas.TabIndex = 0;
            // 
            // label3
            // 
            this.label3.Location = new System.Drawing.Point(5, 257);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(104, 24);
            this.label3.TabIndex = 5;
            this.label3.Text = "OPERA Username";
            this.label3.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // textOPERAUsername
            // 
            this.textOPERAUsername.Location = new System.Drawing.Point(109, 257);
            this.textOPERAUsername.Name = "textOPERAUsername";
            this.textOPERAUsername.Size = new System.Drawing.Size(200, 20);
            this.textOPERAUsername.TabIndex = 6;
            // 
            // textOPERAPwd
            // 
            this.textOPERAPwd.Location = new System.Drawing.Point(109, 281);
            this.textOPERAPwd.Name = "textOPERAPwd";
            this.textOPERAPwd.PasswordChar = '*';
            this.textOPERAPwd.Size = new System.Drawing.Size(200, 20);
            this.textOPERAPwd.TabIndex = 8;
            // 
            // label4
            // 
            this.label4.Location = new System.Drawing.Point(5, 281);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(104, 24);
            this.label4.TabIndex = 7;
            this.label4.Text = "OPERA Password";
            this.label4.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // buttonCancel
            // 
            this.buttonCancel.Location = new System.Drawing.Point(5, 305);
            this.buttonCancel.Name = "buttonCancel";
            this.buttonCancel.Size = new System.Drawing.Size(64, 24);
            this.buttonCancel.TabIndex = 9;
            this.buttonCancel.Text = "Cancel";
            this.buttonCancel.Click += new System.EventHandler(this.buttonCancel_Click);
            // 
            // buttonSend
            // 
            this.buttonSend.Location = new System.Drawing.Point(333, 305);
            this.buttonSend.Name = "buttonSend";
            this.buttonSend.Size = new System.Drawing.Size(64, 24);
            this.buttonSend.TabIndex = 10;
            this.buttonSend.Text = "Send";
            this.buttonSend.Click += new System.EventHandler(this.buttonSend_Click);
            // 
            // frmEasyInterrupt
            // 
            this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
            this.ClientSize = new System.Drawing.Size(425, 333);
            this.Controls.Add(this.buttonSend);
            this.Controls.Add(this.buttonCancel);
            this.Controls.Add(this.textOPERAPwd);
            this.Controls.Add(this.textOPERAUsername);
            this.Controls.Add(this.label4);
            this.Controls.Add(this.label3);
            this.Controls.Add(this.groupScanZones);
            this.Controls.Add(this.comboProcOpId);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.comboBatchManager);
            this.Controls.Add(this.label1);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
            this.Name = "frmEasyInterrupt";
            this.Text = "Easy Interrupt for WideAreaScanDriver";
            this.Load += new System.EventHandler(this.OnLoad);
            this.groupScanZones.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.gridScanningAreas)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

		}
		#endregion

		SySal.OperaDb.OperaDbConnection Conn = null;
		SySal.DAQSystem.BatchManager BM = null;

		private void OnLoad(object sender, System.EventArgs e)
		{
			SySal.OperaDb.OperaDbCredentials cred = SySal.OperaDb.OperaDbCredentials.CreateFromRecord();
			System.Runtime.Remoting.Channels.ChannelServices.RegisterChannel(new TcpChannel(), false);
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
			new SySal.OperaDb.OperaDbDataAdapter("SELECT TB_PROC_OPERATIONS.ID FROM TB_PROC_OPERATIONS INNER JOIN TB_PROGRAMSETTINGS ON (TB_PROC_OPERATIONS.ID_PROGRAMSETTINGS = TB_PROGRAMSETTINGS.ID AND TB_PROGRAMSETTINGS.EXECUTABLE = 'WideAreaScanDriver.exe') WHERE TB_PROC_OPERATIONS.ID IN (" + wherestr + ")", Conn, null).Fill(ds);
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
            //if (checkIgnoreScanFailure.Checked)
            //{
            //    if (interruptdata.Length > 0) interruptdata += ",";
            //    interruptdata += "IgnoreScanFailure " + (radioIgnoreScanFailureTrue.Checked ? "True" : "False");
            //}
            //if (checkIgnoreRecalFailure.Checked)
            //{
            //    if (interruptdata.Length > 0) interruptdata += ",";
            //    interruptdata += "IgnoreRecalFailure " + (radioIgnoreRecalFailureTrue.Checked ? "True" : "False");
            //}
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

        public void FillGridScanningArea()
        {
            string RawDataDir = Convert.ToString(new SySal.OperaDb.OperaDbCommand("select value from LZ_SITEVARS where Name='RawDataDir'", Conn).ExecuteScalar());

            System.Data.DataSet ds = new System.Data.DataSet();
            new SySal.OperaDb.OperaDbDataAdapter("SELECT id_eventbrick, id_parent_operation FROM tb_proc_operations WHERE id = " + comboProcOpId.Text, Conn).Fill(ds);

            string idEventBrick = ds.Tables[0].Rows[0][0].ToString();
            string idParentOperation = ds.Tables[0].Rows[0][1].ToString();

            RawDataDir += "\\cssd_" + idEventBrick + "_" + idParentOperation;

            string searchPattern = "wideareascan_" + comboProcOpId.Text + "_" + idEventBrick + "_monitoring.xml";
            string[] files = System.IO.Directory.GetFiles(RawDataDir, searchPattern, System.IO.SearchOption.TopDirectoryOnly);

            StripLinkStatusInfo status = new StripLinkStatusInfo();
            foreach (string file in files)
            {
                status.Read(file);

                string[] row = { status.Attempts.ToString(), status.XBotShrink.ToString() };

                gridScanningAreas.Rows.Add(row);
            }
        }
    }
}
