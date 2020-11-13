using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;

namespace SySal.Executables.EasyProcess
{
	/// <summary>
	/// Shows detailed information about a process operation. It is a read-only form, whose fields can be copied to the clipboard, but cannot be changed.
	/// </summary>
	public class ProcOpInfo : System.Windows.Forms.Form
	{
		private System.Windows.Forms.Label label1;
		private System.Windows.Forms.TextBox IDText;
		private System.Windows.Forms.TextBox ParentIDText;
		private System.Windows.Forms.Label label2;
		private System.Windows.Forms.TextBox MachineText;
		private System.Windows.Forms.Label label3;
		private System.Windows.Forms.TextBox DescriptionText;
		private System.Windows.Forms.Label label4;
		private System.Windows.Forms.TextBox ExeText;
		private System.Windows.Forms.Label label5;
		private System.Windows.Forms.TextBox DriverLevelText;
		private System.Windows.Forms.Label label6;
		private System.Windows.Forms.TextBox BrickText;
		private System.Windows.Forms.Label label7;
		private System.Windows.Forms.TextBox PlateText;
		private System.Windows.Forms.Label label8;
		private System.Windows.Forms.Label label9;
		private System.Windows.Forms.TextBox StartTimeText;
		private System.Windows.Forms.Label label10;
		private System.Windows.Forms.TextBox FinishTimeText;
		private System.Windows.Forms.Label label11;
		private System.Windows.Forms.TextBox ResultText;
		private System.Windows.Forms.Label label12;
		private System.Windows.Forms.Button ExitButton;
		private System.Windows.Forms.GroupBox groupBox1;
		private System.Windows.Forms.TextBox RequesterText;
		private System.Windows.Forms.Label label13;
		private System.Windows.Forms.Label label14;
		private System.Windows.Forms.TextBox SettingsText;
		private System.Windows.Forms.TextBox AuthorText;
		private System.Windows.Forms.TextBox CalibrationText;
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;

		public ProcOpInfo(SySal.OperaDb.OperaDbCredentials cred, long procopid, ref SySal.OperaDb.OperaDbConnection Conn)
		{
			//
			// Required for Windows Form Designer support
			//
			InitializeComponent();

			//
			// TODO: Add any constructor code after InitializeComponent call
			//
			//SySal.OperaDb.OperaDbConnection Conn = null;
			try
			{
				if (Conn == null)
				{
					Conn = new SySal.OperaDb.OperaDbConnection(cred.DBServer, cred.DBUserName, cred.DBPassword);
					Conn.Open();
				}
				System.Data.DataSet dsprocop = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT /*+INDEX (TB_PROC_OPERATIONS PK_PROC_OPERATIONS) */ ID_MACHINE, ID_PARENT_OPERATION, ID_PROGRAMSETTINGS, DRIVERLEVEL, ID_EVENTBRICK, ID_PLATE, STARTTIME, FINISHTIME, SUCCESS, ID_REQUESTER, ID_CALIBRATION_OPERATION FROM TB_PROC_OPERATIONS WHERE ID = " + procopid, Conn, null).Fill(dsprocop);
				IDText.Text = procopid.ToString();
				ParentIDText.Text = dsprocop.Tables[0].Rows[0][1].ToString();
				DriverLevelText.Text = ((SySal.DAQSystem.Drivers.DriverType)Convert.ToInt32(dsprocop.Tables[0].Rows[0][3])).ToString() + " (" + dsprocop.Tables[0].Rows[0][3].ToString() + ")";
				BrickText.Text = dsprocop.Tables[0].Rows[0][4].ToString();
				PlateText.Text = dsprocop.Tables[0].Rows[0][5].ToString();
				StartTimeText.Text = Convert.ToDateTime(dsprocop.Tables[0].Rows[0][6]).ToString("dd/MM/yyyy HH:mm:ss");
				if (dsprocop.Tables[0].Rows[0][7] != System.DBNull.Value)
				{
					FinishTimeText.Text = Convert.ToDateTime(dsprocop.Tables[0].Rows[0][7]).ToString("dd/MM/yyyy HH:mm:ss");
					switch(dsprocop.Tables[0].Rows[0][8].ToString())
					{
						case "N": 	ResultText.Text = "Failure"; break;
						case "Y":   ResultText.Text = "Success"; break;
						case "R":	ResultText.Text = "Running"; break;
						default:	ResultText.Text = "Other (" + dsprocop.Tables[0].Rows[0][8].ToString() + ")"; break;
					}
				}
				MachineText.Text = new SySal.OperaDb.OperaDbCommand("SELECT /*+INDEX (TB_MACHINES PK_MACHINES) */ NAME FROM TB_MACHINES WHERE ID = " + dsprocop.Tables[0].Rows[0][0].ToString(), Conn, null).ExecuteScalar().ToString();
				RequesterText.Text = new SySal.OperaDb.OperaDbCommand("SELECT /*+INDEX (TB_USERS PK_USERS) */ NAME || ' ' || SURNAME FROM VW_USERS WHERE ID = " + dsprocop.Tables[0].Rows[0][9].ToString(), Conn, null).ExecuteScalar().ToString();
				CalibrationText.Text = (dsprocop.Tables[0].Rows[0][10] == System.DBNull.Value) ? "None" : dsprocop.Tables[0].Rows[0][10].ToString();
				System.Data.DataSet dsprog = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT /*+INDEX (TB_PROGRAMSETTINGS PK_PROGRAMSETTINGS) */ TEMPLATEMARKS, ID_AUTHOR, DESCRIPTION, EXECUTABLE, SETTINGS FROM TB_PROGRAMSETTINGS WHERE ID = " +  dsprocop.Tables[0].Rows[0][2].ToString(), Conn, null).Fill(dsprog);
				//CalibrationText.Text = (Convert.ToInt32(dsprog.Tables[0].Rows[0][0]) == 0) ? "No" : "Yes";
				AuthorText.Text = new SySal.OperaDb.OperaDbCommand("SELECT /*+INDEX (TB_USERS PK_USERS) */ NAME || ' ' || SURNAME FROM VW_USERS WHERE ID = " + dsprog.Tables[0].Rows[0][1].ToString(), Conn, null).ExecuteScalar().ToString();
				DescriptionText.Text = dsprog.Tables[0].Rows[0][2].ToString();
				ExeText.Text = dsprog.Tables[0].Rows[0][3].ToString();
				SettingsText.Text = dsprog.Tables[0].Rows[0][4].ToString();				
			}
			catch (Exception x)
			{
				if (Conn != null) 
				{
					Conn.Close();
					Conn = null;
				}
				MessageBox.Show(x.ToString(), x.Message, MessageBoxButtons.OK, MessageBoxIcon.Error);
			}			
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
			this.IDText = new System.Windows.Forms.TextBox();
			this.ParentIDText = new System.Windows.Forms.TextBox();
			this.label2 = new System.Windows.Forms.Label();
			this.MachineText = new System.Windows.Forms.TextBox();
			this.label3 = new System.Windows.Forms.Label();
			this.DescriptionText = new System.Windows.Forms.TextBox();
			this.label4 = new System.Windows.Forms.Label();
			this.ExeText = new System.Windows.Forms.TextBox();
			this.label5 = new System.Windows.Forms.Label();
			this.DriverLevelText = new System.Windows.Forms.TextBox();
			this.label6 = new System.Windows.Forms.Label();
			this.BrickText = new System.Windows.Forms.TextBox();
			this.label7 = new System.Windows.Forms.Label();
			this.PlateText = new System.Windows.Forms.TextBox();
			this.label8 = new System.Windows.Forms.Label();
			this.CalibrationText = new System.Windows.Forms.TextBox();
			this.label9 = new System.Windows.Forms.Label();
			this.StartTimeText = new System.Windows.Forms.TextBox();
			this.label10 = new System.Windows.Forms.Label();
			this.FinishTimeText = new System.Windows.Forms.TextBox();
			this.label11 = new System.Windows.Forms.Label();
			this.ResultText = new System.Windows.Forms.TextBox();
			this.label12 = new System.Windows.Forms.Label();
			this.ExitButton = new System.Windows.Forms.Button();
			this.groupBox1 = new System.Windows.Forms.GroupBox();
			this.SettingsText = new System.Windows.Forms.TextBox();
			this.AuthorText = new System.Windows.Forms.TextBox();
			this.label14 = new System.Windows.Forms.Label();
			this.RequesterText = new System.Windows.Forms.TextBox();
			this.label13 = new System.Windows.Forms.Label();
			this.groupBox1.SuspendLayout();
			this.SuspendLayout();
			// 
			// label1
			// 
			this.label1.Location = new System.Drawing.Point(8, 8);
			this.label1.Name = "label1";
			this.label1.Size = new System.Drawing.Size(112, 24);
			this.label1.TabIndex = 0;
			this.label1.Text = "ID #";
			// 
			// IDText
			// 
			this.IDText.BackColor = System.Drawing.SystemColors.Info;
			this.IDText.ForeColor = System.Drawing.SystemColors.Highlight;
			this.IDText.Location = new System.Drawing.Point(144, 8);
			this.IDText.Name = "IDText";
			this.IDText.ReadOnly = true;
			this.IDText.Size = new System.Drawing.Size(128, 20);
			this.IDText.TabIndex = 1;
			this.IDText.Text = "";
			this.IDText.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
			// 
			// ParentIDText
			// 
			this.ParentIDText.BackColor = System.Drawing.SystemColors.Info;
			this.ParentIDText.ForeColor = System.Drawing.SystemColors.Highlight;
			this.ParentIDText.Location = new System.Drawing.Point(144, 32);
			this.ParentIDText.Name = "ParentIDText";
			this.ParentIDText.ReadOnly = true;
			this.ParentIDText.Size = new System.Drawing.Size(128, 20);
			this.ParentIDText.TabIndex = 3;
			this.ParentIDText.Text = "";
			this.ParentIDText.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
			// 
			// label2
			// 
			this.label2.Location = new System.Drawing.Point(8, 32);
			this.label2.Name = "label2";
			this.label2.Size = new System.Drawing.Size(112, 24);
			this.label2.TabIndex = 2;
			this.label2.Text = "Parent ID #";
			// 
			// MachineText
			// 
			this.MachineText.BackColor = System.Drawing.SystemColors.Info;
			this.MachineText.ForeColor = System.Drawing.SystemColors.Highlight;
			this.MachineText.Location = new System.Drawing.Point(144, 56);
			this.MachineText.Name = "MachineText";
			this.MachineText.ReadOnly = true;
			this.MachineText.Size = new System.Drawing.Size(128, 20);
			this.MachineText.TabIndex = 5;
			this.MachineText.Text = "";
			this.MachineText.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
			// 
			// label3
			// 
			this.label3.Location = new System.Drawing.Point(8, 56);
			this.label3.Name = "label3";
			this.label3.Size = new System.Drawing.Size(112, 24);
			this.label3.TabIndex = 4;
			this.label3.Text = "Machine";
			// 
			// DescriptionText
			// 
			this.DescriptionText.BackColor = System.Drawing.SystemColors.Info;
			this.DescriptionText.ForeColor = System.Drawing.SystemColors.Highlight;
			this.DescriptionText.Location = new System.Drawing.Point(144, 104);
			this.DescriptionText.Name = "DescriptionText";
			this.DescriptionText.ReadOnly = true;
			this.DescriptionText.Size = new System.Drawing.Size(128, 20);
			this.DescriptionText.TabIndex = 9;
			this.DescriptionText.Text = "";
			this.DescriptionText.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
			// 
			// label4
			// 
			this.label4.Location = new System.Drawing.Point(8, 104);
			this.label4.Name = "label4";
			this.label4.Size = new System.Drawing.Size(112, 24);
			this.label4.TabIndex = 8;
			this.label4.Text = "Task description";
			// 
			// ExeText
			// 
			this.ExeText.BackColor = System.Drawing.SystemColors.Info;
			this.ExeText.ForeColor = System.Drawing.SystemColors.Highlight;
			this.ExeText.Location = new System.Drawing.Point(144, 128);
			this.ExeText.Name = "ExeText";
			this.ExeText.ReadOnly = true;
			this.ExeText.Size = new System.Drawing.Size(128, 20);
			this.ExeText.TabIndex = 11;
			this.ExeText.Text = "";
			this.ExeText.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
			// 
			// label5
			// 
			this.label5.Location = new System.Drawing.Point(8, 128);
			this.label5.Name = "label5";
			this.label5.Size = new System.Drawing.Size(112, 24);
			this.label5.TabIndex = 10;
			this.label5.Text = "Task executable";
			// 
			// DriverLevelText
			// 
			this.DriverLevelText.BackColor = System.Drawing.SystemColors.Info;
			this.DriverLevelText.ForeColor = System.Drawing.SystemColors.Highlight;
			this.DriverLevelText.Location = new System.Drawing.Point(144, 152);
			this.DriverLevelText.Name = "DriverLevelText";
			this.DriverLevelText.ReadOnly = true;
			this.DriverLevelText.Size = new System.Drawing.Size(128, 20);
			this.DriverLevelText.TabIndex = 13;
			this.DriverLevelText.Text = "";
			this.DriverLevelText.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
			// 
			// label6
			// 
			this.label6.Location = new System.Drawing.Point(8, 152);
			this.label6.Name = "label6";
			this.label6.Size = new System.Drawing.Size(112, 24);
			this.label6.TabIndex = 12;
			this.label6.Text = "Driver level";
			// 
			// BrickText
			// 
			this.BrickText.BackColor = System.Drawing.SystemColors.Info;
			this.BrickText.ForeColor = System.Drawing.SystemColors.Highlight;
			this.BrickText.Location = new System.Drawing.Point(144, 176);
			this.BrickText.Name = "BrickText";
			this.BrickText.ReadOnly = true;
			this.BrickText.Size = new System.Drawing.Size(128, 20);
			this.BrickText.TabIndex = 15;
			this.BrickText.Text = "";
			this.BrickText.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
			// 
			// label7
			// 
			this.label7.Location = new System.Drawing.Point(8, 176);
			this.label7.Name = "label7";
			this.label7.Size = new System.Drawing.Size(112, 24);
			this.label7.TabIndex = 14;
			this.label7.Text = "Brick #";
			// 
			// PlateText
			// 
			this.PlateText.BackColor = System.Drawing.SystemColors.Info;
			this.PlateText.ForeColor = System.Drawing.SystemColors.Highlight;
			this.PlateText.Location = new System.Drawing.Point(144, 200);
			this.PlateText.Name = "PlateText";
			this.PlateText.ReadOnly = true;
			this.PlateText.Size = new System.Drawing.Size(128, 20);
			this.PlateText.TabIndex = 17;
			this.PlateText.Text = "";
			this.PlateText.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
			// 
			// label8
			// 
			this.label8.Location = new System.Drawing.Point(8, 200);
			this.label8.Name = "label8";
			this.label8.Size = new System.Drawing.Size(112, 24);
			this.label8.TabIndex = 16;
			this.label8.Text = "Plate #";
			// 
			// CalibrationText
			// 
			this.CalibrationText.BackColor = System.Drawing.SystemColors.Info;
			this.CalibrationText.ForeColor = System.Drawing.SystemColors.Highlight;
			this.CalibrationText.Location = new System.Drawing.Point(144, 224);
			this.CalibrationText.Name = "CalibrationText";
			this.CalibrationText.ReadOnly = true;
			this.CalibrationText.Size = new System.Drawing.Size(128, 20);
			this.CalibrationText.TabIndex = 19;
			this.CalibrationText.Text = "";
			this.CalibrationText.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
			// 
			// label9
			// 
			this.label9.Location = new System.Drawing.Point(8, 224);
			this.label9.Name = "label9";
			this.label9.Size = new System.Drawing.Size(128, 24);
			this.label9.TabIndex = 18;
			this.label9.Text = "Calibration";
			// 
			// StartTimeText
			// 
			this.StartTimeText.BackColor = System.Drawing.SystemColors.Info;
			this.StartTimeText.ForeColor = System.Drawing.SystemColors.Highlight;
			this.StartTimeText.Location = new System.Drawing.Point(144, 248);
			this.StartTimeText.Name = "StartTimeText";
			this.StartTimeText.ReadOnly = true;
			this.StartTimeText.Size = new System.Drawing.Size(128, 20);
			this.StartTimeText.TabIndex = 21;
			this.StartTimeText.Text = "";
			this.StartTimeText.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
			// 
			// label10
			// 
			this.label10.Location = new System.Drawing.Point(8, 248);
			this.label10.Name = "label10";
			this.label10.Size = new System.Drawing.Size(112, 24);
			this.label10.TabIndex = 20;
			this.label10.Text = "Started on";
			// 
			// FinishTimeText
			// 
			this.FinishTimeText.BackColor = System.Drawing.SystemColors.Info;
			this.FinishTimeText.ForeColor = System.Drawing.SystemColors.Highlight;
			this.FinishTimeText.Location = new System.Drawing.Point(144, 272);
			this.FinishTimeText.Name = "FinishTimeText";
			this.FinishTimeText.ReadOnly = true;
			this.FinishTimeText.Size = new System.Drawing.Size(128, 20);
			this.FinishTimeText.TabIndex = 23;
			this.FinishTimeText.Text = "";
			this.FinishTimeText.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
			// 
			// label11
			// 
			this.label11.Location = new System.Drawing.Point(8, 272);
			this.label11.Name = "label11";
			this.label11.Size = new System.Drawing.Size(112, 24);
			this.label11.TabIndex = 22;
			this.label11.Text = "Completed on";
			// 
			// ResultText
			// 
			this.ResultText.BackColor = System.Drawing.SystemColors.Info;
			this.ResultText.ForeColor = System.Drawing.SystemColors.Highlight;
			this.ResultText.Location = new System.Drawing.Point(144, 296);
			this.ResultText.Name = "ResultText";
			this.ResultText.ReadOnly = true;
			this.ResultText.Size = new System.Drawing.Size(128, 20);
			this.ResultText.TabIndex = 25;
			this.ResultText.Text = "";
			this.ResultText.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
			// 
			// label12
			// 
			this.label12.Location = new System.Drawing.Point(8, 296);
			this.label12.Name = "label12";
			this.label12.Size = new System.Drawing.Size(112, 24);
			this.label12.TabIndex = 24;
			this.label12.Text = "Result";
			// 
			// ExitButton
			// 
			this.ExitButton.DialogResult = System.Windows.Forms.DialogResult.Cancel;
			this.ExitButton.Location = new System.Drawing.Point(293, 336);
			this.ExitButton.Name = "ExitButton";
			this.ExitButton.Size = new System.Drawing.Size(48, 24);
			this.ExitButton.TabIndex = 30;
			this.ExitButton.Text = "Exit";
			// 
			// groupBox1
			// 
			this.groupBox1.Controls.Add(this.SettingsText);
			this.groupBox1.Controls.Add(this.AuthorText);
			this.groupBox1.Controls.Add(this.label14);
			this.groupBox1.Location = new System.Drawing.Point(280, 0);
			this.groupBox1.Name = "groupBox1";
			this.groupBox1.Size = new System.Drawing.Size(344, 320);
			this.groupBox1.TabIndex = 26;
			this.groupBox1.TabStop = false;
			this.groupBox1.Text = "Program settings";
			// 
			// SettingsText
			// 
			this.SettingsText.BackColor = System.Drawing.SystemColors.Info;
			this.SettingsText.ForeColor = System.Drawing.SystemColors.Highlight;
			this.SettingsText.Location = new System.Drawing.Point(8, 56);
			this.SettingsText.Multiline = true;
			this.SettingsText.Name = "SettingsText";
			this.SettingsText.ReadOnly = true;
			this.SettingsText.ScrollBars = System.Windows.Forms.ScrollBars.Both;
			this.SettingsText.Size = new System.Drawing.Size(328, 256);
			this.SettingsText.TabIndex = 29;
			this.SettingsText.Text = "";
			// 
			// AuthorText
			// 
			this.AuthorText.BackColor = System.Drawing.SystemColors.Info;
			this.AuthorText.ForeColor = System.Drawing.SystemColors.Highlight;
			this.AuthorText.Location = new System.Drawing.Point(208, 24);
			this.AuthorText.Name = "AuthorText";
			this.AuthorText.ReadOnly = true;
			this.AuthorText.Size = new System.Drawing.Size(128, 20);
			this.AuthorText.TabIndex = 28;
			this.AuthorText.Text = "";
			this.AuthorText.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
			// 
			// label14
			// 
			this.label14.Location = new System.Drawing.Point(8, 24);
			this.label14.Name = "label14";
			this.label14.Size = new System.Drawing.Size(112, 24);
			this.label14.TabIndex = 27;
			this.label14.Text = "Author";
			// 
			// RequesterText
			// 
			this.RequesterText.BackColor = System.Drawing.SystemColors.Info;
			this.RequesterText.ForeColor = System.Drawing.SystemColors.Highlight;
			this.RequesterText.Location = new System.Drawing.Point(144, 80);
			this.RequesterText.Name = "RequesterText";
			this.RequesterText.ReadOnly = true;
			this.RequesterText.Size = new System.Drawing.Size(128, 20);
			this.RequesterText.TabIndex = 7;
			this.RequesterText.Text = "";
			this.RequesterText.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
			// 
			// label13
			// 
			this.label13.Location = new System.Drawing.Point(8, 80);
			this.label13.Name = "label13";
			this.label13.Size = new System.Drawing.Size(112, 24);
			this.label13.TabIndex = 6;
			this.label13.Text = "Requester";
			// 
			// ProcOpInfo
			// 
			this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
			this.CancelButton = this.ExitButton;
			this.ClientSize = new System.Drawing.Size(634, 368);
			this.Controls.Add(this.RequesterText);
			this.Controls.Add(this.label13);
			this.Controls.Add(this.groupBox1);
			this.Controls.Add(this.ExitButton);
			this.Controls.Add(this.ResultText);
			this.Controls.Add(this.label12);
			this.Controls.Add(this.FinishTimeText);
			this.Controls.Add(this.label11);
			this.Controls.Add(this.StartTimeText);
			this.Controls.Add(this.label10);
			this.Controls.Add(this.CalibrationText);
			this.Controls.Add(this.label9);
			this.Controls.Add(this.PlateText);
			this.Controls.Add(this.label8);
			this.Controls.Add(this.BrickText);
			this.Controls.Add(this.label7);
			this.Controls.Add(this.DriverLevelText);
			this.Controls.Add(this.label6);
			this.Controls.Add(this.ExeText);
			this.Controls.Add(this.label5);
			this.Controls.Add(this.DescriptionText);
			this.Controls.Add(this.label4);
			this.Controls.Add(this.MachineText);
			this.Controls.Add(this.label3);
			this.Controls.Add(this.ParentIDText);
			this.Controls.Add(this.label2);
			this.Controls.Add(this.IDText);
			this.Controls.Add(this.label1);
			this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedToolWindow;
			this.MaximizeBox = false;
			this.Name = "ProcOpInfo";
			this.ShowInTaskbar = false;
			this.Text = "Process Operation Information";
			this.groupBox1.ResumeLayout(false);
			this.ResumeLayout(false);

		}
		#endregion
	}
}
