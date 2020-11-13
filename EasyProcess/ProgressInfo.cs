using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;

namespace SySal.Executables.EasyProcess
{
	/// <summary>
	/// This class is not used for the moment.
	/// </summary>
	internal class ProgressInfo : System.Windows.Forms.Form
	{
		private System.Windows.Forms.TextBox IDText;
		private System.Windows.Forms.Label label1;
		private System.Windows.Forms.TextBox StartTimeText;
		private System.Windows.Forms.Label label4;
		private System.Windows.Forms.TextBox EndTimeText;
		private System.Windows.Forms.Label label5;
		private System.Windows.Forms.TextBox ProgressText;
		private System.Windows.Forms.Label label6;
		private System.Windows.Forms.TextBox CompletedText;
		private System.Windows.Forms.Label label7;
		private System.Windows.Forms.TextBox ExitExceptionText;
		private System.Windows.Forms.Button ExitButton;
		private System.Windows.Forms.GroupBox groupBox1;
		private System.Windows.Forms.GroupBox groupBox2;
		private System.Windows.Forms.TextBox CustomInfoText;
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;

		public ProgressInfo(long id, SySal.DAQSystem.Drivers.TaskProgressInfo progrinfo)
		{
			//
			// Required for Windows Form Designer support
			//
			InitializeComponent();

			//
			// TODO: Add any constructor code after InitializeComponent call
			//
			IDText.Text = id.ToString();
			StartTimeText.Text = progrinfo.StartTime.ToShortTimeString();
			EndTimeText.Text = progrinfo.FinishTime.ToShortTimeString();
			ProgressText.Text = progrinfo.Progress.ToString(System.Globalization.CultureInfo.InvariantCulture);
			CompletedText.Text = progrinfo.Complete ? "Yes" : "No";
			ExitExceptionText.Text = progrinfo.ExitException;
			CustomInfoText.Text = (progrinfo.CustomInfo == null) ? "" : progrinfo.CustomInfo;
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
			this.IDText = new System.Windows.Forms.TextBox();
			this.label1 = new System.Windows.Forms.Label();
			this.StartTimeText = new System.Windows.Forms.TextBox();
			this.label4 = new System.Windows.Forms.Label();
			this.EndTimeText = new System.Windows.Forms.TextBox();
			this.label5 = new System.Windows.Forms.Label();
			this.ProgressText = new System.Windows.Forms.TextBox();
			this.label6 = new System.Windows.Forms.Label();
			this.CompletedText = new System.Windows.Forms.TextBox();
			this.label7 = new System.Windows.Forms.Label();
			this.ExitExceptionText = new System.Windows.Forms.TextBox();
			this.ExitButton = new System.Windows.Forms.Button();
			this.groupBox1 = new System.Windows.Forms.GroupBox();
			this.groupBox2 = new System.Windows.Forms.GroupBox();
			this.CustomInfoText = new System.Windows.Forms.TextBox();
			this.groupBox2.SuspendLayout();
			this.SuspendLayout();
			// 
			// IDText
			// 
			this.IDText.BackColor = System.Drawing.SystemColors.Info;
			this.IDText.ForeColor = System.Drawing.SystemColors.Highlight;
			this.IDText.Location = new System.Drawing.Point(184, 8);
			this.IDText.Name = "IDText";
			this.IDText.ReadOnly = true;
			this.IDText.Size = new System.Drawing.Size(128, 20);
			this.IDText.TabIndex = 3;
			this.IDText.Text = "";
			this.IDText.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
			// 
			// label1
			// 
			this.label1.Location = new System.Drawing.Point(8, 8);
			this.label1.Name = "label1";
			this.label1.Size = new System.Drawing.Size(112, 24);
			this.label1.TabIndex = 2;
			this.label1.Text = "ID #";
			// 
			// StartTimeText
			// 
			this.StartTimeText.BackColor = System.Drawing.SystemColors.Info;
			this.StartTimeText.ForeColor = System.Drawing.SystemColors.Highlight;
			this.StartTimeText.Location = new System.Drawing.Point(184, 32);
			this.StartTimeText.Name = "StartTimeText";
			this.StartTimeText.ReadOnly = true;
			this.StartTimeText.Size = new System.Drawing.Size(128, 20);
			this.StartTimeText.TabIndex = 9;
			this.StartTimeText.Text = "";
			this.StartTimeText.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
			// 
			// label4
			// 
			this.label4.Location = new System.Drawing.Point(8, 32);
			this.label4.Name = "label4";
			this.label4.Size = new System.Drawing.Size(112, 24);
			this.label4.TabIndex = 8;
			this.label4.Text = "Start time";
			this.label4.Click += new System.EventHandler(this.label4_Click);
			// 
			// EndTimeText
			// 
			this.EndTimeText.BackColor = System.Drawing.SystemColors.Info;
			this.EndTimeText.ForeColor = System.Drawing.SystemColors.Highlight;
			this.EndTimeText.Location = new System.Drawing.Point(184, 56);
			this.EndTimeText.Name = "EndTimeText";
			this.EndTimeText.ReadOnly = true;
			this.EndTimeText.Size = new System.Drawing.Size(128, 20);
			this.EndTimeText.TabIndex = 11;
			this.EndTimeText.Text = "";
			this.EndTimeText.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
			// 
			// label5
			// 
			this.label5.Location = new System.Drawing.Point(8, 56);
			this.label5.Name = "label5";
			this.label5.Size = new System.Drawing.Size(112, 24);
			this.label5.TabIndex = 10;
			this.label5.Text = "Expected end time";
			// 
			// ProgressText
			// 
			this.ProgressText.BackColor = System.Drawing.SystemColors.Info;
			this.ProgressText.ForeColor = System.Drawing.SystemColors.Highlight;
			this.ProgressText.Location = new System.Drawing.Point(184, 80);
			this.ProgressText.Name = "ProgressText";
			this.ProgressText.ReadOnly = true;
			this.ProgressText.Size = new System.Drawing.Size(128, 20);
			this.ProgressText.TabIndex = 13;
			this.ProgressText.Text = "";
			this.ProgressText.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
			// 
			// label6
			// 
			this.label6.Location = new System.Drawing.Point(8, 80);
			this.label6.Name = "label6";
			this.label6.Size = new System.Drawing.Size(112, 24);
			this.label6.TabIndex = 12;
			this.label6.Text = "Progress";
			// 
			// CompletedText
			// 
			this.CompletedText.BackColor = System.Drawing.SystemColors.Info;
			this.CompletedText.ForeColor = System.Drawing.SystemColors.Highlight;
			this.CompletedText.Location = new System.Drawing.Point(184, 104);
			this.CompletedText.Name = "CompletedText";
			this.CompletedText.ReadOnly = true;
			this.CompletedText.Size = new System.Drawing.Size(128, 20);
			this.CompletedText.TabIndex = 15;
			this.CompletedText.Text = "";
			this.CompletedText.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
			// 
			// label7
			// 
			this.label7.Location = new System.Drawing.Point(8, 104);
			this.label7.Name = "label7";
			this.label7.Size = new System.Drawing.Size(112, 24);
			this.label7.TabIndex = 14;
			this.label7.Text = "Completed";
			// 
			// ExitExceptionText
			// 
			this.ExitExceptionText.BackColor = System.Drawing.SystemColors.Info;
			this.ExitExceptionText.ForeColor = System.Drawing.SystemColors.Highlight;
			this.ExitExceptionText.Location = new System.Drawing.Point(16, 144);
			this.ExitExceptionText.Multiline = true;
			this.ExitExceptionText.Name = "ExitExceptionText";
			this.ExitExceptionText.ReadOnly = true;
			this.ExitExceptionText.ScrollBars = System.Windows.Forms.ScrollBars.Both;
			this.ExitExceptionText.Size = new System.Drawing.Size(288, 232);
			this.ExitExceptionText.TabIndex = 17;
			this.ExitExceptionText.Text = "";
			this.ExitExceptionText.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
			// 
			// ExitButton
			// 
			this.ExitButton.DialogResult = System.Windows.Forms.DialogResult.Cancel;
			this.ExitButton.Location = new System.Drawing.Point(333, 392);
			this.ExitButton.Name = "ExitButton";
			this.ExitButton.Size = new System.Drawing.Size(48, 24);
			this.ExitButton.TabIndex = 31;
			this.ExitButton.Text = "Exit";
			// 
			// groupBox1
			// 
			this.groupBox1.Location = new System.Drawing.Point(8, 128);
			this.groupBox1.Name = "groupBox1";
			this.groupBox1.Size = new System.Drawing.Size(304, 256);
			this.groupBox1.TabIndex = 32;
			this.groupBox1.TabStop = false;
			this.groupBox1.Text = "Exit exception";
			// 
			// groupBox2
			// 
			this.groupBox2.Controls.AddRange(new System.Windows.Forms.Control[] {
																					this.CustomInfoText});
			this.groupBox2.Location = new System.Drawing.Point(320, 0);
			this.groupBox2.Name = "groupBox2";
			this.groupBox2.Size = new System.Drawing.Size(384, 384);
			this.groupBox2.TabIndex = 33;
			this.groupBox2.TabStop = false;
			this.groupBox2.Text = "Custom information";
			// 
			// CustomInfoText
			// 
			this.CustomInfoText.BackColor = System.Drawing.SystemColors.Info;
			this.CustomInfoText.ForeColor = System.Drawing.SystemColors.Highlight;
			this.CustomInfoText.Location = new System.Drawing.Point(8, 16);
			this.CustomInfoText.Multiline = true;
			this.CustomInfoText.Name = "CustomInfoText";
			this.CustomInfoText.ReadOnly = true;
			this.CustomInfoText.ScrollBars = System.Windows.Forms.ScrollBars.Both;
			this.CustomInfoText.Size = new System.Drawing.Size(368, 360);
			this.CustomInfoText.TabIndex = 18;
			this.CustomInfoText.Text = "";
			this.CustomInfoText.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
			// 
			// ProgressInfo
			// 
			this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
			this.ClientSize = new System.Drawing.Size(714, 424);
			this.Controls.AddRange(new System.Windows.Forms.Control[] {
																		  this.groupBox2,
																		  this.ExitButton,
																		  this.ExitExceptionText,
																		  this.CompletedText,
																		  this.label7,
																		  this.ProgressText,
																		  this.label6,
																		  this.EndTimeText,
																		  this.label5,
																		  this.StartTimeText,
																		  this.label4,
																		  this.IDText,
																		  this.label1,
																		  this.groupBox1});
			this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedToolWindow;
			this.Name = "ProgressInfo";
			this.Text = "Progress Information";
			this.groupBox2.ResumeLayout(false);
			this.ResumeLayout(false);

		}
		#endregion

		private void label4_Click(object sender, System.EventArgs e)
		{
		
		}
	}
}
