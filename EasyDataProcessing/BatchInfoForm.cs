using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;

namespace SySal.Executables.EasyDataProcessing
{
	/// <summary>
	/// Summary description for BatchInfoForm.
	/// </summary>
	internal class BatchInfoForm : System.Windows.Forms.Form
	{
		private System.Windows.Forms.Label label1;
		private System.Windows.Forms.TextBox IdText;
		private System.Windows.Forms.TextBox OwnerText;
		private System.Windows.Forms.Label label2;
		private System.Windows.Forms.TextBox ExePathText;
		private System.Windows.Forms.Label label3;
		private System.Windows.Forms.TextBox CommandLineArgsText;
		private System.Windows.Forms.Label label4;
		private System.Windows.Forms.TextBox DescriptionText;
		private System.Windows.Forms.Label label5;
		private System.Windows.Forms.TextBox StartText;
		private System.Windows.Forms.Label label6;
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;

		public BatchInfoForm(SySal.DAQSystem.DataProcessingBatchDesc d)
		{
			//
			// Required for Windows Form Designer support
			//
			InitializeComponent();

			//
			// TODO: Add any constructor code after InitializeComponent call
			//
			IdText.Text = d.Id.ToString("X16");
			OwnerText.Text = d.Username;
			ExePathText.Text = d.Filename;
			CommandLineArgsText.Text = d.CommandLineArguments;
			DescriptionText.Text = d.Description;
			StartText.Text = d.Started.ToString();
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
			this.IdText = new System.Windows.Forms.TextBox();
			this.OwnerText = new System.Windows.Forms.TextBox();
			this.label2 = new System.Windows.Forms.Label();
			this.ExePathText = new System.Windows.Forms.TextBox();
			this.label3 = new System.Windows.Forms.Label();
			this.CommandLineArgsText = new System.Windows.Forms.TextBox();
			this.label4 = new System.Windows.Forms.Label();
			this.DescriptionText = new System.Windows.Forms.TextBox();
			this.label5 = new System.Windows.Forms.Label();
			this.StartText = new System.Windows.Forms.TextBox();
			this.label6 = new System.Windows.Forms.Label();
			this.SuspendLayout();
			// 
			// label1
			// 
			this.label1.Location = new System.Drawing.Point(8, 8);
			this.label1.Name = "label1";
			this.label1.Size = new System.Drawing.Size(32, 24);
			this.label1.TabIndex = 0;
			this.label1.Text = "Id";
			// 
			// IdText
			// 
			this.IdText.Location = new System.Drawing.Point(176, 8);
			this.IdText.Name = "IdText";
			this.IdText.ReadOnly = true;
			this.IdText.Size = new System.Drawing.Size(312, 20);
			this.IdText.TabIndex = 1;
			this.IdText.Text = "";
			// 
			// OwnerText
			// 
			this.OwnerText.Location = new System.Drawing.Point(176, 40);
			this.OwnerText.Name = "OwnerText";
			this.OwnerText.ReadOnly = true;
			this.OwnerText.Size = new System.Drawing.Size(312, 20);
			this.OwnerText.TabIndex = 3;
			this.OwnerText.Text = "";
			// 
			// label2
			// 
			this.label2.Location = new System.Drawing.Point(8, 40);
			this.label2.Name = "label2";
			this.label2.Size = new System.Drawing.Size(152, 24);
			this.label2.TabIndex = 2;
			this.label2.Text = "Owner";
			// 
			// ExePathText
			// 
			this.ExePathText.Location = new System.Drawing.Point(176, 72);
			this.ExePathText.Name = "ExePathText";
			this.ExePathText.ReadOnly = true;
			this.ExePathText.Size = new System.Drawing.Size(312, 20);
			this.ExePathText.TabIndex = 5;
			this.ExePathText.Text = "";
			// 
			// label3
			// 
			this.label3.Location = new System.Drawing.Point(8, 72);
			this.label3.Name = "label3";
			this.label3.Size = new System.Drawing.Size(152, 24);
			this.label3.TabIndex = 4;
			this.label3.Text = "Exe Path";
			// 
			// CommandLineArgsText
			// 
			this.CommandLineArgsText.Location = new System.Drawing.Point(176, 104);
			this.CommandLineArgsText.Name = "CommandLineArgsText";
			this.CommandLineArgsText.ReadOnly = true;
			this.CommandLineArgsText.Size = new System.Drawing.Size(312, 20);
			this.CommandLineArgsText.TabIndex = 7;
			this.CommandLineArgsText.Text = "";
			// 
			// label4
			// 
			this.label4.Location = new System.Drawing.Point(8, 104);
			this.label4.Name = "label4";
			this.label4.Size = new System.Drawing.Size(152, 24);
			this.label4.TabIndex = 6;
			this.label4.Text = "Command Line Arguments";
			// 
			// DescriptionText
			// 
			this.DescriptionText.Location = new System.Drawing.Point(177, 136);
			this.DescriptionText.Name = "DescriptionText";
			this.DescriptionText.ReadOnly = true;
			this.DescriptionText.Size = new System.Drawing.Size(312, 20);
			this.DescriptionText.TabIndex = 9;
			this.DescriptionText.Text = "";
			// 
			// label5
			// 
			this.label5.Location = new System.Drawing.Point(9, 136);
			this.label5.Name = "label5";
			this.label5.Size = new System.Drawing.Size(152, 24);
			this.label5.TabIndex = 8;
			this.label5.Text = "Description";
			// 
			// StartText
			// 
			this.StartText.Location = new System.Drawing.Point(176, 168);
			this.StartText.Name = "StartText";
			this.StartText.ReadOnly = true;
			this.StartText.Size = new System.Drawing.Size(312, 20);
			this.StartText.TabIndex = 11;
			this.StartText.Text = "";
			// 
			// label6
			// 
			this.label6.Location = new System.Drawing.Point(8, 168);
			this.label6.Name = "label6";
			this.label6.Size = new System.Drawing.Size(152, 24);
			this.label6.TabIndex = 10;
			this.label6.Text = "Start time";
			// 
			// BatchInfoForm
			// 
			this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
			this.ClientSize = new System.Drawing.Size(498, 208);
			this.Controls.AddRange(new System.Windows.Forms.Control[] {
																		  this.StartText,
																		  this.label6,
																		  this.DescriptionText,
																		  this.label5,
																		  this.CommandLineArgsText,
																		  this.label4,
																		  this.ExePathText,
																		  this.label3,
																		  this.OwnerText,
																		  this.label2,
																		  this.IdText,
																		  this.label1});
			this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedToolWindow;
			this.Name = "BatchInfoForm";
			this.Text = "Batch Information";
			this.ResumeLayout(false);

		}
		#endregion
	}
}
