using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;
using SySal.Scanning.PostProcessing.PatternMatching;

namespace SySal.Processing.QuickMapping
{
	/// <summary>
	/// This form allows GUI-assisted configuration editing for QuickMapping.
	/// </summary>
	class EditConfigForm : System.Windows.Forms.Form
	{
		private System.Windows.Forms.Label label1;
		private System.Windows.Forms.TextBox SlopeTol;
		private System.Windows.Forms.TextBox PosTol;
		private System.Windows.Forms.Label label2;
		private System.Windows.Forms.Button OK;
		private System.Windows.Forms.Button Cancel;
		private System.Windows.Forms.GroupBox groupBox1;
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;

		public Configuration C;

		public EditConfigForm()
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
			this.SlopeTol = new System.Windows.Forms.TextBox();
			this.PosTol = new System.Windows.Forms.TextBox();
			this.label2 = new System.Windows.Forms.Label();
			this.OK = new System.Windows.Forms.Button();
			this.Cancel = new System.Windows.Forms.Button();
			this.groupBox1 = new System.Windows.Forms.GroupBox();
			this.SuspendLayout();
			// 
			// label1
			// 
			this.label1.Location = new System.Drawing.Point(32, 32);
			this.label1.Name = "label1";
			this.label1.Size = new System.Drawing.Size(128, 24);
			this.label1.TabIndex = 0;
			this.label1.Text = "Slope Tolerance";
			this.label1.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
			// 
			// SlopeTol
			// 
			this.SlopeTol.Location = new System.Drawing.Point(224, 32);
			this.SlopeTol.Name = "SlopeTol";
			this.SlopeTol.Size = new System.Drawing.Size(72, 20);
			this.SlopeTol.TabIndex = 1;
			this.SlopeTol.Text = "0.02";
			this.SlopeTol.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
			// 
			// PosTol
			// 
			this.PosTol.Location = new System.Drawing.Point(224, 64);
			this.PosTol.Name = "PosTol";
			this.PosTol.Size = new System.Drawing.Size(72, 20);
			this.PosTol.TabIndex = 3;
			this.PosTol.Text = "20";
			this.PosTol.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
			// 
			// label2
			// 
			this.label2.Location = new System.Drawing.Point(32, 64);
			this.label2.Name = "label2";
			this.label2.Size = new System.Drawing.Size(128, 24);
			this.label2.TabIndex = 2;
			this.label2.Text = "Position Tolerance";
			this.label2.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
			// 
			// OK
			// 
			this.OK.Location = new System.Drawing.Point(16, 112);
			this.OK.Name = "OK";
			this.OK.Size = new System.Drawing.Size(96, 24);
			this.OK.TabIndex = 4;
			this.OK.Text = "&OK";
			this.OK.Click += new System.EventHandler(this.OK_Click);
			// 
			// Cancel
			// 
			this.Cancel.Location = new System.Drawing.Point(216, 112);
			this.Cancel.Name = "Cancel";
			this.Cancel.Size = new System.Drawing.Size(96, 24);
			this.Cancel.TabIndex = 5;
			this.Cancel.Text = "&Cancel";
			this.Cancel.Click += new System.EventHandler(this.Cancel_Click);
			// 
			// groupBox1
			// 
			this.groupBox1.Location = new System.Drawing.Point(16, 8);
			this.groupBox1.Name = "groupBox1";
			this.groupBox1.Size = new System.Drawing.Size(296, 96);
			this.groupBox1.TabIndex = 6;
			this.groupBox1.TabStop = false;
			// 
			// EditConfigForm
			// 
			this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
			this.ClientSize = new System.Drawing.Size(328, 150);
			this.Controls.AddRange(new System.Windows.Forms.Control[] {
																		  this.Cancel,
																		  this.OK,
																		  this.PosTol,
																		  this.label2,
																		  this.SlopeTol,
																		  this.label1,
																		  this.groupBox1});
			this.Name = "EditConfigForm";
			this.Text = "Edit QuickMapper Configuration";
			this.Load += new System.EventHandler(this.EditConfigForm_Load);
			this.ResumeLayout(false);

		}
		#endregion

		private void EditConfigForm_Load(object sender, System.EventArgs e)
		{
			SlopeTol.Text = C.SlopeTol.ToString();
			PosTol.Text = C.PosTol.ToString();
		}

		private void OK_Click(object sender, System.EventArgs e)
		{
			try
			{
				C.SlopeTol = Convert.ToDouble(SlopeTol.Text);
				C.PosTol = Convert.ToDouble(PosTol.Text);
				this.DialogResult = DialogResult.OK;
				Close();
			}
			catch (System.Exception) {}
		}

		private void Cancel_Click(object sender, System.EventArgs e)
		{
			this.DialogResult = DialogResult.Cancel;
			Close();
		}
	}
}
