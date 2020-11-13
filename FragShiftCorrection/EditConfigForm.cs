using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;

namespace SySal.Processing.FragShiftCorrection
{
	/// <summary>
	/// Summary description for EditConfigForm.
	/// </summary>
	internal class EditConfigForm : System.Windows.Forms.Form
	{
		private System.Windows.Forms.Label label1;
		private System.Windows.Forms.TextBox MinGrains;
		private System.Windows.Forms.TextBox MinSlope;
		private System.Windows.Forms.Label label6;
		private System.Windows.Forms.Button OK;
		private System.Windows.Forms.Button Cancel;
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;
		private System.Windows.Forms.GroupBox groupBox1;
		private System.Windows.Forms.GroupBox groupBox2;
		private System.Windows.Forms.Label label7;
		private System.Windows.Forms.TextBox MergeSlopeTol;
		private System.Windows.Forms.Label label4;
		private System.Windows.Forms.TextBox MergePosTol;
		private System.Windows.Forms.Label label5;
		private System.Windows.Forms.TextBox MinMatches;
		private System.Windows.Forms.Label label12;
		private System.Windows.Forms.TextBox MaxMatchError;
		private System.Windows.Forms.Label label13;
		private System.Windows.Forms.TextBox GrainsOverlapRatio;
		private System.Windows.Forms.GroupBox groupBox3;
		private System.Windows.Forms.TextBox PosTol;
		private System.Windows.Forms.Label label2;
		private System.Windows.Forms.TextBox SlopeTol;
		private System.Windows.Forms.Label label3;
		private System.Windows.Forms.Label label14;
		private System.Windows.Forms.TextBox GrainZTol;
		private System.Windows.Forms.Label label15;
		private System.Windows.Forms.TextBox OverlapTol;
		private System.Windows.Forms.Button HelpButton;
		private System.Windows.Forms.RadioButton SinHysteresisFunction;
		private System.Windows.Forms.RadioButton StepHysteresisFunction;
		private System.Windows.Forms.CheckBox EnableHysteresisCheckBox;

		public FragShiftCorrection.Configuration C;

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
			this.MinGrains = new System.Windows.Forms.TextBox();
			this.MinSlope = new System.Windows.Forms.TextBox();
			this.label6 = new System.Windows.Forms.Label();
			this.OK = new System.Windows.Forms.Button();
			this.Cancel = new System.Windows.Forms.Button();
			this.groupBox1 = new System.Windows.Forms.GroupBox();
			this.groupBox2 = new System.Windows.Forms.GroupBox();
			this.label15 = new System.Windows.Forms.Label();
			this.OverlapTol = new System.Windows.Forms.TextBox();
			this.label14 = new System.Windows.Forms.Label();
			this.GrainZTol = new System.Windows.Forms.TextBox();
			this.label13 = new System.Windows.Forms.Label();
			this.GrainsOverlapRatio = new System.Windows.Forms.TextBox();
			this.label12 = new System.Windows.Forms.Label();
			this.MaxMatchError = new System.Windows.Forms.TextBox();
			this.label5 = new System.Windows.Forms.Label();
			this.MinMatches = new System.Windows.Forms.TextBox();
			this.PosTol = new System.Windows.Forms.TextBox();
			this.label3 = new System.Windows.Forms.Label();
			this.SlopeTol = new System.Windows.Forms.TextBox();
			this.label2 = new System.Windows.Forms.Label();
			this.MergeSlopeTol = new System.Windows.Forms.TextBox();
			this.label4 = new System.Windows.Forms.Label();
			this.label7 = new System.Windows.Forms.Label();
			this.MergePosTol = new System.Windows.Forms.TextBox();
			this.groupBox3 = new System.Windows.Forms.GroupBox();
			this.HelpButton = new System.Windows.Forms.Button();
			this.StepHysteresisFunction = new System.Windows.Forms.RadioButton();
			this.SinHysteresisFunction = new System.Windows.Forms.RadioButton();
			this.EnableHysteresisCheckBox = new System.Windows.Forms.CheckBox();
			this.groupBox1.SuspendLayout();
			this.groupBox2.SuspendLayout();
			this.SuspendLayout();
			// 
			// label1
			// 
			this.label1.Location = new System.Drawing.Point(8, 24);
			this.label1.Name = "label1";
			this.label1.Size = new System.Drawing.Size(72, 16);
			this.label1.TabIndex = 0;
			this.label1.Text = "Min Grains";
			this.label1.TextAlign = System.Drawing.ContentAlignment.MiddleRight;
			// 
			// MinGrains
			// 
			this.MinGrains.Location = new System.Drawing.Point(120, 24);
			this.MinGrains.Name = "MinGrains";
			this.MinGrains.Size = new System.Drawing.Size(48, 20);
			this.MinGrains.TabIndex = 1;
			this.MinGrains.Text = "";
			// 
			// MinSlope
			// 
			this.MinSlope.Location = new System.Drawing.Point(120, 48);
			this.MinSlope.Name = "MinSlope";
			this.MinSlope.Size = new System.Drawing.Size(48, 20);
			this.MinSlope.TabIndex = 3;
			this.MinSlope.Text = "";
			// 
			// label6
			// 
			this.label6.Location = new System.Drawing.Point(8, 48);
			this.label6.Name = "label6";
			this.label6.Size = new System.Drawing.Size(72, 16);
			this.label6.TabIndex = 2;
			this.label6.Text = "Min Slope";
			this.label6.TextAlign = System.Drawing.ContentAlignment.MiddleRight;
			// 
			// OK
			// 
			this.OK.Location = new System.Drawing.Point(8, 264);
			this.OK.Name = "OK";
			this.OK.Size = new System.Drawing.Size(64, 24);
			this.OK.TabIndex = 22;
			this.OK.Text = "OK";
			this.OK.Click += new System.EventHandler(this.OK_Click);
			// 
			// Cancel
			// 
			this.Cancel.Location = new System.Drawing.Point(376, 264);
			this.Cancel.Name = "Cancel";
			this.Cancel.Size = new System.Drawing.Size(64, 24);
			this.Cancel.TabIndex = 23;
			this.Cancel.Text = "Cancel";
			this.Cancel.Click += new System.EventHandler(this.Cancel_Click);
			// 
			// groupBox1
			// 
			this.groupBox1.Controls.AddRange(new System.Windows.Forms.Control[] {
																					this.label1,
																					this.MinSlope,
																					this.MinGrains,
																					this.label6});
			this.groupBox1.Location = new System.Drawing.Point(8, 8);
			this.groupBox1.Name = "groupBox1";
			this.groupBox1.Size = new System.Drawing.Size(184, 88);
			this.groupBox1.TabIndex = 24;
			this.groupBox1.TabStop = false;
			this.groupBox1.Text = "General selection";
			// 
			// groupBox2
			// 
			this.groupBox2.Controls.AddRange(new System.Windows.Forms.Control[] {
																					this.label15,
																					this.OverlapTol,
																					this.label14,
																					this.GrainZTol,
																					this.label13,
																					this.GrainsOverlapRatio,
																					this.label12,
																					this.MaxMatchError,
																					this.label5,
																					this.MinMatches,
																					this.PosTol,
																					this.label3,
																					this.SlopeTol,
																					this.label2});
			this.groupBox2.Location = new System.Drawing.Point(200, 8);
			this.groupBox2.Name = "groupBox2";
			this.groupBox2.Size = new System.Drawing.Size(240, 192);
			this.groupBox2.TabIndex = 26;
			this.groupBox2.TabStop = false;
			this.groupBox2.Text = "Cross-field track matching";
			// 
			// label15
			// 
			this.label15.Location = new System.Drawing.Point(16, 160);
			this.label15.Name = "label15";
			this.label15.Size = new System.Drawing.Size(152, 16);
			this.label15.TabIndex = 20;
			this.label15.Text = "Field Overlap Tol";
			this.label15.TextAlign = System.Drawing.ContentAlignment.MiddleRight;
			// 
			// OverlapTol
			// 
			this.OverlapTol.Location = new System.Drawing.Point(184, 160);
			this.OverlapTol.Name = "OverlapTol";
			this.OverlapTol.Size = new System.Drawing.Size(48, 20);
			this.OverlapTol.TabIndex = 21;
			this.OverlapTol.Text = "";
			// 
			// label14
			// 
			this.label14.Location = new System.Drawing.Point(16, 136);
			this.label14.Name = "label14";
			this.label14.Size = new System.Drawing.Size(152, 16);
			this.label14.TabIndex = 18;
			this.label14.Text = "Grain Z Tol";
			this.label14.TextAlign = System.Drawing.ContentAlignment.MiddleRight;
			// 
			// GrainZTol
			// 
			this.GrainZTol.Location = new System.Drawing.Point(184, 136);
			this.GrainZTol.Name = "GrainZTol";
			this.GrainZTol.Size = new System.Drawing.Size(48, 20);
			this.GrainZTol.TabIndex = 19;
			this.GrainZTol.Text = "";
			// 
			// label13
			// 
			this.label13.Location = new System.Drawing.Point(16, 112);
			this.label13.Name = "label13";
			this.label13.Size = new System.Drawing.Size(152, 16);
			this.label13.TabIndex = 16;
			this.label13.Text = "Grains Overlap Ratio";
			this.label13.TextAlign = System.Drawing.ContentAlignment.MiddleRight;
			// 
			// GrainsOverlapRatio
			// 
			this.GrainsOverlapRatio.Location = new System.Drawing.Point(184, 112);
			this.GrainsOverlapRatio.Name = "GrainsOverlapRatio";
			this.GrainsOverlapRatio.Size = new System.Drawing.Size(48, 20);
			this.GrainsOverlapRatio.TabIndex = 17;
			this.GrainsOverlapRatio.Text = "";
			// 
			// label12
			// 
			this.label12.Location = new System.Drawing.Point(48, 88);
			this.label12.Name = "label12";
			this.label12.Size = new System.Drawing.Size(120, 16);
			this.label12.TabIndex = 14;
			this.label12.Text = "Max Match Error";
			this.label12.TextAlign = System.Drawing.ContentAlignment.MiddleRight;
			// 
			// MaxMatchError
			// 
			this.MaxMatchError.Location = new System.Drawing.Point(184, 88);
			this.MaxMatchError.Name = "MaxMatchError";
			this.MaxMatchError.Size = new System.Drawing.Size(48, 20);
			this.MaxMatchError.TabIndex = 15;
			this.MaxMatchError.Text = "";
			// 
			// label5
			// 
			this.label5.Location = new System.Drawing.Point(48, 64);
			this.label5.Name = "label5";
			this.label5.Size = new System.Drawing.Size(120, 16);
			this.label5.TabIndex = 12;
			this.label5.Text = "Min Matches";
			this.label5.TextAlign = System.Drawing.ContentAlignment.MiddleRight;
			// 
			// MinMatches
			// 
			this.MinMatches.Location = new System.Drawing.Point(184, 64);
			this.MinMatches.Name = "MinMatches";
			this.MinMatches.Size = new System.Drawing.Size(48, 20);
			this.MinMatches.TabIndex = 13;
			this.MinMatches.Text = "";
			// 
			// PosTol
			// 
			this.PosTol.Location = new System.Drawing.Point(184, 16);
			this.PosTol.Name = "PosTol";
			this.PosTol.Size = new System.Drawing.Size(48, 20);
			this.PosTol.TabIndex = 9;
			this.PosTol.Text = "";
			// 
			// label3
			// 
			this.label3.Location = new System.Drawing.Point(72, 40);
			this.label3.Name = "label3";
			this.label3.Size = new System.Drawing.Size(96, 16);
			this.label3.TabIndex = 10;
			this.label3.Text = "Slope Tolerance";
			this.label3.TextAlign = System.Drawing.ContentAlignment.MiddleRight;
			// 
			// SlopeTol
			// 
			this.SlopeTol.Location = new System.Drawing.Point(184, 40);
			this.SlopeTol.Name = "SlopeTol";
			this.SlopeTol.Size = new System.Drawing.Size(48, 20);
			this.SlopeTol.TabIndex = 11;
			this.SlopeTol.Text = "";
			// 
			// label2
			// 
			this.label2.Location = new System.Drawing.Point(72, 16);
			this.label2.Name = "label2";
			this.label2.Size = new System.Drawing.Size(96, 16);
			this.label2.TabIndex = 8;
			this.label2.Text = "Pos Tolerance";
			this.label2.TextAlign = System.Drawing.ContentAlignment.MiddleRight;
			// 
			// MergeSlopeTol
			// 
			this.MergeSlopeTol.Location = new System.Drawing.Point(128, 152);
			this.MergeSlopeTol.Name = "MergeSlopeTol";
			this.MergeSlopeTol.Size = new System.Drawing.Size(48, 20);
			this.MergeSlopeTol.TabIndex = 7;
			this.MergeSlopeTol.Text = "";
			// 
			// label4
			// 
			this.label4.Location = new System.Drawing.Point(24, 152);
			this.label4.Name = "label4";
			this.label4.Size = new System.Drawing.Size(88, 16);
			this.label4.TabIndex = 6;
			this.label4.Text = "Merge Slope Tol";
			this.label4.TextAlign = System.Drawing.ContentAlignment.MiddleRight;
			// 
			// label7
			// 
			this.label7.Location = new System.Drawing.Point(24, 128);
			this.label7.Name = "label7";
			this.label7.Size = new System.Drawing.Size(88, 16);
			this.label7.TabIndex = 4;
			this.label7.Text = "Merge Pos Tol";
			this.label7.TextAlign = System.Drawing.ContentAlignment.MiddleRight;
			// 
			// MergePosTol
			// 
			this.MergePosTol.Location = new System.Drawing.Point(128, 128);
			this.MergePosTol.Name = "MergePosTol";
			this.MergePosTol.Size = new System.Drawing.Size(48, 20);
			this.MergePosTol.TabIndex = 5;
			this.MergePosTol.Text = "";
			// 
			// groupBox3
			// 
			this.groupBox3.Location = new System.Drawing.Point(8, 104);
			this.groupBox3.Name = "groupBox3";
			this.groupBox3.Size = new System.Drawing.Size(184, 96);
			this.groupBox3.TabIndex = 25;
			this.groupBox3.TabStop = false;
			this.groupBox3.Text = "In-field cleaning";
			// 
			// HelpButton
			// 
			this.HelpButton.Location = new System.Drawing.Point(304, 264);
			this.HelpButton.Name = "HelpButton";
			this.HelpButton.Size = new System.Drawing.Size(64, 24);
			this.HelpButton.TabIndex = 27;
			this.HelpButton.Text = "Help";
			this.HelpButton.Click += new System.EventHandler(this.HelpButton_Click);
			// 
			// StepHysteresisFunction
			// 
			this.StepHysteresisFunction.Location = new System.Drawing.Point(8, 208);
			this.StepHysteresisFunction.Name = "StepHysteresisFunction";
			this.StepHysteresisFunction.Size = new System.Drawing.Size(192, 24);
			this.StepHysteresisFunction.TabIndex = 28;
			this.StepHysteresisFunction.Text = "Step Hysteresis Function";
			this.StepHysteresisFunction.CheckedChanged += new System.EventHandler(this.StepHysteresisFunction_CheckedChanged);
			// 
			// SinHysteresisFunction
			// 
			this.SinHysteresisFunction.Location = new System.Drawing.Point(8, 232);
			this.SinHysteresisFunction.Name = "SinHysteresisFunction";
			this.SinHysteresisFunction.Size = new System.Drawing.Size(192, 24);
			this.SinHysteresisFunction.TabIndex = 29;
			this.SinHysteresisFunction.Text = "Sinusoidal Hysteresis Function";
			this.SinHysteresisFunction.CheckedChanged += new System.EventHandler(this.SinHysteresisFunction_CheckedChanged);
			// 
			// EnableHysteresisCheckBox
			// 
			this.EnableHysteresisCheckBox.Location = new System.Drawing.Point(200, 208);
			this.EnableHysteresisCheckBox.Name = "EnableHysteresisCheckBox";
			this.EnableHysteresisCheckBox.Size = new System.Drawing.Size(240, 24);
			this.EnableHysteresisCheckBox.TabIndex = 30;
			this.EnableHysteresisCheckBox.Text = "Enable Hysteresis Estimation";
			// 
			// EditConfigForm
			// 
			this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
			this.ClientSize = new System.Drawing.Size(448, 294);
			this.Controls.AddRange(new System.Windows.Forms.Control[] {
																		  this.EnableHysteresisCheckBox,
																		  this.SinHysteresisFunction,
																		  this.StepHysteresisFunction,
																		  this.HelpButton,
																		  this.groupBox2,
																		  this.groupBox1,
																		  this.Cancel,
																		  this.OK,
																		  this.MergeSlopeTol,
																		  this.MergePosTol,
																		  this.label7,
																		  this.label4,
																		  this.groupBox3});
			this.Name = "EditConfigForm";
			this.Text = "Edit FragShiftCorrection Configuration";
			this.Load += new System.EventHandler(this.EditConfigForm_Load);
			this.groupBox1.ResumeLayout(false);
			this.groupBox2.ResumeLayout(false);
			this.ResumeLayout(false);

		}
		#endregion

		private void EditConfigForm_Load(object sender, System.EventArgs e)
		{
			MinGrains.Text = C.MinGrains.ToString();
			MinSlope.Text = C.MinSlope.ToString();
			PosTol.Text = C.PosTol.ToString();
			SlopeTol.Text = C.SlopeTol.ToString();
			MergePosTol.Text = C.MergePosTol.ToString();
			MergeSlopeTol.Text = C.MergeSlopeTol.ToString();
			MinMatches.Text = C.MinMatches.ToString();
			MaxMatchError.Text = C.MaxMatchError.ToString();
			GrainsOverlapRatio.Text = C.GrainsOverlapRatio.ToString();
			GrainZTol.Text = C.GrainZTol.ToString();
			OverlapTol.Text = C.OverlapTol.ToString();
			StepHysteresisFunction.Checked = C.IsStep;
			SinHysteresisFunction.Checked = !C.IsStep;
			EnableHysteresisCheckBox.Checked = C.EnableHysteresis;
		}

		private void OK_Click(object sender, System.EventArgs e)
		{
			try
			{
				C.MinGrains = Convert.ToInt32(MinGrains.Text);
				C.MinSlope = Convert.ToDouble(MinSlope.Text);
				C.PosTol = Convert.ToDouble(PosTol.Text);
				C.SlopeTol = Convert.ToDouble(SlopeTol.Text);				
				C.MergePosTol = Convert.ToDouble(MergePosTol.Text);
				C.MergeSlopeTol = Convert.ToDouble(MergeSlopeTol.Text);
				C.MinMatches = Convert.ToInt32(MinMatches.Text);
				C.MaxMatchError = Convert.ToDouble(MaxMatchError.Text);
				C.GrainsOverlapRatio = Convert.ToDouble(GrainsOverlapRatio.Text);
				C.GrainZTol = Convert.ToDouble(GrainZTol.Text);
				C.OverlapTol = Convert.ToDouble(OverlapTol.Text);
				C.EnableHysteresis = EnableHysteresisCheckBox.Checked;
				this.DialogResult = DialogResult.OK;
				this.Close();
			}
			catch (System.Exception) {};
		}

		private void Cancel_Click(object sender, System.EventArgs e)
		{
			this.DialogResult = DialogResult.Cancel;
			this.Close();
		}

		private void HelpButton_Click(object sender, System.EventArgs e)
		{
			try
			{
				string loc = GetType().Assembly.Location;
				loc = loc.Remove(loc.Length - 3, 3) + "chm";
				Help.ShowHelp(this, loc, "HelpOverview.htm");
			}
			catch (Exception x)
			{
				MessageBox.Show(x.ToString(), "Help System Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
			}				
		}

		private void StepHysteresisFunction_CheckedChanged(object sender, System.EventArgs e)
		{
			C.IsStep = true;
		}

		private void SinHysteresisFunction_CheckedChanged(object sender, System.EventArgs e)
		{
			C.IsStep = false;
		}

	}
}
