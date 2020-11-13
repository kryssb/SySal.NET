using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;

namespace SySal.Processing.SimpleTrackPostProcessing
{
	/// <summary>
	/// Summary description for EditConfigForm.
	/// </summary>
	public class EditConfigForm : System.Windows.Forms.Form
	{
		private System.Windows.Forms.CheckBox UseTransverseResiduals;
		private System.Windows.Forms.CheckBox CleanDoubleReconstructions;
		private System.Windows.Forms.Label label1;
		private System.Windows.Forms.TextBox MaxDuplicateDistance;
		private System.Windows.Forms.TextBox MaxSigma0;
		private System.Windows.Forms.Label label2;
		private System.Windows.Forms.TextBox MaxSigmaSlope;
		private System.Windows.Forms.Label label3;
		private System.Windows.Forms.Button MyCancelButton;
		private System.Windows.Forms.Button MyOKButton;
		private System.Windows.Forms.Panel SigmaDisplay;
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;
		private System.Windows.Forms.TextBox MaxSlope;
		private System.Windows.Forms.Label label4;

		private SimpleTrackPostProcessing.Configuration C;
		private static double MaxSlopeValue = 1.0;

		public SimpleTrackPostProcessing.Configuration Config
		{
			get
			{
				return (Configuration)C.Clone();
			}
			set
			{
				C = (Configuration)(value.Clone());
				UseTransverseResiduals.Checked = C.UseTransverseResiduals;
				CleanDoubleReconstructions.Checked = C.CleanDoubleReconstructions;
				MaxDuplicateDistance.Text = C.MaxDuplicateDistance.ToString();
				MaxSigma0.Text = C.MaxSigma0.ToString();
				MaxSigmaSlope.Text = C.MaxSigmaSlope.ToString();
				CurvePaint(this, new PaintEventArgs(SigmaDisplay.CreateGraphics(), SigmaDisplay.ClientRectangle));
			}
		}

		public EditConfigForm()
		{
			//
			// Required for Windows Form Designer support
			//
			InitializeComponent();

			//
			// TODO: Add any constructor code after InitializeComponent call
			//
			MaxSlope.Text = MaxSlopeValue.ToString();
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
			this.UseTransverseResiduals = new System.Windows.Forms.CheckBox();
			this.CleanDoubleReconstructions = new System.Windows.Forms.CheckBox();
			this.label1 = new System.Windows.Forms.Label();
			this.MaxDuplicateDistance = new System.Windows.Forms.TextBox();
			this.MaxSigma0 = new System.Windows.Forms.TextBox();
			this.label2 = new System.Windows.Forms.Label();
			this.MaxSigmaSlope = new System.Windows.Forms.TextBox();
			this.label3 = new System.Windows.Forms.Label();
			this.MyCancelButton = new System.Windows.Forms.Button();
			this.MyOKButton = new System.Windows.Forms.Button();
			this.SigmaDisplay = new System.Windows.Forms.Panel();
			this.MaxSlope = new System.Windows.Forms.TextBox();
			this.label4 = new System.Windows.Forms.Label();
			this.SuspendLayout();
			// 
			// UseTransverseResiduals
			// 
			this.UseTransverseResiduals.Location = new System.Drawing.Point(8, 8);
			this.UseTransverseResiduals.Name = "UseTransverseResiduals";
			this.UseTransverseResiduals.Size = new System.Drawing.Size(232, 24);
			this.UseTransverseResiduals.TabIndex = 0;
			this.UseTransverseResiduals.Text = "Use transverse residuals only";
			this.UseTransverseResiduals.CheckedChanged += new System.EventHandler(this.UseTransverseResiduals_CheckedChanged);
			// 
			// CleanDoubleReconstructions
			// 
			this.CleanDoubleReconstructions.Location = new System.Drawing.Point(8, 40);
			this.CleanDoubleReconstructions.Name = "CleanDoubleReconstructions";
			this.CleanDoubleReconstructions.Size = new System.Drawing.Size(232, 24);
			this.CleanDoubleReconstructions.TabIndex = 1;
			this.CleanDoubleReconstructions.Text = "Clean double reconstructions";
			this.CleanDoubleReconstructions.CheckedChanged += new System.EventHandler(this.CleanDoubleReconstructions_CheckedChanged);
			// 
			// label1
			// 
			this.label1.Location = new System.Drawing.Point(8, 80);
			this.label1.Name = "label1";
			this.label1.Size = new System.Drawing.Size(216, 16);
			this.label1.TabIndex = 2;
			this.label1.Text = "Max distance between duplicates (micron)";
			this.label1.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
			// 
			// MaxDuplicateDistance
			// 
			this.MaxDuplicateDistance.Location = new System.Drawing.Point(232, 80);
			this.MaxDuplicateDistance.Name = "MaxDuplicateDistance";
			this.MaxDuplicateDistance.Size = new System.Drawing.Size(40, 20);
			this.MaxDuplicateDistance.TabIndex = 3;
			this.MaxDuplicateDistance.Text = "2";
			this.MaxDuplicateDistance.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
			this.MaxDuplicateDistance.Validating += new System.ComponentModel.CancelEventHandler(this.MaxDuplicateDistance_Validating);
			// 
			// MaxSigma0
			// 
			this.MaxSigma0.Location = new System.Drawing.Point(232, 112);
			this.MaxSigma0.Name = "MaxSigma0";
			this.MaxSigma0.Size = new System.Drawing.Size(40, 20);
			this.MaxSigma0.TabIndex = 5;
			this.MaxSigma0.Text = "0.5";
			this.MaxSigma0.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
			this.MaxSigma0.Validating += new System.ComponentModel.CancelEventHandler(this.MaxSigma0_Validating);
			// 
			// label2
			// 
			this.label2.Location = new System.Drawing.Point(8, 112);
			this.label2.Name = "label2";
			this.label2.Size = new System.Drawing.Size(216, 16);
			this.label2.TabIndex = 4;
			this.label2.Text = "Max Sigma for vertical tracks (micron)";
			this.label2.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
			// 
			// MaxSigmaSlope
			// 
			this.MaxSigmaSlope.Location = new System.Drawing.Point(232, 144);
			this.MaxSigmaSlope.Name = "MaxSigmaSlope";
			this.MaxSigmaSlope.Size = new System.Drawing.Size(40, 20);
			this.MaxSigmaSlope.TabIndex = 7;
			this.MaxSigmaSlope.Text = "0.5";
			this.MaxSigmaSlope.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
			this.MaxSigmaSlope.Validating += new System.ComponentModel.CancelEventHandler(this.MaxSigmaSlope_Validating);
			// 
			// label3
			// 
			this.label3.Location = new System.Drawing.Point(8, 144);
			this.label3.Name = "label3";
			this.label3.Size = new System.Drawing.Size(216, 16);
			this.label3.TabIndex = 6;
			this.label3.Text = "Max Sigma increment (micron)";
			this.label3.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
			// 
			// MyCancelButton
			// 
			this.MyCancelButton.DialogResult = System.Windows.Forms.DialogResult.Cancel;
			this.MyCancelButton.Location = new System.Drawing.Point(8, 208);
			this.MyCancelButton.Name = "MyCancelButton";
			this.MyCancelButton.Size = new System.Drawing.Size(64, 24);
			this.MyCancelButton.TabIndex = 10;
			this.MyCancelButton.Text = "Cancel";
			// 
			// MyOKButton
			// 
			this.MyOKButton.DialogResult = System.Windows.Forms.DialogResult.OK;
			this.MyOKButton.Location = new System.Drawing.Point(208, 208);
			this.MyOKButton.Name = "MyOKButton";
			this.MyOKButton.Size = new System.Drawing.Size(64, 24);
			this.MyOKButton.TabIndex = 11;
			this.MyOKButton.Text = "OK";
			// 
			// SigmaDisplay
			// 
			this.SigmaDisplay.BackColor = System.Drawing.Color.White;
			this.SigmaDisplay.BorderStyle = System.Windows.Forms.BorderStyle.Fixed3D;
			this.SigmaDisplay.Location = new System.Drawing.Point(280, 8);
			this.SigmaDisplay.Name = "SigmaDisplay";
			this.SigmaDisplay.Size = new System.Drawing.Size(256, 224);
			this.SigmaDisplay.TabIndex = 12;
			this.SigmaDisplay.Paint += new System.Windows.Forms.PaintEventHandler(this.CurvePaint);
			// 
			// MaxSlope
			// 
			this.MaxSlope.Location = new System.Drawing.Point(232, 176);
			this.MaxSlope.Name = "MaxSlope";
			this.MaxSlope.Size = new System.Drawing.Size(40, 20);
			this.MaxSlope.TabIndex = 9;
			this.MaxSlope.Text = "1";
			this.MaxSlope.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
			this.MaxSlope.Validating += new System.ComponentModel.CancelEventHandler(this.MaxSlope_Validating);
			// 
			// label4
			// 
			this.label4.Location = new System.Drawing.Point(8, 176);
			this.label4.Name = "label4";
			this.label4.Size = new System.Drawing.Size(216, 16);
			this.label4.TabIndex = 8;
			this.label4.Text = "Max slope (only for display)";
			this.label4.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
			// 
			// EditConfigForm
			// 
			this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
			this.ClientSize = new System.Drawing.Size(544, 238);
			this.Controls.AddRange(new System.Windows.Forms.Control[] {
																		  this.MaxSlope,
																		  this.label4,
																		  this.SigmaDisplay,
																		  this.MyOKButton,
																		  this.MyCancelButton,
																		  this.MaxSigmaSlope,
																		  this.label3,
																		  this.MaxSigma0,
																		  this.label2,
																		  this.MaxDuplicateDistance,
																		  this.label1,
																		  this.CleanDoubleReconstructions,
																		  this.UseTransverseResiduals});
			this.Name = "EditConfigForm";
			this.Text = "Edit SimpleTrackPostProcessor Configuration";
			this.ResumeLayout(false);

		}
		#endregion


		private void UseTransverseResiduals_CheckedChanged(object sender, System.EventArgs e)
		{
			C.UseTransverseResiduals = UseTransverseResiduals.Checked;
		}

		private void CleanDoubleReconstructions_CheckedChanged(object sender, System.EventArgs e)
		{
			C.CleanDoubleReconstructions = CleanDoubleReconstructions.Checked;		
		}

		private void MaxDuplicateDistance_Validating(object sender, System.ComponentModel.CancelEventArgs e)
		{
			try
			{
				double m = Convert.ToDouble(MaxDuplicateDistance.Text);
				if (m < 0.0 || m > 20.0) throw new Exception("Max distance between duplicates must be between 0 and 20 micron.");
				C.MaxDuplicateDistance = m;
			}
			catch (Exception x)
			{
				MessageBox.Show(x.Message, "Input Error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
				e.Cancel = true;
				return;
			}				
		}

		private void MaxSigma0_Validating(object sender, System.ComponentModel.CancelEventArgs e)
		{
			try
			{
				double m = Convert.ToDouble(MaxSigma0.Text);
				if (m < 0.0 || m > 5.0) throw new Exception("Max Sigma for vertical tracks must be between 0 and 5 micron.");
				C.MaxSigma0 = m;
			}
			catch (Exception x)
			{
				MessageBox.Show(x.Message, "Input Error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
				e.Cancel = true;
				return;
			}
			CurvePaint(this, new PaintEventArgs(SigmaDisplay.CreateGraphics(), SigmaDisplay.ClientRectangle));
		}

		private void MaxSigmaSlope_Validating(object sender, System.ComponentModel.CancelEventArgs e)
		{
			try
			{
				double m = Convert.ToDouble(MaxSigmaSlope.Text);
				if (m < 0.0 || m > 5.0) throw new Exception("Max Sigma increment for vertical tracks must be between 0 and 5 micron / slope unit.");
				C.MaxSigmaSlope = m;
			}
			catch (Exception x)
			{
				MessageBox.Show(x.Message, "Input Error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
				e.Cancel = true;
				return;
			}
			CurvePaint(this, new PaintEventArgs(SigmaDisplay.CreateGraphics(), SigmaDisplay.ClientRectangle));	
		}

		private void CurvePaint(object sender, System.Windows.Forms.PaintEventArgs e)
		{
			Graphics g = e.Graphics;
			g.Clear(Color.White);
			if (C == null) return;
			Pen linep = new Pen(Color.CadetBlue, 3);
			Pen axisp = new Pen(Color.DarkGray, 1);
			Pen gridp = new Pen(Color.LightCoral,1);
			Brush textb = new SolidBrush(Color.Black);
			Font numfont = new Font("Arial", 8);
			SizeF ytextsize = new SizeF(0.0f, 0.0f);
			SizeF xtextsize;
			SizeF newsize;
			double ygrid, xgrid, f;
			double lastdivider;
			double maxsigma = (C.MaxSigma0 + C.MaxSigmaSlope * MaxSlopeValue);
			ygrid = 1000.0f;
			lastdivider = 5.0f;
			while (maxsigma / ygrid < 4.0f)
			{
				lastdivider = (lastdivider == 5.0f) ? 2.0f : 5.0f;
				ygrid /= lastdivider;
			}

			xgrid = 1000.0f;
			lastdivider = 5.0f;
			while (MaxSlopeValue / xgrid < 4.0f)
			{
				lastdivider = (lastdivider == 5.0f) ? 2.0f : 5.0f;
				xgrid /= lastdivider;
			}

			for (f = 0; f <= maxsigma; f += ygrid)
			{
				newsize = g.MeasureString(f.ToString("G4"), numfont);
				if (newsize.Width > ytextsize.Width) ytextsize.Width = newsize.Width;
				if (newsize.Height > ytextsize.Height) ytextsize.Height = newsize.Height;
			}
			xtextsize = g.MeasureString(xgrid.ToString("G4"), numfont);
			int ybase = (SigmaDisplay.Height - 12 - (int)xtextsize.Height);
			int xbase = (int)ytextsize.Width + 12;
			double xscale = (double)(SigmaDisplay.Width - xbase - 4 - xtextsize.Width) / MaxSlopeValue;
			double yscale = - (double)(ybase - 4) / maxsigma;
			for (f = ygrid; f <= maxsigma; f += ygrid)
			{
				g.DrawLine(gridp, xbase, ybase + (int)(yscale * f), SigmaDisplay.Width - 4, ybase + (int)(yscale * f));
				g.DrawString(f.ToString("G4"), numfont, textb, 4, ybase + (int)(yscale * f) - (int)(ytextsize.Height) / 2);
			}
			for (f = xgrid; f <= MaxSlopeValue; f += xgrid)
			{
				g.DrawLine(gridp, (int)(xbase + f * xscale), ybase, (int)(xbase + f * xscale), 4);
				g.DrawString(f.ToString("G4"), numfont, textb, (int)(xbase + f * xscale - (int)g.MeasureString(f.ToString("G4"), numfont).Width / 2), ybase + 4);
			}
			g.DrawLine(axisp, xbase, ybase, SigmaDisplay.Width - 4, ybase);
			g.DrawLine(axisp, xbase, 4, xbase, ybase);
			g.DrawLine(linep, xbase, ybase + (int)(C.MaxSigma0 * yscale), xbase + (int)(MaxSlopeValue * xscale), ybase + (int)(maxsigma * yscale));
		}

		private void MaxSlope_Validating(object sender, System.ComponentModel.CancelEventArgs e)
		{
			try
			{
				double m = Convert.ToDouble(MaxSlope.Text);
				if (m < 0.1 || m > 10.0) throw new Exception("Max Sigma increment for vertical tracks must be between 0.1 and 10.");
				MaxSlopeValue = m;
			}
			catch (Exception x)
			{
				MessageBox.Show(x.Message, "Input Error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
				e.Cancel = true;
				return;
			}
			CurvePaint(this, new PaintEventArgs(SigmaDisplay.CreateGraphics(), SigmaDisplay.ClientRectangle));
		}
	}
}
