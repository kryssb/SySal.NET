using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;

namespace SySal.DAQSystem.Drivers.Intercalibration3Driver
{
	/// <summary>
	/// Summary description for ExplanationForm.
	/// </summary>
	internal class ExplanationForm : System.Windows.Forms.Form
	{
		internal System.Windows.Forms.RichTextBox RTFOut;
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;

		public ExplanationForm()
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
			this.RTFOut = new System.Windows.Forms.RichTextBox();
			this.SuspendLayout();
			// 
			// RTFOut
			// 
			this.RTFOut.DetectUrls = false;
			this.RTFOut.Font = new System.Drawing.Font("Lucida Console", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.RTFOut.Location = new System.Drawing.Point(8, 8);
			this.RTFOut.Name = "RTFOut";
			this.RTFOut.ReadOnly = true;
			this.RTFOut.Size = new System.Drawing.Size(704, 400);
			this.RTFOut.TabIndex = 0;
			this.RTFOut.Text = "";
			// 
			// ExplanationForm
			// 
			this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
			this.ClientSize = new System.Drawing.Size(720, 414);
			this.Controls.Add(this.RTFOut);
			this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedToolWindow;
			this.Name = "ExplanationForm";
			this.Text = "Explanation for Intercalibration3Driver";
			this.ResumeLayout(false);

		}
		#endregion
	}
}
