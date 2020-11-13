using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;

namespace SySal.Services.OperaBatchManager_Win
{
	/// <summary>
	/// Summary description for ProgressForm.
	/// </summary>
	public class ProgressForm : System.Windows.Forms.Form
	{
		internal System.Windows.Forms.RichTextBox RTFOut;
		private System.Windows.Forms.Button ApplyButton;
		private System.Windows.Forms.Button ExitButton;
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;

		public ProgressForm()
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
			this.ApplyButton = new System.Windows.Forms.Button();
			this.ExitButton = new System.Windows.Forms.Button();
			this.SuspendLayout();
			// 
			// RTFOut
			// 
			this.RTFOut.Location = new System.Drawing.Point(8, 8);
			this.RTFOut.Name = "RTFOut";
			this.RTFOut.Size = new System.Drawing.Size(600, 232);
			this.RTFOut.TabIndex = 0;
			this.RTFOut.Text = "";
			// 
			// ApplyButton
			// 
			this.ApplyButton.Location = new System.Drawing.Point(496, 248);
			this.ApplyButton.Name = "ApplyButton";
			this.ApplyButton.Size = new System.Drawing.Size(112, 24);
			this.ApplyButton.TabIndex = 2;
			this.ApplyButton.Text = "Apply changes";
			this.ApplyButton.Click += new System.EventHandler(this.ApplyButton_Click);
			// 
			// ExitButton
			// 
			this.ExitButton.Location = new System.Drawing.Point(8, 248);
			this.ExitButton.Name = "ExitButton";
			this.ExitButton.Size = new System.Drawing.Size(112, 24);
			this.ExitButton.TabIndex = 1;
			this.ExitButton.Text = "Exit";
			this.ExitButton.Click += new System.EventHandler(this.ExitButton_Click);
			// 
			// ProgressForm
			// 
			this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
			this.ClientSize = new System.Drawing.Size(618, 280);
			this.Controls.Add(this.ExitButton);
			this.Controls.Add(this.ApplyButton);
			this.Controls.Add(this.RTFOut);
			this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedToolWindow;
			this.Name = "ProgressForm";
			this.Text = "Progress Viewer/Editor";
			this.ResumeLayout(false);

		}
		#endregion

		private void ExitButton_Click(object sender, System.EventArgs e)
		{
			DialogResult = DialogResult.Cancel;
			Close();		
		}

		private void ApplyButton_Click(object sender, System.EventArgs e)
		{
			DialogResult = DialogResult.OK;
			Close();
		}
	}
}
