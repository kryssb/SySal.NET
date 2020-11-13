using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;

namespace SySal.Executables.X3LView
{
	/// <summary>
	/// Owner information viewing form.
	/// </summary>
	internal class OwnerView : System.Windows.Forms.Form
	{
		private System.Windows.Forms.TextBox OwnerText;
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;

		public void SetText(string text)
		{
			OwnerText.Text = text.Replace("\n","\r\n");
		}

		public OwnerView()
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
			this.OwnerText = new System.Windows.Forms.TextBox();
			this.SuspendLayout();
			// 
			// OwnerText
			// 
			this.OwnerText.Location = new System.Drawing.Point(8, 8);
			this.OwnerText.Multiline = true;
			this.OwnerText.Name = "OwnerText";
			this.OwnerText.ReadOnly = true;
			this.OwnerText.Size = new System.Drawing.Size(664, 248);
			this.OwnerText.TabIndex = 0;
			this.OwnerText.Text = "";
			// 
			// OwnerView
			// 
			this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
			this.ClientSize = new System.Drawing.Size(682, 266);
			this.Controls.Add(this.OwnerText);
			this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedToolWindow;
			this.Name = "OwnerView";
			this.Text = "Owner View";
			this.ResumeLayout(false);

		}
		#endregion
	}
}
