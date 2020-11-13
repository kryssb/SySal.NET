using System;
using System.Collections;
using System.ComponentModel;
using System.Drawing;
using System.Data;
using System.Windows.Forms;

namespace SySal.Controls
{
	/// <summary>
	/// Summary description for CheckBox.
	/// </summary>
	public class CheckBox : System.Windows.Forms.UserControl
	{
		/// <summary> 
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;

		public CheckBox()
		{
			// This call is required by the Windows.Forms Form Designer.
			InitializeComponent();

			// TODO: Add any initialization after the InitializeComponent call

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

		static System.Drawing.Image BOn = LoadImage("check_on.bmp");
		static System.Drawing.Image BOff = LoadImage("check_off.bmp");

		#region Component Designer generated code
		/// <summary> 
		/// Required method for Designer support - do not modify 
		/// the contents of this method with the code editor.
		/// </summary>
		private void InitializeComponent()
		{
			// 
			// CheckBox
			// 
			this.BackColor = System.Drawing.Color.White;
			this.Name = "CheckBox";
			this.Size = new System.Drawing.Size(8, 8);
			this.Resize += new System.EventHandler(this.OnResize);
			this.Paint += new System.Windows.Forms.PaintEventHandler(this.OnPaint);
			this.MouseDown += new System.Windows.Forms.MouseEventHandler(this.OnMouseDown);

		}
		#endregion

		static Image LoadImage(string resname)
		{			
			System.IO.Stream myStream;
			System.Reflection.Assembly myAssembly = System.Reflection.Assembly.GetExecutingAssembly();
			myStream = myAssembly.GetManifestResourceStream("SySal.Controls." + resname);
			Image im = new Bitmap(myStream);
			myStream.Close();
			return im;
		}

		bool IsChecked;

		public bool Checked { get { return IsChecked; } set { IsChecked = value; this.Refresh(); } }

		private void OnMouseDown(object sender, System.Windows.Forms.MouseEventArgs e)
		{
			IsChecked = !IsChecked;
			this.Refresh();		
		}

		private void OnPaint(object sender, System.Windows.Forms.PaintEventArgs e)
		{
			e.Graphics.DrawImageUnscaled(IsChecked ? BOn : BOff, 0, 0);
		}

		private void OnResize(object sender, System.EventArgs e)
		{
			this.Width = BOn.Width / 2;
			this.Height = BOn.Height / 2;		
		}
	}
}
