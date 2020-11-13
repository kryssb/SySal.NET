using System;
using System.Collections;
using System.ComponentModel;
using System.Drawing;
using System.Data;
using System.Windows.Forms;

namespace SySal.Controls
{
	/// <summary>
	/// Summary description for BackgroundPanel.
	/// </summary>
	public class BackgroundPanel : System.Windows.Forms.UserControl
	{
		/// <summary> 
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;

		public BackgroundPanel()
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

		#region Component Designer generated code
		/// <summary> 
		/// Required method for Designer support - do not modify 
		/// the contents of this method with the code editor.
		/// </summary>
		private void InitializeComponent()
		{
			// 
			// BackgroundPanel
			// 
			this.BackColor = System.Drawing.Color.White;
			this.Name = "BackgroundPanel";
			this.Resize += new System.EventHandler(this.OnResize);
			this.Paint += new System.Windows.Forms.PaintEventHandler(this.OnPaint);

		}
		#endregion

		static System.Drawing.Image B1UpLeft = LoadImage("ul.bmp");
		static System.Drawing.Image B1Up = LoadImage("up.bmp");
		static System.Drawing.Image B1UpRight = LoadImage("ur.bmp");
		static System.Drawing.Image B1Right = LoadImage("ri.bmp");
		static System.Drawing.Image B1DownRight = LoadImage("dr.bmp");
		static System.Drawing.Image B1Down = LoadImage("do.bmp");
		static System.Drawing.Image B1DownLeft = LoadImage("dl.bmp");
		static System.Drawing.Image B1Left = LoadImage("le.bmp");		

		static System.Drawing.Brush BkgndBrush = new SolidBrush(Color.FromArgb(126, 181, 232));

		static Image LoadImage(string resname)
		{			
			System.IO.Stream myStream;
			System.Reflection.Assembly myAssembly = System.Reflection.Assembly.GetExecutingAssembly();
			myStream = myAssembly.GetManifestResourceStream("SySal.Controls." + resname);
			Image im = new Bitmap(myStream);
			myStream.Close();
			return im;
		}

		private void OnPaint(object sender, System.Windows.Forms.PaintEventArgs e)
		{
			System.Drawing.Graphics g = e.Graphics;
			int w = this.Width - 2;
			int h = this.Height - 2;
			g.FillRectangle(BkgndBrush, 0, 0, w, h);
			int s;
			for (s = B1UpLeft.Width / 2; s < w; s += B1Up.Width / 2)
			{
				g.DrawImageUnscaled(B1Up, s, 0);
				g.DrawImageUnscaled(B1Down, s, h - B1Down.Height / 2);
			}
			for (s = B1UpLeft.Height / 2; s < h; s += B1Left.Height / 2)
			{
				g.DrawImageUnscaled(B1Left, 0, s);
				g.DrawImageUnscaled(B1Right, w - B1Right.Width / 2, s);
			}
			g.DrawImageUnscaled(B1UpLeft, 0, 0);
			g.DrawImageUnscaled(B1UpRight, w - B1UpRight.Width / 2, 0);
			g.DrawImageUnscaled(B1DownLeft, 0, h - B1DownLeft.Height / 2);
			g.DrawImageUnscaled(B1DownRight, w - B1DownRight.Width / 2, h - B1DownRight.Height / 2);
		}

		private void OnResize(object sender, System.EventArgs e)
		{
			this.Refresh();
		}
	}
}
