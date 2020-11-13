using System;
using System.Collections;
using System.ComponentModel;
using System.Drawing;
using System.Data;
using System.Windows.Forms;

namespace SySal.Controls
{
	/// <summary>
	/// Summary description for UserControl1.
	/// </summary>
	public class Button : System.Windows.Forms.UserControl
	{
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;

		public Button()
		{
			// This call is required by the Windows.Forms Form Designer.
			InitializeComponent();

			// TODO: Add any initialization after the InitComponent call
		}

		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		protected override void Dispose( bool disposing )
		{
			if( disposing )
			{
				if( components != null )
					components.Dispose();
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
			// Button
			// 
			this.BackColor = System.Drawing.Color.White;
			this.Name = "Button";
			this.Size = new System.Drawing.Size(24, 24);
			this.Resize += new System.EventHandler(this.OnResize);
			this.MouseUp += new System.Windows.Forms.MouseEventHandler(this.OnMouseUp);
			this.Paint += new System.Windows.Forms.PaintEventHandler(this.OnPaint);
			this.MouseLeave += new System.EventHandler(this.OnMouseLeave);
			this.MouseDown += new System.Windows.Forms.MouseEventHandler(this.OnMouseDown);

		}
		#endregion

		static System.Drawing.Font B1Font = new Font("Arial", 10);		

		static System.Drawing.Image B1UpLeft = LoadImage("b1ul.bmp");
		static System.Drawing.Image B1Up = LoadImage("b1up.bmp");
		static System.Drawing.Image B1UpRight = LoadImage("b1ur.bmp");
		static System.Drawing.Image B1Right = LoadImage("b1ri.bmp");
		static System.Drawing.Image B1DownRight = LoadImage("b1dr.bmp");
		static System.Drawing.Image B1Down = LoadImage("b1do.bmp");
		static System.Drawing.Image B1DownLeft = LoadImage("b1dl.bmp");
		static System.Drawing.Image B1Left = LoadImage("b1le.bmp");
		static System.Drawing.Image B1Center = LoadImage("b1ce.bmp");
		static System.Drawing.Image B2UpLeft = LoadImage("b2ul.bmp");
		static System.Drawing.Image B2Up = LoadImage("b2up.bmp");
		static System.Drawing.Image B2UpRight = LoadImage("b2ur.bmp");
		static System.Drawing.Image B2Right = LoadImage("b2ri.bmp");
		static System.Drawing.Image B2DownRight = LoadImage("b2dr.bmp");
		static System.Drawing.Image B2Down = LoadImage("b2do.bmp");
		static System.Drawing.Image B2DownLeft = LoadImage("b2dl.bmp");
		static System.Drawing.Image B2Left = LoadImage("b2le.bmp");
		static System.Drawing.Image B2Center = LoadImage("b2ce.bmp");		
				
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

		System.Drawing.Brush B1TextBrush = new SolidBrush(System.Drawing.Color.White);
		bool IsDown;

		protected string m_Text = "";

		public string ButtonText { get { return (string)m_Text.Clone(); } set { m_Text = (string)value.Clone(); this.Refresh(); } }

		private void OnPaint(object sender, System.Windows.Forms.PaintEventArgs e)
		{			
			int w = this.Width;
			int h = this.Height;
			System.Drawing.Graphics g = e.Graphics;
			if (IsDown)
			{
				int w2 = w - (B2UpLeft.Width + B2UpRight.Width) / 2;
				int h2 = h - (B2UpLeft.Height + B2DownLeft.Height) / 2;				
				g.FillRectangle(BkgndBrush, B2UpLeft.Width / 2, B2UpLeft.Height / 2, w2 + 1, h2 + 1);
				g.DrawImage(B2Left, 0, 0, B2Left.Width / 2, h + 32);
				g.DrawImage(B2Right, w - B2Right.Width / 2, 0, B2Right.Width, h + 32);
				g.DrawImage(B2Up, 0, 0, w + 32, B2Up.Height / 2);
				g.DrawImage(B2Down, 0, h - B2Down.Height / 2, w + 32, B2Down.Height / 2);
				g.DrawImageUnscaled(B2UpLeft, 0, 0);
				g.DrawImageUnscaled(B2UpRight, w - B2UpRight.Width / 2, 0);
				g.DrawImageUnscaled(B2DownLeft, 0, h - B2DownLeft.Height / 2);
				g.DrawImageUnscaled(B2DownRight, w - B2DownRight.Width / 2, h - B2DownRight.Height / 2);
			}
			else
			{
				int w2 = w - (B2UpLeft.Width + B2UpRight.Width) / 2;
				int h2 = h - (B2UpLeft.Height + B2DownLeft.Height) / 2;				
				g.FillRectangle(BkgndBrush, B1UpLeft.Width / 2, B1UpLeft.Height / 2, w2 + 1, h2 + 1);
				g.DrawImage(B1Left, 0, 0, B1Left.Width / 2, h + 32);
				g.DrawImage(B1Right, w - B1Right.Width / 2, 0, B1Right.Width, h + 32);
				g.DrawImage(B1Up, 0, 0, w + 32, B1Up.Height / 2);
				g.DrawImage(B1Down, 0, h - B1Down.Height / 2, w + 32, B1Down.Height / 2);
				g.DrawImageUnscaled(B1UpLeft, 0, 0);
				g.DrawImageUnscaled(B1UpRight, w - B1UpRight.Width / 2, 0);
				g.DrawImageUnscaled(B1DownLeft, 0, h - B1DownLeft.Height / 2);
				g.DrawImageUnscaled(B1DownRight, w - B1DownRight.Width / 2, h - B1DownRight.Height / 2);
			}
			if (m_Text.Length > 0)
			{
				g.TextRenderingHint = System.Drawing.Text.TextRenderingHint.AntiAlias;
				SizeF textsize = g.MeasureString(m_Text, B1Font);
				if (IsDown) g.DrawString(m_Text, B1Font, B1TextBrush, (w - textsize.Width) * 0.5f + 1, (h - textsize.Height)* 0.5f + 1);
				else g.DrawString(m_Text, B1Font, B1TextBrush, (w - textsize.Width) * 0.5f, (h - textsize.Height) * 0.5f);
			}
		}

		private void OnMouseDown(object sender, System.Windows.Forms.MouseEventArgs e)
		{
			IsDown = true;
			this.Refresh();
		}

		private void OnMouseLeave(object sender, System.EventArgs e)
		{
			IsDown = false;
			this.Refresh();
		}

		private void OnMouseUp(object sender, System.Windows.Forms.MouseEventArgs e)
		{
			IsDown = false;
			this.Refresh();	
		}

		private void OnResize(object sender, System.EventArgs e)
		{
			this.Refresh();
		}
	}
}
