using System;
using System.Collections;
using System.ComponentModel;
using System.Drawing;
using System.Data;
using System.Windows.Forms;

namespace SySal.Controls
{
	/// <summary>
	/// Summary description for ProgressBar.
	/// </summary>
	public class ProgressBar : System.Windows.Forms.UserControl
	{
		/// <summary> 
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;

		static System.Drawing.Brush BkgndBrush = new SolidBrush(Color.FromArgb(126, 181, 232));

		static System.Drawing.Image PGreyBody = LoadImage("progress_grey_body.png");
		static System.Drawing.Image PGreyHead = LoadImage("progress_grey_head.png");
		static System.Drawing.Image PGreyTail = LoadImage("progress_grey_tail.png");
		static System.Drawing.Image PRedBody = LoadImage("progress_red_body.png");
		static System.Drawing.Image PRedHead = LoadImage("progress_red_head.png");
		static System.Drawing.Image PRedTail = LoadImage("progress_red_tail.png");

		static Image LoadImage(string resname)
		{			
			try
			{
				System.IO.Stream myStream;
				System.Reflection.Assembly myAssembly = System.Reflection.Assembly.GetExecutingAssembly();
				myStream = myAssembly.GetManifestResourceStream("SySal.Controls." + resname);
				Image im = new Bitmap(myStream);
				myStream.Close();
				return im;
			}
			catch (Exception x)
			{
				throw new Exception("Unable to load " + resname + "\r\n" + x.Message);
			}
		}

		public ProgressBar()
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

		protected double m_Percent = 0.0;

		public double Percent 
		{ 
			get { return m_Percent; } 
			set 
			{ 
				m_Percent = value; 
				if (m_Percent < 0.0) m_Percent = 0.0;
				else if (m_Percent > 1.0) m_Percent = 1.0;
				this.Refresh(); 
			} 
		}

		#region Component Designer generated code
		/// <summary> 
		/// Required method for Designer support - do not modify 
		/// the contents of this method with the code editor.
		/// </summary>
		private void InitializeComponent()
		{
			// 
			// ProgressBar
			// 
			this.BackColor = System.Drawing.Color.White;
			this.Name = "ProgressBar";
			this.Size = new System.Drawing.Size(128, 8);
			this.Resize += new System.EventHandler(this.OnResize);
			this.Paint += new System.Windows.Forms.PaintEventHandler(this.OnPaint);

		}
		#endregion

		private void OnPaint(object sender, System.Windows.Forms.PaintEventArgs e)
		{
			int h, w, aw, hd, bw, bb, ws;			
			System.Drawing.Graphics g = e.Graphics;
			h = this.Height;
			w = this.Width;
			aw = w - (PGreyHead.Width + PGreyTail.Width);
			hd = (h - PGreyBody.Height) / 2;			
			g.FillRectangle(BkgndBrush, 0, 0, w, h);			
			if (w > 0)
				g.DrawImage(PGreyBody, 0, hd, w, PGreyBody.Height);			
			g.DrawImageUnscaled(PGreyHead, 0, hd);			
			g.DrawImageUnscaled(PGreyTail, w - PGreyTail.Width, hd);
			if (m_Percent > 0.0)
			{
				bw = (int)(Percent * aw);
				if (bw > aw) bw = aw;				
				g.DrawImageUnscaled(PRedHead, 0, hd);
				ws = PRedBody.Width;				
				for (bb = ws; bb <= bw; bb += PRedBody.Width)
					g.DrawImageUnscaled(PRedBody, PRedHead.Width + bb - ws, hd);				
				if (bb - ws < bw)
					g.DrawImage(PRedBody, PRedHead.Width + bb - ws, hd, bw - bb + ws, PRedBody.Height);				
				g.DrawImageUnscaled(PRedTail, bw + PRedHead.Width, hd);
			}
		}

		private void OnResize(object sender, System.EventArgs e)
		{
			this.Height = PGreyBody.Height;
			this.Refresh();
		}
	}
}
