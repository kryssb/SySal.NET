using System;
using System.Collections;
using System.ComponentModel;
using System.Drawing;
using System.Data;
using System.Windows.Forms;

namespace SySal.Controls
{
	/// <summary>
	/// Summary description for GroupBox.
	/// </summary>
	public class GroupBox : System.Windows.Forms.UserControl
	{
		/// <summary> 
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;

		public GroupBox()
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
			// GroupBox
			// 
			this.BackColor = System.Drawing.Color.White;
			this.Name = "GroupBox";
			this.Resize += new System.EventHandler(this.OnResize);
			this.Paint += new System.Windows.Forms.PaintEventHandler(this.OnPaint);
			this.DoubleClick += new System.EventHandler(this.OnDoubleClick);

		}
		#endregion

		static System.Drawing.Image GUpLeft = LoadImage("ul");
		static System.Drawing.Image GUp = LoadImage("up");
		static System.Drawing.Image GUpRight = LoadImage("ur");
		static System.Drawing.Image GRight = LoadImage("ri");
		static System.Drawing.Image GDownRight = LoadImage("dr");
		static System.Drawing.Image GDown = LoadImage("do");
		static System.Drawing.Image GDownLeft = LoadImage("dl");
		static System.Drawing.Image GLeft = LoadImage("le");		

		static System.Drawing.Brush BkgndBrush = new SolidBrush(Color.FromArgb(126, 181, 232));

		static System.Drawing.Font GFont = new Font("Arial", 8);		

		static System.Drawing.Brush GTextBrush = new SolidBrush(System.Drawing.Color.White);

		static Image LoadImage(string resname)
		{			
			System.IO.Stream myStream;
			System.Reflection.Assembly myAssembly = System.Reflection.Assembly.GetExecutingAssembly();
			myStream = myAssembly.GetManifestResourceStream("SySal.Controls.group_" + resname + ".png");
			Image im = new Bitmap(myStream);
			myStream.Close();
			return im;
		}

		protected bool m_IsStatic = true;

		public bool IsStatic
		{
			get { return m_IsStatic; }
			set 
			{
				if (m_IsStatic == value || DesignMode == false)
				{
					m_IsStatic = value;
					return;
				}
				m_IsStatic = value;
				m_IsOpen = true;
				this.Refresh();
			}
		}

		public delegate void dOpenCloseEvent(GroupBox sender, bool isopen);

		protected bool m_IsOpen = true;		

		public bool IsOpen
		{
			get { return m_IsOpen; }
			set
			{
				m_IsOpen = value;
				if (m_IsOpen)
				{				
					this.SuspendLayout();
					this.Location = new Point(m_OpenPosition.Left, m_OpenPosition.Top);					
					this.Width = m_OpenPosition.Width;
					this.Height = m_OpenPosition.Height;
					foreach (System.Windows.Forms.Control ctl in Controls) ctl.Visible = true;
					this.ResumeLayout();
					this.Refresh();
				}
				else
				{
					this.SuspendLayout();
					this.Location = new Point(m_ClosedPosition.Left, m_ClosedPosition.Top);
					this.Width = m_ClosedPosition.Width;
					this.Height = m_ClosedPosition.Height;
					foreach (System.Windows.Forms.Control ctl in Controls) ctl.Visible = false;
					this.ResumeLayout();
					this.Refresh();
				}
			}
		}

		protected Rectangle m_OpenPosition;

		public Rectangle OpenPosition
		{
			get { return m_OpenPosition; }
			set { m_OpenPosition = value; this.Refresh(); }
		}

		protected Rectangle m_ClosedPosition;

		public Rectangle ClosedPosition
		{
			get { return m_ClosedPosition; }
			set { m_ClosedPosition = value; this.Refresh(); }
		}

		public event dOpenCloseEvent OpenCloseEvent;

		protected string m_LabelText = "";

		public string LabelText
		{
			get { return (string)m_LabelText.Clone(); }
			set
			{
				if (value == null) m_LabelText = "";
				else m_LabelText = (string)value.Clone();
				this.Refresh();
			}
		}

		private void OnPaint(object sender, System.Windows.Forms.PaintEventArgs e)
		{
			System.Drawing.Graphics g = e.Graphics;
			g.TextRenderingHint = System.Drawing.Text.TextRenderingHint.AntiAlias;
			SizeF textsize = g.MeasureString(m_LabelText, GFont);

			int w = this.Width - 2;
			int h = this.Height - 2;
			int hl = (int)textsize.Height + (GUpLeft.Height + GDownLeft.Height) / 2;
			int h2 = h - hl / 2;
			int hs = ((int)textsize.Height + GUp.Height) / 2;			
			g.FillRectangle(BkgndBrush, 0, 0, w + 2, h + 2);			
			int s;
			for (s = GUpLeft.Width; s < w; s += GUp.Width / 2)
			{				
				g.DrawImageUnscaled(GDown, s, h - GDown.Height / 2);
			}
			g.DrawImage(GLeft, GUpLeft.Width, 0, GLeft.Width, hl);
			g.DrawImage(GRight, GUpLeft.Width + (int)textsize.Width + GLeft.Width, 0, GRight.Width, hl);
			g.DrawImageUnscaled(GUpLeft, GUpLeft.Width, 0, GUpLeft.Width, GUpLeft.Height);
			g.DrawImageUnscaled(GDownLeft, GUpLeft.Width, GUpLeft.Height + (int)textsize.Height, GDownLeft.Width, GDownLeft.Height);
			for (s = 2 * GUpLeft.Width; (s + GUp.Width) < 2 * GUpLeft.Width + (int)textsize.Width + GRight.Width; s += GUp.Width / 2)
			{				
				g.DrawImageUnscaled(GUp, s, 0);
				g.DrawImageUnscaled(GDown, s, GUp.Height + (int)textsize.Height);
			}
			g.DrawImage(GUp, s, 0, (2 * GUpLeft.Width + (int)textsize.Width + GRight.Width - s), GUp.Height);
			g.DrawImage(GDown, s, GUp.Height + (int)textsize.Height, (2 * GUpLeft.Width + (int)textsize.Width + GRight.Width - s), GDown.Height);
			for (s = 2 * GUpLeft.Width + (int)textsize.Width + GRight.Width; s < w; s += GUp.Width / 2)
			{				
				g.DrawImageUnscaled(GUp, s, hs);
			}
			g.DrawImageUnscaled(GUpRight, 2 * GUpLeft.Width + (int)textsize.Width, 0, GUpRight.Width, GUpRight.Height);
			g.DrawImageUnscaled(GDownRight, 2 * GUpLeft.Width + (int)textsize.Width, GUpRight.Height + (int)textsize.Height, GDownRight.Width, GDownRight.Height);			
			for (s = GUpLeft.Height + hs; s < h; s += GLeft.Height / 2)
			{
				g.DrawImageUnscaled(GLeft, 0, s);
				g.DrawImageUnscaled(GRight, w - GRight.Width / 2, s);
			}
			g.DrawImageUnscaled(GUpLeft, 0, hs);
			g.DrawImageUnscaled(GUpRight, w - GUpRight.Width / 2, hs);
			g.DrawImageUnscaled(GDownLeft, 0, h - GDownLeft.Height / 2);
			g.DrawImageUnscaled(GDownRight, w - GDownRight.Width / 2, h - GDownRight.Height / 2);
			g.DrawString(m_LabelText, GFont, GTextBrush, GUpLeft.Width + GLeft.Width, GUp.Height);
		}

		private void OnResize(object sender, System.EventArgs e)
		{
			this.Refresh();
		}

		private void OnDoubleClick(object sender, System.EventArgs e)
		{
			if (m_IsStatic == false) 
			{
				IsOpen = !IsOpen;
				if (OpenCloseEvent != null) OpenCloseEvent(this, m_IsOpen);
			}
		}

		public void AdoptChild(System.Windows.Forms.Control child)
		{
			Point p = new Point();
			p.X = (child.Location.X - this.Location.X);
			p.Y = (child.Location.Y - this.Location.Y);
			child.Location = p;
			this.Controls.Add(child);
		}
	}
}
