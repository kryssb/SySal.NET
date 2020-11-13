using System;
using System.Collections;
using System.ComponentModel;
using System.Drawing;
using System.Data;
using System.Windows.Forms;

namespace SySal.Controls
{
	/// <summary>
	/// Summary description for PercentChart.
	/// </summary>
	public class PercentChart : System.Windows.Forms.UserControl
	{
		/// <summary> 
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;

		public PercentChart()
		{
			// This call is required by the Windows.Forms Form Designer.
			InitializeComponent();

			// TODO: Add any initialization after the InitializeComponent call			
			m_Items = new PercentChartItemCollection(this);
		}

		internal void ViewRefresh()
		{
			PageTotal = m_Items.Count * 16;
			this.Refresh();
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
			this.ScrollUp = new System.Windows.Forms.Panel();
			this.ScrollDown = new System.Windows.Forms.Panel();
			this.SuspendLayout();
			// 
			// ScrollUp
			// 
			this.ScrollUp.Location = new System.Drawing.Point(128, 0);
			this.ScrollUp.Name = "ScrollUp";
			this.ScrollUp.Size = new System.Drawing.Size(16, 16);
			this.ScrollUp.TabIndex = 0;
			this.ScrollUp.Click += new System.EventHandler(this.OnScrollUpClick);
			this.ScrollUp.MouseUp += new System.Windows.Forms.MouseEventHandler(this.OnScrollUpMouseUp);
			this.ScrollUp.Paint += new System.Windows.Forms.PaintEventHandler(this.OnUpPaint);
			this.ScrollUp.DoubleClick += new System.EventHandler(this.OnScrollUpClick);
			this.ScrollUp.MouseLeave += new System.EventHandler(this.OnScrollUpMouseLeave);
			this.ScrollUp.MouseDown += new System.Windows.Forms.MouseEventHandler(this.OnScrollUpMouseDown);
			// 
			// ScrollDown
			// 
			this.ScrollDown.Location = new System.Drawing.Point(128, 128);
			this.ScrollDown.Name = "ScrollDown";
			this.ScrollDown.Size = new System.Drawing.Size(16, 16);
			this.ScrollDown.TabIndex = 1;
			this.ScrollDown.Click += new System.EventHandler(this.OnScrollDownClick);
			this.ScrollDown.MouseUp += new System.Windows.Forms.MouseEventHandler(this.OnScrollDownMouseUp);
			this.ScrollDown.Paint += new System.Windows.Forms.PaintEventHandler(this.OnDownPaint);
			this.ScrollDown.DoubleClick += new System.EventHandler(this.OnScrollDownClick);
			this.ScrollDown.MouseLeave += new System.EventHandler(this.OnScrollDownMouseLeave);
			this.ScrollDown.MouseDown += new System.Windows.Forms.MouseEventHandler(this.OnScrollDownMouseDown);
			// 
			// PercentChart
			// 
			this.BackColor = System.Drawing.Color.White;
			this.Controls.Add(this.ScrollDown);
			this.Controls.Add(this.ScrollUp);
			this.Name = "PercentChart";
			this.Resize += new System.EventHandler(this.OnResize);
			this.Paint += new System.Windows.Forms.PaintEventHandler(this.OnPaint);
			this.ResumeLayout(false);

		}
		#endregion

		static System.Drawing.Image SUpReleased = LoadImage("scroll_up_released");
		static System.Drawing.Image SUpPressed = LoadImage("scroll_up_pressed");
		static System.Drawing.Image SDownReleased = LoadImage("scroll_down_released");
		static System.Drawing.Image SDownPressed = LoadImage("scroll_down_pressed");
		static System.Drawing.Image RoundCornerUpLeft = LoadImage("round_ul");
		static System.Drawing.Image RoundCornerDownLeft = LoadImage("round_dl");
		static System.Drawing.Image [,] BarImages = MultiLoadImage(new string [] {"red", "yellow", "green", "blue", "orange", "cyan", "magenta", "grey" });
		private System.Windows.Forms.Panel ScrollUp;
		private System.Windows.Forms.Panel ScrollDown;

		static Image LoadImage(string resname)
		{			
			System.IO.Stream myStream;
			System.Reflection.Assembly myAssembly = System.Reflection.Assembly.GetExecutingAssembly();
			myStream = myAssembly.GetManifestResourceStream("SySal.Controls." + resname + ".png");
			Image im = new Bitmap(myStream);
			myStream.Close();
			return im;
		}

		static Image [,] MultiLoadImage(string [] colors)
		{
			Image [,] ret = new Image[colors.Length, 3];
			int i;
			for (i = 0; i < colors.Length; i++)
			{
				ret[i, 0] = LoadImage("bar_" + colors[i] + "_head");
				ret[i, 1] = LoadImage("bar_" + colors[i] + "_body");
				ret[i, 2] = LoadImage("bar_" + colors[i] + "_tail");
			}
			return ret;
		}

		static System.Drawing.Font PFont = new Font("Arial", 8);	

		static System.Drawing.Brush PTextBrush = new SolidBrush(System.Drawing.Color.Navy);

		bool ScrollUpIsPressed = false;
		bool ScrollDownIsPressed = false;
		int PageStart = 0;
		int PageTotal = 0;
		const int PageStep = 4;

		private void OnScrollUpMouseDown(object sender, System.Windows.Forms.MouseEventArgs e)
		{
			ScrollUpIsPressed = true;
			ScrollUp.Refresh();
		}

		private void OnScrollUpMouseLeave(object sender, System.EventArgs e)
		{
			ScrollUpIsPressed = false;
			ScrollUp.Refresh();
		}

		private void OnScrollUpMouseUp(object sender, System.Windows.Forms.MouseEventArgs e)
		{
			ScrollUpIsPressed = false;
			ScrollUp.Refresh();
		}

		private void OnScrollDownMouseDown(object sender, System.Windows.Forms.MouseEventArgs e)
		{
			ScrollDownIsPressed = true;
			ScrollDown.Refresh();
		}

		private void OnScrollDownMouseLeave(object sender, System.EventArgs e)
		{
			ScrollDownIsPressed = false;
			ScrollDown.Refresh();
		}

		private void OnScrollDownMouseUp(object sender, System.Windows.Forms.MouseEventArgs e)
		{
			ScrollDownIsPressed = false;
			ScrollDown.Refresh();
		}

		private void OnResize(object sender, System.EventArgs e)
		{
			ScrollUp.Left = this.Width - ScrollUp.Width;
			ScrollUp.Top = 0;
			ScrollDown.Left = this.Width - ScrollDown.Width;
			ScrollDown.Top = this.Height - ScrollDown.Height;
			if (PageStart > PageTotal - this.Height)
			{
				PageStart = PageTotal - this.Height;
				if (PageStart < 0) PageStart = 0;
			}
			this.Refresh();
		}

		private void OnUpPaint(object sender, System.Windows.Forms.PaintEventArgs e)
		{
			e.Graphics.DrawImageUnscaled(ScrollUpIsPressed ? SUpPressed : SUpReleased, 0, 0, SUpReleased.Width, SUpReleased.Height);
		}

		private void OnDownPaint(object sender, System.Windows.Forms.PaintEventArgs e)
		{
			e.Graphics.DrawImageUnscaled(ScrollDownIsPressed ? SDownPressed : SDownReleased, 0, 0, SDownReleased.Width, SDownReleased.Height);		
		}

		private void OnDownLeftPaint(object sender, System.Windows.Forms.PaintEventArgs e)
		{
			e.Graphics.DrawImageUnscaled(RoundCornerDownLeft, 0, 0, RoundCornerDownLeft.Width, RoundCornerDownLeft.Height);
		}

		private void OnUpLeftPaint(object sender, System.Windows.Forms.PaintEventArgs e)
		{
			e.Graphics.DrawImageUnscaled(RoundCornerUpLeft, 0, 0, RoundCornerUpLeft.Width, RoundCornerUpLeft.Height);		
		}

		private void OnScrollUpClick(object sender, System.EventArgs e)
		{
			PageStart -= PageStep;
			if (PageStart < 0) PageStart = 0;
			this.Refresh();
		}

		private void OnScrollDownClick(object sender, System.EventArgs e)
		{
			PageStart += PageStep;
			if (PageStart >= PageTotal - this.Height) 
			{
				PageStart = PageTotal - this.Height;
				if (PageStart < 0) PageStart = 0;
			}
			this.Refresh();
		}

		protected PercentChartItemCollection m_Items;

		private void OnPaint(object sender, System.Windows.Forms.PaintEventArgs e)
		{
			System.Drawing.Graphics g = e.Graphics;
			g.TextRenderingHint = System.Drawing.Text.TextRenderingHint.AntiAlias;			
			int wi = 0;
			if (m_Items.Count > 0)
			{
				SizeF [] size_entries = new SizeF[m_Items.Count];
				int i;
				for (i = 0; i < m_Items.Count; i++)
				{
					string s = ((PercentChartItem)(m_Items[i])).Text;
					size_entries[i] = g.MeasureString(s, PFont);
					if (size_entries[i].Width > wi) wi = (int)(size_entries[i].Width);
				}
				int ps = this.Width - RoundCornerUpLeft.Width - SUpReleased.Width;
				for (i = 0; i < m_Items.Count; i++)
				{
					PercentChartItem pi = (PercentChartItem)m_Items[i];
					g.DrawString(pi.Text, PFont, PTextBrush, 2 + wi - size_entries[i].Width, 16 * i - PageStart);
					if (pi.Percent > 0.0)
					{
						int bs;
						int ci = i % 8;
						int pblen = (int)(pi.Percent * (ps - wi - 8 - BarImages[ci, 0].Width - BarImages[ci, 2].Width));
						int pend = pblen + wi + 4 + BarImages[ci, 0].Width;
						for (bs = wi + 4 + BarImages[ci, 0].Width; (bs + BarImages[ci, 1].Width) < pend; bs += (int)BarImages[ci, 1].Width)
							g.DrawImageUnscaled(BarImages[ci, 1], bs, i * 16 - PageStart);
						g.DrawImage(BarImages[ci, 1], bs, i * 16 - PageStart, pend - bs, BarImages[ci, 1].Height);
						g.DrawImageUnscaled(BarImages[ci, 0], wi + 4, i * 16 - PageStart);
						g.DrawImageUnscaled(BarImages[ci, 2], pblen + wi + 4 + BarImages[ci, 0].Width , i * 16 - PageStart);
					}
				}				
			}
			g.DrawImageUnscaled(RoundCornerUpLeft, 0, 0);
			g.DrawImageUnscaled(RoundCornerDownLeft, 0, this.Height - RoundCornerDownLeft.Height);
		}

		public PercentChartItemCollection Items { get { return m_Items; } }
	}

	public class PercentChartItem : ICloneable
	{
		public string Text;
		public double Percent;

		public PercentChartItem(string t, double p) { Text = (string)t.Clone(); Percent = p; }

		#region ICloneable Members

		public object Clone()
		{
			return new PercentChartItem(Text, Percent);
		}

		#endregion
	}

	public class PercentChartItemCollection : ICollection
	{
		PercentChart m_Owner;

		public PercentChartItemCollection(PercentChart owner)
		{
			m_Owner = owner;
		}

		#region ICollection Members

		public bool IsSynchronized
		{
			get
			{
				return false;
			}
		}

		System.Collections.ArrayList m_Entries = new System.Collections.ArrayList();

		public int Count
		{
			get
			{				
				return m_Entries.Count;
			}
		}

		public void CopyTo(Array array, int index)
		{
			m_Entries.CopyTo(array, index);
		}

		public object SyncRoot
		{
			get
			{
				return null;
			}
		}

		#endregion

		#region IEnumerable Members

		public IEnumerator GetEnumerator()
		{			
			return m_Entries.GetEnumerator();
		}

		#endregion

		public void Add(PercentChartItem pi)
		{
			m_Entries.Add(pi);
			m_Owner.ViewRefresh();
		}

		public void Insert(int index, PercentChartItem pi)
		{
			m_Entries.Insert(index, pi);
			m_Owner.ViewRefresh();
		}

		public void Clear()
		{
			m_Entries.Clear();
			m_Owner.ViewRefresh();
		}

		public void RemoveAt(int index)
		{
			m_Entries.RemoveAt(index);
			m_Owner.ViewRefresh();
		}

		public PercentChartItem this[int index] { get { return (PercentChartItem)m_Entries[index]; } }
	}		
}
