using System;
using System.Collections;
using System.ComponentModel;
using System.Drawing;
using System.Data;
using System.Windows.Forms;

namespace SySal.Controls
{
	/// <summary>
	/// Summary description for StaticText.
	/// </summary>
	public class StaticText : System.Windows.Forms.UserControl
	{
		/// <summary> 
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;

		public StaticText()
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
			// StaticText
			// 
			this.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(126)), ((System.Byte)(181)), ((System.Byte)(232)));
			this.Name = "StaticText";
			this.Resize += new System.EventHandler(this.OnResize);
			this.Paint += new System.Windows.Forms.PaintEventHandler(this.OnPaint);

		}
		#endregion

		static Brush BkgndBrush = new SolidBrush(Color.FromArgb(126,181,232));

		System.Drawing.Brush TextBrush = new SolidBrush(System.Drawing.Color.White);

		static System.Drawing.Font TFont = new Font("Arial", 10);		

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

		private ContentAlignment m_TextAlign = ContentAlignment.MiddleCenter;

		public ContentAlignment TextAlign 
		{
			get { return m_TextAlign; }
			set
			{
				m_TextAlign = value;
				this.Refresh();
			}
		}

		private void OnPaint(object sender, System.Windows.Forms.PaintEventArgs e)
		{
			System.Drawing.Graphics g = e.Graphics;
			g.TextRenderingHint = System.Drawing.Text.TextRenderingHint.AntiAlias;
			g.FillRectangle(BkgndBrush, 0, 0, this.Width, this.Height);
			SizeF textmeas = g.MeasureString(m_LabelText, TFont);
			int top, left;
			switch (m_TextAlign)
			{
				case ContentAlignment.BottomCenter:		top = (int)(this.Height - textmeas.Height); left = (int)((this.Width - textmeas.Width) / 2); break;
				case ContentAlignment.BottomLeft:		top = (int)(this.Height - textmeas.Height); left = 0; break;
				case ContentAlignment.BottomRight:		top = (int)(this.Height - textmeas.Height); left = (int)(this.Width - textmeas.Width); break;
				case ContentAlignment.MiddleCenter:		top = (int)((this.Height - textmeas.Height) / 2); left = (int)((this.Width - textmeas.Width) / 2); break;
				case ContentAlignment.MiddleLeft:		top = (int)((this.Height - textmeas.Height) / 2); left = 0; break;
				case ContentAlignment.MiddleRight:		top = (int)((this.Height - textmeas.Height) / 2); left = (int)(this.Width - textmeas.Width); break;
				case ContentAlignment.TopCenter:		top = 0; left = (int)((this.Width - textmeas.Width) / 2); break;
				case ContentAlignment.TopLeft:			top = 0; left = 0; break;
				case ContentAlignment.TopRight:			top = 0; left = (int)(this.Width - textmeas.Width); break;
				default:	return;
			}
			g.DrawString(m_LabelText, TFont, TextBrush, left, top);
		}

		private void OnResize(object sender, System.EventArgs e)
		{
			this.Refresh();
		}
	}
}
