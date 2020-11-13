using System;
using System.Collections;
using System.ComponentModel;
using System.Drawing;
using System.Data;
using System.Windows.Forms;

namespace SySal.Controls
{
	/// <summary>
	/// Summary description for RadioButton.
	/// </summary>
	public class RadioButton : System.Windows.Forms.UserControl
	{
		/// <summary> 
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;

		public RadioButton()
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
			// RadioButton
			// 
			this.BackColor = System.Drawing.Color.White;
			this.Name = "RadioButton";
			this.Size = new System.Drawing.Size(16, 16);
			this.Resize += new System.EventHandler(this.OnResize);
			this.Paint += new System.Windows.Forms.PaintEventHandler(this.OnPaint);
			this.MouseDown += new System.Windows.Forms.MouseEventHandler(this.OnMouseDown);

		}
		#endregion


		static System.Drawing.Image BOn = LoadImage("radio_filled.bmp");
		static System.Drawing.Image BOff = LoadImage("radio_empty.bmp");

		static Image LoadImage(string resname)
		{			
			System.IO.Stream myStream;
			System.Reflection.Assembly myAssembly = System.Reflection.Assembly.GetExecutingAssembly();
			myStream = myAssembly.GetManifestResourceStream("SySal.Controls." + resname);
			Image im = new Bitmap(myStream);
			myStream.Close();
			return im;
		}

		internal bool IsChecked;

		public bool Checked 
		{ 
			get 
			{ 
				return IsChecked; 
			} 
			set 
			{ 
				IsChecked = value;
				if (IsChecked) GroupUncheck();
				this.Refresh(); 
			} 
		}

		internal void GroupUncheck()
		{
			if (Parent != null && Parent.Controls != null)
				foreach (Control ctl in Parent.Controls)
				{
					if (ctl.GetType() == typeof(RadioButton) && ctl != this)
					{
						RadioButton r = (RadioButton)ctl;
						r.IsChecked = false;
						r.Refresh();
					}
				}
		}

		private void OnPaint(object sender, System.Windows.Forms.PaintEventArgs e)
		{
			e.Graphics.DrawImage(IsChecked ? BOn : BOff, 0, 0, 16, 16);		
		}

		private void OnMouseDown(object sender, System.Windows.Forms.MouseEventArgs e)
		{
			Checked = true;
		}

		private void OnResize(object sender, System.EventArgs e)
		{
			this.Width = BOn.Width;
			this.Height = BOn.Height;
		}
	}
}
