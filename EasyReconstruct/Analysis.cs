using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;

namespace SySal.Executables.EasyReconstruct
{
	/// <summary>
	/// Analysis form - allows quick data analysis of quantities relevant for alignment.
	/// </summary>
	/// <remarks>
	/// <para>The functions supported are those of <see cref="NumericalTools.AnalysisControl">StatisticalAnalysisManager</see>.</para>
	/// </remarks>
	public class AnalysisForm : System.Windows.Forms.Form
	{
		private System.Windows.Forms.MainMenu mainMenu1;
		private System.Windows.Forms.MenuItem menuItem1;
		private System.Windows.Forms.MenuItem menuItem2;
		private System.Windows.Forms.SaveFileDialog saveFileDialog1;
		private System.Windows.Forms.MenuItem menuItem3;
        internal NumericalTools.AnalysisControl analysisControl1;
        private IContainer components;

		public AnalysisForm()
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
            this.components = new System.ComponentModel.Container();
            this.mainMenu1 = new System.Windows.Forms.MainMenu(this.components);
            this.menuItem1 = new System.Windows.Forms.MenuItem();
            this.menuItem2 = new System.Windows.Forms.MenuItem();
            this.menuItem3 = new System.Windows.Forms.MenuItem();
            this.saveFileDialog1 = new System.Windows.Forms.SaveFileDialog();
            this.analysisControl1 = new NumericalTools.AnalysisControl();
            this.SuspendLayout();
            // 
            // mainMenu1
            // 
            this.mainMenu1.MenuItems.AddRange(new System.Windows.Forms.MenuItem[] {
            this.menuItem1});
            // 
            // menuItem1
            // 
            this.menuItem1.Index = 0;
            this.menuItem1.MenuItems.AddRange(new System.Windows.Forms.MenuItem[] {
            this.menuItem2,
            this.menuItem3});
            this.menuItem1.Text = "File";
            // 
            // menuItem2
            // 
            this.menuItem2.Index = 0;
            this.menuItem2.Text = "Save Plot";
            this.menuItem2.Click += new System.EventHandler(this.menuItem2_Click);
            // 
            // menuItem3
            // 
            this.menuItem3.Index = 1;
            this.menuItem3.Text = "Save DataSet";
            this.menuItem3.Click += new System.EventHandler(this.menuItem3_Click);
            // 
            // saveFileDialog1
            // 
            this.saveFileDialog1.FileName = "doc1";
            // 
            // analysisControl1
            // 
            this.analysisControl1.CurrentDataSet = -1;
            this.analysisControl1.LabelFont = new System.Drawing.Font("Comic Sans MS", 9F, System.Drawing.FontStyle.Bold);
            this.analysisControl1.Location = new System.Drawing.Point(0, 0);
            this.analysisControl1.Name = "analysisControl1";
            this.analysisControl1.Palette = NumericalTools.Plot.PaletteType.RGBContinuous;
            this.analysisControl1.Panel = null;
            this.analysisControl1.PanelFont = new System.Drawing.Font("Comic Sans MS", 9F, System.Drawing.FontStyle.Bold);
            this.analysisControl1.PanelFormat = null;
            this.analysisControl1.PanelX = 1;
            this.analysisControl1.PanelY = 0;
            this.analysisControl1.PlotColor = System.Drawing.Color.Red;
            this.analysisControl1.Size = new System.Drawing.Size(966, 679);
            this.analysisControl1.TabIndex = 0;
            // 
            // Form3
            // 
            this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
            this.ClientSize = new System.Drawing.Size(963, 670);
            this.Controls.Add(this.analysisControl1);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
            this.MaximizeBox = false;
            this.Menu = this.mainMenu1;
            this.Name = "Form3";
            this.Text = "Analysis";
            this.ResumeLayout(false);

		}
		#endregion

		private void menuItem2_Click(object sender, System.EventArgs e)
		{
			
			saveFileDialog1.CheckPathExists=true;
			saveFileDialog1.Filter= "Graphics Interchange Format (*.gif)|*.gif|" +
				"Joint Photographic Experts Group (*.jpeg)|*.jpeg|" +
				"Bitmap (*.bmp)|*.bmp|" +
				"Portable Network Graphics (*.png)|*.png|" +
				"Windows Metafile (*.wmf)|*.wmf";

			//saveFileDialog1.FilterIndex=1;
			saveFileDialog1.ShowDialog();

			string fname=saveFileDialog1.FileName;
			if(fname=="") return;

			Bitmap b = analysisControl1.BitmapPlot;
			if(saveFileDialog1.FilterIndex==1)
				b.Save(fname, System.Drawing.Imaging.ImageFormat.Gif);
			else if(saveFileDialog1.FilterIndex==2)
				b.Save(fname, System.Drawing.Imaging.ImageFormat.Jpeg);
			else if(saveFileDialog1.FilterIndex==3)
				b.Save(fname, System.Drawing.Imaging.ImageFormat.Bmp);
			else if(saveFileDialog1.FilterIndex==4)
				b.Save(fname, System.Drawing.Imaging.ImageFormat.Png);
			else if(saveFileDialog1.FilterIndex==5)
				b.Save(fname, System.Drawing.Imaging.ImageFormat.Wmf);


		}

		private void menuItem3_Click(object sender, System.EventArgs e)
		{
			SaveFileDialog sf = new SaveFileDialog();
			sf.Filter = "Text files (*.txt)|*.txt|All files (*.*)|*.*";
			sf.FilterIndex = 0;
			sf.CheckPathExists = true;
			sf.OverwritePrompt = true;
			if (sf.ShowDialog() == DialogResult.OK)	analysisControl1.DumpCurrentDataSetIntoFile(sf.FileName);

		}
	}
}
