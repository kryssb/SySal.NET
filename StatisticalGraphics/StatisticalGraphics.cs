using System;
using System.Collections;
using System.ComponentModel;
using System.Drawing;
using System.Data;
using System.Windows.Forms;

namespace NumericalTools
{
	/// <summary>
	/// Control that shows statistical plots.
	/// </summary>

	public class PlotControl : System.Windows.Forms.UserControl
	{
        internal System.Windows.Forms.GroupBox GroupBox1;
		internal System.Windows.Forms.Label LabelY;
		internal System.Windows.Forms.Label LabelX;

		enum LastPlot 
		{
			Nothing, Histo, HistoSkyline, Scatter, GScatter, SymbolArea,
			GArea, GAreaValues, GAreaCValues, Pie, LEGO, Scatter3D, ScatterHue, ArrowPlot};

		/// <summary>
		/// Required designer variable.
		/// </summary>

		private int ORIGINALWIDTH =568;
		private int ORIGINALHEIGHT=464;

		private System.ComponentModel.Container components = null;

		private Pen myPen = new Pen(System.Drawing.Color.Black, 1);
		private Pen myRedPen = new Pen(System.Drawing.Color.Red, 1);
		private Font myFont = new Font("Arial", 8);
		private Brush myBrush = new SolidBrush(System.Drawing.Color.Black);
		private System.Windows.Forms.Panel panel1;
		private System.Windows.Forms.TextBox m_ResultsBox;
		private System.Windows.Forms.Button m_SetFontButton;
		private System.Windows.Forms.Label label1;
		private System.Windows.Forms.TextBox m_LabelXText;
		private System.Windows.Forms.TextBox m_LabelYText;
		private System.Windows.Forms.Label label2;
		private System.Windows.Forms.TextBox m_FormatText;
		private System.Windows.Forms.Button m_SetAxisFontButton;
		private System.Windows.Forms.Button m_SetPlotColorButton;
		private System.Windows.Forms.TextBox m_PenWidthText;
		private System.Windows.Forms.Label label3;
		private System.Windows.Forms.ColorDialog PlotColorDialog;
        internal Label LabelZ;
        private ComboBox m_Marker;
        private Label label4;
        private TextBox m_MarkerSize;
        private CheckBox m_ShowErrorBars;        
		private Brush myRedBrush = new SolidBrush(System.Drawing.Color.Red);
		//private Graphics gPB;

		public PlotControl()
		{
			// This call is required by the Windows.Forms Form Designer.
			InitializeComponent();

			// TODO: Add any initialization after the InitForm call
			//gPB = panel1.CreateGraphics();
			gA.HistoColor = Color.Red;	

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
            this.GroupBox1 = new System.Windows.Forms.GroupBox();
            this.LabelZ = new System.Windows.Forms.Label();
            this.m_PenWidthText = new System.Windows.Forms.TextBox();
            this.label3 = new System.Windows.Forms.Label();
            this.m_SetPlotColorButton = new System.Windows.Forms.Button();
            this.m_SetAxisFontButton = new System.Windows.Forms.Button();
            this.m_FormatText = new System.Windows.Forms.TextBox();
            this.label2 = new System.Windows.Forms.Label();
            this.m_LabelYText = new System.Windows.Forms.TextBox();
            this.m_LabelXText = new System.Windows.Forms.TextBox();
            this.label1 = new System.Windows.Forms.Label();
            this.m_SetFontButton = new System.Windows.Forms.Button();
            this.m_ResultsBox = new System.Windows.Forms.TextBox();
            this.LabelY = new System.Windows.Forms.Label();
            this.LabelX = new System.Windows.Forms.Label();
            this.panel1 = new System.Windows.Forms.Panel();
            this.PlotColorDialog = new System.Windows.Forms.ColorDialog();
            this.m_Marker = new System.Windows.Forms.ComboBox();
            this.m_MarkerSize = new System.Windows.Forms.TextBox();
            this.label4 = new System.Windows.Forms.Label();
            this.m_ShowErrorBars = new System.Windows.Forms.CheckBox();
            this.GroupBox1.SuspendLayout();
            this.SuspendLayout();
            // 
            // GroupBox1
            // 
            this.GroupBox1.Controls.Add(this.m_ShowErrorBars);
            this.GroupBox1.Controls.Add(this.label4);
            this.GroupBox1.Controls.Add(this.m_MarkerSize);
            this.GroupBox1.Controls.Add(this.m_Marker);
            this.GroupBox1.Controls.Add(this.LabelZ);
            this.GroupBox1.Controls.Add(this.m_PenWidthText);
            this.GroupBox1.Controls.Add(this.label3);
            this.GroupBox1.Controls.Add(this.m_SetPlotColorButton);
            this.GroupBox1.Controls.Add(this.m_SetAxisFontButton);
            this.GroupBox1.Controls.Add(this.m_FormatText);
            this.GroupBox1.Controls.Add(this.label2);
            this.GroupBox1.Controls.Add(this.m_LabelYText);
            this.GroupBox1.Controls.Add(this.m_LabelXText);
            this.GroupBox1.Controls.Add(this.label1);
            this.GroupBox1.Controls.Add(this.m_SetFontButton);
            this.GroupBox1.Controls.Add(this.m_ResultsBox);
            this.GroupBox1.Controls.Add(this.LabelY);
            this.GroupBox1.Controls.Add(this.LabelX);
            this.GroupBox1.Location = new System.Drawing.Point(8, 451);
            this.GroupBox1.Name = "GroupBox1";
            this.GroupBox1.Size = new System.Drawing.Size(646, 136);
            this.GroupBox1.TabIndex = 7;
            this.GroupBox1.TabStop = false;
            // 
            // LabelZ
            // 
            this.LabelZ.Location = new System.Drawing.Point(8, 48);
            this.LabelZ.Name = "LabelZ";
            this.LabelZ.Size = new System.Drawing.Size(152, 16);
            this.LabelZ.TabIndex = 21;
            // 
            // m_PenWidthText
            // 
            this.m_PenWidthText.Location = new System.Drawing.Point(594, 50);
            this.m_PenWidthText.Name = "m_PenWidthText";
            this.m_PenWidthText.Size = new System.Drawing.Size(40, 20);
            this.m_PenWidthText.TabIndex = 20;
            this.m_PenWidthText.Text = "1";
            this.m_PenWidthText.Leave += new System.EventHandler(this.OnPenTextLeave);
            // 
            // label3
            // 
            this.label3.Location = new System.Drawing.Point(546, 50);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(48, 24);
            this.label3.TabIndex = 19;
            this.label3.Text = "Pen:";
            this.label3.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // m_SetPlotColorButton
            // 
            this.m_SetPlotColorButton.Location = new System.Drawing.Point(466, 50);
            this.m_SetPlotColorButton.Name = "m_SetPlotColorButton";
            this.m_SetPlotColorButton.Size = new System.Drawing.Size(64, 24);
            this.m_SetPlotColorButton.TabIndex = 18;
            this.m_SetPlotColorButton.Text = "Plot Color";
            this.m_SetPlotColorButton.Click += new System.EventHandler(this.m_SetPlotColorButton_Click);
            // 
            // m_SetAxisFontButton
            // 
            this.m_SetAxisFontButton.Location = new System.Drawing.Point(466, 18);
            this.m_SetAxisFontButton.Name = "m_SetAxisFontButton";
            this.m_SetAxisFontButton.Size = new System.Drawing.Size(64, 24);
            this.m_SetAxisFontButton.TabIndex = 17;
            this.m_SetAxisFontButton.Text = "Axis Font";
            this.m_SetAxisFontButton.Click += new System.EventHandler(this.m_SetAxisFontButton_Click);
            // 
            // m_FormatText
            // 
            this.m_FormatText.Location = new System.Drawing.Point(594, 82);
            this.m_FormatText.Name = "m_FormatText";
            this.m_FormatText.Size = new System.Drawing.Size(40, 20);
            this.m_FormatText.TabIndex = 16;
            this.m_FormatText.Leave += new System.EventHandler(this.OnFormatTextLeave);
            // 
            // label2
            // 
            this.label2.Location = new System.Drawing.Point(546, 82);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(48, 24);
            this.label2.TabIndex = 15;
            this.label2.Text = "Format:";
            this.label2.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // m_LabelYText
            // 
            this.m_LabelYText.Location = new System.Drawing.Point(594, 106);
            this.m_LabelYText.Name = "m_LabelYText";
            this.m_LabelYText.Size = new System.Drawing.Size(40, 20);
            this.m_LabelYText.TabIndex = 14;
            this.m_LabelYText.Text = "0";
            this.m_LabelYText.Leave += new System.EventHandler(this.OnYLabelTextLeave);
            // 
            // m_LabelXText
            // 
            this.m_LabelXText.Location = new System.Drawing.Point(546, 106);
            this.m_LabelXText.Name = "m_LabelXText";
            this.m_LabelXText.Size = new System.Drawing.Size(40, 20);
            this.m_LabelXText.TabIndex = 13;
            this.m_LabelXText.Text = "1";
            this.m_LabelXText.Leave += new System.EventHandler(this.OnXLabelTextLeave);
            // 
            // label1
            // 
            this.label1.Location = new System.Drawing.Point(466, 106);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(64, 24);
            this.label1.TabIndex = 12;
            this.label1.Text = "Label XY:";
            this.label1.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // m_SetFontButton
            // 
            this.m_SetFontButton.Location = new System.Drawing.Point(538, 18);
            this.m_SetFontButton.Name = "m_SetFontButton";
            this.m_SetFontButton.Size = new System.Drawing.Size(96, 24);
            this.m_SetFontButton.TabIndex = 11;
            this.m_SetFontButton.Text = "Panel Font";
            this.m_SetFontButton.Click += new System.EventHandler(this.m_SetFontButton_Click);
            // 
            // m_ResultsBox
            // 
            this.m_ResultsBox.Location = new System.Drawing.Point(260, 18);
            this.m_ResultsBox.Multiline = true;
            this.m_ResultsBox.Name = "m_ResultsBox";
            this.m_ResultsBox.ReadOnly = true;
            this.m_ResultsBox.Size = new System.Drawing.Size(200, 112);
            this.m_ResultsBox.TabIndex = 10;
            // 
            // LabelY
            // 
            this.LabelY.Location = new System.Drawing.Point(8, 32);
            this.LabelY.Name = "LabelY";
            this.LabelY.Size = new System.Drawing.Size(152, 16);
            this.LabelY.TabIndex = 8;
            this.LabelY.Text = "Y:";
            // 
            // LabelX
            // 
            this.LabelX.Location = new System.Drawing.Point(8, 16);
            this.LabelX.Name = "LabelX";
            this.LabelX.Size = new System.Drawing.Size(152, 16);
            this.LabelX.TabIndex = 7;
            this.LabelX.Text = "X:";
            // 
            // panel1
            // 
            this.panel1.BackColor = System.Drawing.SystemColors.ActiveCaptionText;
            this.panel1.Location = new System.Drawing.Point(3, 0);
            this.panel1.Name = "panel1";
            this.panel1.Size = new System.Drawing.Size(639, 445);
            this.panel1.TabIndex = 8;
            this.panel1.MouseDown += new System.Windows.Forms.MouseEventHandler(this.panel1_MouseDown);
            // 
            // m_Marker
            // 
            this.m_Marker.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.m_Marker.FormattingEnabled = true;
            this.m_Marker.Location = new System.Drawing.Point(62, 106);
            this.m_Marker.Name = "m_Marker";
            this.m_Marker.Size = new System.Drawing.Size(113, 21);
            this.m_Marker.TabIndex = 22;
            this.m_Marker.SelectedIndexChanged += new System.EventHandler(this.OnMarkerChanged);
            // 
            // m_MarkerSize
            // 
            this.m_MarkerSize.Location = new System.Drawing.Point(181, 107);
            this.m_MarkerSize.Name = "m_MarkerSize";
            this.m_MarkerSize.Size = new System.Drawing.Size(47, 20);
            this.m_MarkerSize.TabIndex = 23;
            this.m_MarkerSize.Leave += new System.EventHandler(this.OnMarkerSizeLeave);
            // 
            // label4
            // 
            this.label4.Location = new System.Drawing.Point(8, 103);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(48, 24);
            this.label4.TabIndex = 24;
            this.label4.Text = "Marker:";
            this.label4.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // m_ShowErrorBars
            // 
            this.m_ShowErrorBars.AutoSize = true;
            this.m_ShowErrorBars.Location = new System.Drawing.Point(11, 82);
            this.m_ShowErrorBars.Name = "m_ShowErrorBars";
            this.m_ShowErrorBars.Size = new System.Drawing.Size(102, 17);
            this.m_ShowErrorBars.TabIndex = 25;
            this.m_ShowErrorBars.Text = "Show Error Bars";
            this.m_ShowErrorBars.UseVisualStyleBackColor = true;
            this.m_ShowErrorBars.CheckedChanged += new System.EventHandler(this.OnShowErrorBarsChecked);
            // 
            // PlotControl
            // 
            this.Controls.Add(this.panel1);
            this.Controls.Add(this.GroupBox1);
            this.Name = "PlotControl";
            this.Size = new System.Drawing.Size(657, 590);
            this.Load += new System.EventHandler(this.OnLoad);
            this.Resize += new System.EventHandler(this.GraphicsControl_Resize);
            this.GroupBox1.ResumeLayout(false);
            this.GroupBox1.PerformLayout();
            this.ResumeLayout(false);

		}
		#endregion

		#region Component Methods

		public void Clear()
		{
			m_lastplotpainted = LastPlot.Nothing;
			m_ResultsBox.Text = "";
			Bitmap b = new Bitmap((int)(panel1.Width),(int)(panel1.Height), System.Drawing.Imaging.PixelFormat.Format24bppRgb);
			Graphics gPB = Graphics.FromImage(b);
			//Graphics gPB = panel1.CreateGraphics();
			gPB.Clear(System.Drawing.Color.White);
			panel1.BackgroundImage = b;
		}

		public double[][] Histo()
		{            		
			m_lastplotpainted = LastPlot.Histo;
			//Bitmap b = new Bitmap((int)(panel1.Width),(int)(panel1.Height));            
			//Graphics gPB = Graphics.FromImage(b);
            Graphics gPB = Graphics.FromImage(panel1.BackgroundImage);

			//panel1.BackgroundImage = b;			
            m_RepeatDrawing = gA.Histo;
            double[][] o_val = m_RepeatDrawing(gPB, panel1.Width, panel1.Height);
            m_ResultsBox.Text = "Total entries: " + m_Entries;
			if(gA.FitPar!= null)
			{
				int n = gA.FitPar.GetLength(0);
				for (int i=0;i<n;i++)
					if (i<2 || (i>1 && gA.HistoFit != -2)) 
						m_ResultsBox.Text += "\r\n" + gA.ParDescr[i] + ": " + gA.FitPar[i];
			}
            return o_val;
		}

        public double[][] HistoSkyline()
        {
            m_lastplotpainted = LastPlot.HistoSkyline;
            //Bitmap b = new Bitmap((int)(panel1.Width),(int)(panel1.Height));            
            //Graphics gPB = Graphics.FromImage(b);
            Graphics gPB = Graphics.FromImage(panel1.BackgroundImage);

            //panel1.BackgroundImage = b;
            m_RepeatDrawing = gA.HistoSkyline;
            double[][] o_val = m_RepeatDrawing(gPB, panel1.Width, panel1.Height);
            m_ResultsBox.Text = "Total entries: " + m_Entries;
            if (gA.FitPar != null)
            {
                int n = gA.FitPar.GetLength(0);
                for (int i = 0; i < n; i++)
                    if (i < 2 || (i > 1 && gA.HistoFit != -2))
                        m_ResultsBox.Text += "\r\n" + gA.ParDescr[i] + ": " + gA.FitPar[i];
            }
            return o_val;
        }


		public double [][] GroupScatter()
		{

			m_lastplotpainted = LastPlot.GScatter;
            //Bitmap b = new Bitmap((int)(panel1.Width),(int)(panel1.Height));            
            //Graphics gPB = Graphics.FromImage(b);
            Graphics gPB = Graphics.FromImage(panel1.BackgroundImage);

            //panel1.BackgroundImage = b;
            m_RepeatDrawing = gA.GroupScatter;
            double[][] o_val = m_RepeatDrawing(gPB, panel1.Width, panel1.Height);
			m_ResultsBox.Text = "Total entries: " + m_Entries;
			if(gA.FitPar!= null && gA.ScatterFit !=0)
			{
				int n = gA.FitPar.GetLength(0);
				for (int i=0;i<n;i++)
					m_ResultsBox.Text += "\r\n" + gA.ParDescr[i] + ": " + gA.FitPar[i];
			};
            return o_val;
		}


		public void Scatter()
		{

			m_lastplotpainted = LastPlot.Scatter;
            //Bitmap b = new Bitmap((int)(panel1.Width),(int)(panel1.Height));            
            //Graphics gPB = Graphics.FromImage(b);
            Graphics gPB = Graphics.FromImage(panel1.BackgroundImage);

            //panel1.BackgroundImage = b;
            m_RepeatDrawing = gA.Scatter;
            m_RepeatDrawing(gPB, panel1.Width, panel1.Height);
            m_ResultsBox.Text = "Total entries: " + m_Entries;
			if(gA.FitPar!= null && gA.ScatterFit !=0)
			{
				int n = gA.FitPar.GetLength(0);
				for (int i=0;i<n;i++)
					m_ResultsBox.Text += "\r\n" + gA.ParDescr[i] + ": " + gA.FitPar[i];
			};
		}

        public void ScatterHue()
        {

            m_lastplotpainted = LastPlot.ScatterHue;
            //Bitmap b = new Bitmap((int)(panel1.Width),(int)(panel1.Height));            
            //Graphics gPB = Graphics.FromImage(b);
            Graphics gPB = Graphics.FromImage(panel1.BackgroundImage);

            //panel1.BackgroundImage = b;
            m_RepeatDrawing = gA.ScatterHue;
            m_RepeatDrawing(gPB, panel1.Width, panel1.Height);
            m_ResultsBox.Text = "Total entries: " + m_Entries;
            if (gA.FitPar != null && gA.ScatterFit != 0)
            {
                int n = gA.FitPar.GetLength(0);
                for (int i = 0; i < n; i++)
                    m_ResultsBox.Text += "\r\n" + gA.ParDescr[i] + ": " + gA.FitPar[i];
            };
        }

        public void ArrowPlot()
        {
            m_lastplotpainted = LastPlot.ArrowPlot;
            //Bitmap b = new Bitmap((int)(panel1.Width),(int)(panel1.Height));            
            //Graphics gPB = Graphics.FromImage(b);
            Graphics gPB = Graphics.FromImage(panel1.BackgroundImage);

            //panel1.BackgroundImage = b;
            m_RepeatDrawing = gA.ArrowPlot;
            m_RepeatDrawing(gPB, panel1.Width, panel1.Height);
            m_ResultsBox.Text = "Total entries: " + m_Entries;
        }

        public double[][] GreyLevelArea()
		{

			m_lastplotpainted = LastPlot.GArea;
            //Bitmap b = new Bitmap((int)(panel1.Width),(int)(panel1.Height));            
            //Graphics gPB = Graphics.FromImage(b);
            Graphics gPB = Graphics.FromImage(panel1.BackgroundImage);

            //panel1.BackgroundImage = b;    
            m_RepeatDrawing = gA.GreyLevelArea;
            double[][] o_val = m_RepeatDrawing(gPB, panel1.Width, panel1.Height);
            m_ResultsBox.Text = "Total entries: " + m_Entries;
            return o_val;
		}
		
		public double[][] LEGO()
		{

			m_lastplotpainted = LastPlot.LEGO;
            //Bitmap b = new Bitmap((int)(panel1.Width),(int)(panel1.Height));            
            //Graphics gPB = Graphics.FromImage(b);
            Graphics gPB = Graphics.FromImage(panel1.BackgroundImage);

            //panel1.BackgroundImage = b;            
            m_RepeatDrawing = gA.LEGOPlot;
            double[][] o_val = m_RepeatDrawing(gPB, panel1.Width, panel1.Height);
            m_ResultsBox.Text = "Total entries: " + m_Entries;
            return o_val;
		}
		
		public void Scatter3D()
		{
			m_lastplotpainted = LastPlot.LEGO;
            //Bitmap b = new Bitmap((int)(panel1.Width),(int)(panel1.Height));            
            //Graphics gPB = Graphics.FromImage(b);
            Graphics gPB = Graphics.FromImage(panel1.BackgroundImage);

            //panel1.BackgroundImage = b;
            m_RepeatDrawing = gA.Scatter3D;
            m_RepeatDrawing(gPB, panel1.Width, panel1.Height);
            m_ResultsBox.Text = "Total entries: " + m_Entries;
		}
		
		public double[][] GAreaValues()
		{

			m_lastplotpainted = LastPlot.GAreaValues;
            //Bitmap b = new Bitmap((int)(panel1.Width),(int)(panel1.Height));            
            //Graphics gPB = Graphics.FromImage(b);
            Graphics gPB = Graphics.FromImage(panel1.BackgroundImage);

            //panel1.BackgroundImage = b;
            m_RepeatDrawing = gA.GAreaValues;
            double[][] o_val = m_RepeatDrawing(gPB, panel1.Width, panel1.Height);
            m_ResultsBox.Text = "Total entries: " + m_Entries;
            return o_val;
		}

		public double[][] GAreaComputedValues()
		{
			m_lastplotpainted = LastPlot.GAreaCValues;
            //Bitmap b = new Bitmap((int)(panel1.Width),(int)(panel1.Height));            
            //Graphics gPB = Graphics.FromImage(b);
            Graphics gPB = Graphics.FromImage(panel1.BackgroundImage);

            //panel1.BackgroundImage = b;
            m_RepeatDrawing = gA.GAreaComputedValues;
            double[][] o_val = m_RepeatDrawing(gPB, panel1.Width, panel1.Height);
            m_ResultsBox.Text = "Total entries: " + m_Entries;
            return o_val;
		}

        public double[][] SymbolAreaValues()
        {
            m_lastplotpainted = LastPlot.SymbolArea;
            //Bitmap b = new Bitmap((int)(panel1.Width),(int)(panel1.Height));            
            //Graphics gPB = Graphics.FromImage(b);
            Graphics gPB = Graphics.FromImage(panel1.BackgroundImage);

            //panel1.BackgroundImage = b;
            m_RepeatDrawing = gA.SymbolArea;
            double[][] o_val = m_RepeatDrawing(gPB, panel1.Width, panel1.Height);
            m_ResultsBox.Text = "Total entries: " + m_Entries;
            return o_val;
        }

		public double[][] HueAreaValues()
		{
			m_lastplotpainted = LastPlot.GAreaValues;
            //Bitmap b = new Bitmap((int)(panel1.Width),(int)(panel1.Height));            
            //Graphics gPB = Graphics.FromImage(b);
            Graphics gPB = Graphics.FromImage(panel1.BackgroundImage);

            //panel1.BackgroundImage = b;
            m_RepeatDrawing = gA.HueArea;
            double[][] o_val = m_RepeatDrawing(gPB, panel1.Width, panel1.Height);
            m_ResultsBox.Text = "Total entries: " + m_Entries;
            return o_val;
		}

		public double[][] HueAreaComputedValues()
		{

			m_lastplotpainted = LastPlot.GAreaCValues;
            //Bitmap b = new Bitmap((int)(panel1.Width),(int)(panel1.Height));            
            //Graphics gPB = Graphics.FromImage(b);
            Graphics gPB = Graphics.FromImage(panel1.BackgroundImage);

            //panel1.BackgroundImage = b;
            m_RepeatDrawing = gA.HueAreaComputedValues;
            double[][] o_val = m_RepeatDrawing(gPB, panel1.Width, panel1.Height);
            m_ResultsBox.Text = "Total entries: " + m_Entries;
            return o_val;
		}

		public void Pie()
		{
			m_lastplotpainted = LastPlot.Pie;
            //Bitmap b = new Bitmap((int)(panel1.Width),(int)(panel1.Height));            
            //Graphics gPB = Graphics.FromImage(b);
            Graphics gPB = Graphics.FromImage(panel1.BackgroundImage);

            //panel1.BackgroundImage = b;
            m_RepeatDrawing = gA.Pie;
            m_RepeatDrawing(gPB, panel1.Width, panel1.Height);
            m_ResultsBox.Text = "Total entries: " + m_Entries;
		}

		#endregion

		#region External Properties

		private int m_Entries = 0;

		public Bitmap BitmapPlot
		{
			get
			{
/*				Bitmap b = new Bitmap((int)(panel1.Width),(int)(panel1.Height));
				Graphics gPB = Graphics.FromImage(b);

				if (m_lastplotpainted == LastPlot.Histo)
				{
					gA.Histo(gPB, panel1.Width, panel1.Height);
				}
				else if (m_lastplotpainted == LastPlot.GScatter)
				{
					gA.GroupScatter(gPB, panel1.Width, panel1.Height);
				}
				else if (m_lastplotpainted == LastPlot.Scatter)
				{
					gA.Scatter(gPB, panel1.Width, panel1.Height);
				}
				else if (m_lastplotpainted == LastPlot.GArea)
				{
					gA.GreyLevelArea(gPB, panel1.Width, panel1.Height);
				}
				else if (m_lastplotpainted == LastPlot.GAreaValues)
				{
					gA.GAreaValues(gPB, panel1.Width, panel1.Height);
				}
				else if (m_lastplotpainted == LastPlot.GAreaCValues)
				{
					gA.GAreaComputedValues(gPB, panel1.Width, panel1.Height);
				}
				else if (m_lastplotpainted == LastPlot.Pie)
				{
					gA.Pie(gPB, panel1.Width, panel1.Height);
				};
				return (Bitmap)b.Clone();	
*/
				return (Bitmap)panel1.BackgroundImage.Clone();
			}

		}

		public double[] FitPar
		{
			get
			{
				return (double[])gA.FitPar.Clone();	
			}

		}

		public string Function
		{
			set
			{
				gA.Function = (string)value.Clone();	
			}

		}

		public string[] CommentX
		{
			set
			{
				gA.CommentX = (string[])value.Clone();	
			}

		}

		public double[] VecX
		{
			set
			{
				gA.VecX = value;	
				m_Entries = value.Length;
			}

		}

		public double[] VecY
		{
			set
			{
				gA.VecY = value;
			}

		}

        public double[] VecDX
        {
            set
            {
                gA.VecDX = value;
            }

        }

        public double[] VecDY
        {
            set
            {
                gA.VecDY = value;
            }

        }

        public double[] VecZ
		{
			set
			{
				gA.VecZ = value;	
			}

		}

		public double[,] MatZ
		{
			set
			{
				gA.MatZ = value;	
			}

		}


		public bool FunctionOverlayed
		{
			get
			{
				return gA.FunctionOverlayed;
			}
			set
			{
				gA.FunctionOverlayed = value;	
			}
		}

		public bool FittingOnlyDataInPlot
		{
			get
			{
				return gA.FittingOnlyDataInPlot;
			}

			set
			{
				gA.FittingOnlyDataInPlot = value;	
			}

		}

		public bool SetXDefaultLimits
		{
			get
			{
				return gA.SetXDefaultLimits;
			}

			set
			{
				gA.SetXDefaultLimits = value;	
			}

		}

		public bool SetYDefaultLimits
		{
			get
			{
				return gA.SetYDefaultLimits;
			}

			set
			{
				gA.SetYDefaultLimits = value;	
			}

		}

		public float DX
		{
			get
			{
				return gA.DX;
			}

			set
			{
				gA.DX = value;	
			}

		}

		public double MaxX
		{
			get
			{
				return gA.MaxX;
			}

			set
			{
				gA.MaxX = value;	
			}

		}

		public double MinX
		{
			get
			{
				return gA.MinX;
			}

			set
			{
				gA.MinX = value;	
			}

		}

		public float DY
		{
			get
			{
				return gA.DY;
			}

			set
			{
				gA.DY = value;	
			}

		}

		public double MaxY
		{
			get
			{
				return gA.MaxY;
			}

			set
			{
				gA.MaxY = value;	
			}

		}

		public double MinY
		{
			get
			{
				return gA.MinY;
			}

			set
			{
				gA.MinY = value;	
			}

		}

		/*		public float DZ
				{
					get
					{
						return gA.DZ;
					}

					set
					{
						gA.DZ = value;	
					}

				}
		*/
		public short HistoFit
		{
			get
			{
				return gA.HistoFit;
			}

			set
			{
				gA.HistoFit = value;	
			}

		}

		public short ScatterFit
		{
			get
			{
				return gA.ScatterFit;
			}

			set
			{
				gA.ScatterFit = value;	
			}

		}


		public string XTitle
		{
			get
			{
				return gA.XTitle;
			}

			set
			{
				gA.XTitle = value;	
			}

		}

		public string YTitle
		{
			get
			{
				return gA.YTitle;
			}

			set
			{
				gA.YTitle = value;	
			}

		}

		public string ZTitle
		{
			get
			{
				return gA.ZTitle;
			}

			set
			{
				gA.ZTitle = value;	
			}

		}

		public bool LinearFitWE
		{
			get
			{
				return gA.LinearFitWE;
			}

			set
			{
				gA.LinearFitWE = value;	
			}

		}

		public bool HistoFill
		{
			get
			{
				return gA.HistoFill;
			}

			set
			{
				gA.HistoFill = value;	
			}

		}

		public NumericalTools.Plot.PaletteType Palette
		{
			get { return gA.Palette; }
			set 
			{
				gA.Palette = value;
			}
		}

		public Color HistoColor
		{
			get
			{
				return gA.HistoColor;
			}

			set
			{
				gA.HistoColor = value;	
			}

		}

		public double Skewedness
		{
			get { return gA.Skewedness; }
			set { gA.Skewedness = value; }
		}
		
		public double ObservationAngle
		{
			get { return gA.ObservationAngle; }
			set { gA.ObservationAngle = value; }
		}

		private LastPlot m_lastplotpainted = LastPlot.Nothing;

		private LastPlot lastplotpainted
		{
			get
			{
				return m_lastplotpainted;
			}

			set
			{
				m_lastplotpainted = value;	
			}

		}
	
		#endregion

		/*
		*
		*		private Graphics gPB;
		*
		*/

		Plot gA = new Plot();

/*
		private void GraphicsControl_Paint(object sender, System.Windows.Forms.PaintEventArgs e)
		{
			Graphics gPB = e.Graphics;
			gPB = panel1.CreateGraphics();
			//gA = new GraphicsAnalysis();

			if (m_lastplotpainted == LastPlot.Histo)
			{
				gA.Histo(gPB, panel1.Width, panel1.Height);
			}
			else if (m_lastplotpainted == LastPlot.GScatter)
			{
				gA.GroupScatter(gPB, panel1.Width, panel1.Height);
			}
			else if (m_lastplotpainted == LastPlot.Scatter)
			{
				gA.Scatter(gPB, panel1.Width, panel1.Height);
			}
			else if (m_lastplotpainted == LastPlot.GArea)
			{
				gA.GreyLevelArea(gPB, panel1.Width, panel1.Height);
			}
			else if (m_lastplotpainted == LastPlot.GAreaValues)
			{
				gA.GAreaValues(gPB, panel1.Width, panel1.Height);
			}
			else if (m_lastplotpainted == LastPlot.GAreaCValues)
			{
				gA.GAreaComputedValues(gPB, panel1.Width, panel1.Height);
			}
			else if (m_lastplotpainted == LastPlot.Pie)
			{
				gA.Pie(gPB, panel1.Width, panel1.Height);
			};
		}
*/

		private void panel1_MouseDown(object sender, System.Windows.Forms.MouseEventArgs e)
		{
			
			double MinDist, tmp;
			int i,j,n,m, minindx, minindy;
			
			if(m_lastplotpainted != LastPlot.Nothing)
			{
			
				if(e.Button == MouseButtons.Left)
				{
					LabelX.Text = "X: " + gA.RevAffineX(e.X);
					LabelY.Text = "Y: " + gA.RevAffineY(e.Y);
					LabelZ.Text = "";
				}
				else if(e.Button == MouseButtons.Right)
				{
					MinDist = Math.Sqrt((gA.PlottedX[0] - gA.RevAffineX(e.X))*(gA.PlottedX[0] - gA.RevAffineX(e.X))+
						(gA.PlottedY[0] - gA.RevAffineY(e.Y))*(gA.PlottedY[0] - gA.RevAffineY(e.Y)));
					n= gA.PlottedX.GetLength(0);
					minindx=0;
					if ((m_lastplotpainted == LastPlot.Histo)||
						(m_lastplotpainted == LastPlot.Scatter)||
						(m_lastplotpainted == LastPlot.GScatter))
					{
						for(i=1;i<n;i++)
						{
							tmp = Math.Sqrt((gA.PlottedX[i] - gA.RevAffineX(e.X))*(gA.PlottedX[i] - gA.RevAffineX(e.X))+
								(gA.PlottedY[i] - gA.RevAffineY(e.Y))*(gA.PlottedY[i] - gA.RevAffineY(e.Y)));
							if(tmp<MinDist) 
							{
								MinDist=tmp;
								minindx=i;
							};

						};
						LabelX.Text = "X: " + gA.PlottedX[minindx];
						LabelY.Text = "Y: " + gA.PlottedY[minindx];
						if ((m_lastplotpainted == LastPlot.Histo)||
							(m_lastplotpainted == LastPlot.Scatter))
						{
							LabelZ.Text = "";
						}
						else if(m_lastplotpainted == LastPlot.GScatter)
						{
							LabelZ.Text = "S(Y): " + gA.PlottedSY[minindx];
						};
					}
					else if ((m_lastplotpainted == LastPlot.GArea)||
						(m_lastplotpainted == LastPlot.GAreaValues)||
						(m_lastplotpainted == LastPlot.GAreaCValues))
					{
						m= gA.PlottedY.GetLength(0);
						minindy=0;
						for(i=0;i<n;i++)
							for(j=0;j<m;j++)
							{
								tmp = Math.Sqrt((gA.PlottedX[i] - gA.RevAffineX(e.X))*(gA.PlottedX[i] - gA.RevAffineX(e.X))+
									(gA.PlottedY[j] - gA.RevAffineY(e.Y))*(gA.PlottedY[j] - gA.RevAffineY(e.Y)));
								if(tmp<MinDist) 
								{
									MinDist=tmp;
									minindx=i;
									minindy=j;
								};

							};
						LabelX.Text = "X: " + gA.PlottedX[minindx];
						LabelY.Text = "Y: " + gA.PlottedY[minindy];
						LabelZ.Text = "Z: " + gA.PlottedMatZ[minindx,minindy];
					};
				};
			};

		}

		private void GraphicsControl_Resize(object sender, System.EventArgs e)
		{
			//this.Width = ORIGINALWIDTH;
			//this.Height = ORIGINALHEIGHT;
		}

		private void m_SetFontButton_Click(object sender, System.EventArgs e)
		{
			FontDialog fdlg = new FontDialog();
			fdlg.Font = gA.PanelFont;
			if (fdlg.ShowDialog() == DialogResult.OK)
				gA.PanelFont = fdlg.Font;
			fdlg.Dispose();
		}

		private void OnXLabelTextLeave(object sender, System.EventArgs e)
		{
			try
			{
				gA.PanelX = Convert.ToDouble(m_LabelXText.Text, System.Globalization.CultureInfo.InvariantCulture);
			}
			catch(Exception)
			{
				gA.PanelX = 1.0;
				m_LabelXText.Text = gA.PanelX.ToString(System.Globalization.CultureInfo.InvariantCulture);
			}
		}

		private void OnYLabelTextLeave(object sender, System.EventArgs e)
		{
			try
			{
				gA.PanelY = Convert.ToDouble(m_LabelYText.Text, System.Globalization.CultureInfo.InvariantCulture);
			}
			catch(Exception)
			{
				gA.PanelY = 0.0;
				m_LabelYText.Text = gA.PanelY.ToString(System.Globalization.CultureInfo.InvariantCulture);
			}
		}

		private void OnFormatTextLeave(object sender, System.EventArgs e)
		{
			try
			{
				gA.PanelFormat = m_FormatText.Text.Trim();
				string s = (1.0).ToString(gA.PanelFormat, System.Globalization.CultureInfo.InvariantCulture);
			}
			catch (Exception)
			{
				gA.PanelFormat = m_FormatText.Text = "";
			}
		}

		private void m_SetAxisFontButton_Click(object sender, System.EventArgs e)
		{
			FontDialog fdlg = new FontDialog();
			fdlg.Font = gA.LabelFont;
			if (fdlg.ShowDialog() == DialogResult.OK)
				gA.LabelFont = fdlg.Font;
			fdlg.Dispose();		
		}

		private void OnPenTextLeave(object sender, System.EventArgs e)
		{
            try
            {
                gA.PlotThickness = Convert.ToInt32(m_PenWidthText.Text);
            }
            catch (Exception)
            {
                m_PenWidthText.Text = gA.PlotThickness.ToString();
            }			
		}

		private void m_SetPlotColorButton_Click(object sender, System.EventArgs e)
		{
			if (PlotColorDialog.ShowDialog() == DialogResult.OK)
				gA.HistoColor = PlotColorDialog.Color;
		}

		public int PlotThickness
		{
			get { return gA.PlotThickness; }
			set { gA.PlotThickness = value; }
		}

		public Font PanelFont
		{
			get { return gA.PanelFont; }
			set { gA.PanelFont = value; }
		}

		public Font LabelFont
		{
			get { return gA.LabelFont; }
			set { gA.LabelFont = value; }
		}

		public string PanelFormat
		{
			get { return gA.PanelFormat; }
			set { gA.PanelFormat = value; }
		}

		public double PanelX
		{
			get { return gA.PanelX; }
			set { gA.PanelX = value; }
		}

		public double PanelY
		{
			get { return gA.PanelY; }
			set { gA.PanelY = value; }
		}

		public string Panel
		{
			get { return gA.Panel; }
			set { gA.Panel = value; }
		}

        public double ArrowScale
        {
            get { return gA.ArrowScale; }
            set { gA.ArrowScale = value; }
        }

        public double ArrowSample
        {
            get { return gA.ArrowSample; }
            set { gA.ArrowSample = value; }
        }

        public double ArrowSize
        {
            get { return gA.ArrowSize; }
            set { gA.ArrowSize = value; }
        }

        public void SaveMetafile(System.Drawing.Graphics myg)
        {
            if (m_RepeatDrawing != null)
            {
                m_RepeatDrawing(myg, panel1.Width, panel1.Height);
            }
        }

        delegate double[][] dMakeDrawing(System.Drawing.Graphics g, int w, int h);

        private dMakeDrawing m_RepeatDrawing = null;

        public System.Drawing.Bitmap GetCompatibleBitmap()
        {
            return new System.Drawing.Bitmap(panel1.Width, panel1.Height);
        }

        public void RefreshPlot()
        {
            Graphics gPB = Graphics.FromImage(panel1.BackgroundImage);
            if (m_RepeatDrawing != null)
            {
                m_RepeatDrawing(gPB, panel1.Width, panel1.Height);
            }
            panel1.Refresh();
        }

        private void OnLoad(object sender, EventArgs e)
        {
            m_Marker.Items.Clear();
            string [] markers = NumericalTools.Plot.Marker.KnownMarkers;
            m_Marker.Items.AddRange(markers);
            m_MarkerSize.Text = gA.CurrentMarkerSize.ToString();
            m_Marker.SelectedValue = markers[0];
            foreach (var m in markers)
                if (String.Compare(m, "None", true) != 0)
                {
                    m_Marker.SelectedValue = m;
                    gA.CurrentMarker = m;
                    break;
                }
            gA.PlotThickness = Convert.ToInt32(m_PenWidthText.Text);
            m_ShowErrorBars.Checked = gA.ShowErrorBars;
        }

        private void OnMarkerChanged(object sender, EventArgs e)
        {
            gA.CurrentMarker = m_Marker.Items[m_Marker.SelectedIndex].ToString();
        }

        private void OnMarkerSizeLeave(object sender, EventArgs e)
        {
            try
            {
                gA.CurrentMarkerSize = uint.Parse(m_MarkerSize.Text);
            }
            catch (Exception)
            {
                m_MarkerSize.Text = gA.CurrentMarkerSize.ToString();
            }
        }

        private void OnShowErrorBarsChecked(object sender, EventArgs e)
        {
            gA.ShowErrorBars = m_ShowErrorBars.Checked;
        }
	}	
}
 