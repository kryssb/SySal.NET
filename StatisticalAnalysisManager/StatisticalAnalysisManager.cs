#define NEWSTYLE 

using System;
using System.Collections;
using System.ComponentModel;
using System.Drawing;
using System.Data;
using System.Windows.Forms;
using System.Collections.Generic;
using System.Linq;

namespace NumericalTools
{
	public class SySalDataRow
	{
		internal ArrayList _Data = new ArrayList();
	
		public object this[int index]
		{
			get { return _Data[index]; }
		}
		
		public int Length
		{
			get { return _Data.Count; }
		}
		
		public void AddColumn(object data, int i = -1)
		{
			_Data.Insert((i < 0) ? _Data.Count : i, data);
		}
		
		public void RemoveColumn(int i)
		{
			_Data.RemoveAt(i);
		}
	}
	
	public class SySalDataRowSet
	{
		internal List<SySalDataRow> _Rows;
		
		public int Count
		{
			get { return _Rows.Count; }
		}
		
		public SySalDataRow this[int index]
		{
			get { return _Rows[index]; }			
		}
		
		public void AddRow(object [] data)
		{
			var nr = new SySalDataRow();
			nr._Data.AddRange(data);
			_Rows.Add(nr);
		}
		
		public void AddColumn(object data, int i = -1)
		{
			foreach (var r in _Rows)
				r.AddColumn(data, i);
		}
		
		public void RemoveColumn(int i)
		{
			foreach (var r in _Rows)
				r.RemoveColumn(i);
		}		
	}
	
	public class SySalDataColumn
	{
		public string Name = "";
		public string Unit = "";		
	}

	public class SySalDataTable
	{
		internal List<SySalDataColumn> _Columns = new List<SySalDataColumn>();
	
		public SySalDataColumn [] Columns
		{
			get { return _Columns.ToArray(); }
		}
		
		public SySalDataColumn AddColumn(object defaultdata, string name = null, string unit = "", int index = -1)
		{
			if (name == null) name = "v" + _Columns.Count;
			SySalDataColumn sdc = new SySalDataColumn() { Name = name, Unit = unit };
			_Rows.AddColumn(defaultdata, index);
			return sdc;
		}
		
		public void RemoveColumn(int index = -1)
		{
			_Rows.RemoveColumn(index);			
		}
		
		internal SySalDataRowSet _Rows;
		
		public SySalDataRowSet Rows
		{
			get { return _Rows; }
		}		
	}

	public class SySalDataSet
	{
	
	}
	/// <summary>
	/// Analysis manager control.
	/// </summary>
	/// <remarks>
	/// <para>The Analysis manager control offers the following features:
	/// <list type="bullet">
	/// <term>Dataset manipulation</term>
	/// <term>Plot generation</term>
	/// <term>Data export</term>
	/// <term>Plot export</term>
	/// </list>	
	/// </para>
	/// <para><b>Datasets</b></para>
	/// <para>The analysis control can host more than one dataset. 
	/// Each dataset has its own name and its own set of variables. 
	/// All operations are performed on the current dataset. 
	/// To change the current dataset, select it and click on the <i>Switch</i> button. 
	/// To remove a dataset, select it and click on the <i>Remove</i> button.
	/// You can have direct access to the dataset values, by clicking on the 
	/// <i>Show data</i> button. This opens the <see cref="NumericalTools.ShowDataForm">ShowDataForm</see> 
	/// which you can use to view/modify the values.</para>
	/// <para><b>Variables</b></para>
	/// <para>You can add/remove variables to a dataset. At each moment, a dataset has 
	/// one variable selected for the X axis, one for the Y axis and one for the Z axis.
	/// Variables have a name and can have a measurement unit. Variable names are case-preserving
	/// but not case-sensitive (i.e., case is remembered, but variables are recognized in
	/// expressions even if the case is not identical letter-by-letter). A variable can be
	/// added by generating it through a mathematical expression, to be typed in the 
	/// dedicated text box; then, one has to select the <i>Add variable</i> function and
	/// click on the <i>Apply</i> button. A variable is removed by selecting it and 
	/// clicking on the <i>Remove</i> button in the <i>Remove variable</i> panel.</para>
	/// <para>The <b>rownum</b> pseudocolumn is equal to the row number in the recordset.</para>
	/// <para><b>Cuts</b></para>
	/// <para>A subset of a dataset can be generated through a cut. Records are selected on the 
	/// basis of a mathematical expression, to be typed in the dedicated text box; the rows that
	/// make the expression non-zero pass the selection. In order to perform the cut, one needs to
	/// select the <i>Apply cut</i> function and then click on the <i>Apply</i> button. A prompt
	/// appears asking whether a new dataset is to be generated or not; if no new dataset is 
	/// generated, the current one is reused, dropping (and forgetting) all records that do not 
	/// pass the selection.</para>
	/// <para><b>Plots</b></para>
	/// <para>Several plot types are available, and each plot can be completed with a fit curve. 
	/// See <see cref="NumericalTools.AnalysisControl.Plot"/> for a list of plot types and fit types.
	/// To obtain a plot, select the X, Y, Z variables (as needed by the plot dimensionality), 
	/// and click on the <i>Plot</i> Button. One line can be added to 2D plots: you can specify its
	/// mathematical expression in the dedicated text box, then select the <i>Plot</i> function and
	/// click on the <i>Apply</i> button. Beware to use single-variable expressions, with the same
	/// variable as the X axis of the plot.</para>
	/// <para>For LEGO plots, the 3D viewing parameters must be specified. You have to set the 
	/// Skewedness (i.e. the viewing slope) and the rotation angle around the Z axis (XY angle).</para>
	/// <para><b>Formatting</b></para>
	/// <para>A subpanel on the left hosts some controls that can be used to tune the style of the plot.
	/// The axis and label text fonts can be set. The number of digits in the label window can be changed using the following 
	/// format strings:
	/// <list type="bullet">
	/// <item><term><c>Fn</c> where "n" is the number of digits after the point (e.g. <c>F2</c>, <c>F5</c>).</term></item>
	/// <item><term><c>Gn</c> where "n" is the number of significant digits (e.g. <c>G3</c>, <c>G8</c>).</term></item>
	/// </list>
	/// The line plotting color can be set. The position of the label can be defined at will. 
	/// On either axis (X or Y), the 0 value of the coordinate means "label position at the lower extent of the axis",
	/// and 1 means "label position at the upper extent of the axis". Therefore, (0,0) means "label at the
	/// upper left corner"; (1,1) means "label at the lower right corner"; (0.5, 0) means "label centered on X
	/// and on the top edge".</para>
	/// </remarks>
	public class AnalysisControl : System.Windows.Forms.UserControl
	{
		private System.Windows.Forms.GroupBox groupBox1;
		private System.Windows.Forms.TextBox txtXbin;
		private System.Windows.Forms.Label label1;
		private System.Windows.Forms.Label label2;
		private System.Windows.Forms.TextBox txtMaxX;
		private System.Windows.Forms.TextBox txtMinX;
		private System.Windows.Forms.Label label3;
		private System.Windows.Forms.GroupBox groupBox2;
		private System.Windows.Forms.Label label4;
		private System.Windows.Forms.Label label5;
		private System.Windows.Forms.Label label6;
		private System.Windows.Forms.GroupBox groupBox3;
		private System.Windows.Forms.Label label8;
		private System.Windows.Forms.Label label9;
		private System.Windows.Forms.TextBox txtMinY;
		private System.Windows.Forms.TextBox txtMaxY;
		private System.Windows.Forms.TextBox txtYbin;
		private System.Windows.Forms.Label label7;
		private System.Windows.Forms.Label label10;
		private System.Windows.Forms.Button Plotbutton;
		/// <summary>
		/// Required designer variable.
		/// </summary>

		private int ORIGINALWIDTH =848;
		private int ORIGINALHEIGHT=512;
		private System.Windows.Forms.CheckBox chkFit;
		private System.Windows.Forms.ComboBox cmbX;
		private System.Windows.Forms.ComboBox cmbY;
		private System.Windows.Forms.ComboBox cmbZ;
		private System.Windows.Forms.ComboBox cmbXDeg;
		private System.Windows.Forms.ComboBox cmbYDeg;
		private System.Windows.Forms.ComboBox cmbPlotType;
		private System.Windows.Forms.GroupBox groupBox4;
		private System.Windows.Forms.RadioButton radME0;
		private System.Windows.Forms.RadioButton radME1;
		private System.Windows.Forms.RadioButton radME2;
		private System.Windows.Forms.TextBox txtME;
		private System.Windows.Forms.GroupBox groupBox5;
		private System.Windows.Forms.ComboBox cmbRemove;
		private System.Windows.Forms.Button Removebutton;
		private System.Windows.Forms.Button Applybutton;
		private System.Windows.Forms.Label label11;
		private System.Windows.Forms.Label label12;
		private System.Windows.Forms.TextBox txtVariableName;
		private System.Windows.Forms.TextBox txtMeasurementUnit;
		private System.Windows.Forms.GroupBox groupBox6;
		private System.Windows.Forms.Button RemoveDataSet;
		private System.Windows.Forms.ComboBox cmbDataSet;
		private System.Windows.Forms.Button SwitchDataSet;
		private System.Windows.Forms.TextBox m_SkewednessText;
		private System.Windows.Forms.Label label13;
		private System.Windows.Forms.TextBox m_ObservationAngleText;
		private System.Windows.Forms.Label label14;
		private System.Windows.Forms.ComboBox cmbPalette;
		private System.Windows.Forms.Label label15;
		private System.Windows.Forms.Button ShowDataButton;
		private NumericalTools.PlotControl graphicsControl1;
        private ComboBox cmbDX;
        private Label label16;
        private ComboBox cmbDY;
        private Label label17;
        private TextBox m_ArrowSizeText;
        private Label label18;
        private TextBox m_ArrowScaleText;
        private Label label19;
        private TextBox m_ArrowSampleText;
        private Label label20;
        private CheckBox chkGraphDataSet;

		private System.ComponentModel.Container components = null;

		public AnalysisControl()
		{
			// This call is required by the Windows.Forms Form Designer.
			InitializeComponent();

			// TODO: Add any initialization after the InitForm call
			m_SkewednessText.Text = graphicsControl1.Skewedness.ToString();
			m_ObservationAngleText.Text = graphicsControl1.ObservationAngle.ToString();

			cmbPlotType.Items.Add("GScatter");
			cmbPlotType.Items.Add("GLEntries");
			cmbPlotType.Items.Add("GLQuantities");
			cmbPlotType.Items.Add("Histo");
            cmbPlotType.Items.Add("HSkyline");
            cmbPlotType.Items.Add("SymbolEntries");
			cmbPlotType.Items.Add("HueEntries");
			cmbPlotType.Items.Add("HueQuantities");
			cmbPlotType.Items.Add("LEGO");
			cmbPlotType.Items.Add("Scatter");            
            cmbPlotType.Items.Add("ScatterHue");            
			cmbPlotType.Items.Add("Scatter3D");            
            cmbPlotType.Items.Add("ArrowPlot");
			cmbPlotType.SelectedIndex = 3;

			int n;
			cmbXDeg.Items.Add("No Fit");
			cmbXDeg.SelectedIndex=0;
			cmbXDeg.Items.Add("Inv.Gauss");
			cmbXDeg.Items.Add("Gauss");
			for (n = 1; n < 16; n++)
				cmbXDeg.Items.Add(n.ToString());

			cmbYDeg.Items.Add("No Fit");
			cmbYDeg.SelectedIndex=0;
			for(n = 1; n < 16; n++)
				cmbYDeg.Items.Add(n.ToString());
            graphicsControl1.ArrowScale = 10.0;
            graphicsControl1.ArrowSize = 5;
            graphicsControl1.ArrowSample = 0.1;
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
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.txtXbin = new System.Windows.Forms.TextBox();
            this.txtMinX = new System.Windows.Forms.TextBox();
            this.cmbX = new System.Windows.Forms.ComboBox();
            this.label3 = new System.Windows.Forms.Label();
            this.txtMaxX = new System.Windows.Forms.TextBox();
            this.label2 = new System.Windows.Forms.Label();
            this.label1 = new System.Windows.Forms.Label();
            this.groupBox2 = new System.Windows.Forms.GroupBox();
            this.txtMaxY = new System.Windows.Forms.TextBox();
            this.txtMinY = new System.Windows.Forms.TextBox();
            this.txtYbin = new System.Windows.Forms.TextBox();
            this.cmbY = new System.Windows.Forms.ComboBox();
            this.label4 = new System.Windows.Forms.Label();
            this.label5 = new System.Windows.Forms.Label();
            this.label6 = new System.Windows.Forms.Label();
            this.groupBox3 = new System.Windows.Forms.GroupBox();
            this.cmbYDeg = new System.Windows.Forms.ComboBox();
            this.cmbXDeg = new System.Windows.Forms.ComboBox();
            this.chkFit = new System.Windows.Forms.CheckBox();
            this.label8 = new System.Windows.Forms.Label();
            this.label9 = new System.Windows.Forms.Label();
            this.label7 = new System.Windows.Forms.Label();
            this.label10 = new System.Windows.Forms.Label();
            this.Plotbutton = new System.Windows.Forms.Button();
            this.cmbZ = new System.Windows.Forms.ComboBox();
            this.cmbPlotType = new System.Windows.Forms.ComboBox();
            this.groupBox4 = new System.Windows.Forms.GroupBox();
            this.label12 = new System.Windows.Forms.Label();
            this.txtMeasurementUnit = new System.Windows.Forms.TextBox();
            this.txtVariableName = new System.Windows.Forms.TextBox();
            this.label11 = new System.Windows.Forms.Label();
            this.Applybutton = new System.Windows.Forms.Button();
            this.txtME = new System.Windows.Forms.TextBox();
            this.radME2 = new System.Windows.Forms.RadioButton();
            this.radME1 = new System.Windows.Forms.RadioButton();
            this.radME0 = new System.Windows.Forms.RadioButton();
            this.groupBox5 = new System.Windows.Forms.GroupBox();
            this.Removebutton = new System.Windows.Forms.Button();
            this.cmbRemove = new System.Windows.Forms.ComboBox();
            this.groupBox6 = new System.Windows.Forms.GroupBox();
            this.SwitchDataSet = new System.Windows.Forms.Button();
            this.RemoveDataSet = new System.Windows.Forms.Button();
            this.cmbDataSet = new System.Windows.Forms.ComboBox();
            this.m_SkewednessText = new System.Windows.Forms.TextBox();
            this.label13 = new System.Windows.Forms.Label();
            this.m_ObservationAngleText = new System.Windows.Forms.TextBox();
            this.label14 = new System.Windows.Forms.Label();
            this.cmbPalette = new System.Windows.Forms.ComboBox();
            this.label15 = new System.Windows.Forms.Label();
            this.ShowDataButton = new System.Windows.Forms.Button();
            this.graphicsControl1 = new NumericalTools.PlotControl();
            this.cmbDX = new System.Windows.Forms.ComboBox();
            this.label16 = new System.Windows.Forms.Label();
            this.cmbDY = new System.Windows.Forms.ComboBox();
            this.label17 = new System.Windows.Forms.Label();
            this.m_ArrowSizeText = new System.Windows.Forms.TextBox();
            this.label18 = new System.Windows.Forms.Label();
            this.m_ArrowScaleText = new System.Windows.Forms.TextBox();
            this.label19 = new System.Windows.Forms.Label();
            this.m_ArrowSampleText = new System.Windows.Forms.TextBox();
            this.label20 = new System.Windows.Forms.Label();
            this.chkGraphDataSet = new System.Windows.Forms.CheckBox();
            this.groupBox1.SuspendLayout();
            this.groupBox2.SuspendLayout();
            this.groupBox3.SuspendLayout();
            this.groupBox4.SuspendLayout();
            this.groupBox5.SuspendLayout();
            this.groupBox6.SuspendLayout();
            this.SuspendLayout();
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.txtXbin);
            this.groupBox1.Controls.Add(this.txtMinX);
            this.groupBox1.Controls.Add(this.cmbX);
            this.groupBox1.Controls.Add(this.label3);
            this.groupBox1.Controls.Add(this.txtMaxX);
            this.groupBox1.Controls.Add(this.label2);
            this.groupBox1.Controls.Add(this.label1);
            this.groupBox1.Location = new System.Drawing.Point(691, 0);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(264, 70);
            this.groupBox1.TabIndex = 1;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "X Variable";
            // 
            // txtXbin
            // 
            this.txtXbin.Location = new System.Drawing.Point(56, 40);
            this.txtXbin.Name = "txtXbin";
            this.txtXbin.Size = new System.Drawing.Size(72, 20);
            this.txtXbin.TabIndex = 1;
            this.txtXbin.TextChanged += new System.EventHandler(this.txtXbin_TextChanged);
            // 
            // txtMinX
            // 
            this.txtMinX.Location = new System.Drawing.Point(184, 40);
            this.txtMinX.Name = "txtMinX";
            this.txtMinX.Size = new System.Drawing.Size(72, 20);
            this.txtMinX.TabIndex = 3;
            this.txtMinX.TextChanged += new System.EventHandler(this.txtMinX_TextChanged);
            // 
            // cmbX
            // 
            this.cmbX.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cmbX.Location = new System.Drawing.Point(8, 16);
            this.cmbX.Name = "cmbX";
            this.cmbX.Size = new System.Drawing.Size(120, 21);
            this.cmbX.TabIndex = 7;
            this.cmbX.SelectedIndexChanged += new System.EventHandler(this.cmbX_SelectedIndexChanged);
            // 
            // label3
            // 
            this.label3.Location = new System.Drawing.Point(124, 41);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(64, 16);
            this.label3.TabIndex = 6;
            this.label3.Text = "Min Scale:";
            this.label3.TextAlign = System.Drawing.ContentAlignment.MiddleRight;
            // 
            // txtMaxX
            // 
            this.txtMaxX.Location = new System.Drawing.Point(184, 16);
            this.txtMaxX.Name = "txtMaxX";
            this.txtMaxX.Size = new System.Drawing.Size(72, 20);
            this.txtMaxX.TabIndex = 2;
            this.txtMaxX.TextChanged += new System.EventHandler(this.txtMaxX_TextChanged);
            // 
            // label2
            // 
            this.label2.Location = new System.Drawing.Point(124, 16);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(64, 16);
            this.label2.TabIndex = 5;
            this.label2.Text = "Max Scale:";
            this.label2.TextAlign = System.Drawing.ContentAlignment.MiddleRight;
            // 
            // label1
            // 
            this.label1.Location = new System.Drawing.Point(6, 41);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(56, 16);
            this.label1.TabIndex = 4;
            this.label1.Text = "Bin Size:";
            this.label1.TextAlign = System.Drawing.ContentAlignment.MiddleRight;
            // 
            // groupBox2
            // 
            this.groupBox2.Controls.Add(this.txtMaxY);
            this.groupBox2.Controls.Add(this.txtMinY);
            this.groupBox2.Controls.Add(this.txtYbin);
            this.groupBox2.Controls.Add(this.cmbY);
            this.groupBox2.Controls.Add(this.label4);
            this.groupBox2.Controls.Add(this.label5);
            this.groupBox2.Controls.Add(this.label6);
            this.groupBox2.Location = new System.Drawing.Point(691, 70);
            this.groupBox2.Name = "groupBox2";
            this.groupBox2.Size = new System.Drawing.Size(264, 72);
            this.groupBox2.TabIndex = 2;
            this.groupBox2.TabStop = false;
            this.groupBox2.Text = "Y Variable";
            // 
            // txtMaxY
            // 
            this.txtMaxY.Location = new System.Drawing.Point(184, 17);
            this.txtMaxY.Name = "txtMaxY";
            this.txtMaxY.Size = new System.Drawing.Size(72, 20);
            this.txtMaxY.TabIndex = 2;
            this.txtMaxY.TextChanged += new System.EventHandler(this.txtMaxY_TextChanged);
            // 
            // txtMinY
            // 
            this.txtMinY.Location = new System.Drawing.Point(184, 41);
            this.txtMinY.Name = "txtMinY";
            this.txtMinY.Size = new System.Drawing.Size(72, 20);
            this.txtMinY.TabIndex = 3;
            this.txtMinY.TextChanged += new System.EventHandler(this.txtMinY_TextChanged);
            // 
            // txtYbin
            // 
            this.txtYbin.Location = new System.Drawing.Point(56, 40);
            this.txtYbin.Name = "txtYbin";
            this.txtYbin.Size = new System.Drawing.Size(72, 20);
            this.txtYbin.TabIndex = 1;
            this.txtYbin.TextChanged += new System.EventHandler(this.txtYbin_TextChanged);
            // 
            // cmbY
            // 
            this.cmbY.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cmbY.Location = new System.Drawing.Point(8, 16);
            this.cmbY.Name = "cmbY";
            this.cmbY.Size = new System.Drawing.Size(120, 21);
            this.cmbY.TabIndex = 7;
            this.cmbY.SelectedIndexChanged += new System.EventHandler(this.cmbY_SelectedIndexChanged);
            // 
            // label4
            // 
            this.label4.Location = new System.Drawing.Point(123, 42);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(64, 16);
            this.label4.TabIndex = 6;
            this.label4.Text = "Min Scale:";
            this.label4.TextAlign = System.Drawing.ContentAlignment.MiddleRight;
            // 
            // label5
            // 
            this.label5.Location = new System.Drawing.Point(124, 18);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(64, 16);
            this.label5.TabIndex = 5;
            this.label5.Text = "Max Scale:";
            this.label5.TextAlign = System.Drawing.ContentAlignment.MiddleRight;
            // 
            // label6
            // 
            this.label6.Location = new System.Drawing.Point(4, 41);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(56, 16);
            this.label6.TabIndex = 4;
            this.label6.Text = "Bin Size:";
            this.label6.TextAlign = System.Drawing.ContentAlignment.MiddleRight;
            // 
            // groupBox3
            // 
            this.groupBox3.Controls.Add(this.cmbYDeg);
            this.groupBox3.Controls.Add(this.cmbXDeg);
            this.groupBox3.Controls.Add(this.chkFit);
            this.groupBox3.Controls.Add(this.label8);
            this.groupBox3.Controls.Add(this.label9);
            this.groupBox3.Location = new System.Drawing.Point(691, 519);
            this.groupBox3.Name = "groupBox3";
            this.groupBox3.Size = new System.Drawing.Size(267, 54);
            this.groupBox3.TabIndex = 3;
            this.groupBox3.TabStop = false;
            this.groupBox3.Text = "Fitting";
            // 
            // cmbYDeg
            // 
            this.cmbYDeg.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cmbYDeg.Location = new System.Drawing.Point(173, 23);
            this.cmbYDeg.Name = "cmbYDeg";
            this.cmbYDeg.Size = new System.Drawing.Size(88, 21);
            this.cmbYDeg.TabIndex = 9;
            // 
            // cmbXDeg
            // 
            this.cmbXDeg.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cmbXDeg.Location = new System.Drawing.Point(80, 24);
            this.cmbXDeg.Name = "cmbXDeg";
            this.cmbXDeg.Size = new System.Drawing.Size(88, 21);
            this.cmbXDeg.TabIndex = 8;
            // 
            // chkFit
            // 
            this.chkFit.Checked = true;
            this.chkFit.CheckState = System.Windows.Forms.CheckState.Checked;
            this.chkFit.Location = new System.Drawing.Point(8, 16);
            this.chkFit.Name = "chkFit";
            this.chkFit.Size = new System.Drawing.Size(72, 26);
            this.chkFit.TabIndex = 7;
            this.chkFit.Text = "Only Data Plot";
            // 
            // label8
            // 
            this.label8.Location = new System.Drawing.Point(173, 8);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(64, 16);
            this.label8.TabIndex = 5;
            this.label8.Text = "Scatter:";
            this.label8.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // label9
            // 
            this.label9.Location = new System.Drawing.Point(80, 8);
            this.label9.Name = "label9";
            this.label9.Size = new System.Drawing.Size(64, 16);
            this.label9.TabIndex = 4;
            this.label9.Text = "Histogram:";
            this.label9.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // label7
            // 
            this.label7.Location = new System.Drawing.Point(699, 150);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(64, 16);
            this.label7.TabIndex = 5;
            this.label7.Text = "Z Variable:";
            this.label7.TextAlign = System.Drawing.ContentAlignment.MiddleRight;
            // 
            // label10
            // 
            this.label10.Location = new System.Drawing.Point(835, 150);
            this.label10.Name = "label10";
            this.label10.Size = new System.Drawing.Size(56, 15);
            this.label10.TabIndex = 6;
            this.label10.Text = "Plot Type:";
            this.label10.TextAlign = System.Drawing.ContentAlignment.MiddleRight;
            // 
            // Plotbutton
            // 
            this.Plotbutton.Location = new System.Drawing.Point(690, 631);
            this.Plotbutton.Name = "Plotbutton";
            this.Plotbutton.Size = new System.Drawing.Size(81, 24);
            this.Plotbutton.TabIndex = 8;
            this.Plotbutton.Text = "Plot";
            this.Plotbutton.Click += new System.EventHandler(this.Plotbutton_Click);
            // 
            // cmbZ
            // 
            this.cmbZ.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cmbZ.Location = new System.Drawing.Point(699, 174);
            this.cmbZ.Name = "cmbZ";
            this.cmbZ.Size = new System.Drawing.Size(123, 21);
            this.cmbZ.TabIndex = 10;
            // 
            // cmbPlotType
            // 
            this.cmbPlotType.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cmbPlotType.Location = new System.Drawing.Point(835, 174);
            this.cmbPlotType.Name = "cmbPlotType";
            this.cmbPlotType.Size = new System.Drawing.Size(123, 21);
            this.cmbPlotType.TabIndex = 11;
            // 
            // groupBox4
            // 
            this.groupBox4.Controls.Add(this.label12);
            this.groupBox4.Controls.Add(this.txtMeasurementUnit);
            this.groupBox4.Controls.Add(this.txtVariableName);
            this.groupBox4.Controls.Add(this.label11);
            this.groupBox4.Controls.Add(this.Applybutton);
            this.groupBox4.Controls.Add(this.txtME);
            this.groupBox4.Controls.Add(this.radME2);
            this.groupBox4.Controls.Add(this.radME1);
            this.groupBox4.Controls.Add(this.radME0);
            this.groupBox4.Location = new System.Drawing.Point(691, 391);
            this.groupBox4.Name = "groupBox4";
            this.groupBox4.Size = new System.Drawing.Size(267, 120);
            this.groupBox4.TabIndex = 12;
            this.groupBox4.TabStop = false;
            this.groupBox4.Text = "Math Expression";
            // 
            // label12
            // 
            this.label12.AutoSize = true;
            this.label12.Location = new System.Drawing.Point(134, 75);
            this.label12.Name = "label12";
            this.label12.Size = new System.Drawing.Size(96, 13);
            this.label12.TabIndex = 8;
            this.label12.Text = "Measurement Unit:";
            // 
            // txtMeasurementUnit
            // 
            this.txtMeasurementUnit.Location = new System.Drawing.Point(135, 89);
            this.txtMeasurementUnit.Name = "txtMeasurementUnit";
            this.txtMeasurementUnit.Size = new System.Drawing.Size(118, 20);
            this.txtMeasurementUnit.TabIndex = 7;
            // 
            // txtVariableName
            // 
            this.txtVariableName.Location = new System.Drawing.Point(9, 89);
            this.txtVariableName.Name = "txtVariableName";
            this.txtVariableName.Size = new System.Drawing.Size(118, 20);
            this.txtVariableName.TabIndex = 6;
            // 
            // label11
            // 
            this.label11.AutoSize = true;
            this.label11.Location = new System.Drawing.Point(8, 76);
            this.label11.Name = "label11";
            this.label11.Size = new System.Drawing.Size(38, 13);
            this.label11.TabIndex = 5;
            this.label11.Text = "Name:";
            // 
            // Applybutton
            // 
            this.Applybutton.Location = new System.Drawing.Point(96, 48);
            this.Applybutton.Name = "Applybutton";
            this.Applybutton.Size = new System.Drawing.Size(160, 24);
            this.Applybutton.TabIndex = 4;
            this.Applybutton.Text = "Apply";
            this.Applybutton.Click += new System.EventHandler(this.Applybutton_Click);
            // 
            // txtME
            // 
            this.txtME.Location = new System.Drawing.Point(96, 20);
            this.txtME.Name = "txtME";
            this.txtME.ScrollBars = System.Windows.Forms.ScrollBars.Horizontal;
            this.txtME.Size = new System.Drawing.Size(160, 20);
            this.txtME.TabIndex = 3;
            // 
            // radME2
            // 
            this.radME2.Location = new System.Drawing.Point(8, 54);
            this.radME2.Name = "radME2";
            this.radME2.Size = new System.Drawing.Size(88, 16);
            this.radME2.TabIndex = 2;
            this.radME2.Text = "Add Variable";
            // 
            // radME1
            // 
            this.radME1.Location = new System.Drawing.Point(8, 36);
            this.radME1.Name = "radME1";
            this.radME1.Size = new System.Drawing.Size(88, 16);
            this.radME1.TabIndex = 1;
            this.radME1.Text = "Apply Cut";
            // 
            // radME0
            // 
            this.radME0.Checked = true;
            this.radME0.Location = new System.Drawing.Point(8, 19);
            this.radME0.Name = "radME0";
            this.radME0.Size = new System.Drawing.Size(88, 16);
            this.radME0.TabIndex = 0;
            this.radME0.TabStop = true;
            this.radME0.Text = "Plot";
            // 
            // groupBox5
            // 
            this.groupBox5.Controls.Add(this.Removebutton);
            this.groupBox5.Controls.Add(this.cmbRemove);
            this.groupBox5.Location = new System.Drawing.Point(691, 575);
            this.groupBox5.Name = "groupBox5";
            this.groupBox5.Size = new System.Drawing.Size(267, 48);
            this.groupBox5.TabIndex = 13;
            this.groupBox5.TabStop = false;
            this.groupBox5.Text = "Remove Variable";
            // 
            // Removebutton
            // 
            this.Removebutton.Location = new System.Drawing.Point(144, 16);
            this.Removebutton.Name = "Removebutton";
            this.Removebutton.Size = new System.Drawing.Size(112, 24);
            this.Removebutton.TabIndex = 1;
            this.Removebutton.Text = "Remove";
            this.Removebutton.Click += new System.EventHandler(this.Removebutton_Click);
            // 
            // cmbRemove
            // 
            this.cmbRemove.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cmbRemove.Location = new System.Drawing.Point(4, 16);
            this.cmbRemove.Name = "cmbRemove";
            this.cmbRemove.Size = new System.Drawing.Size(123, 21);
            this.cmbRemove.TabIndex = 0;
            // 
            // groupBox6
            // 
            this.groupBox6.Controls.Add(this.SwitchDataSet);
            this.groupBox6.Controls.Add(this.RemoveDataSet);
            this.groupBox6.Controls.Add(this.cmbDataSet);
            this.groupBox6.Location = new System.Drawing.Point(16, 615);
            this.groupBox6.Name = "groupBox6";
            this.groupBox6.Size = new System.Drawing.Size(669, 48);
            this.groupBox6.TabIndex = 14;
            this.groupBox6.TabStop = false;
            this.groupBox6.Text = "Data Sets";
            // 
            // SwitchDataSet
            // 
            this.SwitchDataSet.Location = new System.Drawing.Point(447, 16);
            this.SwitchDataSet.Name = "SwitchDataSet";
            this.SwitchDataSet.Size = new System.Drawing.Size(104, 24);
            this.SwitchDataSet.TabIndex = 2;
            this.SwitchDataSet.Text = "Switch to";
            this.SwitchDataSet.Click += new System.EventHandler(this.SwitchDataSet_Click);
            // 
            // RemoveDataSet
            // 
            this.RemoveDataSet.Location = new System.Drawing.Point(559, 16);
            this.RemoveDataSet.Name = "RemoveDataSet";
            this.RemoveDataSet.Size = new System.Drawing.Size(104, 24);
            this.RemoveDataSet.TabIndex = 1;
            this.RemoveDataSet.Text = "Remove";
            this.RemoveDataSet.Click += new System.EventHandler(this.RemoveDataSet_Click);
            // 
            // cmbDataSet
            // 
            this.cmbDataSet.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cmbDataSet.Location = new System.Drawing.Point(8, 16);
            this.cmbDataSet.Name = "cmbDataSet";
            this.cmbDataSet.Size = new System.Drawing.Size(433, 21);
            this.cmbDataSet.TabIndex = 0;
            // 
            // m_SkewednessText
            // 
            this.m_SkewednessText.Location = new System.Drawing.Point(779, 206);
            this.m_SkewednessText.Name = "m_SkewednessText";
            this.m_SkewednessText.Size = new System.Drawing.Size(40, 20);
            this.m_SkewednessText.TabIndex = 15;
            this.m_SkewednessText.Leave += new System.EventHandler(this.OnSkewednessTextLeave);
            // 
            // label13
            // 
            this.label13.Location = new System.Drawing.Point(699, 206);
            this.label13.Name = "label13";
            this.label13.Size = new System.Drawing.Size(72, 16);
            this.label13.TabIndex = 16;
            this.label13.Text = "Skewedness";
            this.label13.TextAlign = System.Drawing.ContentAlignment.MiddleRight;
            // 
            // m_ObservationAngleText
            // 
            this.m_ObservationAngleText.Location = new System.Drawing.Point(899, 206);
            this.m_ObservationAngleText.Name = "m_ObservationAngleText";
            this.m_ObservationAngleText.Size = new System.Drawing.Size(56, 20);
            this.m_ObservationAngleText.TabIndex = 17;
            this.m_ObservationAngleText.Leave += new System.EventHandler(this.OnObservationAngleTextLeave);
            // 
            // label14
            // 
            this.label14.Location = new System.Drawing.Point(835, 206);
            this.label14.Name = "label14";
            this.label14.Size = new System.Drawing.Size(56, 16);
            this.label14.TabIndex = 18;
            this.label14.Text = "XY Angle";
            this.label14.TextAlign = System.Drawing.ContentAlignment.MiddleRight;
            // 
            // cmbPalette
            // 
            this.cmbPalette.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cmbPalette.Items.AddRange(new object[] {
            "RGBContinuous",
            "Flat16",
            "GreyContinuous",
            "Grey16"});
            this.cmbPalette.Location = new System.Drawing.Point(771, 238);
            this.cmbPalette.Name = "cmbPalette";
            this.cmbPalette.Size = new System.Drawing.Size(184, 21);
            this.cmbPalette.TabIndex = 20;
            this.cmbPalette.SelectedIndexChanged += new System.EventHandler(this.OnPaletteSelChanged);
            // 
            // label15
            // 
            this.label15.Location = new System.Drawing.Point(699, 238);
            this.label15.Name = "label15";
            this.label15.Size = new System.Drawing.Size(64, 16);
            this.label15.TabIndex = 19;
            this.label15.Text = "Palette:";
            this.label15.TextAlign = System.Drawing.ContentAlignment.MiddleRight;
            // 
            // ShowDataButton
            // 
            this.ShowDataButton.Location = new System.Drawing.Point(855, 631);
            this.ShowDataButton.Name = "ShowDataButton";
            this.ShowDataButton.Size = new System.Drawing.Size(92, 24);
            this.ShowDataButton.TabIndex = 21;
            this.ShowDataButton.Text = "Show Data";
            this.ShowDataButton.Click += new System.EventHandler(this.ShowDataButton_Click);
            // 
            // graphicsControl1
            // 
            this.graphicsControl1.ArrowSample = 1D;
            this.graphicsControl1.ArrowScale = 1D;
            this.graphicsControl1.ArrowSize = 5D;
            this.graphicsControl1.DX = 0F;
            this.graphicsControl1.DY = 0F;
            this.graphicsControl1.FittingOnlyDataInPlot = true;
            this.graphicsControl1.FunctionOverlayed = false;
            this.graphicsControl1.HistoColor = System.Drawing.Color.Red;
            this.graphicsControl1.HistoFill = false;
            this.graphicsControl1.HistoFit = ((short)(0));
            this.graphicsControl1.LabelFont = new System.Drawing.Font("Comic Sans MS", 9F, System.Drawing.FontStyle.Bold);
            this.graphicsControl1.LinearFitWE = false;
            this.graphicsControl1.Location = new System.Drawing.Point(8, 8);
            this.graphicsControl1.MaxX = 0D;
            this.graphicsControl1.MaxY = 0D;
            this.graphicsControl1.MinX = 0D;
            this.graphicsControl1.MinY = 0D;
            this.graphicsControl1.Name = "graphicsControl1";
            this.graphicsControl1.ObservationAngle = 30D;
            this.graphicsControl1.Palette = NumericalTools.Plot.PaletteType.RGBContinuous;
            this.graphicsControl1.Panel = null;
            this.graphicsControl1.PanelFont = new System.Drawing.Font("Comic Sans MS", 9F, System.Drawing.FontStyle.Bold);
            this.graphicsControl1.PanelFormat = null;
            this.graphicsControl1.PanelX = 1D;
            this.graphicsControl1.PanelY = 0D;
            this.graphicsControl1.PlotThickness = 0;
            this.graphicsControl1.ScatterFit = ((short)(0));
            this.graphicsControl1.SetXDefaultLimits = true;
            this.graphicsControl1.SetYDefaultLimits = true;
            this.graphicsControl1.Size = new System.Drawing.Size(677, 601);
            this.graphicsControl1.Skewedness = 0.3D;
            this.graphicsControl1.TabIndex = 22;
            this.graphicsControl1.XTitle = null;
            this.graphicsControl1.YTitle = null;
            this.graphicsControl1.ZTitle = null;
            // 
            // cmbDX
            // 
            this.cmbDX.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cmbDX.Location = new System.Drawing.Point(699, 298);
            this.cmbDX.Name = "cmbDX";
            this.cmbDX.Size = new System.Drawing.Size(123, 21);
            this.cmbDX.TabIndex = 24;
            // 
            // label16
            // 
            this.label16.Location = new System.Drawing.Point(699, 275);
            this.label16.Name = "label16";
            this.label16.Size = new System.Drawing.Size(123, 15);
            this.label16.TabIndex = 23;
            this.label16.Text = "DX Variable/Up err:";
            this.label16.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // cmbDY
            // 
            this.cmbDY.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cmbDY.Location = new System.Drawing.Point(832, 298);
            this.cmbDY.Name = "cmbDY";
            this.cmbDY.Size = new System.Drawing.Size(123, 21);
            this.cmbDY.TabIndex = 26;
            // 
            // label17
            // 
            this.label17.Location = new System.Drawing.Point(832, 275);
            this.label17.Name = "label17";
            this.label17.Size = new System.Drawing.Size(115, 15);
            this.label17.TabIndex = 25;
            this.label17.Text = "DY Variable/Down err:";
            this.label17.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // m_ArrowSizeText
            // 
            this.m_ArrowSizeText.Location = new System.Drawing.Point(899, 333);
            this.m_ArrowSizeText.Name = "m_ArrowSizeText";
            this.m_ArrowSizeText.Size = new System.Drawing.Size(56, 20);
            this.m_ArrowSizeText.TabIndex = 29;
            this.m_ArrowSizeText.Text = "5";
            this.m_ArrowSizeText.Leave += new System.EventHandler(this.OnArrowSizeTextLeave);
            // 
            // label18
            // 
            this.label18.Location = new System.Drawing.Point(827, 333);
            this.label18.Name = "label18";
            this.label18.Size = new System.Drawing.Size(64, 16);
            this.label18.TabIndex = 30;
            this.label18.Text = "Arrow Size";
            this.label18.TextAlign = System.Drawing.ContentAlignment.MiddleRight;
            // 
            // m_ArrowScaleText
            // 
            this.m_ArrowScaleText.Location = new System.Drawing.Point(779, 333);
            this.m_ArrowScaleText.Name = "m_ArrowScaleText";
            this.m_ArrowScaleText.Size = new System.Drawing.Size(40, 20);
            this.m_ArrowScaleText.TabIndex = 27;
            this.m_ArrowScaleText.Text = "10";
            this.m_ArrowScaleText.Leave += new System.EventHandler(this.OnArrowScaleTextLeave);
            // 
            // label19
            // 
            this.label19.Location = new System.Drawing.Point(699, 333);
            this.label19.Name = "label19";
            this.label19.Size = new System.Drawing.Size(72, 16);
            this.label19.TabIndex = 28;
            this.label19.Text = "Arrow Scale";
            this.label19.TextAlign = System.Drawing.ContentAlignment.MiddleRight;
            // 
            // m_ArrowSampleText
            // 
            this.m_ArrowSampleText.Location = new System.Drawing.Point(779, 358);
            this.m_ArrowSampleText.Name = "m_ArrowSampleText";
            this.m_ArrowSampleText.Size = new System.Drawing.Size(40, 20);
            this.m_ArrowSampleText.TabIndex = 31;
            this.m_ArrowSampleText.Text = "0.1";
            this.m_ArrowSampleText.Leave += new System.EventHandler(this.OnArrowSampleTextLeave);
            // 
            // label20
            // 
            this.label20.Location = new System.Drawing.Point(679, 358);
            this.label20.Name = "label20";
            this.label20.Size = new System.Drawing.Size(92, 20);
            this.label20.TabIndex = 32;
            this.label20.Text = "Arrow Sample";
            this.label20.TextAlign = System.Drawing.ContentAlignment.MiddleRight;
            // 
            // chkGraphDataSet
            // 
            this.chkGraphDataSet.Location = new System.Drawing.Point(777, 626);
            this.chkGraphDataSet.Name = "chkGraphDataSet";
            this.chkGraphDataSet.Size = new System.Drawing.Size(72, 33);
            this.chkGraphDataSet.TabIndex = 33;
            this.chkGraphDataSet.Text = "Data Set from plot";
            // 
            // AnalysisControl
            // 
            this.Controls.Add(this.chkGraphDataSet);
            this.Controls.Add(this.m_ArrowSampleText);
            this.Controls.Add(this.label20);
            this.Controls.Add(this.m_ArrowSizeText);
            this.Controls.Add(this.label18);
            this.Controls.Add(this.m_ArrowScaleText);
            this.Controls.Add(this.label19);
            this.Controls.Add(this.cmbDY);
            this.Controls.Add(this.label17);
            this.Controls.Add(this.cmbDX);
            this.Controls.Add(this.label16);
            this.Controls.Add(this.graphicsControl1);
            this.Controls.Add(this.ShowDataButton);
            this.Controls.Add(this.cmbPalette);
            this.Controls.Add(this.label15);
            this.Controls.Add(this.m_ObservationAngleText);
            this.Controls.Add(this.label14);
            this.Controls.Add(this.m_SkewednessText);
            this.Controls.Add(this.label13);
            this.Controls.Add(this.groupBox6);
            this.Controls.Add(this.groupBox5);
            this.Controls.Add(this.groupBox4);
            this.Controls.Add(this.cmbPlotType);
            this.Controls.Add(this.cmbZ);
            this.Controls.Add(this.Plotbutton);
            this.Controls.Add(this.label10);
            this.Controls.Add(this.label7);
            this.Controls.Add(this.groupBox3);
            this.Controls.Add(this.groupBox2);
            this.Controls.Add(this.groupBox1);
            this.Name = "AnalysisControl";
            this.Size = new System.Drawing.Size(958, 667);
            this.Resize += new System.EventHandler(this.UserControl1_Resize);
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            this.groupBox2.ResumeLayout(false);
            this.groupBox2.PerformLayout();
            this.groupBox3.ResumeLayout(false);
            this.groupBox4.ResumeLayout(false);
            this.groupBox4.PerformLayout();
            this.groupBox5.ResumeLayout(false);
            this.groupBox6.ResumeLayout(false);
            this.ResumeLayout(false);
            this.PerformLayout();

		}
		#endregion

		#region Internal Variables
		//sono i dati del solo dataset corrente
		string [] names;
		double [] bin;
		double [] max;
		double [] min;
		double [,] data;
		//è dimensionato secondo il numero di dataset
		int[] entries;
		//contiene tutti i dati
		ArrayList ArrData = new ArrayList();
		//contiene tutti i nomi dei dati ed i dataset a cui appartengono
		//nell'ordine preciso in cui sono stati inseriti
		ArrayList HistData = new ArrayList();
		
		ArrayList DataSets = new ArrayList();
		#endregion

		#region External Methods


		public void AutoVariableStatistics(int index, double [] var)
		{
			if (m_CurrentDataSet < 0) return;
			System.Data.DataTable dt = ((System.Data.DataSet)DataSets[m_CurrentDataSet]).Tables[0];
			System.Data.DataTable dm = ((System.Data.DataSet)DataSets[m_CurrentDataSet]).Tables[1];

			bool isinteger = true;
			int i;				
			if (var == null)
			{
				var = new double[dt.Rows.Count];
				for (i = 0; i < dt.Rows.Count; i++)
					var[i] = Convert.ToDouble(dt.Rows[i][index]);
			}
			foreach (double x in var)
				if (Math.Round(x) != x) 
				{
					isinteger = false;
					break;
				}

			double vMin = -0.5;
			double vMax = 0.5;
			double vAvg = 0.0;
			double vRMS = 0.0;
			double vBin = 1.0;
			if (var.Length > 0)
			{
				Fitting.FindStatistics(var, ref vMax, ref vMin, ref vAvg, ref vRMS);
				if (vMax <= vMin)
				{
					vMax = vAvg + 1.0;
					vMin = vAvg - 1.0;
					vBin = 0.5;
				}
				else
				{
					vBin = Math.Pow(10.0, 1.0 + Math.Round(Math.Log10(vRMS / Math.Sqrt(var.Length))));
					if (isinteger) 
						vBin = Math.Ceiling(vBin);
					vMin = Math.Floor(vMin / vBin) * vBin;
					vMax = Math.Ceiling(vMax / vBin) * vBin;
				}
			}
			else
			{
				vMin = -1.0;
				vMax = 1.0;
				vBin = 0.5;
			}			
			dm.Rows[1][index] = vMin;
			dm.Rows[2][index] = vMax;
			dm.Rows[3][index] = vBin;

			if (cmbX.SelectedIndex == index)
			{
				txtMaxX.Text = vMax.ToString(System.Globalization.CultureInfo.InvariantCulture);
				txtMinX.Text = vMin.ToString(System.Globalization.CultureInfo.InvariantCulture);
				txtXbin.Text = vBin.ToString(System.Globalization.CultureInfo.InvariantCulture);
			}

			if (cmbY.SelectedIndex == index)
			{
				txtMaxY.Text = vMax.ToString(System.Globalization.CultureInfo.InvariantCulture);
				txtMinY.Text = vMin.ToString(System.Globalization.CultureInfo.InvariantCulture);
				txtYbin.Text = vBin.ToString(System.Globalization.CultureInfo.InvariantCulture);
			}

			GC.Collect();
		}

        public double GetDouble(int row, int col)
        {
            return (double)((System.Data.DataSet)DataSets[m_CurrentDataSet]).Tables[0].Rows[row][col];
        }

        public int CurrentDataRows
        {
            get { return ((System.Data.DataSet)DataSets[m_CurrentDataSet]).Tables[0].Rows.Count; }
        }

		public void AddRow(double [] r)
		{
			System.Data.DataTable dt = ((System.Data.DataSet)DataSets[m_CurrentDataSet]).Tables[0];
			System.Data.DataRow dr = dt.Rows.Add(new object [r.Length]);
			int i;
			for (i = 0; i < r.Length; i++)
				dr[i] = r[i];
		}

		public bool AddVariable(double[] var, string name, string unit_meas)
		{
			if (m_CurrentDataSet < 0) return false;
			System.Data.DataTable dt = ((System.Data.DataSet)DataSets[m_CurrentDataSet]).Tables[0];
			System.Data.DataTable dm = ((System.Data.DataSet)DataSets[m_CurrentDataSet]).Tables[1];
			int i, j;
			if (dt.Rows.Count != 0 && var.Length != dt.Rows.Count) return false;
			name = name.Trim();
			foreach (System.Data.DataColumn dc in dt.Columns)
			{
				string s = dc.ColumnName;
				if (String.Compare(s, name, true) == 0) return false;
			}
			dt.Columns.Add(name, typeof(double));
			dm.Columns.Add(name);			
			j = dt.Columns.Count - 1;			
			if (j == 0)
			{
				dm.Rows.Add(new object[1] {""});
				dm.Rows.Add(new object[1] {0.0});
				dm.Rows.Add(new object[1] {0.0});
				dm.Rows.Add(new object[1] {0.0});
			}
			dm.Rows[0][j] = unit_meas;			
			for (i = 0; i < var.Length; i++)
			{
				if (j == 0) dt.Rows.Add(new object[1] {0.0});
				dt.Rows[i][j] = var[i];
			}

			AutoVariableStatistics(j, var);

			cmbX.Items.Add(name);
			cmbY.Items.Add(name);
			cmbZ.Items.Add(name);
            cmbDX.Items.Add(name);
            cmbDY.Items.Add(name);
			cmbRemove.Items.Add(name);
			
			if (j == 0) cmbX.SelectedIndex = 0;
			if (j <= 1) cmbY.SelectedIndex = Math.Min(1, cmbY.Items.Count - 1);
			if (j <= 2) cmbZ.SelectedIndex = Math.Min(2, cmbZ.Items.Count - 1);
            if (j <= 3) cmbDX.SelectedIndex = Math.Min(3, cmbDX.Items.Count - 1);
            if (j <= 4) cmbDY.SelectedIndex = Math.Min(4, cmbDY.Items.Count - 1);

			return true;
		}


		public void DumpCurrentDataSetIntoFile(string FilePath)
		{
			if (m_CurrentDataSet < 0) return;
			System.IO.StreamWriter w = null;
			try
			{
				int i;
				w = new System.IO.StreamWriter(FilePath);
				System.Data.DataTable dt = ((System.Data.DataSet)DataSets[m_CurrentDataSet]).Tables[0];
				foreach (System.Data.DataColumn dc in dt.Columns)
				{
					if (dc.Ordinal > 0) w.Write("\t");
					w.Write(dc.ColumnName);
				}
				w.WriteLine();
				foreach (System.Data.DataRow dr in dt.Rows)
				{
					for (i = 0; i < dr.ItemArray.Length; i++)
					{
						if (i > 0) w.Write("\t");
						w.Write(Convert.ToDouble(dr.ItemArray[i]).ToString(System.Globalization.CultureInfo.InvariantCulture));
					}
					w.WriteLine();						
				}
				w.Flush();
				w.Close();				
			}
			catch(Exception x)
			{
				if (w != null) w.Close();
				MessageBox.Show(x.ToString(), "Error dumping dataset", MessageBoxButtons.OK, MessageBoxIcon.Error);	
			};
		}

		public bool RemoveVariable(int Index)
		{
			if (m_CurrentDataSet < 0) return false;
			System.Data.DataTable dt = ((System.Data.DataSet)DataSets[m_CurrentDataSet]).Tables[0];			
			System.Data.DataTable dm = ((System.Data.DataSet)DataSets[m_CurrentDataSet]).Tables[1];			
			dt.Columns.RemoveAt(Index);
			dm.Columns.RemoveAt(Index);
			cmbX.Items.RemoveAt(Index);
			cmbY.Items.RemoveAt(Index);
			cmbZ.Items.RemoveAt(Index);
            cmbDX.Items.RemoveAt(Index);
            cmbDY.Items.RemoveAt(Index);
			cmbRemove.Items.RemoveAt(Index);
			return true;
		}

		public bool RemoveVariable(string Name)
		{
			if (m_CurrentDataSet < 0) return false;
			System.Data.DataTable dt = ((System.Data.DataSet)DataSets[m_CurrentDataSet]).Tables[0];			
			foreach (System.Data.DataColumn dc in dt.Columns)
				if (String.Compare(Name, dc.ColumnName, true) == 0)
					return RemoveVariable(dc.Ordinal);
			return false;
		}


		public bool RemoveDSet(int Index)
		{
			if (Index == m_CurrentDataSet)
			{
				cmbX.Items.Clear();
				cmbY.Items.Clear();
				cmbZ.Items.Clear();
				cmbRemove.Items.Clear();
			}
			DataSets.RemoveAt(Index);
			cmbDataSet.Items.RemoveAt(Index);
			while (m_CurrentDataSet >= DataSets.Count) m_CurrentDataSet--;
			return true;
		}

		public bool AddDataSet(string DataSetName)
		{
			System.Data.DataSet ds = new System.Data.DataSet();
			ds.Tables.Add(); ds.Tables[0].TableName = "Data";
			ds.Tables.Add(); ds.Tables[1].TableName = "Statistics";
			DataSets.Add(ds);
			cmbDataSet.Items.Add(DataSetName);
			CurrentDataSet = cmbDataSet.Items.Count - 1;
			return true;
		}

		#endregion

		#region External Properties

		public Bitmap BitmapPlot
		{
			get
			{
				return (Bitmap)graphicsControl1.BitmapPlot.Clone();	
			}

		}

		public int Variables
		{
			get
			{
				if (m_CurrentDataSet >= 0) return ((System.Data.DataSet)DataSets[m_CurrentDataSet]).Tables[0].Columns.Count;
				return 0;
			}
		}

		private int m_CurrentDataSet = -1;

		public int CurrentDataSet
		{
			get
			{
				return m_CurrentDataSet;
			}

			set
			{
				if (value > DataSets.Count) throw new System.Exception("Invalid Index");				
				if (value < 0) return;

				m_CurrentDataSet = value;
				System.Data.DataTable dt = ((System.Data.DataSet)DataSets[m_CurrentDataSet]).Tables[0];
				System.Data.DataTable dm = ((System.Data.DataSet)DataSets[m_CurrentDataSet]).Tables[1];

				cmbX.Items.Clear();
				cmbY.Items.Clear();
				cmbZ.Items.Clear();
                cmbDX.Items.Clear();
                cmbDY.Items.Clear();
				cmbRemove.Items.Clear();

				foreach (System.Data.DataColumn dc in dm.Columns)
				{
					cmbX.Items.Add(dc.ColumnName);
					cmbY.Items.Add(dc.ColumnName);
					cmbZ.Items.Add(dc.ColumnName);
                    cmbDX.Items.Add(dc.ColumnName);
                    cmbDY.Items.Add(dc.ColumnName);
					cmbRemove.Items.Add(dc.ColumnName);
				}

				if (cmbX.Items.Count > 0) cmbX.SelectedIndex = 0;
				if (cmbY.Items.Count > 0) 
					cmbY.SelectedIndex = (cmbY.Items.Count > 1) ? 1 : 0;
				if (cmbZ.Items.Count > 0) 
					cmbZ.SelectedIndex = (cmbZ.Items.Count > 2) ? 2 : cmbZ.Items.Count - 1;
                if (cmbDX.Items.Count > 0)
                    cmbDX.SelectedIndex = (cmbDX.Items.Count > 3) ? 3 : cmbDX.Items.Count - 1;
                if (cmbDY.Items.Count > 0)
                    cmbDY.SelectedIndex = (cmbDY.Items.Count > 4) ? 4 : cmbDY.Items.Count - 1;
            }
		}


		public int DataSetNumber
		{
			get
			{
				return DataSets.Count;
			}

		}
		#endregion

		#region Events

        static int GraphPlotCount = 1;
	
		private void Plotbutton_Click(object sender, System.EventArgs e)
		{
			if (m_CurrentDataSet < 0) return;
			System.Data.DataTable dt = ((System.Data.DataSet)DataSets[m_CurrentDataSet]).Tables[0];
			System.Data.DataTable dm = ((System.Data.DataSet)DataSets[m_CurrentDataSet]).Tables[1];
			int n = dt.Rows.Count;
			if (n > 0)
			{
				double[] tmpx= new double[n];
				double[] tmpy= new double[n];
				double[] tmpz= new double[n];
                double[] tmpdx = new double[n];
                double[] tmpdy = new double[n];

				int ix, iy, iz, idx, idy, j;

				ix = cmbX.SelectedIndex;
				iy = cmbY.SelectedIndex;
				iz = cmbZ.SelectedIndex;
                idx = cmbDX.SelectedIndex;
                idy = cmbDY.SelectedIndex;

				if (ix >= 0)
					for (j = 0; j < n; j++)
						tmpx[j] = Convert.ToDouble(dt.Rows[j][ix]);

				if (iy >= 0)
					for (j = 0; j < n; j++)
						tmpy[j] = Convert.ToDouble(dt.Rows[j][iy]);

				if (iz >= 0)
					for (j = 0; j < n; j++)
						tmpz[j] = Convert.ToDouble(dt.Rows[j][iz]);

                if (idx >= 0)
                    for (j = 0; j < n; j++)
                        tmpdx[j] = Convert.ToDouble(dt.Rows[j][idx]);

                if (idy >= 0)
                    for (j = 0; j < n; j++)
                        tmpdy[j] = Convert.ToDouble(dt.Rows[j][idy]);

                //Fit
				if (cmbYDeg.Text == "No Fit")
					graphicsControl1.ScatterFit = 0;
				else
					graphicsControl1.ScatterFit = Convert.ToInt16(cmbYDeg.SelectedIndex);

				if (cmbXDeg.Text == "No Fit")
					graphicsControl1.HistoFit =-2;			
				else if (cmbXDeg.Text == "Inv.Gauss")
					graphicsControl1.HistoFit =-1;			
				else if (cmbXDeg.Text == "Gauss")
					graphicsControl1.HistoFit =0;			
				else
					graphicsControl1.HistoFit = Convert.ToInt16(cmbXDeg.Text/*.SelectedIndex*/);

				graphicsControl1.FunctionOverlayed = false;
				graphicsControl1.SetXDefaultLimits = false;
				graphicsControl1.SetYDefaultLimits = false;
				graphicsControl1.FittingOnlyDataInPlot= chkFit.Checked;

				try
				{
					
					graphicsControl1.DX = Convert.ToSingle(txtXbin.Text, System.Globalization.CultureInfo.InvariantCulture);
					graphicsControl1.MaxX =Convert.ToDouble(txtMaxX.Text, System.Globalization.CultureInfo.InvariantCulture);
					graphicsControl1.MinX = Convert.ToDouble(txtMinX.Text, System.Globalization.CultureInfo.InvariantCulture);
                    if (cmbPlotType.Text != "Histo" && cmbPlotType.Text != "HSkyline")
					{
						graphicsControl1.DY = Convert.ToSingle(txtYbin.Text, System.Globalization.CultureInfo.InvariantCulture);
						graphicsControl1.MaxY =Convert.ToDouble(txtMaxY.Text, System.Globalization.CultureInfo.InvariantCulture);
						graphicsControl1.MinY = Convert.ToDouble(txtMinY.Text, System.Globalization.CultureInfo.InvariantCulture);
					};

				} 
				catch(Exception ec)
				{
					System.Windows.Forms.MessageBox.Show(this,"Not a numerical value in text box.\r\n" + ec.ToString(),
						"Analysis Control",MessageBoxButtons.OK, MessageBoxIcon.Exclamation);
					return;
				};


				if (graphicsControl1.DX < 0)
				{
					System.Windows.Forms.MessageBox.Show(this,"Wrong X bin size.\r\n" ,
						"Analysis Control",MessageBoxButtons.OK, MessageBoxIcon.Exclamation);
					return;
				};

				if (graphicsControl1.MaxX - graphicsControl1.MinX < graphicsControl1.DX)
				{
					System.Windows.Forms.MessageBox.Show(this,"X values allow less than two bins.\r\n" ,
						"Analysis Control",MessageBoxButtons.OK, MessageBoxIcon.Exclamation);
					return;
				};

                if (cmbPlotType.Text != "Histo" && cmbPlotType.Text != "HSkyline" && (graphicsControl1.DY < 0))
				{
					System.Windows.Forms.MessageBox.Show(this,"Wrong Y bin size.\r\n" ,
						"Analysis Control",MessageBoxButtons.OK, MessageBoxIcon.Exclamation);
					return;
				};

                if (cmbPlotType.Text != "Histo" && cmbPlotType.Text != "HSkyline" && (graphicsControl1.MaxY - graphicsControl1.MinY < graphicsControl1.DY))
				{
					System.Windows.Forms.MessageBox.Show(this,"Y values allow less than two bins.\r\n" ,
						"Analysis Control",MessageBoxButtons.OK, MessageBoxIcon.Exclamation);
					return;
				};

				graphicsControl1.XTitle = cmbX.Text + ((dm.Rows[0][ix].ToString().Trim().Length > 0) ? (" (" + dm.Rows[0][ix].ToString() + ")") : ""); //cmbX.Text;
				graphicsControl1.YTitle = cmbY.Text + ((dm.Rows[0][iy].ToString().Trim().Length > 0) ? (" (" + dm.Rows[0][iy].ToString() + ")") : "");//cmbY.Text;
				graphicsControl1.ZTitle = cmbZ.Text + ((dm.Rows[0][iz].ToString().Trim().Length > 0) ? (" (" + dm.Rows[0][iz].ToString() + ")") : "");//cmbZ.Text;
				graphicsControl1.VecX = tmpx;
				graphicsControl1.VecY = tmpy;
				graphicsControl1.VecZ = tmpz;
                graphicsControl1.VecDX = tmpdx;
                graphicsControl1.VecDY = tmpdy;

                graphicsControl1.Clear();
                double [][] nds = null;
                string xname = cmbX.Text;
                string yname = cmbY.Text;
                string zname = cmbZ.Text;
                string newdataset = (txtVariableName.Text.Length == 0) ? ("Graph_" + GraphPlotCount++) : txtVariableName.Text; 
                if (cmbPlotType.Text == "Histo")
                {
                    nds = graphicsControl1.Histo();
                    if (chkGraphDataSet.Checked)
                    {
                        AddDataSet(newdataset);
                        AddVariable(nds[0], xname, "");
                        AddVariable(nds[1], "Counts", "");
                        AddVariable(nds[2], "Cumulated", "");
                    }
                }
                else if (cmbPlotType.Text == "HSkyline")
                {
                    nds = graphicsControl1.HistoSkyline();
                    if (chkGraphDataSet.Checked)
                    {
                        AddDataSet(newdataset);
                        AddVariable(nds[0], xname, "");
                        AddVariable(nds[1], "Counts", "");
                        AddVariable(nds[2], "Cumulated", "");
                    }
                }
                else if (cmbPlotType.Text == "Scatter")                
                    graphicsControl1.Scatter();                    
                else if (cmbPlotType.Text == "ScatterHue")
                    graphicsControl1.ScatterHue();
                else if (cmbPlotType.Text == "ArrowPlot")
                    graphicsControl1.ArrowPlot();
                else if (cmbPlotType.Text == "GScatter")
                {
                    nds = graphicsControl1.GroupScatter();
                    if (chkGraphDataSet.Checked)
                    {
                        AddDataSet(newdataset);
                        AddVariable(nds[0], xname, "");
                        AddVariable(nds[1], yname, "");
                        AddVariable(nds[2], "StdDev", "");
                        AddVariable(nds[2], "Counts", "");
                    }
                }
                else if (cmbPlotType.Text == "GLEntries")
                {
                    nds = graphicsControl1.GreyLevelArea();
                    if (chkGraphDataSet.Checked)
                    {
                        AddDataSet(newdataset);
                        AddVariable(nds[0], xname, "");
                        AddVariable(nds[1], yname, "");
                        AddVariable(nds[2], "Counts", "");
                    }
                }
                else if (cmbPlotType.Text == "SymbolEntries")
                {
                    nds = graphicsControl1.SymbolAreaValues();
                    if (chkGraphDataSet.Checked)
                    {
                        AddDataSet(newdataset);
                        AddVariable(nds[0], xname, "");
                        AddVariable(nds[1], yname, "");
                        AddVariable(nds[2], "Counts", "");
                    }
                }
                else if (cmbPlotType.Text == "HueEntries")
                {
                    nds = graphicsControl1.HueAreaValues();
                    if (chkGraphDataSet.Checked)
                    {
                        AddDataSet(newdataset);
                        AddVariable(nds[0], xname, "");
                        AddVariable(nds[1], yname, "");
                        AddVariable(nds[2], "Counts", "");
                    }
                }
                else if (cmbPlotType.Text == "LEGO")
                {
                    nds = graphicsControl1.LEGO();
                    if (chkGraphDataSet.Checked)
                    {
                        AddDataSet(newdataset);
                        AddVariable(nds[0], xname, "");
                        AddVariable(nds[1], yname, "");
                        AddVariable(nds[2], "Counts", "");
                    }
                }
                else if (cmbPlotType.Text == "Scatter3D")
                    graphicsControl1.Scatter3D();
                else if (cmbPlotType.Text == "GLQuantities")
                {
                    nds = graphicsControl1.GAreaComputedValues();
                    if (chkGraphDataSet.Checked)
                    {
                        AddDataSet(newdataset);
                        AddVariable(nds[0], xname, "");
                        AddVariable(nds[1], yname, "");
                        AddVariable(nds[2], zname, "");
                        AddVariable(nds[3], "Counts", "");
                    }
                }
                else if (cmbPlotType.Text == "HueQuantities")
                {
                    nds = graphicsControl1.HueAreaComputedValues();
                    if (chkGraphDataSet.Checked)
                    {
                        AddDataSet(newdataset);
                        AddVariable(nds[0], xname, "");
                        AddVariable(nds[1], yname, "");
                        AddVariable(nds[2], zname, "");
                        AddVariable(nds[3], "Counts", "");
                    }
                }
			};
		}

		private void UserControl1_Resize(object sender, System.EventArgs e)
		{
			//this.Width = ORIGINALWIDTH;
			//this.Height = ORIGINALHEIGHT;
		}

		private void txtXbin_TextChanged(object sender, System.EventArgs e)
		{
			try
			{
				if (m_CurrentDataSet < 0) return;
				System.Data.DataTable dm = ((System.Data.DataSet)DataSets[m_CurrentDataSet]).Tables[1];
				if (cmbX.SelectedIndex >= 0) dm.Rows[3][cmbX.SelectedIndex] = Convert.ToDouble(txtXbin.Text, System.Globalization.CultureInfo.InvariantCulture);
			}
			catch
			{
				return;
			};

		}

		private void txtMaxX_TextChanged(object sender, System.EventArgs e)
		{
			try
			{
				if (m_CurrentDataSet < 0) return;
				System.Data.DataTable dm = ((System.Data.DataSet)DataSets[m_CurrentDataSet]).Tables[1];
				if (cmbX.SelectedIndex >= 0) dm.Rows[2][cmbX.SelectedIndex] = Convert.ToDouble(txtMaxX.Text, System.Globalization.CultureInfo.InvariantCulture);
			}
			catch
			{
				return;
			};
		}

		private void txtMinX_TextChanged(object sender, System.EventArgs e)
		{
			try
			{
				if (m_CurrentDataSet < 0) return;
				System.Data.DataTable dm = ((System.Data.DataSet)DataSets[m_CurrentDataSet]).Tables[1];
				if (cmbX.SelectedIndex >= 0) dm.Rows[1][cmbX.SelectedIndex] = Convert.ToDouble(txtMinX.Text, System.Globalization.CultureInfo.InvariantCulture);
			}
			catch
			{
				return;
			};

		}

		private void txtYbin_TextChanged(object sender, System.EventArgs e)
		{
			try
			{
				if (m_CurrentDataSet < 0) return;
				System.Data.DataTable dm = ((System.Data.DataSet)DataSets[m_CurrentDataSet]).Tables[1];
				if (cmbY.SelectedIndex >= 0) dm.Rows[3][cmbY.SelectedIndex] = Convert.ToDouble(txtYbin.Text, System.Globalization.CultureInfo.InvariantCulture);
			}
			catch
			{
				return;
			};
		}

		private void txtMaxY_TextChanged(object sender, System.EventArgs e)
		{
			try
			{
				if (m_CurrentDataSet < 0) return;
				System.Data.DataTable dm = ((System.Data.DataSet)DataSets[m_CurrentDataSet]).Tables[1];
				if (cmbY.SelectedIndex >= 0) dm.Rows[2][cmbY.SelectedIndex] = Convert.ToDouble(txtMaxY.Text, System.Globalization.CultureInfo.InvariantCulture);
			}
			catch
			{
				return;
			};
		}

		private void txtMinY_TextChanged(object sender, System.EventArgs e)
		{
			try
			{
				if (m_CurrentDataSet < 0) return;
				System.Data.DataTable dm = ((System.Data.DataSet)DataSets[m_CurrentDataSet]).Tables[1];
				if (cmbY.SelectedIndex >= 0) dm.Rows[1][cmbY.SelectedIndex] = Convert.ToDouble(txtMinY.Text, System.Globalization.CultureInfo.InvariantCulture);
			}
			catch
			{
				return;
			};
		}

		private void cmbX_SelectedIndexChanged(object sender, System.EventArgs e)
		{
			if (cmbX.SelectedIndex >= 0)
			{
				if (m_CurrentDataSet < 0) return;
				System.Data.DataTable dm = ((System.Data.DataSet)DataSets[m_CurrentDataSet]).Tables[1];
				txtMaxX.Text = Convert.ToDouble(dm.Rows[2][cmbX.SelectedIndex]).ToString(System.Globalization.CultureInfo.InvariantCulture);
				txtXbin.Text = Convert.ToDouble(dm.Rows[3][cmbX.SelectedIndex]).ToString(System.Globalization.CultureInfo.InvariantCulture);
				txtMinX.Text = Convert.ToDouble(dm.Rows[1][cmbX.SelectedIndex]).ToString(System.Globalization.CultureInfo.InvariantCulture);
			};		
		}

		private void cmbY_SelectedIndexChanged(object sender, System.EventArgs e)
		{
			if (cmbY.SelectedIndex >= 0)
			{
				if (m_CurrentDataSet < 0) return;
				System.Data.DataTable dm = ((System.Data.DataSet)DataSets[m_CurrentDataSet]).Tables[1];
				txtMaxY.Text = Convert.ToDouble(dm.Rows[2][cmbY.SelectedIndex]).ToString(System.Globalization.CultureInfo.InvariantCulture);
				txtYbin.Text = Convert.ToDouble(dm.Rows[3][cmbY.SelectedIndex]).ToString(System.Globalization.CultureInfo.InvariantCulture);
				txtMinY.Text = Convert.ToDouble(dm.Rows[1][cmbY.SelectedIndex]).ToString(System.Globalization.CultureInfo.InvariantCulture);
			};
		}

		private void Removebutton_Click(object sender, System.EventArgs e)
		{
			if (m_CurrentDataSet < 0) return;
			if (cmbRemove.SelectedIndex < 0) return;

			System.Windows.Forms.DialogResult dr = System.Windows.Forms.MessageBox.Show(this,"Remove Variable:\r\n" + cmbRemove.Text,
				"Analysis Control",MessageBoxButtons.YesNo, MessageBoxIcon.Exclamation);

			if(dr == System.Windows.Forms.DialogResult.No) return;
			RemoveVariable(cmbRemove.SelectedIndex);
		}

		public enum FunctionUse { Cut, CutNew, CutInteractive, AddVariable, Overlay }

		public void ApplyFunction(string functionexpr, FunctionUse use, string fname)
		{					
			if (m_CurrentDataSet < 0) return;
			functionexpr = functionexpr.Trim();
			if (functionexpr=="") return;
			System.Data.DataTable dt = ((System.Data.DataSet)DataSets[m_CurrentDataSet]).Tables[0];

			int i, k = 0, j, nne = 0,l;

			Function f = null;
			try
			{
				f = new CStyleParsedFunction(functionexpr);
			}
			catch (Exception x)
			{
				MessageBox.Show(x.Message, "Error in formula");
				return;
			}
			string varname;
			int npar = f.ParameterList.Length;
			int[] indx=new int[npar];			

			foreach(string parname in f.ParameterList)
				if (String.Compare(parname, "rownumber", true) == 0)
				{
					indx[k] = -1;
					k++;					
				}
				else
					for(i=0; i< dt.Columns.Count; i++) 
					{
						varname = dt.Columns[i].ColumnName;
						if(varname != "" && String.Compare(varname, parname, true) == 0) 
						{
							indx[k]=i;
							k++;
							break;
						};
					};
			
			if(k != npar)
			{
				i = npar - k;
				System.Windows.Forms.MessageBox.Show(this, i + " unknown parameters.",
					"Analysis Control",MessageBoxButtons.OK, MessageBoxIcon.Exclamation);
			
				return;
			};

			if (use == FunctionUse.Overlay)
			{
			
				if (npar > 1)
				{
					System.Windows.Forms.MessageBox.Show(this, "Only 1 parameter is allowed.",
						"Analysis Control",MessageBoxButtons.OK, MessageBoxIcon.Exclamation);
			
					return;
				};

				if (npar == 1)
					if(String.Compare(dt.Columns[cmbX.SelectedIndex].ColumnName, f.ParameterList[0], true) != 0)
					{
						System.Windows.Forms.MessageBox.Show(this, "Only X Variable as parameter is allowed.",
							"Analysis Control",MessageBoxButtons.OK, MessageBoxIcon.Exclamation);
				
						return;
					};

				graphicsControl1.Function = functionexpr;
				graphicsControl1.FunctionOverlayed = true;
                graphicsControl1.RefreshPlot();
                /*
				if (cmbPlotType.Text=="Histo")
					graphicsControl1.Histo();
                else if (cmbPlotType.Text == "HSkyline")
                    graphicsControl1.HistoSkyline();
                else if (cmbPlotType.Text == "Scatter")
					graphicsControl1.Scatter();
				else if (cmbPlotType.Text =="GScatter")
					graphicsControl1.GroupScatter();
				else if (cmbPlotType.Text =="GLEntries")
					graphicsControl1.GreyLevelArea();
				else if (cmbPlotType.Text =="HueEntries")
					graphicsControl1.HueAreaValues();
				else if (cmbPlotType.Text == "GLQuantities")
					graphicsControl1.GAreaComputedValues();
				else if (cmbPlotType.Text == "HueQuantities")
					graphicsControl1.HueAreaComputedValues();				*/
			}
			else if (use == FunctionUse.Cut || use == FunctionUse.CutNew || use == FunctionUse.CutInteractive)
			{
				for (i = 0; i < dt.Rows.Count; i++)				
				{
					System.Data.DataRow dr = dt.Rows[i];
					for (j = 0; j < npar; j++) f[j] = (indx[j] < 0) ? (double)i : Convert.ToDouble(dr[indx[j]]);
					if (Convert.ToBoolean(f.Evaluate())) nne++;
				}

				if (use == FunctionUse.CutInteractive)
				{
					System.Windows.Forms.DialogResult dr = System.Windows.Forms.MessageBox.Show(this, 
						nne + " entries selected\r\n Do you wish to create a new data set?\r\n (If not current dataset will be modified)",
						"Analysis Control",MessageBoxButtons.YesNoCancel, MessageBoxIcon.Exclamation);

					if(dr == System.Windows.Forms.DialogResult.Cancel) return;
					use = (dr == DialogResult.Yes) ? FunctionUse.CutNew : FunctionUse.Cut;
				}

				if(use == FunctionUse.Cut) 
				{
					dt.AcceptChanges();
					for (i = 0; i < dt.Rows.Count; i++)
					{
						System.Data.DataRow dr = dt.Rows[i];
						for (j = 0; j < npar; j++) f[j] = (indx[j] < 0) ? (double)i : Convert.ToDouble(dr[indx[j]]);
						if (!Convert.ToBoolean(f.Evaluate()))
						{
							dr.Delete();
							//dt.Rows.RemoveAt(i);
							//i--;
						}
					}
					dt.AcceptChanges();
					GC.Collect();
					for (i = 0; i < dt.Columns.Count; i++)
						AutoVariableStatistics(i, null);
					GC.Collect();
				}
				else if(use == FunctionUse.CutNew)
				{
					System.Data.DataSet ds = ((System.Data.DataSet)DataSets[m_CurrentDataSet]);
					System.Data.DataSet nds = (System.Data.DataSet)ds.Clone();
					foreach (System.Data.DataRow dr in ds.Tables[1].Rows)
						nds.Tables[1].Rows.Add(dr.ItemArray);
					for (i = 0; i < dt.Rows.Count; i++)					
					{
						System.Data.DataRow dr = dt.Rows[i];
						for (j = 0; j < npar; j++) f[j] = (indx[j] < 0) ? (double)i : Convert.ToDouble(dr[indx[j]]);
						if (Convert.ToBoolean(f.Evaluate())) 
							nds.Tables[0].Rows.Add(dr.ItemArray);
					}

					string dsname = (fname.Trim().Length > 0) ? fname.Trim() : (cmbDataSet.Items[m_CurrentDataSet] + "_" + functionexpr);
					AddDataSet(dsname);
					DataSets[DataSets.Count - 1] = nds;
					CurrentDataSet = DataSets.Count - 1;
					cmbDataSet.SelectedIndex = CurrentDataSet;
					GC.Collect();
					for (i = 0; i < dt.Columns.Count; i++)
						AutoVariableStatistics(i, null);
					GC.Collect();
				}
			}
			else
			{
				if(fname.Trim() == "")
				{
					System.Windows.Forms.MessageBox.Show(this, "Insert a Name for this Variable",
						"Analysis Control",MessageBoxButtons.OK, MessageBoxIcon.Exclamation);

					return;				
				};

				for(i = 0; i < dt.Columns.Count; i++) 
					if(String.Compare(dt.Columns[i].ColumnName, fname, true) == 0)
					{
						System.Windows.Forms.MessageBox.Show(this, "Variable Name already in use",
							"Analysis Control",MessageBoxButtons.OK, MessageBoxIcon.Exclamation);					
						return;
					};

				dt.Columns.Add(fname, typeof(double));
				k = dt.Columns.Count - 1;
				System.Data.DataTable dm = ((System.Data.DataSet)DataSets[m_CurrentDataSet]).Tables[1];
				dm.Columns.Add(fname);

				for(i = 0; i < dt.Rows.Count; i++) 
				{
					System.Data.DataRow dr = dt.Rows[i];
					for (j = 0; j < npar; j++) f[j] = (indx[j] < 0) ? (double)i : Convert.ToDouble(dr[indx[j]]);
					dr[k] = f.Evaluate();
				};
				AutoVariableStatistics(k, null);
				cmbX.Items.Add(fname);
				cmbY.Items.Add(fname);
				cmbZ.Items.Add(fname);
                cmbDX.Items.Add(fname);
                cmbDY.Items.Add(fname);
				cmbRemove.Items.Add(fname);
				GC.Collect();
			};
		}

		private void Applybutton_Click(object sender, System.EventArgs e)
		{
			ApplyFunction(txtME.Text, radME0.Checked ? FunctionUse.Overlay : (radME1.Checked ? FunctionUse.CutInteractive : FunctionUse.AddVariable), txtVariableName.Text);
			return;
		}


		#endregion

		private void SwitchDataSet_Click(object sender, System.EventArgs e)
		{
			try
			{
				if (cmbDataSet.SelectedIndex==-1)
					CurrentDataSet = 0;
				else
					CurrentDataSet = cmbDataSet.SelectedIndex;
			}
			catch
			{
				return;
			};
		
		}

		private void RemoveDataSet_Click(object sender, System.EventArgs e)
		{
			if(cmbDataSet.SelectedIndex<0) return;
			System.Windows.Forms.DialogResult dr = System.Windows.Forms.MessageBox.Show(this, 
				"Do you wish to delete " + cmbDataSet.Items[cmbDataSet.SelectedIndex] + " data set?\r\n",
				"Analysis Control",MessageBoxButtons.YesNo, MessageBoxIcon.Exclamation);
			if(dr==System.Windows.Forms.DialogResult.Yes) RemoveDSet(cmbDataSet.SelectedIndex);
		}

		private void OnSkewednessTextLeave(object sender, System.EventArgs e)
		{
			try
			{
				double sk = Convert.ToDouble(m_SkewednessText.Text);
				if (sk < 0.1 || sk > 0.9) throw new Exception("The skewedness angle should be a number between 0.1 and 0.9.");
				graphicsControl1.Skewedness = sk;
			}
			catch (Exception x)
			{
				MessageBox.Show(x.ToString(), "Error");
			}		
		}

		private void OnObservationAngleTextLeave(object sender, System.EventArgs e)
		{
			try
			{
				double oba = Convert.ToDouble(m_ObservationAngleText.Text);
				if (oba < 0 || oba > 360.0) throw new Exception("The observation angle should be a number between 0 and 360 degrees.");
				graphicsControl1.ObservationAngle = oba;
			}
			catch (Exception x)
			{
				MessageBox.Show(x.ToString(), "Error");
			}		
		}

		/// <summary>
		/// Retrieves the name of a dataset.
		/// </summary>
		/// <param name="i">the dataset for which the name is to be read.</param>
		/// <returns>the dataset name.</returns>
		public string DataSetName(int i)
		{
			return cmbDataSet.Items[i].ToString();
		}
		/// <summary>
		/// Selects a dataset into the statistical analysis manager panel.
		/// </summary>
		/// <param name="name">the name of the dataset to be selected (case insensitive).</param>
		public void SelectDataSet(string name)
		{
			int i;
			for (i = 0; i < cmbDataSet.Items.Count && String.Compare(cmbDataSet.Items[i].ToString(), name, true) != 0; i++);
			if (i == cmbDataSet.Items.Count) throw new Exception("Unknown dataset \"" + name + "\".");
			CurrentDataSet = i;
		}
		/// <summary>
		/// Retrieves the name of a variable in the current dataset.
		/// </summary>
		/// <param name="i">the variable for which the name is sought.</param>
		/// <returns>the variable name.</returns>		
		public string VariableName(int i)
		{
			if (m_CurrentDataSet < 0) return "";
			return ((System.Data.DataSet)DataSets[m_CurrentDataSet]).Tables[0].Columns[i].ColumnName;
		}
		/// <summary>
		/// Selects a variable into the X axis, using default extents and binning.
		/// </summary>
		/// <param name="varname">the name of the variable to be selected (case insensitive).</param>
		public void SetX(string varname)
		{
			if (m_CurrentDataSet < 0) return;
			System.Data.DataTable dt = ((System.Data.DataSet)DataSets[m_CurrentDataSet]).Tables[0];
			System.Data.DataTable dm = ((System.Data.DataSet)DataSets[m_CurrentDataSet]).Tables[1];

			int iv, i, n;
			for (iv = 0; iv < dt.Columns.Count && String.Compare(dt.Columns[iv].ColumnName, varname, true) != 0; iv++);
			if (iv == dt.Columns.Count) throw new Exception("Unknown variable \"" + varname + "\".");
			n = dt.Rows.Count;
			double [] tmp = new double[n];
			double min = 0.0, max = 0.0, bin = 1.0; 
			if (n > 0)
			{
				bool isinteger = true;
				min = max = Convert.ToDouble(dt.Rows[0][iv]);
				for (i = 0; i < n; i++) 
				{
					tmp[i] = Convert.ToDouble(dt.Rows[i][iv]);
					if (min > tmp[i]) min = tmp[i];
					else if (max < tmp[i]) max = tmp[i];
					if (Math.Round(tmp[i]) != tmp[i]) isinteger = false;
				}				
				bin = (max - min) / Math.Sqrt(n); 
				if (isinteger && bin < 1.0) bin = 1.0;
				if (bin <= 0.0) bin = 1.0;
				min -= 0.5 * bin;
				max += 0.5 * bin;				
			}
			graphicsControl1.VecX = tmp;
			graphicsControl1.MinX = min;
			graphicsControl1.MaxX = max;
			graphicsControl1.DX = (float)bin;
			graphicsControl1.SetXDefaultLimits = false;
			graphicsControl1.XTitle = dt.Columns[iv].ColumnName + " (" + dm.Rows[0][iv].ToString() + ")";
			GC.Collect();
		}

		/// <summary>
		/// Selects a variable into the X axis, specifying extents and binning.
		/// </summary>
		/// <param name="varname">the name of the variable to be selected (case insensitive).</param>
		/// <param name="min">the minimum extent.</param>
		/// <param name="max">the maximum extent.</param>
		/// <param name="bin">the bin size.</param>
		public void SetX(string varname, double min, double max, double bin)
		{
			if (m_CurrentDataSet < 0) return;
			System.Data.DataTable dt = ((System.Data.DataSet)DataSets[m_CurrentDataSet]).Tables[0];
			System.Data.DataTable dm = ((System.Data.DataSet)DataSets[m_CurrentDataSet]).Tables[1];

			int iv, i, n;
			for (iv = 0; iv < dt.Columns.Count && String.Compare(dt.Columns[iv].ColumnName, varname, true) != 0; iv++);
			if (iv == dt.Columns.Count) throw new Exception("Unknown variable \"" + varname + "\".");
			n = dt.Rows.Count;
			double [] tmp = new double[n];
			for (i = 0; i < n; i++) tmp[i] = Convert.ToDouble(dt.Rows[i][iv]);
			graphicsControl1.VecX = tmp;
			graphicsControl1.MinX = min;
			graphicsControl1.MaxX = max;
			graphicsControl1.DX = (float)bin;
			graphicsControl1.SetXDefaultLimits = false;
			graphicsControl1.XTitle = dt.Columns[iv].ColumnName + " (" + dm.Rows[0][iv].ToString() + ")";
			GC.Collect();
		}

		/// <summary>
		/// Selects a variable into the Y axis, using default extents and binning.
		/// </summary>
		/// <param name="varname">the name of the variable to be selected (case insensitive).</param>
		public void SetY(string varname)
		{
			if (m_CurrentDataSet < 0) return;
			System.Data.DataTable dt = ((System.Data.DataSet)DataSets[m_CurrentDataSet]).Tables[0];
			System.Data.DataTable dm = ((System.Data.DataSet)DataSets[m_CurrentDataSet]).Tables[1];

			int iv, i, n;
			for (iv = 0; iv < dt.Columns.Count && String.Compare(dt.Columns[iv].ColumnName, varname, true) != 0; iv++);
			if (iv == dt.Columns.Count) throw new Exception("Unknown variable \"" + varname + "\".");
			n = dt.Rows.Count;
			double [] tmp = new double[n];
			double min = 0.0, max = 0.0, bin = 1.0; 
			if (n > 0)
			{
				bool isinteger = true;
				min = max = Convert.ToDouble(dt.Rows[0][iv]);
				for (i = 0; i < n; i++) 
				{
					tmp[i] = Convert.ToDouble(dt.Rows[i][iv]);
					if (min > tmp[i]) min = tmp[i];
					else if (max < tmp[i]) max = tmp[i];
					if (Math.Round(tmp[i]) != tmp[i]) isinteger = false;
				}				
				bin = (max - min) / Math.Sqrt(n); 
				if (isinteger && bin < 1.0) bin = 1.0;
				if (bin <= 0.0) bin = 1.0;
				min -= 0.5 * bin;
				max += 0.5 * bin;				
			}
			graphicsControl1.VecY = tmp;
			graphicsControl1.MinY = min;
			graphicsControl1.MaxY = max;
			graphicsControl1.DY = (float)bin;
			graphicsControl1.SetYDefaultLimits = false;
			graphicsControl1.YTitle = dt.Columns[iv].ColumnName + " (" + dm.Rows[0][iv].ToString() + ")";
			GC.Collect();
		}

		/// <summary>
		/// Selects a variable into the Y axis, specifying extents and binning.
		/// </summary>
		/// <param name="varname">the name of the variable to be selected (case insensitive).</param>
		/// <param name="min">the minimum extent.</param>
		/// <param name="max">the maximum extent.</param>
		/// <param name="bin">the bin size.</param>
		public void SetY(string varname, double min, double max, double bin)
		{
			if (m_CurrentDataSet < 0) return;
			System.Data.DataTable dt = ((System.Data.DataSet)DataSets[m_CurrentDataSet]).Tables[0];
			System.Data.DataTable dm = ((System.Data.DataSet)DataSets[m_CurrentDataSet]).Tables[1];

			int iv, i, n;
			for (iv = 0; iv < dt.Columns.Count && String.Compare(dt.Columns[iv].ColumnName, varname, true) != 0; iv++);
			if (iv == dt.Columns.Count) throw new Exception("Unknown variable \"" + varname + "\".");
			n = dt.Rows.Count;
			double [] tmp = new double[n];
			for (i = 0; i < n; i++) tmp[i] = Convert.ToDouble(dt.Rows[i][iv]);
			graphicsControl1.VecY = tmp;
			graphicsControl1.MinY = min;
			graphicsControl1.MaxY = max;
			graphicsControl1.DY = (float)bin;
			graphicsControl1.SetYDefaultLimits = false;
			graphicsControl1.YTitle = dt.Columns[iv].ColumnName + " (" + dm.Rows[0][iv].ToString() + ")";
			GC.Collect();
		}

		/// <summary>
		/// Selects a variable into the Z axis.
		/// </summary>
		/// <param name="varname">the name of the variable to be selected (case insensitive).</param>
		public void SetZ(string varname)
		{
			if (m_CurrentDataSet < 0) return;
			System.Data.DataTable dt = ((System.Data.DataSet)DataSets[m_CurrentDataSet]).Tables[0];
			System.Data.DataTable dm = ((System.Data.DataSet)DataSets[m_CurrentDataSet]).Tables[1];

			int iv, i, n;
			for (iv = 0; iv < dt.Columns.Count && String.Compare(dt.Columns[iv].ColumnName, varname, true) != 0; iv++);
			if (iv == dt.Columns.Count) throw new Exception("Unknown variable \"" + varname + "\".");
			n = dt.Rows.Count;
			double [] tmp = new double[n];
			for (i = 0; i < n; i++) tmp[i] = Convert.ToDouble(dt.Rows[i][iv]);
			graphicsControl1.VecZ = tmp;
			graphicsControl1.ZTitle = dt.Columns[iv].ColumnName + " (" + dm.Rows[0][iv].ToString() + ")";
			GC.Collect();						
		}

		/// <summary>
		/// Builds a plot.
		/// </summary>
		/// <param name="plottype">a string that specifies the plot type, through one of these possible values:
		/// <list type="table">
		/// <listheader><term>String</term><description>Plot type</description></listheader>
		/// <item><term><c>histo</c></term><description>1D histogram</description></item>
        /// <item><term><c>hskyline</c></term><description>1D histogram, "skyline" only</description></item>
		/// <item><term><c>glent</c></term><description>grey level plot of entry density</description></item>
        /// <item><term><c>symbent</c></term><description>symbol plot of entry density</description></item>
		/// <item><term><c>hueent</c></term><description>hue plot of entry density</description></item>
		/// <item><term><c>glquant</c></term><description>grey level plot of Z quantity</description></item>
		/// <item><term><c>huequant</c></term><description>hue plot of Z quantity</description></item>
		/// <item><term><c>gscatter</c></term><description>group-scatter plot</description></item>
		/// <item><term><c>lego</c></term><description>LEGO plot</description></item>
		/// <item><term><c>scatter</c></term><description>scatter plots</description></item>
		/// <item><term><c>scatter3d</c></term><description>3D scatter plots</description></item>
        /// <item><term><c>arrowplot</c></term><description>2D scatter plots with arrows</description></item>
		/// </list>
		/// </param>
		/// <param name="fittype">a string that specifies the fit type, through one of these possible values:
		/// <list type="table">
		/// <listheader><term>String</term><description>Fit type</description></listheader>
		/// <item><term><c>null</c> or <c>""</c></term><description>add no fit</description></item>
		/// <item><term><c>gauss</c></term><description>gaussian fit</description></item>
		/// <item><term><c>igauss</c></term><description>inverse gaussian fit</description></item>
		/// <item><term><c>1</c></term><description>linear fit</description></item>
		/// <item><term><c>2</c></term><description>parabolic fit</description></item>
		/// <item><term><c>n</c> (an integer)</term><description>n-th order polynomial fit</description></item>
		/// </list>
		/// </param>
		public void Plot(string plottype, string fittype)
		{			
			if (fittype == null || fittype.Length == 0)
			{
				this.graphicsControl1.HistoFit = -2;
				this.graphicsControl1.ScatterFit = -2;
			}
			else if (String.Compare(fittype, "gauss", true) == 0)
			{
				this.graphicsControl1.HistoFit = 0;
				this.graphicsControl1.ScatterFit = 0;
			}
			else if (String.Compare(fittype, "igauss", true) == 0)
			{
				this.graphicsControl1.HistoFit = -1;
				this.graphicsControl1.ScatterFit = -1;
			}
			else 
			{
				this.graphicsControl1.HistoFit = Convert.ToInt16(fittype);
				this.graphicsControl1.ScatterFit = Convert.ToInt16(fittype);				
			}
			if (String.Compare(plottype, "histo", true) == 0)
			{
				this.graphicsControl1.Histo();
			}
            else if (String.Compare(plottype, "hskyline", true) == 0)
            {
                this.graphicsControl1.HistoSkyline();
            }
            else if (String.Compare(plottype, "glent", true) == 0)
			{
				this.graphicsControl1.GreyLevelArea();
			}
            else if (String.Compare(plottype, "symbent", true) == 0)
            {
                this.graphicsControl1.HueAreaValues();
            }
            else if (String.Compare(plottype, "hueent", true) == 0)
			{
				this.graphicsControl1.HueAreaValues();
			}
			else if (String.Compare(plottype, "glquant", true) == 0)
			{
				this.graphicsControl1.GAreaComputedValues();
			}
			else if (String.Compare(plottype, "huequant", true) == 0)
			{
				this.graphicsControl1.HueAreaComputedValues();
			}
			else if (String.Compare(plottype, "gscatter", true) == 0)
			{
				this.graphicsControl1.GroupScatter();
			}
			else if (String.Compare(plottype, "lego", true) == 0)
			{
				this.graphicsControl1.LEGO();
			}
			else if (String.Compare(plottype, "scatter", true) == 0)
			{
				this.graphicsControl1.Scatter();
			}
			else if (String.Compare(plottype, "scatter3d", true) == 0)
			{
				this.graphicsControl1.Scatter3D();
			}
		}

		private void OnPaletteSelChanged(object sender, System.EventArgs e)
		{
			graphicsControl1.Palette = (NumericalTools.Plot.PaletteType)cmbPalette.SelectedIndex;
		}

		private void ShowDataButton_Click(object sender, System.EventArgs e)
		{
			if (m_CurrentDataSet < 0) return;
			new ShowDataForm(cmbDataSet.Items[m_CurrentDataSet].ToString(), (System.Data.DataSet)DataSets[m_CurrentDataSet]).Show();	
		}

		/// <summary>
		/// The palette type to be used for plotting color plots.
        /// </summary>
		public NumericalTools.Plot.PaletteType Palette
		{
			get { return graphicsControl1.Palette; }
			set 
			{
				graphicsControl1.Palette = value;
			}
		}

		public Color PlotColor
		{
			get { return graphicsControl1.HistoColor; }
			set { graphicsControl1.HistoColor = value; }
		}

		public Font LabelFont
		{
			get { return graphicsControl1.LabelFont; }
			set { graphicsControl1.LabelFont = value; }
		}

		public Font PanelFont
		{
			get { return graphicsControl1.PanelFont; }
			set { graphicsControl1.PanelFont = value; }
		}

		public string PanelFormat
		{
			get { return graphicsControl1.PanelFormat; }
			set { graphicsControl1.PanelFormat = value; }
		}

		public double PanelX
		{
			get { return graphicsControl1.PanelX; }
			set { graphicsControl1.PanelX = value; }
		}

		public double PanelY
		{
			get { return graphicsControl1.PanelY; }
			set { graphicsControl1.PanelY = value; }
		}

		public string Panel
		{
			get { return graphicsControl1.Panel; }
			set { graphicsControl1.Panel = value; }
		}

        private void OnArrowScaleTextLeave(object sender, EventArgs e)
        {
            try
            {
                double oba = Convert.ToDouble(m_ArrowScaleText.Text);
                if (oba <= 0.0) throw new Exception("Arrow scale must be positive.");
                graphicsControl1.ArrowScale = oba;
            }
            catch (Exception x)
            {
                MessageBox.Show(x.ToString(), "Error");
            }		
        }

        private void OnArrowSizeTextLeave(object sender, EventArgs e)
        {
            try
            {
                double oba = Convert.ToDouble(m_ArrowSizeText.Text);
                if (oba < 1.0) throw new Exception("Arrow size must be 1 at least.");
                graphicsControl1.ArrowSize = oba;
            }
            catch (Exception x)
            {
                MessageBox.Show(x.ToString(), "Error");
            }		
        }

        private void OnArrowSampleTextLeave(object sender, EventArgs e)
        {
            try
            {
                double oba = Convert.ToDouble(m_ArrowSampleText.Text);
                if (oba <= 0.0) throw new Exception("Arrow sample must be positive.");
                graphicsControl1.ArrowSample = oba;
            }
            catch (Exception x)
            {
                MessageBox.Show(x.ToString(), "Error");
            }		
        }

        public void SaveMetafile(string f)
        {
            System.Drawing.Bitmap bmp = graphicsControl1.GetCompatibleBitmap();
            System.Drawing.Graphics g = System.Drawing.Graphics.FromImage(bmp);
            IntPtr hdc = g.GetHdc();
            System.Drawing.Imaging.Metafile mf = new System.Drawing.Imaging.Metafile(f, hdc, new Rectangle(0, 0, bmp.Width, bmp.Height), System.Drawing.Imaging.MetafileFrameUnit.Pixel);            
            g.ReleaseHdc();
            g.Dispose();            
            g = System.Drawing.Graphics.FromImage(mf);
            g.ScaleTransform(1.0f, 1.0f);
            graphicsControl1.SaveMetafile(g);            
            g.Dispose();
            mf.Dispose();
        }
	}
}
