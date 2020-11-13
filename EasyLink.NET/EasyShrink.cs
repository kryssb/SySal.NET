using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;
using NumericalTools;

namespace SySal.Executables.EasyLinkNET
{
	/// <summary>
	/// Summary description for EasyShrink.
	/// </summary>
	internal class EasyShrink : System.Windows.Forms.Form
	{
		private System.Windows.Forms.GroupBox groupBox1;
		private System.Windows.Forms.RadioButton radBottom;
		private System.Windows.Forms.RadioButton radTop;
		private System.Windows.Forms.GroupBox groupBox2;
		private System.Windows.Forms.TextBox txtConstant;
		private System.Windows.Forms.Label label4;
		private System.Windows.Forms.TextBox txtPLSlopeMultiplier;
		private System.Windows.Forms.Label label3;
		private System.Windows.Forms.Button cmdPlotLine;
		private System.Windows.Forms.RadioButton radBoth;
		private System.Windows.Forms.Button cmdFitData;
		private System.Windows.Forms.Panel PlotPanelY;
		private System.Windows.Forms.Panel PlotPanelX;
		private System.Windows.Forms.GroupBox groupBox3;
		private System.Windows.Forms.Label label1;
		private System.Windows.Forms.Label label2;
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;
		private System.Windows.Forms.GroupBox groupBox4;
		private System.Windows.Forms.Label label6;
		private System.Windows.Forms.Label label7;
		private System.Windows.Forms.RadioButton radGLevelXZ;
		private System.Windows.Forms.RadioButton radScatterXZ;
		private System.Windows.Forms.RadioButton radGLevelYZ;
		private System.Windows.Forms.RadioButton radScatterYZ;
		private System.Windows.Forms.TextBox txtYBinXZ;
		private System.Windows.Forms.TextBox txtXBinXZ;
		private System.Windows.Forms.TextBox txtYBinYZ;
		private System.Windows.Forms.TextBox txtXBinYZ;
		private SySal.Scanning.Plate.LinkedZone Ret;
		private double[] x;
		private double[] dx;
		private double[] y;
		private double[] dy;
		private double[] angle;
		private double[] chi2;

		private System.Windows.Forms.TextBox txtYMaxXZ;
		private System.Windows.Forms.Label label8;
		private System.Windows.Forms.TextBox txtXMaxXZ;
		private System.Windows.Forms.Label label9;
		private System.Windows.Forms.TextBox txtYMinXZ;
		private System.Windows.Forms.Label label10;
		private System.Windows.Forms.TextBox txtXMinXZ;
		private System.Windows.Forms.Label label11;
		private System.Windows.Forms.TextBox txtYMinYZ;
		private System.Windows.Forms.Label label12;
		private System.Windows.Forms.TextBox txtXMinYZ;
		private System.Windows.Forms.Label label13;
		private System.Windows.Forms.TextBox txtYMaxYZ;
		private System.Windows.Forms.Label label14;
		private System.Windows.Forms.TextBox txtXMaxYZ;
		private System.Windows.Forms.Label label15;
		private System.Windows.Forms.RadioButton radYZ;
		private System.Windows.Forms.RadioButton radXZ;
		private System.Windows.Forms.Label lblXMultiplier;
		private System.Windows.Forms.TextBox txtYMult;
		private System.Windows.Forms.Label lblYMultiplier;
		private System.Windows.Forms.TextBox txtXMult;
		private System.Windows.Forms.GroupBox groupBox5;
		private System.Windows.Forms.Label label5;
		private System.Windows.Forms.Label label16;
		private System.Windows.Forms.Label label17;
		private System.Windows.Forms.TextBox txtMinchi;
		private System.Windows.Forms.TextBox txtMaxchi;
		private System.Windows.Forms.TextBox txtBinchi;
		private System.Windows.Forms.Label label18;
		private System.Windows.Forms.Label label19;
		private System.Windows.Forms.Label label20;
		private System.Windows.Forms.Label label21;
		private System.Windows.Forms.Label label22;
		private System.Windows.Forms.Label label23;
		private System.Windows.Forms.Label label24;
		private System.Windows.Forms.Label label25;
		private System.Windows.Forms.Label label26;
		private System.Windows.Forms.Label label27;
		private System.Windows.Forms.Label label28;
		private System.Windows.Forms.Label label29;
		private System.Windows.Forms.GroupBox groupBox6;
		private System.Windows.Forms.RadioButton optShrinkage;
		private System.Windows.Forms.RadioButton optChi2;
		private System.Windows.Forms.Button cmdPlotChi2;
		private System.Windows.Forms.TextBox txtTopXs0;
		private System.Windows.Forms.TextBox txtTopXdeg;
		private System.Windows.Forms.TextBox txtTopYs0;
		private System.Windows.Forms.TextBox txtTopYdeg;
		private System.Windows.Forms.TextBox txtBotYs0;
		private System.Windows.Forms.TextBox txtBotYdeg;
		private System.Windows.Forms.TextBox txtBotXs0;
		private System.Windows.Forms.TextBox txtBotXdeg;
		private System.Windows.Forms.GroupBox groupBox7;
		private System.Windows.Forms.GroupBox groupBox8;
		private System.Windows.Forms.RadioButton optds_vs_s;
		private System.Windows.Forms.RadioButton optds_vs_angle;
		private System.Windows.Forms.CheckBox chkGaussFit;
		private System.Windows.Forms.Button cmdCut;
		private System.Windows.Forms.TextBox txtchi2cut;
		private System.Windows.Forms.Label label30;
		private System.Windows.Forms.Label lblXDelta;
		private System.Windows.Forms.Label lblYDelta;
		private System.Windows.Forms.TextBox txtYDelta;
		private System.Windows.Forms.TextBox txtXDelta;
		private int NData;
		private double ExternalTopMultiplier;
		private double ExternalBotMultiplier;
		private double ExternalTopXDelta;
		private double ExternalBotXDelta;
		private double ExternalTopYDelta;
		private System.Windows.Forms.Label lblTracksBelow;
		private double ExternalBotYDelta;

		public EasyShrink(SySal.Scanning.Plate.LinkedZone lzone, double topmult, double botmult, double topxdel, double botxdel, double topydel, double botydel)
		{
			//
			// Required for Windows Form Designer support
			//
			InitializeComponent();
			
			Ret = lzone;
			NData = Ret.Length;
			x = new double[NData];
			dx = new double[NData];
			y = new double[NData];
			dy = new double[NData];
			chi2 = new double[NData];
			angle = new double[NData];
			for(int i=0;  i<NData; i++) angle[i]= Math.Sqrt(Ret[i].Info.Slope.X*Ret[i].Info.Slope.X +	Ret[i].Info.Slope.Y*Ret[i].Info.Slope.Y);
			ExternalTopMultiplier = topmult;
			ExternalBotMultiplier = botmult;
			ExternalTopXDelta = topxdel;
			ExternalBotXDelta = botxdel;
			ExternalTopYDelta = topydel;
			ExternalBotYDelta = botydel;
			
			SetData();
			SetChi2Data();
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
			this.groupBox1 = new System.Windows.Forms.GroupBox();
			this.txtYDelta = new System.Windows.Forms.TextBox();
			this.txtXDelta = new System.Windows.Forms.TextBox();
			this.lblXDelta = new System.Windows.Forms.Label();
			this.lblYDelta = new System.Windows.Forms.Label();
			this.txtYMult = new System.Windows.Forms.TextBox();
			this.txtXMult = new System.Windows.Forms.TextBox();
			this.cmdFitData = new System.Windows.Forms.Button();
			this.groupBox4 = new System.Windows.Forms.GroupBox();
			this.txtYMinXZ = new System.Windows.Forms.TextBox();
			this.label10 = new System.Windows.Forms.Label();
			this.txtXMinXZ = new System.Windows.Forms.TextBox();
			this.label11 = new System.Windows.Forms.Label();
			this.txtYMaxXZ = new System.Windows.Forms.TextBox();
			this.label8 = new System.Windows.Forms.Label();
			this.txtXMaxXZ = new System.Windows.Forms.TextBox();
			this.label9 = new System.Windows.Forms.Label();
			this.txtYBinXZ = new System.Windows.Forms.TextBox();
			this.label6 = new System.Windows.Forms.Label();
			this.txtXBinXZ = new System.Windows.Forms.TextBox();
			this.label7 = new System.Windows.Forms.Label();
			this.radGLevelXZ = new System.Windows.Forms.RadioButton();
			this.radScatterXZ = new System.Windows.Forms.RadioButton();
			this.groupBox3 = new System.Windows.Forms.GroupBox();
			this.txtYMinYZ = new System.Windows.Forms.TextBox();
			this.label12 = new System.Windows.Forms.Label();
			this.txtXMinYZ = new System.Windows.Forms.TextBox();
			this.label13 = new System.Windows.Forms.Label();
			this.txtYMaxYZ = new System.Windows.Forms.TextBox();
			this.label14 = new System.Windows.Forms.Label();
			this.txtXMaxYZ = new System.Windows.Forms.TextBox();
			this.label15 = new System.Windows.Forms.Label();
			this.txtYBinYZ = new System.Windows.Forms.TextBox();
			this.label1 = new System.Windows.Forms.Label();
			this.txtXBinYZ = new System.Windows.Forms.TextBox();
			this.label2 = new System.Windows.Forms.Label();
			this.radGLevelYZ = new System.Windows.Forms.RadioButton();
			this.radScatterYZ = new System.Windows.Forms.RadioButton();
			this.lblXMultiplier = new System.Windows.Forms.Label();
			this.lblYMultiplier = new System.Windows.Forms.Label();
			this.radBottom = new System.Windows.Forms.RadioButton();
			this.radTop = new System.Windows.Forms.RadioButton();
			this.groupBox2 = new System.Windows.Forms.GroupBox();
			this.groupBox7 = new System.Windows.Forms.GroupBox();
			this.radBoth = new System.Windows.Forms.RadioButton();
			this.radXZ = new System.Windows.Forms.RadioButton();
			this.radYZ = new System.Windows.Forms.RadioButton();
			this.groupBox6 = new System.Windows.Forms.GroupBox();
			this.optChi2 = new System.Windows.Forms.RadioButton();
			this.optShrinkage = new System.Windows.Forms.RadioButton();
			this.txtConstant = new System.Windows.Forms.TextBox();
			this.label4 = new System.Windows.Forms.Label();
			this.txtPLSlopeMultiplier = new System.Windows.Forms.TextBox();
			this.label3 = new System.Windows.Forms.Label();
			this.cmdPlotLine = new System.Windows.Forms.Button();
			this.PlotPanelY = new System.Windows.Forms.Panel();
			this.PlotPanelX = new System.Windows.Forms.Panel();
			this.groupBox5 = new System.Windows.Forms.GroupBox();
			this.label30 = new System.Windows.Forms.Label();
			this.txtchi2cut = new System.Windows.Forms.TextBox();
			this.cmdCut = new System.Windows.Forms.Button();
			this.chkGaussFit = new System.Windows.Forms.CheckBox();
			this.cmdPlotChi2 = new System.Windows.Forms.Button();
			this.txtBotYs0 = new System.Windows.Forms.TextBox();
			this.txtBotYdeg = new System.Windows.Forms.TextBox();
			this.label24 = new System.Windows.Forms.Label();
			this.label25 = new System.Windows.Forms.Label();
			this.label26 = new System.Windows.Forms.Label();
			this.txtBotXs0 = new System.Windows.Forms.TextBox();
			this.txtBotXdeg = new System.Windows.Forms.TextBox();
			this.label27 = new System.Windows.Forms.Label();
			this.label28 = new System.Windows.Forms.Label();
			this.label29 = new System.Windows.Forms.Label();
			this.txtTopYs0 = new System.Windows.Forms.TextBox();
			this.txtTopYdeg = new System.Windows.Forms.TextBox();
			this.label21 = new System.Windows.Forms.Label();
			this.label22 = new System.Windows.Forms.Label();
			this.label23 = new System.Windows.Forms.Label();
			this.txtTopXs0 = new System.Windows.Forms.TextBox();
			this.txtTopXdeg = new System.Windows.Forms.TextBox();
			this.label18 = new System.Windows.Forms.Label();
			this.label19 = new System.Windows.Forms.Label();
			this.label20 = new System.Windows.Forms.Label();
			this.txtBinchi = new System.Windows.Forms.TextBox();
			this.txtMaxchi = new System.Windows.Forms.TextBox();
			this.txtMinchi = new System.Windows.Forms.TextBox();
			this.label17 = new System.Windows.Forms.Label();
			this.label16 = new System.Windows.Forms.Label();
			this.label5 = new System.Windows.Forms.Label();
			this.groupBox8 = new System.Windows.Forms.GroupBox();
			this.optds_vs_s = new System.Windows.Forms.RadioButton();
			this.optds_vs_angle = new System.Windows.Forms.RadioButton();
			this.lblTracksBelow = new System.Windows.Forms.Label();
			this.groupBox1.SuspendLayout();
			this.groupBox4.SuspendLayout();
			this.groupBox3.SuspendLayout();
			this.groupBox2.SuspendLayout();
			this.groupBox7.SuspendLayout();
			this.groupBox6.SuspendLayout();
			this.groupBox5.SuspendLayout();
			this.groupBox8.SuspendLayout();
			this.SuspendLayout();
			// 
			// groupBox1
			// 
			this.groupBox1.Controls.AddRange(new System.Windows.Forms.Control[] {
																					this.txtYDelta,
																					this.txtXDelta,
																					this.lblXDelta,
																					this.lblYDelta,
																					this.txtYMult,
																					this.txtXMult,
																					this.cmdFitData,
																					this.groupBox4,
																					this.groupBox3,
																					this.lblXMultiplier,
																					this.lblYMultiplier});
			this.groupBox1.Location = new System.Drawing.Point(8, 418);
			this.groupBox1.Name = "groupBox1";
			this.groupBox1.Size = new System.Drawing.Size(868, 128);
			this.groupBox1.TabIndex = 32;
			this.groupBox1.TabStop = false;
			this.groupBox1.Text = "Shrinkage";
			// 
			// txtYDelta
			// 
			this.txtYDelta.Location = new System.Drawing.Point(715, 92);
			this.txtYDelta.Name = "txtYDelta";
			this.txtYDelta.Size = new System.Drawing.Size(82, 20);
			this.txtYDelta.TabIndex = 39;
			this.txtYDelta.Text = "";
			// 
			// txtXDelta
			// 
			this.txtXDelta.Location = new System.Drawing.Point(715, 68);
			this.txtXDelta.Name = "txtXDelta";
			this.txtXDelta.Size = new System.Drawing.Size(82, 20);
			this.txtXDelta.TabIndex = 42;
			this.txtXDelta.Text = "";
			// 
			// lblXDelta
			// 
			this.lblXDelta.AutoSize = true;
			this.lblXDelta.Location = new System.Drawing.Point(579, 68);
			this.lblXDelta.Name = "lblXDelta";
			this.lblXDelta.Size = new System.Drawing.Size(124, 13);
			this.lblXDelta.TabIndex = 40;
			this.lblXDelta.Text = "X Suggested Top Delta:";
			// 
			// lblYDelta
			// 
			this.lblYDelta.AutoSize = true;
			this.lblYDelta.Location = new System.Drawing.Point(579, 92);
			this.lblYDelta.Name = "lblYDelta";
			this.lblYDelta.Size = new System.Drawing.Size(124, 13);
			this.lblYDelta.TabIndex = 41;
			this.lblYDelta.Text = "Y Suggested Top Delta:";
			// 
			// txtYMult
			// 
			this.txtYMult.Location = new System.Drawing.Point(715, 44);
			this.txtYMult.Name = "txtYMult";
			this.txtYMult.Size = new System.Drawing.Size(82, 20);
			this.txtYMult.TabIndex = 34;
			this.txtYMult.Text = "";
			// 
			// txtXMult
			// 
			this.txtXMult.Location = new System.Drawing.Point(715, 20);
			this.txtXMult.Name = "txtXMult";
			this.txtXMult.Size = new System.Drawing.Size(82, 20);
			this.txtXMult.TabIndex = 38;
			this.txtXMult.Text = "";
			// 
			// cmdFitData
			// 
			this.cmdFitData.Location = new System.Drawing.Point(803, 16);
			this.cmdFitData.Name = "cmdFitData";
			this.cmdFitData.Size = new System.Drawing.Size(56, 40);
			this.cmdFitData.TabIndex = 28;
			this.cmdFitData.Text = "Fit Data";
			this.cmdFitData.Click += new System.EventHandler(this.cmdFitData_Click);
			// 
			// groupBox4
			// 
			this.groupBox4.Controls.AddRange(new System.Windows.Forms.Control[] {
																					this.txtYMinXZ,
																					this.label10,
																					this.txtXMinXZ,
																					this.label11,
																					this.txtYMaxXZ,
																					this.label8,
																					this.txtXMaxXZ,
																					this.label9,
																					this.txtYBinXZ,
																					this.label6,
																					this.txtXBinXZ,
																					this.label7,
																					this.radGLevelXZ,
																					this.radScatterXZ});
			this.groupBox4.Location = new System.Drawing.Point(6, 14);
			this.groupBox4.Name = "groupBox4";
			this.groupBox4.Size = new System.Drawing.Size(282, 104);
			this.groupBox4.TabIndex = 34;
			this.groupBox4.TabStop = false;
			this.groupBox4.Text = "Plot Type XZ";
			// 
			// txtYMinXZ
			// 
			this.txtYMinXZ.Location = new System.Drawing.Point(228, 72);
			this.txtYMinXZ.Name = "txtYMinXZ";
			this.txtYMinXZ.Size = new System.Drawing.Size(42, 20);
			this.txtYMinXZ.TabIndex = 45;
			this.txtYMinXZ.Text = "";
			// 
			// label10
			// 
			this.label10.AutoSize = true;
			this.label10.Location = new System.Drawing.Point(194, 76);
			this.label10.Name = "label10";
			this.label10.Size = new System.Drawing.Size(26, 13);
			this.label10.TabIndex = 44;
			this.label10.Text = "Min:";
			// 
			// txtXMinXZ
			// 
			this.txtXMinXZ.Location = new System.Drawing.Point(228, 43);
			this.txtXMinXZ.Name = "txtXMinXZ";
			this.txtXMinXZ.Size = new System.Drawing.Size(42, 20);
			this.txtXMinXZ.TabIndex = 43;
			this.txtXMinXZ.Text = "";
			// 
			// label11
			// 
			this.label11.AutoSize = true;
			this.label11.Location = new System.Drawing.Point(194, 47);
			this.label11.Name = "label11";
			this.label11.Size = new System.Drawing.Size(26, 13);
			this.label11.TabIndex = 42;
			this.label11.Text = "Min:";
			// 
			// txtYMaxXZ
			// 
			this.txtYMaxXZ.Location = new System.Drawing.Point(145, 74);
			this.txtYMaxXZ.Name = "txtYMaxXZ";
			this.txtYMaxXZ.Size = new System.Drawing.Size(42, 20);
			this.txtYMaxXZ.TabIndex = 41;
			this.txtYMaxXZ.Text = "";
			// 
			// label8
			// 
			this.label8.AutoSize = true;
			this.label8.Location = new System.Drawing.Point(115, 78);
			this.label8.Name = "label8";
			this.label8.Size = new System.Drawing.Size(29, 13);
			this.label8.TabIndex = 40;
			this.label8.Text = "Max:";
			// 
			// txtXMaxXZ
			// 
			this.txtXMaxXZ.Location = new System.Drawing.Point(145, 44);
			this.txtXMaxXZ.Name = "txtXMaxXZ";
			this.txtXMaxXZ.Size = new System.Drawing.Size(42, 20);
			this.txtXMaxXZ.TabIndex = 39;
			this.txtXMaxXZ.Text = "";
			// 
			// label9
			// 
			this.label9.AutoSize = true;
			this.label9.Location = new System.Drawing.Point(115, 47);
			this.label9.Name = "label9";
			this.label9.Size = new System.Drawing.Size(29, 13);
			this.label9.TabIndex = 38;
			this.label9.Text = "Max:";
			// 
			// txtYBinXZ
			// 
			this.txtYBinXZ.Location = new System.Drawing.Point(68, 75);
			this.txtYBinXZ.Name = "txtYBinXZ";
			this.txtYBinXZ.Size = new System.Drawing.Size(42, 20);
			this.txtYBinXZ.TabIndex = 37;
			this.txtYBinXZ.Text = "";
			// 
			// label6
			// 
			this.label6.AutoSize = true;
			this.label6.Location = new System.Drawing.Point(10, 78);
			this.label6.Name = "label6";
			this.label6.Size = new System.Drawing.Size(59, 13);
			this.label6.TabIndex = 36;
			this.label6.Text = "Y Bin Size:";
			// 
			// txtXBinXZ
			// 
			this.txtXBinXZ.Location = new System.Drawing.Point(68, 45);
			this.txtXBinXZ.Name = "txtXBinXZ";
			this.txtXBinXZ.Size = new System.Drawing.Size(42, 20);
			this.txtXBinXZ.TabIndex = 35;
			this.txtXBinXZ.Text = "";
			// 
			// label7
			// 
			this.label7.AutoSize = true;
			this.label7.Location = new System.Drawing.Point(10, 48);
			this.label7.Name = "label7";
			this.label7.Size = new System.Drawing.Size(59, 13);
			this.label7.TabIndex = 34;
			this.label7.Text = "X Bin Size:";
			// 
			// radGLevelXZ
			// 
			this.radGLevelXZ.Location = new System.Drawing.Point(78, 22);
			this.radGLevelXZ.Name = "radGLevelXZ";
			this.radGLevelXZ.Size = new System.Drawing.Size(87, 16);
			this.radGLevelXZ.TabIndex = 3;
			this.radGLevelXZ.Text = "Gray Levels";
			// 
			// radScatterXZ
			// 
			this.radScatterXZ.Checked = true;
			this.radScatterXZ.Location = new System.Drawing.Point(16, 21);
			this.radScatterXZ.Name = "radScatterXZ";
			this.radScatterXZ.Size = new System.Drawing.Size(71, 16);
			this.radScatterXZ.TabIndex = 2;
			this.radScatterXZ.TabStop = true;
			this.radScatterXZ.Text = "Scatter";
			// 
			// groupBox3
			// 
			this.groupBox3.Controls.AddRange(new System.Windows.Forms.Control[] {
																					this.txtYMinYZ,
																					this.label12,
																					this.txtXMinYZ,
																					this.label13,
																					this.txtYMaxYZ,
																					this.label14,
																					this.txtXMaxYZ,
																					this.label15,
																					this.txtYBinYZ,
																					this.label1,
																					this.txtXBinYZ,
																					this.label2,
																					this.radGLevelYZ,
																					this.radScatterYZ});
			this.groupBox3.Location = new System.Drawing.Point(292, 14);
			this.groupBox3.Name = "groupBox3";
			this.groupBox3.Size = new System.Drawing.Size(283, 104);
			this.groupBox3.TabIndex = 33;
			this.groupBox3.TabStop = false;
			this.groupBox3.Text = "Plot Type YZ";
			// 
			// txtYMinYZ
			// 
			this.txtYMinYZ.Location = new System.Drawing.Point(232, 72);
			this.txtYMinYZ.Name = "txtYMinYZ";
			this.txtYMinYZ.Size = new System.Drawing.Size(42, 20);
			this.txtYMinYZ.TabIndex = 53;
			this.txtYMinYZ.Text = "";
			// 
			// label12
			// 
			this.label12.AutoSize = true;
			this.label12.Location = new System.Drawing.Point(200, 72);
			this.label12.Name = "label12";
			this.label12.Size = new System.Drawing.Size(26, 13);
			this.label12.TabIndex = 52;
			this.label12.Text = "Min:";
			// 
			// txtXMinYZ
			// 
			this.txtXMinYZ.Location = new System.Drawing.Point(232, 40);
			this.txtXMinYZ.Name = "txtXMinYZ";
			this.txtXMinYZ.Size = new System.Drawing.Size(42, 20);
			this.txtXMinYZ.TabIndex = 51;
			this.txtXMinYZ.Text = "";
			// 
			// label13
			// 
			this.label13.AutoSize = true;
			this.label13.Location = new System.Drawing.Point(200, 48);
			this.label13.Name = "label13";
			this.label13.Size = new System.Drawing.Size(26, 13);
			this.label13.TabIndex = 50;
			this.label13.Text = "Min:";
			// 
			// txtYMaxYZ
			// 
			this.txtYMaxYZ.Location = new System.Drawing.Point(151, 72);
			this.txtYMaxYZ.Name = "txtYMaxYZ";
			this.txtYMaxYZ.Size = new System.Drawing.Size(42, 20);
			this.txtYMaxYZ.TabIndex = 49;
			this.txtYMaxYZ.Text = "";
			// 
			// label14
			// 
			this.label14.AutoSize = true;
			this.label14.Location = new System.Drawing.Point(119, 79);
			this.label14.Name = "label14";
			this.label14.Size = new System.Drawing.Size(29, 13);
			this.label14.TabIndex = 48;
			this.label14.Text = "Max:";
			// 
			// txtXMaxYZ
			// 
			this.txtXMaxYZ.Location = new System.Drawing.Point(151, 41);
			this.txtXMaxYZ.Name = "txtXMaxYZ";
			this.txtXMaxYZ.Size = new System.Drawing.Size(42, 20);
			this.txtXMaxYZ.TabIndex = 47;
			this.txtXMaxYZ.Text = "";
			// 
			// label15
			// 
			this.label15.AutoSize = true;
			this.label15.Location = new System.Drawing.Point(119, 47);
			this.label15.Name = "label15";
			this.label15.Size = new System.Drawing.Size(29, 13);
			this.label15.TabIndex = 46;
			this.label15.Text = "Max:";
			// 
			// txtYBinYZ
			// 
			this.txtYBinYZ.Location = new System.Drawing.Point(70, 75);
			this.txtYBinYZ.Name = "txtYBinYZ";
			this.txtYBinYZ.Size = new System.Drawing.Size(42, 20);
			this.txtYBinYZ.TabIndex = 37;
			this.txtYBinYZ.Text = "";
			// 
			// label1
			// 
			this.label1.AutoSize = true;
			this.label1.Location = new System.Drawing.Point(11, 77);
			this.label1.Name = "label1";
			this.label1.Size = new System.Drawing.Size(59, 13);
			this.label1.TabIndex = 36;
			this.label1.Text = "Y Bin Size:";
			// 
			// txtXBinYZ
			// 
			this.txtXBinYZ.Location = new System.Drawing.Point(70, 44);
			this.txtXBinYZ.Name = "txtXBinYZ";
			this.txtXBinYZ.Size = new System.Drawing.Size(42, 20);
			this.txtXBinYZ.TabIndex = 35;
			this.txtXBinYZ.Text = "";
			// 
			// label2
			// 
			this.label2.AutoSize = true;
			this.label2.Location = new System.Drawing.Point(11, 45);
			this.label2.Name = "label2";
			this.label2.Size = new System.Drawing.Size(59, 13);
			this.label2.TabIndex = 34;
			this.label2.Text = "X Bin Size:";
			// 
			// radGLevelYZ
			// 
			this.radGLevelYZ.Location = new System.Drawing.Point(79, 22);
			this.radGLevelYZ.Name = "radGLevelYZ";
			this.radGLevelYZ.Size = new System.Drawing.Size(87, 16);
			this.radGLevelYZ.TabIndex = 3;
			this.radGLevelYZ.Text = "Gray Levels";
			// 
			// radScatterYZ
			// 
			this.radScatterYZ.Checked = true;
			this.radScatterYZ.Location = new System.Drawing.Point(16, 21);
			this.radScatterYZ.Name = "radScatterYZ";
			this.radScatterYZ.Size = new System.Drawing.Size(71, 16);
			this.radScatterYZ.TabIndex = 2;
			this.radScatterYZ.TabStop = true;
			this.radScatterYZ.Text = "Scatter";
			// 
			// lblXMultiplier
			// 
			this.lblXMultiplier.AutoSize = true;
			this.lblXMultiplier.Location = new System.Drawing.Point(579, 20);
			this.lblXMultiplier.Name = "lblXMultiplier";
			this.lblXMultiplier.Size = new System.Drawing.Size(143, 13);
			this.lblXMultiplier.TabIndex = 36;
			this.lblXMultiplier.Text = "X Suggested Top Multiplier:";
			// 
			// lblYMultiplier
			// 
			this.lblYMultiplier.AutoSize = true;
			this.lblYMultiplier.Location = new System.Drawing.Point(579, 44);
			this.lblYMultiplier.Name = "lblYMultiplier";
			this.lblYMultiplier.Size = new System.Drawing.Size(143, 13);
			this.lblYMultiplier.TabIndex = 37;
			this.lblYMultiplier.Text = "Y Suggested Top Multiplier:";
			// 
			// radBottom
			// 
			this.radBottom.Location = new System.Drawing.Point(760, 32);
			this.radBottom.Name = "radBottom";
			this.radBottom.Size = new System.Drawing.Size(88, 16);
			this.radBottom.TabIndex = 1;
			this.radBottom.Text = "Bottom Side";
			this.radBottom.CheckedChanged += new System.EventHandler(this.radBottom_CheckedChanged);
			// 
			// radTop
			// 
			this.radTop.Checked = true;
			this.radTop.Location = new System.Drawing.Point(760, 8);
			this.radTop.Name = "radTop";
			this.radTop.Size = new System.Drawing.Size(80, 16);
			this.radTop.TabIndex = 0;
			this.radTop.TabStop = true;
			this.radTop.Text = "Top Side";
			this.radTop.CheckedChanged += new System.EventHandler(this.radTop_CheckedChanged);
			// 
			// groupBox2
			// 
			this.groupBox2.Controls.AddRange(new System.Windows.Forms.Control[] {
																					this.groupBox7,
																					this.groupBox6,
																					this.txtConstant,
																					this.label4,
																					this.txtPLSlopeMultiplier,
																					this.label3,
																					this.cmdPlotLine});
			this.groupBox2.Location = new System.Drawing.Point(755, 48);
			this.groupBox2.Name = "groupBox2";
			this.groupBox2.Size = new System.Drawing.Size(120, 240);
			this.groupBox2.TabIndex = 32;
			this.groupBox2.TabStop = false;
			this.groupBox2.Text = "Correction";
			// 
			// groupBox7
			// 
			this.groupBox7.Controls.AddRange(new System.Windows.Forms.Control[] {
																					this.radBoth,
																					this.radXZ,
																					this.radYZ});
			this.groupBox7.Location = new System.Drawing.Point(16, 88);
			this.groupBox7.Name = "groupBox7";
			this.groupBox7.Size = new System.Drawing.Size(96, 72);
			this.groupBox7.TabIndex = 35;
			this.groupBox7.TabStop = false;
			this.groupBox7.Text = "Plot on";
			// 
			// radBoth
			// 
			this.radBoth.Checked = true;
			this.radBoth.Location = new System.Drawing.Point(8, 16);
			this.radBoth.Name = "radBoth";
			this.radBoth.Size = new System.Drawing.Size(64, 16);
			this.radBoth.TabIndex = 2;
			this.radBoth.TabStop = true;
			this.radBoth.Text = "Both";
			// 
			// radXZ
			// 
			this.radXZ.Location = new System.Drawing.Point(8, 32);
			this.radXZ.Name = "radXZ";
			this.radXZ.Size = new System.Drawing.Size(72, 16);
			this.radXZ.TabIndex = 0;
			this.radXZ.Text = "XZ plane";
			// 
			// radYZ
			// 
			this.radYZ.Location = new System.Drawing.Point(8, 48);
			this.radYZ.Name = "radYZ";
			this.radYZ.Size = new System.Drawing.Size(72, 16);
			this.radYZ.TabIndex = 1;
			this.radYZ.Text = "YZ plane";
			// 
			// groupBox6
			// 
			this.groupBox6.Controls.AddRange(new System.Windows.Forms.Control[] {
																					this.optChi2,
																					this.optShrinkage});
			this.groupBox6.Location = new System.Drawing.Point(16, 168);
			this.groupBox6.Name = "groupBox6";
			this.groupBox6.Size = new System.Drawing.Size(96, 64);
			this.groupBox6.TabIndex = 34;
			this.groupBox6.TabStop = false;
			this.groupBox6.Text = "Apply to";
			// 
			// optChi2
			// 
			this.optChi2.Location = new System.Drawing.Point(8, 32);
			this.optChi2.Name = "optChi2";
			this.optChi2.Size = new System.Drawing.Size(80, 24);
			this.optChi2.TabIndex = 1;
			this.optChi2.Text = "Chi2";
			// 
			// optShrinkage
			// 
			this.optShrinkage.Checked = true;
			this.optShrinkage.Location = new System.Drawing.Point(8, 16);
			this.optShrinkage.Name = "optShrinkage";
			this.optShrinkage.Size = new System.Drawing.Size(80, 16);
			this.optShrinkage.TabIndex = 0;
			this.optShrinkage.TabStop = true;
			this.optShrinkage.Text = "Shrinkage";
			// 
			// txtConstant
			// 
			this.txtConstant.Location = new System.Drawing.Point(49, 34);
			this.txtConstant.Name = "txtConstant";
			this.txtConstant.Size = new System.Drawing.Size(63, 20);
			this.txtConstant.TabIndex = 33;
			this.txtConstant.Text = "0";
			// 
			// label4
			// 
			this.label4.AutoSize = true;
			this.label4.Location = new System.Drawing.Point(8, 37);
			this.label4.Name = "label4";
			this.label4.Size = new System.Drawing.Size(40, 13);
			this.label4.TabIndex = 32;
			this.label4.Text = "Const.:";
			// 
			// txtPLSlopeMultiplier
			// 
			this.txtPLSlopeMultiplier.Location = new System.Drawing.Point(48, 13);
			this.txtPLSlopeMultiplier.Name = "txtPLSlopeMultiplier";
			this.txtPLSlopeMultiplier.Size = new System.Drawing.Size(64, 20);
			this.txtPLSlopeMultiplier.TabIndex = 31;
			this.txtPLSlopeMultiplier.Text = "0";
			// 
			// label3
			// 
			this.label3.AutoSize = true;
			this.label3.Location = new System.Drawing.Point(8, 16);
			this.label3.Name = "label3";
			this.label3.Size = new System.Drawing.Size(39, 13);
			this.label3.TabIndex = 30;
			this.label3.Text = "Linear:";
			// 
			// cmdPlotLine
			// 
			this.cmdPlotLine.Location = new System.Drawing.Point(14, 58);
			this.cmdPlotLine.Name = "cmdPlotLine";
			this.cmdPlotLine.Size = new System.Drawing.Size(98, 24);
			this.cmdPlotLine.TabIndex = 29;
			this.cmdPlotLine.Text = "Plot Line";
			this.cmdPlotLine.Click += new System.EventHandler(this.cmdPlotLine_Click);
			// 
			// PlotPanelY
			// 
			this.PlotPanelY.BackColor = System.Drawing.SystemColors.ActiveCaptionText;
			this.PlotPanelY.Location = new System.Drawing.Point(380, 8);
			this.PlotPanelY.Name = "PlotPanelY";
			this.PlotPanelY.Size = new System.Drawing.Size(368, 320);
			this.PlotPanelY.TabIndex = 35;
			// 
			// PlotPanelX
			// 
			this.PlotPanelX.BackColor = System.Drawing.SystemColors.ActiveCaptionText;
			this.PlotPanelX.Location = new System.Drawing.Point(8, 8);
			this.PlotPanelX.Name = "PlotPanelX";
			this.PlotPanelX.Size = new System.Drawing.Size(368, 320);
			this.PlotPanelX.TabIndex = 33;
			// 
			// groupBox5
			// 
			this.groupBox5.Controls.AddRange(new System.Windows.Forms.Control[] {
																					this.lblTracksBelow,
																					this.label30,
																					this.txtchi2cut,
																					this.cmdCut,
																					this.chkGaussFit,
																					this.cmdPlotChi2,
																					this.txtBotYs0,
																					this.txtBotYdeg,
																					this.label24,
																					this.label25,
																					this.label26,
																					this.txtBotXs0,
																					this.txtBotXdeg,
																					this.label27,
																					this.label28,
																					this.label29,
																					this.txtTopYs0,
																					this.txtTopYdeg,
																					this.label21,
																					this.label22,
																					this.label23,
																					this.txtTopXs0,
																					this.txtTopXdeg,
																					this.label18,
																					this.label19,
																					this.label20,
																					this.txtBinchi,
																					this.txtMaxchi,
																					this.txtMinchi,
																					this.label17,
																					this.label16,
																					this.label5});
			this.groupBox5.Location = new System.Drawing.Point(8, 328);
			this.groupBox5.Name = "groupBox5";
			this.groupBox5.Size = new System.Drawing.Size(624, 88);
			this.groupBox5.TabIndex = 36;
			this.groupBox5.TabStop = false;
			this.groupBox5.Text = "Chi2";
			// 
			// label30
			// 
			this.label30.AutoSize = true;
			this.label30.Location = new System.Drawing.Point(544, 44);
			this.label30.Name = "label30";
			this.label30.Size = new System.Drawing.Size(28, 13);
			this.label30.TabIndex = 71;
			this.label30.Text = "X2 <";
			// 
			// txtchi2cut
			// 
			this.txtchi2cut.Location = new System.Drawing.Point(576, 41);
			this.txtchi2cut.Name = "txtchi2cut";
			this.txtchi2cut.Size = new System.Drawing.Size(40, 20);
			this.txtchi2cut.TabIndex = 70;
			this.txtchi2cut.Text = "2.5";
			// 
			// cmdCut
			// 
			this.cmdCut.Location = new System.Drawing.Point(488, 38);
			this.cmdCut.Name = "cmdCut";
			this.cmdCut.Size = new System.Drawing.Size(40, 24);
			this.cmdCut.TabIndex = 69;
			this.cmdCut.Text = "Cut";
			this.cmdCut.Click += new System.EventHandler(this.cmdCut_Click);
			// 
			// chkGaussFit
			// 
			this.chkGaussFit.Location = new System.Drawing.Point(560, 16);
			this.chkGaussFit.Name = "chkGaussFit";
			this.chkGaussFit.Size = new System.Drawing.Size(56, 16);
			this.chkGaussFit.TabIndex = 68;
			this.chkGaussFit.Text = "Gauss";
			// 
			// cmdPlotChi2
			// 
			this.cmdPlotChi2.Location = new System.Drawing.Point(488, 10);
			this.cmdPlotChi2.Name = "cmdPlotChi2";
			this.cmdPlotChi2.Size = new System.Drawing.Size(56, 24);
			this.cmdPlotChi2.TabIndex = 67;
			this.cmdPlotChi2.Text = "Plot";
			this.cmdPlotChi2.Click += new System.EventHandler(this.cmdPlotChi2_Click);
			// 
			// txtBotYs0
			// 
			this.txtBotYs0.Location = new System.Drawing.Point(424, 26);
			this.txtBotYs0.Name = "txtBotYs0";
			this.txtBotYs0.Size = new System.Drawing.Size(56, 20);
			this.txtBotYs0.TabIndex = 66;
			this.txtBotYs0.Text = "";
			// 
			// txtBotYdeg
			// 
			this.txtBotYdeg.Location = new System.Drawing.Point(424, 48);
			this.txtBotYdeg.Name = "txtBotYdeg";
			this.txtBotYdeg.Size = new System.Drawing.Size(56, 20);
			this.txtBotYdeg.TabIndex = 65;
			this.txtBotYdeg.Text = "";
			// 
			// label24
			// 
			this.label24.AutoSize = true;
			this.label24.Location = new System.Drawing.Point(400, 53);
			this.label24.Name = "label24";
			this.label24.Size = new System.Drawing.Size(26, 13);
			this.label24.TabIndex = 64;
			this.label24.Text = "deg:";
			// 
			// label25
			// 
			this.label25.AutoSize = true;
			this.label25.Location = new System.Drawing.Point(400, 32);
			this.label25.Name = "label25";
			this.label25.Size = new System.Drawing.Size(19, 13);
			this.label25.TabIndex = 63;
			this.label25.Text = "s0:";
			// 
			// label26
			// 
			this.label26.AutoSize = true;
			this.label26.Location = new System.Drawing.Point(400, 11);
			this.label26.Name = "label26";
			this.label26.Size = new System.Drawing.Size(51, 13);
			this.label26.TabIndex = 62;
			this.label26.Text = "Bottom Y";
			// 
			// txtBotXs0
			// 
			this.txtBotXs0.Location = new System.Drawing.Point(336, 27);
			this.txtBotXs0.Name = "txtBotXs0";
			this.txtBotXs0.Size = new System.Drawing.Size(56, 20);
			this.txtBotXs0.TabIndex = 61;
			this.txtBotXs0.Text = "";
			// 
			// txtBotXdeg
			// 
			this.txtBotXdeg.Location = new System.Drawing.Point(336, 49);
			this.txtBotXdeg.Name = "txtBotXdeg";
			this.txtBotXdeg.Size = new System.Drawing.Size(56, 20);
			this.txtBotXdeg.TabIndex = 60;
			this.txtBotXdeg.Text = "";
			// 
			// label27
			// 
			this.label27.AutoSize = true;
			this.label27.Location = new System.Drawing.Point(312, 54);
			this.label27.Name = "label27";
			this.label27.Size = new System.Drawing.Size(26, 13);
			this.label27.TabIndex = 59;
			this.label27.Text = "deg:";
			// 
			// label28
			// 
			this.label28.AutoSize = true;
			this.label28.Location = new System.Drawing.Point(312, 33);
			this.label28.Name = "label28";
			this.label28.Size = new System.Drawing.Size(19, 13);
			this.label28.TabIndex = 58;
			this.label28.Text = "s0:";
			// 
			// label29
			// 
			this.label29.AutoSize = true;
			this.label29.Location = new System.Drawing.Point(312, 12);
			this.label29.Name = "label29";
			this.label29.Size = new System.Drawing.Size(51, 13);
			this.label29.TabIndex = 57;
			this.label29.Text = "Bottom X";
			// 
			// txtTopYs0
			// 
			this.txtTopYs0.Location = new System.Drawing.Point(248, 27);
			this.txtTopYs0.Name = "txtTopYs0";
			this.txtTopYs0.Size = new System.Drawing.Size(56, 20);
			this.txtTopYs0.TabIndex = 56;
			this.txtTopYs0.Text = "";
			// 
			// txtTopYdeg
			// 
			this.txtTopYdeg.Location = new System.Drawing.Point(248, 49);
			this.txtTopYdeg.Name = "txtTopYdeg";
			this.txtTopYdeg.Size = new System.Drawing.Size(56, 20);
			this.txtTopYdeg.TabIndex = 55;
			this.txtTopYdeg.Text = "";
			// 
			// label21
			// 
			this.label21.AutoSize = true;
			this.label21.Location = new System.Drawing.Point(224, 54);
			this.label21.Name = "label21";
			this.label21.Size = new System.Drawing.Size(26, 13);
			this.label21.TabIndex = 54;
			this.label21.Text = "deg:";
			// 
			// label22
			// 
			this.label22.AutoSize = true;
			this.label22.Location = new System.Drawing.Point(224, 33);
			this.label22.Name = "label22";
			this.label22.Size = new System.Drawing.Size(19, 13);
			this.label22.TabIndex = 53;
			this.label22.Text = "s0:";
			// 
			// label23
			// 
			this.label23.AutoSize = true;
			this.label23.Location = new System.Drawing.Point(224, 12);
			this.label23.Name = "label23";
			this.label23.Size = new System.Drawing.Size(34, 13);
			this.label23.TabIndex = 52;
			this.label23.Text = "Top Y";
			// 
			// txtTopXs0
			// 
			this.txtTopXs0.Location = new System.Drawing.Point(159, 28);
			this.txtTopXs0.Name = "txtTopXs0";
			this.txtTopXs0.Size = new System.Drawing.Size(56, 20);
			this.txtTopXs0.TabIndex = 51;
			this.txtTopXs0.Text = "";
			// 
			// txtTopXdeg
			// 
			this.txtTopXdeg.Location = new System.Drawing.Point(159, 50);
			this.txtTopXdeg.Name = "txtTopXdeg";
			this.txtTopXdeg.Size = new System.Drawing.Size(56, 20);
			this.txtTopXdeg.TabIndex = 50;
			this.txtTopXdeg.Text = "";
			// 
			// label18
			// 
			this.label18.AutoSize = true;
			this.label18.Location = new System.Drawing.Point(131, 55);
			this.label18.Name = "label18";
			this.label18.Size = new System.Drawing.Size(26, 13);
			this.label18.TabIndex = 49;
			this.label18.Text = "deg:";
			// 
			// label19
			// 
			this.label19.AutoSize = true;
			this.label19.Location = new System.Drawing.Point(131, 34);
			this.label19.Name = "label19";
			this.label19.Size = new System.Drawing.Size(19, 13);
			this.label19.TabIndex = 48;
			this.label19.Text = "s0:";
			// 
			// label20
			// 
			this.label20.AutoSize = true;
			this.label20.Location = new System.Drawing.Point(131, 13);
			this.label20.Name = "label20";
			this.label20.Size = new System.Drawing.Size(34, 13);
			this.label20.TabIndex = 47;
			this.label20.Text = "Top X";
			// 
			// txtBinchi
			// 
			this.txtBinchi.Location = new System.Drawing.Point(67, 10);
			this.txtBinchi.Name = "txtBinchi";
			this.txtBinchi.Size = new System.Drawing.Size(56, 20);
			this.txtBinchi.TabIndex = 46;
			this.txtBinchi.Text = "";
			// 
			// txtMaxchi
			// 
			this.txtMaxchi.Location = new System.Drawing.Point(67, 32);
			this.txtMaxchi.Name = "txtMaxchi";
			this.txtMaxchi.Size = new System.Drawing.Size(56, 20);
			this.txtMaxchi.TabIndex = 45;
			this.txtMaxchi.Text = "";
			// 
			// txtMinchi
			// 
			this.txtMinchi.Location = new System.Drawing.Point(67, 54);
			this.txtMinchi.Name = "txtMinchi";
			this.txtMinchi.Size = new System.Drawing.Size(56, 20);
			this.txtMinchi.TabIndex = 44;
			this.txtMinchi.Text = "";
			// 
			// label17
			// 
			this.label17.AutoSize = true;
			this.label17.Location = new System.Drawing.Point(40, 59);
			this.label17.Name = "label17";
			this.label17.Size = new System.Drawing.Size(26, 13);
			this.label17.TabIndex = 43;
			this.label17.Text = "Min:";
			// 
			// label16
			// 
			this.label16.AutoSize = true;
			this.label16.Location = new System.Drawing.Point(38, 38);
			this.label16.Name = "label16";
			this.label16.Size = new System.Drawing.Size(29, 13);
			this.label16.TabIndex = 39;
			this.label16.Text = "Max:";
			// 
			// label5
			// 
			this.label5.AutoSize = true;
			this.label5.Location = new System.Drawing.Point(11, 18);
			this.label5.Name = "label5";
			this.label5.Size = new System.Drawing.Size(59, 13);
			this.label5.TabIndex = 35;
			this.label5.Text = "X Bin Size:";
			// 
			// groupBox8
			// 
			this.groupBox8.Controls.AddRange(new System.Windows.Forms.Control[] {
																					this.optds_vs_s,
																					this.optds_vs_angle});
			this.groupBox8.Location = new System.Drawing.Point(640, 328);
			this.groupBox8.Name = "groupBox8";
			this.groupBox8.Size = new System.Drawing.Size(136, 88);
			this.groupBox8.TabIndex = 37;
			this.groupBox8.TabStop = false;
			this.groupBox8.Text = "Variables";
			// 
			// optds_vs_s
			// 
			this.optds_vs_s.Checked = true;
			this.optds_vs_s.Location = new System.Drawing.Point(8, 16);
			this.optds_vs_s.Name = "optds_vs_s";
			this.optds_vs_s.Size = new System.Drawing.Size(120, 32);
			this.optds_vs_s.TabIndex = 0;
			this.optds_vs_s.TabStop = true;
			this.optds_vs_s.Text = "DS(side) vs S(side)";
			this.optds_vs_s.CheckedChanged += new System.EventHandler(this.optds_vs_s_CheckedChanged);
			// 
			// optds_vs_angle
			// 
			this.optds_vs_angle.Location = new System.Drawing.Point(8, 48);
			this.optds_vs_angle.Name = "optds_vs_angle";
			this.optds_vs_angle.Size = new System.Drawing.Size(120, 32);
			this.optds_vs_angle.TabIndex = 1;
			this.optds_vs_angle.Text = "DS(side) vs Angle";
			this.optds_vs_angle.CheckedChanged += new System.EventHandler(this.optds_vs_angle_CheckedChanged);
			// 
			// lblTracksBelow
			// 
			this.lblTracksBelow.AutoSize = true;
			this.lblTracksBelow.Location = new System.Drawing.Point(489, 67);
			this.lblTracksBelow.Name = "lblTracksBelow";
			this.lblTracksBelow.Size = new System.Drawing.Size(75, 13);
			this.lblTracksBelow.TabIndex = 72;
			this.lblTracksBelow.Text = "Track below: -";
			// 
			// EasyShrink
			// 
			this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
			this.ClientSize = new System.Drawing.Size(880, 550);
			this.Controls.AddRange(new System.Windows.Forms.Control[] {
																		  this.groupBox8,
																		  this.groupBox5,
																		  this.PlotPanelX,
																		  this.PlotPanelY,
																		  this.groupBox2,
																		  this.radTop,
																		  this.radBottom,
																		  this.groupBox1});
			this.Name = "EasyShrink";
			this.Text = "EasyShrink";
			this.groupBox1.ResumeLayout(false);
			this.groupBox4.ResumeLayout(false);
			this.groupBox3.ResumeLayout(false);
			this.groupBox2.ResumeLayout(false);
			this.groupBox7.ResumeLayout(false);
			this.groupBox6.ResumeLayout(false);
			this.groupBox5.ResumeLayout(false);
			this.groupBox8.ResumeLayout(false);
			this.ResumeLayout(false);

		}
		#endregion

		private void cmdPlotLine_Click(object sender, System.EventArgs e)
		{
			int i, n, j, m;
			Bitmap b;
			Graphics gPB;
			Plot gA = new Plot();
			bool top = radTop.Checked;
			bool bot = radBottom.Checked;
			bool xscatter = radScatterXZ.Checked;
			bool yscatter = radScatterYZ.Checked;
			bool shri = optShrinkage.Checked;
			
			try
			{
				gA.SetXDefaultLimits = false;
				gA.SetYDefaultLimits = false;
				if(this.optds_vs_s.Checked)
				{
					gA.VecX = x;
					gA.XTitle = "Side Slope X";
				}
				else
				{
					gA.VecX = angle;
					gA.XTitle = "Angle";
				}

				gA.DX = Convert.ToSingle(txtXBinXZ.Text);
				gA.MaxX = Convert.ToSingle(txtXMaxXZ.Text);
				gA.MinX = Convert.ToSingle(txtXMinXZ.Text);
				gA.VecY = dx;
				gA.YTitle = "Linked Slope X - Side Slope X";
				gA.DY = Convert.ToSingle(txtYBinXZ.Text);
				gA.MaxY = Convert.ToSingle(txtYMaxXZ.Text);
				gA.MinY = Convert.ToSingle(txtYMinXZ.Text);
				b = new Bitmap((int)(PlotPanelX.Width),(int)(PlotPanelX.Height));
				gPB = Graphics.FromImage(b);
				//gA.ScatterFit = 1;
				if (radBoth.Checked ||  radXZ.Checked)
				{
					gA.FunctionOverlayed = true;
					gA.Function = this.txtPLSlopeMultiplier.Text + "*x +(" + this.txtConstant.Text+")"; 
				}
				if (xscatter) gA.Scatter(gPB, PlotPanelX.Width, PlotPanelX.Height);
				else gA.GreyLevelArea(gPB, PlotPanelX.Width, PlotPanelX.Height);
				PlotPanelX.BackgroundImage = b;

				gA.FunctionOverlayed = false;
				if(this.optds_vs_s.Checked)
				{
					gA.VecX = y;
					gA.XTitle = "Side Slope Y";
				}
				else
				{
					gA.VecX = angle;
					gA.XTitle = "Angle";
				}
				gA.DX = Convert.ToSingle(txtXBinYZ.Text);
				gA.MaxX = Convert.ToSingle(txtXMaxYZ.Text);
				gA.MinX = Convert.ToSingle(txtXMinYZ.Text);
				gA.VecY = dy;
				gA.YTitle = "Linked Slope Y - Side Slope Y";
				gA.DY = Convert.ToSingle(txtYBinYZ.Text);
				gA.MaxY = Convert.ToSingle(txtYMaxYZ.Text);
				gA.MinY = Convert.ToSingle(txtYMinYZ.Text);
				b = new Bitmap((int)(PlotPanelY.Width),(int)(PlotPanelY.Height));
				gPB = Graphics.FromImage(b);
				//gA.ScatterFit = 1;
				if (radBoth.Checked ||  radYZ.Checked)
				{
					gA.FunctionOverlayed = true;
					gA.Function = this.txtPLSlopeMultiplier.Text + "*x +(" + this.txtConstant.Text + ")"; 
				}
				if (yscatter) gA.Scatter(gPB, PlotPanelY.Width, PlotPanelY.Height);
				else gA.GreyLevelArea(gPB, PlotPanelY.Width, PlotPanelY.Height);
				PlotPanelY.BackgroundImage = b;

				if(shri)
				{
					//Corregge lo shrinkage
					if (top)
					{
						if(radBoth.Checked ||  radXZ.Checked)
						{
							this.lblXMultiplier.Text =  "X Suggested Top Multiplier: ";
							this.txtXMult.Text = Fitting.ExtendedRound((ExternalTopMultiplier*(1+Convert.ToDouble(this.txtPLSlopeMultiplier.Text))), 5, RoundOption.MathematicalRound).ToString(); 
							this.lblXDelta.Text =  "X Suggested Top Delta: ";
							this.txtXDelta.Text = Fitting.ExtendedRound(ExternalTopXDelta+Convert.ToDouble(this.txtConstant.Text), 5, RoundOption.MathematicalRound).ToString(); 
						}
						if(radBoth.Checked ||  radYZ.Checked)
						{
							this.lblYMultiplier.Text =  "Y Suggested Top Multiplier: ";
							this.txtYMult.Text = Fitting.ExtendedRound((ExternalTopMultiplier*(1+Convert.ToDouble(this.txtPLSlopeMultiplier.Text))), 5, RoundOption.MathematicalRound).ToString(); 
							this.lblYDelta.Text =  "Y Suggested Top Delta: ";
							this.txtYDelta.Text = Fitting.ExtendedRound(ExternalTopYDelta+Convert.ToDouble(this.txtConstant.Text), 5, RoundOption.MathematicalRound).ToString(); 
						}
					}
					else
					{
						if(radBoth.Checked ||  radXZ.Checked)
						{
							this.lblXMultiplier.Text =  "X Suggested Bottom Multiplier: "; 
							this.txtXMult.Text = Fitting.ExtendedRound((ExternalBotMultiplier*(1+Convert.ToDouble(this.txtPLSlopeMultiplier.Text))), 5, RoundOption.MathematicalRound).ToString(); 
							this.lblXDelta.Text =  "X Suggested Bottom Delta: "; 
							this.txtXDelta.Text = Fitting.ExtendedRound(ExternalBotXDelta+Convert.ToDouble(this.txtConstant.Text), 5, RoundOption.MathematicalRound).ToString(); 
						}
						if(radBoth.Checked ||  radYZ.Checked)
						{
							this.lblYMultiplier.Text =  "Y Suggested Bottom Multiplier: ";
							this.txtYMult.Text = Fitting.ExtendedRound((ExternalBotMultiplier*(1+Convert.ToDouble(this.txtPLSlopeMultiplier.Text))), 5, RoundOption.MathematicalRound).ToString(); 
							this.lblYDelta.Text =  "Y Suggested Bottom Delta: ";
							this.txtYDelta.Text = Fitting.ExtendedRound(ExternalBotYDelta+Convert.ToDouble(this.txtConstant.Text), 5, RoundOption.MathematicalRound).ToString(); 
						}
					}
				}
				else
				{
					//Parametrizza il chi2
					if (top)
					{
						if(radBoth.Checked ||  radXZ.Checked)
						{
							this.txtTopXs0.Text = Math.Abs(Fitting.ExtendedRound(Convert.ToDouble(this.txtConstant.Text)/3, 5, RoundOption.MathematicalRound)).ToString(); 
							this.txtTopXdeg.Text = Math.Abs(Fitting.ExtendedRound(3*Convert.ToDouble(this.txtPLSlopeMultiplier.Text)/Convert.ToDouble(this.txtConstant.Text), 5, RoundOption.MathematicalRound)).ToString(); 
						}
						if(radBoth.Checked ||  radYZ.Checked)
						{
							this.txtTopYs0.Text = Math.Abs(Fitting.ExtendedRound(Convert.ToDouble(this.txtConstant.Text)/3, 5, RoundOption.MathematicalRound)).ToString(); 
							this.txtTopYdeg.Text = Math.Abs(Fitting.ExtendedRound(3*Convert.ToDouble(this.txtPLSlopeMultiplier.Text)/Convert.ToDouble(this.txtConstant.Text), 5, RoundOption.MathematicalRound)).ToString(); 
						}
					}
					else
					{
						if(radBoth.Checked ||  radXZ.Checked)
						{
							this.txtBotXs0.Text = Math.Abs(Fitting.ExtendedRound(Convert.ToDouble(this.txtConstant.Text)/3, 5, RoundOption.MathematicalRound)).ToString(); 
							this.txtBotXdeg.Text = Math.Abs(Fitting.ExtendedRound(3*Convert.ToDouble(this.txtPLSlopeMultiplier.Text)/Convert.ToDouble(this.txtConstant.Text), 5, RoundOption.MathematicalRound)).ToString(); 
						}
						if(radBoth.Checked ||  radYZ.Checked)
						{
							this.txtBotYs0.Text = Math.Abs(Fitting.ExtendedRound(Convert.ToDouble(this.txtConstant.Text)/3, 5, RoundOption.MathematicalRound)).ToString(); 
							this.txtBotYdeg.Text = Math.Abs(Fitting.ExtendedRound(3*Convert.ToDouble(this.txtPLSlopeMultiplier.Text)/Convert.ToDouble(this.txtConstant.Text), 5, RoundOption.MathematicalRound)).ToString(); 
						}
					}
				
				}
			}
			catch(Exception exc)
			{
				System.Windows.Forms.MessageBox.Show(exc.ToString(), "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
			
			}
		}

		private void cmdFitData_Click(object sender, System.EventArgs e)
		{
			int i, n, j, m;
			Bitmap b;
			Graphics gPB;
			Plot gA = new Plot();
			bool top = radTop.Checked;
			bool xscatter = radScatterXZ.Checked;
			bool yscatter = radScatterYZ.Checked;
			

			gA.SetXDefaultLimits = false;
			gA.SetYDefaultLimits = false;
			if(this.optds_vs_s.Checked)
			{
				gA.VecX = x;
				gA.XTitle = "Side Slope X";
			}
			else
			{
				gA.VecX = angle;
				gA.XTitle = "Angle";
			}
			gA.DX = Convert.ToSingle(txtXBinXZ.Text);
			gA.MaxX = Convert.ToSingle(txtXMaxXZ.Text);
			gA.MinX = Convert.ToSingle(txtXMinXZ.Text);
			gA.VecY = dx;
			gA.YTitle = "Linked Slope X - Side Slope X";
			gA.DY = Convert.ToSingle(txtYBinXZ.Text);
			gA.MaxY = Convert.ToSingle(txtYMaxXZ.Text);
			gA.MinY = Convert.ToSingle(txtYMinXZ.Text);
			b = new Bitmap((int)(PlotPanelX.Width),(int)(PlotPanelX.Height));
			gPB = Graphics.FromImage(b);
			gA.ScatterFit = 1;
			if (xscatter) gA.Scatter(gPB, PlotPanelX.Width, PlotPanelX.Height);
			else gA.GreyLevelArea(gPB, PlotPanelX.Width, PlotPanelX.Height);
			PlotPanelX.BackgroundImage = b;

			//Corregge lo shrinkage
			if (top && xscatter)
			{
				this.lblXMultiplier.Text =  "X Suggested Top Multiplier: ";
				this.txtXMult.Text = Fitting.ExtendedRound((ExternalTopMultiplier*(1+gA.FitPar[1])), 5, RoundOption.MathematicalRound).ToString(); 
			}
			else if(!top && xscatter)
			{
				this.lblXMultiplier.Text =  "X Suggested Bottom Multiplier: "; 
				this.txtXMult.Text = Fitting.ExtendedRound((ExternalBotMultiplier*(1+gA.FitPar[1])), 5, RoundOption.MathematicalRound).ToString(); 
			}

			if(this.optds_vs_s.Checked)
			{
				gA.VecX = y;
				gA.XTitle = "Side Slope Y";
			}
			else
			{
				gA.VecX = angle;
				gA.XTitle = "Angle";
			}
			gA.DX = Convert.ToSingle(txtXBinYZ.Text);
			gA.MaxX = Convert.ToSingle(txtXMaxYZ.Text);
			gA.MinX = Convert.ToSingle(txtXMinYZ.Text);
			gA.VecY = dy;
			gA.YTitle = "Linked Slope Y - Side Slope Y";
			gA.DY = Convert.ToSingle(txtYBinYZ.Text);
			gA.MaxY = Convert.ToSingle(txtYMaxYZ.Text);
			gA.MinY = Convert.ToSingle(txtYMinYZ.Text);
			b = new Bitmap((int)(PlotPanelY.Width),(int)(PlotPanelY.Height));
			gPB = Graphics.FromImage(b);
			gA.ScatterFit = 1;
			if (yscatter) gA.Scatter(gPB, PlotPanelY.Width, PlotPanelY.Height);
			else gA.GreyLevelArea(gPB, PlotPanelY.Width, PlotPanelY.Height);
			PlotPanelY.BackgroundImage = b;

			if (top && yscatter)
			{
				this.lblYMultiplier.Text =  "Y Suggested Top Multiplier: "; 
				this.txtYMult.Text = Fitting.ExtendedRound((ExternalTopMultiplier*(1+gA.FitPar[1])), 5, RoundOption.MathematicalRound).ToString(); 
			}
			else if (!top && yscatter)
			{
				this.lblYMultiplier.Text =  "Y Suggested Bottom Multiplier: "; 
				this.txtYMult.Text = Fitting.ExtendedRound((ExternalBotMultiplier*(1+gA.FitPar[1])), 5, RoundOption.MathematicalRound).ToString(); 
			}

		}

		private void radBottom_CheckedChanged(object sender, System.EventArgs e)
		{
			SetData();
		
		}

		private void radTop_CheckedChanged(object sender, System.EventArgs e)
		{
			SetData();

		}

		private void SetData()
		{
			int i;
			bool top = radTop.Checked;
			double maxx, minx, binx;
			double maxy, miny, biny;
			double maxdx, mindx, bindx;
			double maxdy, mindy, bindy;

			//bool xscatter = radScatterXZ.Checked;
			//bool yscatter = radScatterYZ.Checked;
			for(i = 0; i < NData; i++)
			{
				if(top) x[i] = Ret.Top[i].Info.Slope.X;
				else x[i] = Ret.Bottom[i].Info.Slope.X;
				dx[i] = Ret[i].Info.Slope.X - x[i];
				if(top) y[i] = Ret.Top[i].Info.Slope.Y;
				else y[i] = Ret.Bottom[i].Info.Slope.Y;
				dy[i] = Ret[i].Info.Slope.Y - y[i];
			}

			if(this.optds_vs_s.Checked)
			{
				maxx = Fitting.Maximum(x);
				minx = Fitting.Minimum(x);
				binx = 0.01*(maxx-minx);
				maxy = Fitting.Maximum(y);
				miny = Fitting.Minimum(y);
				biny = 0.01*(maxy-miny);
			}
			else
			{
				maxy = maxx = Fitting.Maximum(angle);
				miny = minx = Fitting.Minimum(angle);
				biny = binx = 0.01*(maxx-minx);
			}
			maxdx = Fitting.Maximum(dx);
			mindx = Fitting.Minimum(dx);
			bindx = 0.01*(maxdx-mindx);
			maxdy = Fitting.Maximum(dy);
			mindy = Fitting.Minimum(dy);
			bindy = 0.01*(maxdy-mindy);
			txtXBinXZ.Text = Fitting.ExtendedRound(binx,3,RoundOption.MathematicalRound).ToString();
			txtXBinYZ.Text = Fitting.ExtendedRound(biny,3,RoundOption.MathematicalRound).ToString();
			txtYBinXZ.Text = Fitting.ExtendedRound(bindx,3,RoundOption.MathematicalRound).ToString();
			txtYBinYZ.Text = Fitting.ExtendedRound(bindy,3,RoundOption.MathematicalRound).ToString();
			txtXMaxXZ.Text = Fitting.ExtendedRound(maxx,3,RoundOption.CeilingRound).ToString();
			txtXMinXZ.Text = Fitting.ExtendedRound(minx,3,RoundOption.FloorRound).ToString();
			txtYMaxXZ.Text = Fitting.ExtendedRound(maxdx,3,RoundOption.CeilingRound).ToString();
			txtYMinXZ.Text = Fitting.ExtendedRound(mindx,3,RoundOption.FloorRound).ToString();
			txtXMaxYZ.Text = Fitting.ExtendedRound(maxy,3,RoundOption.CeilingRound).ToString();
			txtXMinYZ.Text = Fitting.ExtendedRound(miny,3,RoundOption.FloorRound).ToString();
			txtYMaxYZ.Text = Fitting.ExtendedRound(maxdy,3,RoundOption.CeilingRound).ToString();
			txtYMinYZ.Text = Fitting.ExtendedRound(mindy,3,RoundOption.FloorRound).ToString();
			
		}
		
		private void SetChi2Data()
		{

			txtTopXs0.Text = "0.015";
			txtTopYs0.Text = "0.015";
			txtBotXs0.Text = "0.015";
			txtBotYs0.Text = "0.015";
			txtTopXdeg.Text = "0";
			txtTopYdeg.Text = "0";
			txtBotXdeg.Text = "0";
			txtBotYdeg.Text = "0";
			
			ComputeChi2();
			double maxx = Fitting.Maximum(chi2);
			double minx = Fitting.Minimum(chi2);
			double binx = 0.1*(maxx-minx);
			txtBinchi.Text = Fitting.ExtendedRound(binx,3,RoundOption.MathematicalRound).ToString();
			txtMaxchi.Text = Fitting.ExtendedRound(maxx,3,RoundOption.CeilingRound).ToString();
			txtMinchi.Text = Fitting.ExtendedRound(minx,3,RoundOption.FloorRound).ToString();
		
		}

		private void ComputeChi2()
		{
			int i;
			double stx0 = Convert.ToDouble(txtTopXs0.Text);
			double sty0 = Convert.ToDouble(txtTopYs0.Text);
			double sbx0 = Convert.ToDouble(txtBotXs0.Text);
			double sby0 = Convert.ToDouble(txtBotYs0.Text);
			double degtx = Convert.ToDouble(txtTopXdeg.Text);
			double degty = Convert.ToDouble(txtTopYdeg.Text);
			double degbx = Convert.ToDouble(txtBotXdeg.Text);
			double degby = Convert.ToDouble(txtBotYdeg.Text);

			for(i = 0; i < NData; i++)
			{
				double sigtx = stx0*(1+degtx*angle[i]);
				double sigty = sty0*(1+degty*angle[i]);
				double sigbx = sbx0*(1+degbx*angle[i]);
				double sigby = sby0*(1+degby*angle[i]);
				chi2[i] = 0.5*Math.Sqrt(
					((Ret.Top[i].Info.Slope.X-Ret[i].Info.Slope.X)/sigtx)*((Ret.Top[i].Info.Slope.X-Ret[i].Info.Slope.X)/sigtx) +
					((Ret.Top[i].Info.Slope.Y-Ret[i].Info.Slope.Y)/sigty)*((Ret.Top[i].Info.Slope.Y-Ret[i].Info.Slope.Y)/sigty) +
					((Ret.Bottom[i].Info.Slope.X-Ret[i].Info.Slope.X)/sigbx)*((Ret.Bottom[i].Info.Slope.X-Ret[i].Info.Slope.X)/sigbx) +
					((Ret.Bottom[i].Info.Slope.Y-Ret[i].Info.Slope.Y)/sigby)*((Ret.Bottom[i].Info.Slope.Y-Ret[i].Info.Slope.Y)/sigby)
					);
			}

		}

		private void cmdPlotChi2_Click(object sender, System.EventArgs e)
		{
			Bitmap b;
			Graphics gPB;
			Plot gA = new Plot();

			ComputeChi2();
			if (chkGaussFit.Checked) gA.HistoFit = 0;
			else gA.HistoFit = -2;
			gA.SetXDefaultLimits = false;
			gA.SetYDefaultLimits = false;
			gA.VecX = chi2;
			gA.XTitle = "Chi2";
			gA.DX = Convert.ToSingle(txtBinchi.Text);
			gA.MaxX = Convert.ToSingle(txtMaxchi.Text);
			gA.MinX = Convert.ToSingle(txtMinchi.Text);
			b = new Bitmap((int)(PlotPanelX.Width),(int)(PlotPanelX.Height));
			gPB = Graphics.FromImage(b);
			gA.Histo(gPB, PlotPanelX.Width, PlotPanelX.Height);
			PlotPanelX.BackgroundImage = b;

		}

		private void optds_vs_s_CheckedChanged(object sender, System.EventArgs e)
		{
			SetData();
		}

		private void optds_vs_angle_CheckedChanged(object sender, System.EventArgs e)
		{
			SetData();
		}

		private void cmdCut_Click(object sender, System.EventArgs e)
		{
			int i, k=0;
			double stx0 = Convert.ToDouble(txtTopXs0.Text);
			double sty0 = Convert.ToDouble(txtTopYs0.Text);
			double sbx0 = Convert.ToDouble(txtBotXs0.Text);
			double sby0 = Convert.ToDouble(txtBotYs0.Text);
			double degtx = Convert.ToDouble(txtTopXdeg.Text);
			double degty = Convert.ToDouble(txtTopYdeg.Text);
			double degbx = Convert.ToDouble(txtBotXdeg.Text);
			double degby = Convert.ToDouble(txtBotYdeg.Text);
			double thre = Convert.ToDouble(txtchi2cut.Text);

			for(i = 0; i < NData; i++)
			{
				double sigtx = stx0*(1+degtx*angle[i]);
				double sigty = sty0*(1+degty*angle[i]);
				double sigbx = sbx0*(1+degbx*angle[i]);
				double sigby = sby0*(1+degby*angle[i]);
				chi2[i] = 0.5*Math.Sqrt(
					((Ret.Top[i].Info.Slope.X-Ret[i].Info.Slope.X)/sigtx)*((Ret.Top[i].Info.Slope.X-Ret[i].Info.Slope.X)/sigtx) +
					((Ret.Top[i].Info.Slope.Y-Ret[i].Info.Slope.Y)/sigty)*((Ret.Top[i].Info.Slope.Y-Ret[i].Info.Slope.Y)/sigty) +
					((Ret.Bottom[i].Info.Slope.X-Ret[i].Info.Slope.X)/sigbx)*((Ret.Bottom[i].Info.Slope.X-Ret[i].Info.Slope.X)/sigbx) +
					((Ret.Bottom[i].Info.Slope.Y-Ret[i].Info.Slope.Y)/sigby)*((Ret.Bottom[i].Info.Slope.Y-Ret[i].Info.Slope.Y)/sigby)
					);
				if (chi2[i]<thre) k++;
			}
			lblTracksBelow.Text = "Tracks below: " + k;
		}

	}
}
