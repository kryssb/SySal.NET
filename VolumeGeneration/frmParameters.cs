using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;

namespace SySal.Processing.VolumeGeneration
{
	/// <summary>
	/// Form to edit configuration parameters.
	/// </summary>
	internal class frmParameters : System.Windows.Forms.Form
	{
		private System.Windows.Forms.GroupBox groupBox1;
		private System.Windows.Forms.Label label1;
		private System.Windows.Forms.TextBox txtOBVMaxX;
		private System.Windows.Forms.Label label2;
		private System.Windows.Forms.Label label3;
		private System.Windows.Forms.TextBox txtOBVMinX;
		private System.Windows.Forms.TextBox txtOBVMinY;
		private System.Windows.Forms.Label label4;
		private System.Windows.Forms.Label label5;
		private System.Windows.Forms.TextBox txtOBVMaxY;
		private System.Windows.Forms.TextBox txtOBVMaxZ;
		private System.Windows.Forms.TextBox txtOBVMinZ;
		private System.Windows.Forms.Label label6;
		private System.Windows.Forms.Label label7;
		private System.Windows.Forms.TextBox txtVMaxZ;
		private System.Windows.Forms.TextBox txtVMinZ;
		private System.Windows.Forms.Label label8;
		private System.Windows.Forms.Label label9;
		private System.Windows.Forms.TextBox txtVMaxY;
		private System.Windows.Forms.TextBox txtVMaxX;
		private System.Windows.Forms.TextBox txtVMinY;
		private System.Windows.Forms.Label label10;
		private System.Windows.Forms.Label label11;
		private System.Windows.Forms.TextBox txtVMinX;
		private System.Windows.Forms.Label label12;
		private System.Windows.Forms.Label label13;
		private System.Windows.Forms.Label label14;
		private System.Windows.Forms.Label label15;
		private System.Windows.Forms.Label label16;
		private System.Windows.Forms.Label label17;
		private System.Windows.Forms.TextBox txtTracking;
		private System.Windows.Forms.TextBox txtNonTracking;
		private System.Windows.Forms.TextBox txtMostUpstreamPlate;
		private System.Windows.Forms.Label label18;
		private System.Windows.Forms.GroupBox groupBox2;
		private System.Windows.Forms.Label label20;
		private System.Windows.Forms.Label label21;
		private System.Windows.Forms.Label label22;
		private System.Windows.Forms.TextBox txtCoordErrX;
		private System.Windows.Forms.TextBox txtCoordErrY;
		private System.Windows.Forms.Label label24;
		private System.Windows.Forms.TextBox txtSlopeErrX;
		private System.Windows.Forms.TextBox txtSlopeErrY;
		private System.Windows.Forms.Label label25;
		private System.Windows.Forms.Label label26;
		private System.Windows.Forms.Label label27;
		private System.Windows.Forms.GroupBox groupBox3;
		private System.Windows.Forms.Label label28;
		private System.Windows.Forms.Label label29;
		private System.Windows.Forms.TextBox txtMinimumEnergyLoss;
		private System.Windows.Forms.TextBox txtRadiationLength;
		private System.Windows.Forms.GroupBox groupBox4;
		private System.Windows.Forms.Label label30;
		private System.Windows.Forms.Label label31;
		private System.Windows.Forms.Label label32;
		private System.Windows.Forms.Label label34;
		private System.Windows.Forms.ColumnHeader columnName;
		private System.Windows.Forms.ColumnHeader columnValue;
		private System.Windows.Forms.TextBox txtConfigName;
		private System.Windows.Forms.Label label35;
		private System.Windows.Forms.Button cmdCancel;
		private System.Windows.Forms.Button cmdOk;
		private System.Windows.Forms.Button cmdDefault;
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;
		private System.Windows.Forms.Label label36;
		private System.Windows.Forms.Label lblHMdensity;
		private System.Windows.Forms.Label lblELdensity;
		private System.Windows.Forms.Label lblJunkdensity;
		private System.Windows.Forms.TextBox txtHMTracks;
		private System.Windows.Forms.TextBox txtELTracks;
		private System.Windows.Forms.TextBox txtJunkTracks;
		private System.Windows.Forms.GroupBox groupBox5;
		private System.Windows.Forms.Label label33;
		private System.Windows.Forms.Label label38;
		private System.Windows.Forms.TextBox txtTFE;
		private System.Windows.Forms.Label label40;
		private System.Windows.Forms.TextBox flytxtSlopeX;
		private System.Windows.Forms.ListView lstSlopeX;
		private System.Windows.Forms.ComboBox cmbSlopeXDistrib;
		private System.Windows.Forms.TextBox flytxtSlopeY;
		private System.Windows.Forms.ListView lstSlopeY;
		private System.Windows.Forms.ColumnHeader columnHeader1;
		private System.Windows.Forms.ColumnHeader columnHeader2;
		private System.Windows.Forms.ComboBox cmbSlopeYDistrib;
		private System.Windows.Forms.Label label39;
		private System.Windows.Forms.TextBox flytxtMomentum;
		private System.Windows.Forms.ListView lstMomentum;
		private System.Windows.Forms.ColumnHeader columnHeader3;
		private System.Windows.Forms.ColumnHeader columnHeader4;
		private System.Windows.Forms.ComboBox cmbMomentumDistrib;
		private System.Windows.Forms.Label label41;
		private System.Windows.Forms.TextBox txtSCoeffMin;
		private System.Windows.Forms.TextBox txtSCoeffMax;
		private System.Windows.Forms.TextBox txtCoordAlignMax;
		private System.Windows.Forms.Label label23;
		private System.Windows.Forms.TextBox txtCoordAlignMin;
		private System.Windows.Forms.Label label44;
		private System.Windows.Forms.TextBox txtDiagMin;
		private System.Windows.Forms.Label label42;
		private System.Windows.Forms.TextBox txtDiagMax;
		private System.Windows.Forms.Label label45;
		private System.Windows.Forms.TextBox txtOutDMin;
		private System.Windows.Forms.Label label46;
		private System.Windows.Forms.TextBox txtOutDMax;
		private System.Windows.Forms.Label label47;
		private System.Windows.Forms.Label label43;
		private System.Windows.Forms.Label label48;
		private System.Windows.Forms.TextBox txtSShiftMax;
		private System.Windows.Forms.TextBox txtSShiftMin;
		private System.Windows.Forms.TextBox txtLongAlignMin;
		private System.Windows.Forms.Label label37;
		private System.Windows.Forms.TextBox txtLongAlignMax;
		private System.Windows.Forms.Label label49;
		private System.Windows.Forms.Label label50;
		private System.Windows.Forms.Button cmdHelp;
		private System.Windows.Forms.GroupBox groupBox6;
		private System.Windows.Forms.TextBox txtLocalVertexDepth;
		private System.Windows.Forms.Label label19;
		private System.Windows.Forms.Label label51;
		private System.Windows.Forms.TextBox txtOutgoingTracks;
		private System.Windows.Forms.CheckBox chkPrimaryTrack;

		//Local Variables
		public SySal.Processing.VolumeGeneration.Configuration VConfig;

		public frmParameters()
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
			this.groupBox1 = new System.Windows.Forms.GroupBox();
			this.txtMostUpstreamPlate = new System.Windows.Forms.TextBox();
			this.label18 = new System.Windows.Forms.Label();
			this.txtTracking = new System.Windows.Forms.TextBox();
			this.txtNonTracking = new System.Windows.Forms.TextBox();
			this.label16 = new System.Windows.Forms.Label();
			this.label17 = new System.Windows.Forms.Label();
			this.label15 = new System.Windows.Forms.Label();
			this.txtVMaxZ = new System.Windows.Forms.TextBox();
			this.txtVMinZ = new System.Windows.Forms.TextBox();
			this.label8 = new System.Windows.Forms.Label();
			this.label9 = new System.Windows.Forms.Label();
			this.txtVMaxY = new System.Windows.Forms.TextBox();
			this.txtVMaxX = new System.Windows.Forms.TextBox();
			this.txtVMinY = new System.Windows.Forms.TextBox();
			this.label10 = new System.Windows.Forms.Label();
			this.label11 = new System.Windows.Forms.Label();
			this.txtVMinX = new System.Windows.Forms.TextBox();
			this.label12 = new System.Windows.Forms.Label();
			this.label13 = new System.Windows.Forms.Label();
			this.label14 = new System.Windows.Forms.Label();
			this.txtOBVMaxZ = new System.Windows.Forms.TextBox();
			this.txtOBVMinZ = new System.Windows.Forms.TextBox();
			this.label6 = new System.Windows.Forms.Label();
			this.label7 = new System.Windows.Forms.Label();
			this.txtOBVMaxY = new System.Windows.Forms.TextBox();
			this.txtOBVMaxX = new System.Windows.Forms.TextBox();
			this.txtOBVMinY = new System.Windows.Forms.TextBox();
			this.label4 = new System.Windows.Forms.Label();
			this.label5 = new System.Windows.Forms.Label();
			this.txtOBVMinX = new System.Windows.Forms.TextBox();
			this.label3 = new System.Windows.Forms.Label();
			this.label2 = new System.Windows.Forms.Label();
			this.label1 = new System.Windows.Forms.Label();
			this.groupBox2 = new System.Windows.Forms.GroupBox();
			this.txtTFE = new System.Windows.Forms.TextBox();
			this.label40 = new System.Windows.Forms.Label();
			this.label27 = new System.Windows.Forms.Label();
			this.txtSlopeErrX = new System.Windows.Forms.TextBox();
			this.txtSlopeErrY = new System.Windows.Forms.TextBox();
			this.label25 = new System.Windows.Forms.Label();
			this.label26 = new System.Windows.Forms.Label();
			this.txtCoordErrX = new System.Windows.Forms.TextBox();
			this.txtCoordErrY = new System.Windows.Forms.TextBox();
			this.label21 = new System.Windows.Forms.Label();
			this.label22 = new System.Windows.Forms.Label();
			this.label20 = new System.Windows.Forms.Label();
			this.txtSCoeffMin = new System.Windows.Forms.TextBox();
			this.label36 = new System.Windows.Forms.Label();
			this.txtSCoeffMax = new System.Windows.Forms.TextBox();
			this.label24 = new System.Windows.Forms.Label();
			this.groupBox3 = new System.Windows.Forms.GroupBox();
			this.txtMinimumEnergyLoss = new System.Windows.Forms.TextBox();
			this.label28 = new System.Windows.Forms.Label();
			this.txtRadiationLength = new System.Windows.Forms.TextBox();
			this.label29 = new System.Windows.Forms.Label();
			this.groupBox4 = new System.Windows.Forms.GroupBox();
			this.flytxtMomentum = new System.Windows.Forms.TextBox();
			this.lstMomentum = new System.Windows.Forms.ListView();
			this.columnHeader3 = new System.Windows.Forms.ColumnHeader();
			this.columnHeader4 = new System.Windows.Forms.ColumnHeader();
			this.cmbMomentumDistrib = new System.Windows.Forms.ComboBox();
			this.label41 = new System.Windows.Forms.Label();
			this.flytxtSlopeY = new System.Windows.Forms.TextBox();
			this.lstSlopeY = new System.Windows.Forms.ListView();
			this.columnHeader1 = new System.Windows.Forms.ColumnHeader();
			this.columnHeader2 = new System.Windows.Forms.ColumnHeader();
			this.cmbSlopeYDistrib = new System.Windows.Forms.ComboBox();
			this.label39 = new System.Windows.Forms.Label();
			this.lblJunkdensity = new System.Windows.Forms.Label();
			this.lblELdensity = new System.Windows.Forms.Label();
			this.lblHMdensity = new System.Windows.Forms.Label();
			this.flytxtSlopeX = new System.Windows.Forms.TextBox();
			this.lstSlopeX = new System.Windows.Forms.ListView();
			this.columnName = new System.Windows.Forms.ColumnHeader();
			this.columnValue = new System.Windows.Forms.ColumnHeader();
			this.cmbSlopeXDistrib = new System.Windows.Forms.ComboBox();
			this.txtHMTracks = new System.Windows.Forms.TextBox();
			this.label30 = new System.Windows.Forms.Label();
			this.txtELTracks = new System.Windows.Forms.TextBox();
			this.txtJunkTracks = new System.Windows.Forms.TextBox();
			this.label31 = new System.Windows.Forms.Label();
			this.label32 = new System.Windows.Forms.Label();
			this.label34 = new System.Windows.Forms.Label();
			this.txtConfigName = new System.Windows.Forms.TextBox();
			this.label35 = new System.Windows.Forms.Label();
			this.cmdCancel = new System.Windows.Forms.Button();
			this.cmdOk = new System.Windows.Forms.Button();
			this.cmdDefault = new System.Windows.Forms.Button();
			this.groupBox5 = new System.Windows.Forms.GroupBox();
			this.label50 = new System.Windows.Forms.Label();
			this.txtLongAlignMin = new System.Windows.Forms.TextBox();
			this.label37 = new System.Windows.Forms.Label();
			this.txtLongAlignMax = new System.Windows.Forms.TextBox();
			this.label49 = new System.Windows.Forms.Label();
			this.label43 = new System.Windows.Forms.Label();
			this.label48 = new System.Windows.Forms.Label();
			this.txtSShiftMax = new System.Windows.Forms.TextBox();
			this.txtSShiftMin = new System.Windows.Forms.TextBox();
			this.txtOutDMin = new System.Windows.Forms.TextBox();
			this.label46 = new System.Windows.Forms.Label();
			this.txtOutDMax = new System.Windows.Forms.TextBox();
			this.label47 = new System.Windows.Forms.Label();
			this.txtDiagMin = new System.Windows.Forms.TextBox();
			this.label42 = new System.Windows.Forms.Label();
			this.txtDiagMax = new System.Windows.Forms.TextBox();
			this.label45 = new System.Windows.Forms.Label();
			this.txtCoordAlignMin = new System.Windows.Forms.TextBox();
			this.label44 = new System.Windows.Forms.Label();
			this.label33 = new System.Windows.Forms.Label();
			this.label38 = new System.Windows.Forms.Label();
			this.txtCoordAlignMax = new System.Windows.Forms.TextBox();
			this.label23 = new System.Windows.Forms.Label();
			this.cmdHelp = new System.Windows.Forms.Button();
			this.groupBox6 = new System.Windows.Forms.GroupBox();
			this.txtOutgoingTracks = new System.Windows.Forms.TextBox();
			this.label51 = new System.Windows.Forms.Label();
			this.txtLocalVertexDepth = new System.Windows.Forms.TextBox();
			this.label19 = new System.Windows.Forms.Label();
			this.chkPrimaryTrack = new System.Windows.Forms.CheckBox();
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
			this.groupBox1.Controls.AddRange(new System.Windows.Forms.Control[] {
																					this.txtMostUpstreamPlate,
																					this.label18,
																					this.txtTracking,
																					this.txtNonTracking,
																					this.label16,
																					this.label17,
																					this.label15,
																					this.txtVMaxZ,
																					this.txtVMinZ,
																					this.label8,
																					this.label9,
																					this.txtVMaxY,
																					this.txtVMaxX,
																					this.txtVMinY,
																					this.label10,
																					this.label11,
																					this.txtVMinX,
																					this.label12,
																					this.label13,
																					this.label14,
																					this.txtOBVMaxZ,
																					this.txtOBVMinZ,
																					this.label6,
																					this.label7,
																					this.txtOBVMaxY,
																					this.txtOBVMaxX,
																					this.txtOBVMinY,
																					this.label4,
																					this.label5,
																					this.txtOBVMinX,
																					this.label3,
																					this.label2,
																					this.label1});
			this.groupBox1.Location = new System.Drawing.Point(8, 8);
			this.groupBox1.Name = "groupBox1";
			this.groupBox1.Size = new System.Drawing.Size(248, 264);
			this.groupBox1.TabIndex = 0;
			this.groupBox1.TabStop = false;
			this.groupBox1.Text = "Geometry";
			// 
			// txtMostUpstreamPlate
			// 
			this.txtMostUpstreamPlate.Location = new System.Drawing.Point(168, 232);
			this.txtMostUpstreamPlate.Name = "txtMostUpstreamPlate";
			this.txtMostUpstreamPlate.Size = new System.Drawing.Size(65, 20);
			this.txtMostUpstreamPlate.TabIndex = 33;
			this.txtMostUpstreamPlate.Text = "";
			// 
			// label18
			// 
			this.label18.AutoSize = true;
			this.label18.Location = new System.Drawing.Point(56, 240);
			this.label18.Name = "label18";
			this.label18.Size = new System.Drawing.Size(110, 13);
			this.label18.TabIndex = 34;
			this.label18.Text = "Most upstream plate:";
			// 
			// txtTracking
			// 
			this.txtTracking.Location = new System.Drawing.Point(56, 208);
			this.txtTracking.Name = "txtTracking";
			this.txtTracking.Size = new System.Drawing.Size(51, 20);
			this.txtTracking.TabIndex = 27;
			this.txtTracking.Text = "";
			// 
			// txtNonTracking
			// 
			this.txtNonTracking.Location = new System.Drawing.Point(184, 208);
			this.txtNonTracking.Name = "txtNonTracking";
			this.txtNonTracking.Size = new System.Drawing.Size(51, 20);
			this.txtNonTracking.TabIndex = 29;
			this.txtNonTracking.Text = "";
			// 
			// label16
			// 
			this.label16.AutoSize = true;
			this.label16.Location = new System.Drawing.Point(113, 211);
			this.label16.Name = "label16";
			this.label16.Size = new System.Drawing.Size(75, 13);
			this.label16.TabIndex = 30;
			this.label16.Text = "Non-Tracking:";
			// 
			// label17
			// 
			this.label17.AutoSize = true;
			this.label17.Location = new System.Drawing.Point(7, 211);
			this.label17.Name = "label17";
			this.label17.Size = new System.Drawing.Size(51, 13);
			this.label17.TabIndex = 28;
			this.label17.Text = "Tracking:";
			// 
			// label15
			// 
			this.label15.AutoSize = true;
			this.label15.Location = new System.Drawing.Point(7, 192);
			this.label15.Name = "label15";
			this.label15.Size = new System.Drawing.Size(55, 13);
			this.label15.TabIndex = 26;
			this.label15.Text = "Thickness";
			// 
			// txtVMaxZ
			// 
			this.txtVMaxZ.Location = new System.Drawing.Point(46, 168);
			this.txtVMaxZ.Name = "txtVMaxZ";
			this.txtVMaxZ.Size = new System.Drawing.Size(72, 20);
			this.txtVMaxZ.TabIndex = 22;
			this.txtVMaxZ.Text = "";
			// 
			// txtVMinZ
			// 
			this.txtVMinZ.Location = new System.Drawing.Point(164, 168);
			this.txtVMinZ.Name = "txtVMinZ";
			this.txtVMinZ.Size = new System.Drawing.Size(72, 20);
			this.txtVMinZ.TabIndex = 24;
			this.txtVMinZ.Text = "";
			// 
			// label8
			// 
			this.label8.AutoSize = true;
			this.label8.Location = new System.Drawing.Point(129, 172);
			this.label8.Name = "label8";
			this.label8.Size = new System.Drawing.Size(36, 13);
			this.label8.TabIndex = 25;
			this.label8.Text = "Min Z:";
			// 
			// label9
			// 
			this.label9.AutoSize = true;
			this.label9.Location = new System.Drawing.Point(8, 172);
			this.label9.Name = "label9";
			this.label9.Size = new System.Drawing.Size(39, 13);
			this.label9.TabIndex = 23;
			this.label9.Text = "Max Z:";
			// 
			// txtVMaxY
			// 
			this.txtVMaxY.Location = new System.Drawing.Point(46, 144);
			this.txtVMaxY.Name = "txtVMaxY";
			this.txtVMaxY.Size = new System.Drawing.Size(72, 20);
			this.txtVMaxY.TabIndex = 18;
			this.txtVMaxY.Text = "";
			// 
			// txtVMaxX
			// 
			this.txtVMaxX.Location = new System.Drawing.Point(46, 120);
			this.txtVMaxX.Name = "txtVMaxX";
			this.txtVMaxX.Size = new System.Drawing.Size(72, 20);
			this.txtVMaxX.TabIndex = 14;
			this.txtVMaxX.Text = "";
			// 
			// txtVMinY
			// 
			this.txtVMinY.Location = new System.Drawing.Point(164, 144);
			this.txtVMinY.Name = "txtVMinY";
			this.txtVMinY.Size = new System.Drawing.Size(72, 20);
			this.txtVMinY.TabIndex = 20;
			this.txtVMinY.Text = "";
			// 
			// label10
			// 
			this.label10.AutoSize = true;
			this.label10.Location = new System.Drawing.Point(129, 148);
			this.label10.Name = "label10";
			this.label10.Size = new System.Drawing.Size(36, 13);
			this.label10.TabIndex = 21;
			this.label10.Text = "Min Y:";
			// 
			// label11
			// 
			this.label11.AutoSize = true;
			this.label11.Location = new System.Drawing.Point(8, 148);
			this.label11.Name = "label11";
			this.label11.Size = new System.Drawing.Size(39, 13);
			this.label11.TabIndex = 19;
			this.label11.Text = "Max Y:";
			// 
			// txtVMinX
			// 
			this.txtVMinX.Location = new System.Drawing.Point(164, 120);
			this.txtVMinX.Name = "txtVMinX";
			this.txtVMinX.Size = new System.Drawing.Size(72, 20);
			this.txtVMinX.TabIndex = 16;
			this.txtVMinX.Text = "";
			// 
			// label12
			// 
			this.label12.AutoSize = true;
			this.label12.Location = new System.Drawing.Point(130, 124);
			this.label12.Name = "label12";
			this.label12.Size = new System.Drawing.Size(36, 13);
			this.label12.TabIndex = 17;
			this.label12.Text = "Min X:";
			// 
			// label13
			// 
			this.label13.AutoSize = true;
			this.label13.Location = new System.Drawing.Point(9, 124);
			this.label13.Name = "label13";
			this.label13.Size = new System.Drawing.Size(39, 13);
			this.label13.TabIndex = 15;
			this.label13.Text = "Max X:";
			// 
			// label14
			// 
			this.label14.AutoSize = true;
			this.label14.Location = new System.Drawing.Point(9, 104);
			this.label14.Name = "label14";
			this.label14.Size = new System.Drawing.Size(43, 13);
			this.label14.TabIndex = 13;
			this.label14.Text = "Volume";
			// 
			// txtOBVMaxZ
			// 
			this.txtOBVMaxZ.Location = new System.Drawing.Point(45, 80);
			this.txtOBVMaxZ.Name = "txtOBVMaxZ";
			this.txtOBVMaxZ.Size = new System.Drawing.Size(72, 20);
			this.txtOBVMaxZ.TabIndex = 9;
			this.txtOBVMaxZ.Text = "";
			// 
			// txtOBVMinZ
			// 
			this.txtOBVMinZ.Location = new System.Drawing.Point(163, 80);
			this.txtOBVMinZ.Name = "txtOBVMinZ";
			this.txtOBVMinZ.Size = new System.Drawing.Size(72, 20);
			this.txtOBVMinZ.TabIndex = 11;
			this.txtOBVMinZ.Text = "";
			// 
			// label6
			// 
			this.label6.AutoSize = true;
			this.label6.Location = new System.Drawing.Point(128, 83);
			this.label6.Name = "label6";
			this.label6.Size = new System.Drawing.Size(36, 13);
			this.label6.TabIndex = 12;
			this.label6.Text = "Min Z:";
			// 
			// label7
			// 
			this.label7.AutoSize = true;
			this.label7.Location = new System.Drawing.Point(7, 83);
			this.label7.Name = "label7";
			this.label7.Size = new System.Drawing.Size(39, 13);
			this.label7.TabIndex = 10;
			this.label7.Text = "Max Z:";
			// 
			// txtOBVMaxY
			// 
			this.txtOBVMaxY.Location = new System.Drawing.Point(45, 56);
			this.txtOBVMaxY.Name = "txtOBVMaxY";
			this.txtOBVMaxY.Size = new System.Drawing.Size(72, 20);
			this.txtOBVMaxY.TabIndex = 5;
			this.txtOBVMaxY.Text = "";
			this.txtOBVMaxY.TextChanged += new System.EventHandler(this.txtOBVMaxY_TextChanged);
			// 
			// txtOBVMaxX
			// 
			this.txtOBVMaxX.Location = new System.Drawing.Point(45, 32);
			this.txtOBVMaxX.Name = "txtOBVMaxX";
			this.txtOBVMaxX.Size = new System.Drawing.Size(72, 20);
			this.txtOBVMaxX.TabIndex = 1;
			this.txtOBVMaxX.Text = "";
			this.txtOBVMaxX.TextChanged += new System.EventHandler(this.txtOBVMaxX_TextChanged);
			// 
			// txtOBVMinY
			// 
			this.txtOBVMinY.Location = new System.Drawing.Point(163, 56);
			this.txtOBVMinY.Name = "txtOBVMinY";
			this.txtOBVMinY.Size = new System.Drawing.Size(72, 20);
			this.txtOBVMinY.TabIndex = 7;
			this.txtOBVMinY.Text = "";
			this.txtOBVMinY.TextChanged += new System.EventHandler(this.txtOBVMinY_TextChanged);
			// 
			// label4
			// 
			this.label4.AutoSize = true;
			this.label4.Location = new System.Drawing.Point(128, 61);
			this.label4.Name = "label4";
			this.label4.Size = new System.Drawing.Size(36, 13);
			this.label4.TabIndex = 8;
			this.label4.Text = "Min Y:";
			// 
			// label5
			// 
			this.label5.AutoSize = true;
			this.label5.Location = new System.Drawing.Point(7, 60);
			this.label5.Name = "label5";
			this.label5.Size = new System.Drawing.Size(39, 13);
			this.label5.TabIndex = 6;
			this.label5.Text = "Max Y:";
			// 
			// txtOBVMinX
			// 
			this.txtOBVMinX.Location = new System.Drawing.Point(163, 32);
			this.txtOBVMinX.Name = "txtOBVMinX";
			this.txtOBVMinX.Size = new System.Drawing.Size(72, 20);
			this.txtOBVMinX.TabIndex = 3;
			this.txtOBVMinX.Text = "";
			this.txtOBVMinX.TextChanged += new System.EventHandler(this.txtOBVMinX_TextChanged);
			// 
			// label3
			// 
			this.label3.AutoSize = true;
			this.label3.Location = new System.Drawing.Point(129, 36);
			this.label3.Name = "label3";
			this.label3.Size = new System.Drawing.Size(36, 13);
			this.label3.TabIndex = 4;
			this.label3.Text = "Min X:";
			// 
			// label2
			// 
			this.label2.AutoSize = true;
			this.label2.Location = new System.Drawing.Point(8, 35);
			this.label2.Name = "label2";
			this.label2.Size = new System.Drawing.Size(39, 13);
			this.label2.TabIndex = 2;
			this.label2.Text = "Max X:";
			// 
			// label1
			// 
			this.label1.AutoSize = true;
			this.label1.Location = new System.Drawing.Point(8, 16);
			this.label1.Name = "label1";
			this.label1.Size = new System.Drawing.Size(106, 13);
			this.label1.TabIndex = 0;
			this.label1.Text = "Out-Bounds Volume";
			// 
			// groupBox2
			// 
			this.groupBox2.Controls.AddRange(new System.Windows.Forms.Control[] {
																					this.txtTFE,
																					this.label40,
																					this.label27,
																					this.txtSlopeErrX,
																					this.txtSlopeErrY,
																					this.label25,
																					this.label26,
																					this.txtCoordErrX,
																					this.txtCoordErrY,
																					this.label21,
																					this.label22,
																					this.label20});
			this.groupBox2.Location = new System.Drawing.Point(264, 200);
			this.groupBox2.Name = "groupBox2";
			this.groupBox2.Size = new System.Drawing.Size(376, 104);
			this.groupBox2.TabIndex = 1;
			this.groupBox2.TabStop = false;
			this.groupBox2.Text = "Errors";
			// 
			// txtTFE
			// 
			this.txtTFE.Location = new System.Drawing.Point(320, 12);
			this.txtTFE.Name = "txtTFE";
			this.txtTFE.Size = new System.Drawing.Size(48, 20);
			this.txtTFE.TabIndex = 19;
			this.txtTFE.Text = "";
			// 
			// label40
			// 
			this.label40.AutoSize = true;
			this.label40.Location = new System.Drawing.Point(200, 16);
			this.label40.Name = "label40";
			this.label40.Size = new System.Drawing.Size(128, 13);
			this.label40.TabIndex = 18;
			this.label40.Text = "Track Finding Efficiency:";
			// 
			// label27
			// 
			this.label27.AutoSize = true;
			this.label27.Location = new System.Drawing.Point(6, 56);
			this.label27.Name = "label27";
			this.label27.Size = new System.Drawing.Size(39, 13);
			this.label27.TabIndex = 17;
			this.label27.Text = "Slopes";
			// 
			// txtSlopeErrX
			// 
			this.txtSlopeErrX.Location = new System.Drawing.Point(48, 75);
			this.txtSlopeErrX.Name = "txtSlopeErrX";
			this.txtSlopeErrX.Size = new System.Drawing.Size(48, 20);
			this.txtSlopeErrX.TabIndex = 11;
			this.txtSlopeErrX.Text = "";
			// 
			// txtSlopeErrY
			// 
			this.txtSlopeErrY.Location = new System.Drawing.Point(136, 75);
			this.txtSlopeErrY.Name = "txtSlopeErrY";
			this.txtSlopeErrY.Size = new System.Drawing.Size(48, 20);
			this.txtSlopeErrY.TabIndex = 13;
			this.txtSlopeErrY.Text = "";
			// 
			// label25
			// 
			this.label25.AutoSize = true;
			this.label25.Location = new System.Drawing.Point(96, 78);
			this.label25.Name = "label25";
			this.label25.Size = new System.Drawing.Size(43, 13);
			this.label25.TabIndex = 14;
			this.label25.Text = "Error Y:";
			// 
			// label26
			// 
			this.label26.AutoSize = true;
			this.label26.Location = new System.Drawing.Point(8, 78);
			this.label26.Name = "label26";
			this.label26.Size = new System.Drawing.Size(43, 13);
			this.label26.TabIndex = 12;
			this.label26.Text = "Error X:";
			// 
			// txtCoordErrX
			// 
			this.txtCoordErrX.Location = new System.Drawing.Point(48, 32);
			this.txtCoordErrX.Name = "txtCoordErrX";
			this.txtCoordErrX.Size = new System.Drawing.Size(48, 20);
			this.txtCoordErrX.TabIndex = 5;
			this.txtCoordErrX.Text = "";
			// 
			// txtCoordErrY
			// 
			this.txtCoordErrY.Location = new System.Drawing.Point(136, 32);
			this.txtCoordErrY.Name = "txtCoordErrY";
			this.txtCoordErrY.Size = new System.Drawing.Size(48, 20);
			this.txtCoordErrY.TabIndex = 7;
			this.txtCoordErrY.Text = "";
			// 
			// label21
			// 
			this.label21.AutoSize = true;
			this.label21.Location = new System.Drawing.Point(96, 35);
			this.label21.Name = "label21";
			this.label21.Size = new System.Drawing.Size(43, 13);
			this.label21.TabIndex = 8;
			this.label21.Text = "Error Y:";
			// 
			// label22
			// 
			this.label22.AutoSize = true;
			this.label22.Location = new System.Drawing.Point(8, 35);
			this.label22.Name = "label22";
			this.label22.Size = new System.Drawing.Size(43, 13);
			this.label22.TabIndex = 6;
			this.label22.Text = "Error X:";
			// 
			// label20
			// 
			this.label20.AutoSize = true;
			this.label20.Location = new System.Drawing.Point(3, 16);
			this.label20.Name = "label20";
			this.label20.Size = new System.Drawing.Size(65, 13);
			this.label20.TabIndex = 0;
			this.label20.Text = "Coordinates";
			// 
			// txtSCoeffMin
			// 
			this.txtSCoeffMin.Location = new System.Drawing.Point(616, 64);
			this.txtSCoeffMin.Name = "txtSCoeffMin";
			this.txtSCoeffMin.Size = new System.Drawing.Size(48, 20);
			this.txtSCoeffMin.TabIndex = 20;
			this.txtSCoeffMin.Text = "";
			// 
			// label36
			// 
			this.label36.AutoSize = true;
			this.label36.Location = new System.Drawing.Point(560, 64);
			this.label36.Name = "label36";
			this.label36.Size = new System.Drawing.Size(56, 13);
			this.label36.TabIndex = 21;
			this.label36.Text = "Min Coeff:";
			// 
			// txtSCoeffMax
			// 
			this.txtSCoeffMax.Location = new System.Drawing.Point(616, 40);
			this.txtSCoeffMax.Name = "txtSCoeffMax";
			this.txtSCoeffMax.Size = new System.Drawing.Size(48, 20);
			this.txtSCoeffMax.TabIndex = 15;
			this.txtSCoeffMax.Text = "";
			// 
			// label24
			// 
			this.label24.AutoSize = true;
			this.label24.Location = new System.Drawing.Point(560, 40);
			this.label24.Name = "label24";
			this.label24.Size = new System.Drawing.Size(59, 13);
			this.label24.TabIndex = 16;
			this.label24.Text = "Max Coeff:";
			// 
			// groupBox3
			// 
			this.groupBox3.Controls.AddRange(new System.Windows.Forms.Control[] {
																					this.txtMinimumEnergyLoss,
																					this.label28,
																					this.txtRadiationLength,
																					this.label29});
			this.groupBox3.Location = new System.Drawing.Point(648, 200);
			this.groupBox3.Name = "groupBox3";
			this.groupBox3.Size = new System.Drawing.Size(168, 67);
			this.groupBox3.TabIndex = 2;
			this.groupBox3.TabStop = false;
			this.groupBox3.Text = "Kinematics";
			// 
			// txtMinimumEnergyLoss
			// 
			this.txtMinimumEnergyLoss.Location = new System.Drawing.Point(104, 15);
			this.txtMinimumEnergyLoss.Name = "txtMinimumEnergyLoss";
			this.txtMinimumEnergyLoss.Size = new System.Drawing.Size(56, 20);
			this.txtMinimumEnergyLoss.TabIndex = 13;
			this.txtMinimumEnergyLoss.Text = "";
			// 
			// label28
			// 
			this.label28.AutoSize = true;
			this.label28.Location = new System.Drawing.Point(8, 18);
			this.label28.Name = "label28";
			this.label28.Size = new System.Drawing.Size(104, 13);
			this.label28.TabIndex = 14;
			this.label28.Text = "MinimuEnergyLoss:";
			// 
			// txtRadiationLength
			// 
			this.txtRadiationLength.Location = new System.Drawing.Point(104, 38);
			this.txtRadiationLength.Name = "txtRadiationLength";
			this.txtRadiationLength.Size = new System.Drawing.Size(56, 20);
			this.txtRadiationLength.TabIndex = 11;
			this.txtRadiationLength.Text = "";
			// 
			// label29
			// 
			this.label29.AutoSize = true;
			this.label29.Location = new System.Drawing.Point(8, 39);
			this.label29.Name = "label29";
			this.label29.Size = new System.Drawing.Size(93, 13);
			this.label29.TabIndex = 12;
			this.label29.Text = "Radiation Length:";
			// 
			// groupBox4
			// 
			this.groupBox4.Controls.AddRange(new System.Windows.Forms.Control[] {
																					this.flytxtMomentum,
																					this.lstMomentum,
																					this.cmbMomentumDistrib,
																					this.label41,
																					this.flytxtSlopeY,
																					this.lstSlopeY,
																					this.cmbSlopeYDistrib,
																					this.label39,
																					this.lblJunkdensity,
																					this.lblELdensity,
																					this.lblHMdensity,
																					this.flytxtSlopeX,
																					this.lstSlopeX,
																					this.cmbSlopeXDistrib,
																					this.txtHMTracks,
																					this.label30,
																					this.txtELTracks,
																					this.txtJunkTracks,
																					this.label31,
																					this.label32,
																					this.label34});
			this.groupBox4.Location = new System.Drawing.Point(264, 40);
			this.groupBox4.Name = "groupBox4";
			this.groupBox4.Size = new System.Drawing.Size(584, 160);
			this.groupBox4.TabIndex = 3;
			this.groupBox4.TabStop = false;
			this.groupBox4.Text = "Tracks";
			// 
			// flytxtMomentum
			// 
			this.flytxtMomentum.AutoSize = false;
			this.flytxtMomentum.BackColor = System.Drawing.SystemColors.InactiveBorder;
			this.flytxtMomentum.BorderStyle = System.Windows.Forms.BorderStyle.None;
			this.flytxtMomentum.Location = new System.Drawing.Point(512, 128);
			this.flytxtMomentum.Name = "flytxtMomentum";
			this.flytxtMomentum.Size = new System.Drawing.Size(56, 16);
			this.flytxtMomentum.TabIndex = 32;
			this.flytxtMomentum.Text = "flytxtMomentum";
			this.flytxtMomentum.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
			this.flytxtMomentum.Visible = false;
			// 
			// lstMomentum
			// 
			this.lstMomentum.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
																						  this.columnHeader3,
																						  this.columnHeader4});
			this.lstMomentum.FullRowSelect = true;
			this.lstMomentum.GridLines = true;
			this.lstMomentum.HeaderStyle = System.Windows.Forms.ColumnHeaderStyle.Nonclickable;
			this.lstMomentum.Location = new System.Drawing.Point(448, 64);
			this.lstMomentum.MultiSelect = false;
			this.lstMomentum.Name = "lstMomentum";
			this.lstMomentum.Size = new System.Drawing.Size(124, 80);
			this.lstMomentum.Sorting = System.Windows.Forms.SortOrder.Ascending;
			this.lstMomentum.TabIndex = 31;
			this.lstMomentum.View = System.Windows.Forms.View.Details;
			// 
			// columnHeader3
			// 
			this.columnHeader3.Text = "Name";
			// 
			// columnHeader4
			// 
			this.columnHeader4.Text = "Value";
			this.columnHeader4.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
			// 
			// cmbMomentumDistrib
			// 
			this.cmbMomentumDistrib.Location = new System.Drawing.Point(448, 32);
			this.cmbMomentumDistrib.Name = "cmbMomentumDistrib";
			this.cmbMomentumDistrib.Size = new System.Drawing.Size(124, 21);
			this.cmbMomentumDistrib.Sorted = true;
			this.cmbMomentumDistrib.TabIndex = 29;
			this.cmbMomentumDistrib.SelectedIndexChanged += new System.EventHandler(this.cmbMomentumDistrib_SelectedIndexChanged);
			// 
			// label41
			// 
			this.label41.AutoSize = true;
			this.label41.Location = new System.Drawing.Point(448, 16);
			this.label41.Name = "label41";
			this.label41.Size = new System.Drawing.Size(122, 13);
			this.label41.TabIndex = 30;
			this.label41.Text = "Momentum distribution:";
			// 
			// flytxtSlopeY
			// 
			this.flytxtSlopeY.AutoSize = false;
			this.flytxtSlopeY.BackColor = System.Drawing.SystemColors.InactiveBorder;
			this.flytxtSlopeY.BorderStyle = System.Windows.Forms.BorderStyle.None;
			this.flytxtSlopeY.Location = new System.Drawing.Point(376, 128);
			this.flytxtSlopeY.Name = "flytxtSlopeY";
			this.flytxtSlopeY.Size = new System.Drawing.Size(56, 16);
			this.flytxtSlopeY.TabIndex = 28;
			this.flytxtSlopeY.Text = "flytxtSlopeY";
			this.flytxtSlopeY.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
			this.flytxtSlopeY.Visible = false;
			// 
			// lstSlopeY
			// 
			this.lstSlopeY.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
																						this.columnHeader1,
																						this.columnHeader2});
			this.lstSlopeY.FullRowSelect = true;
			this.lstSlopeY.GridLines = true;
			this.lstSlopeY.HeaderStyle = System.Windows.Forms.ColumnHeaderStyle.Nonclickable;
			this.lstSlopeY.Location = new System.Drawing.Point(312, 64);
			this.lstSlopeY.MultiSelect = false;
			this.lstSlopeY.Name = "lstSlopeY";
			this.lstSlopeY.Size = new System.Drawing.Size(124, 80);
			this.lstSlopeY.Sorting = System.Windows.Forms.SortOrder.Ascending;
			this.lstSlopeY.TabIndex = 27;
			this.lstSlopeY.View = System.Windows.Forms.View.Details;
			// 
			// columnHeader1
			// 
			this.columnHeader1.Text = "Name";
			// 
			// columnHeader2
			// 
			this.columnHeader2.Text = "Value";
			this.columnHeader2.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
			// 
			// cmbSlopeYDistrib
			// 
			this.cmbSlopeYDistrib.Location = new System.Drawing.Point(312, 32);
			this.cmbSlopeYDistrib.Name = "cmbSlopeYDistrib";
			this.cmbSlopeYDistrib.Size = new System.Drawing.Size(124, 21);
			this.cmbSlopeYDistrib.Sorted = true;
			this.cmbSlopeYDistrib.TabIndex = 25;
			this.cmbSlopeYDistrib.SelectedIndexChanged += new System.EventHandler(this.cmbSlopeYDistrib_SelectedIndexChanged);
			// 
			// label39
			// 
			this.label39.AutoSize = true;
			this.label39.Location = new System.Drawing.Point(312, 16);
			this.label39.Name = "label39";
			this.label39.Size = new System.Drawing.Size(102, 13);
			this.label39.TabIndex = 26;
			this.label39.Text = "SlopeY distribution:";
			// 
			// lblJunkdensity
			// 
			this.lblJunkdensity.AutoSize = true;
			this.lblJunkdensity.Location = new System.Drawing.Point(8, 128);
			this.lblJunkdensity.Name = "lblJunkdensity";
			this.lblJunkdensity.Size = new System.Drawing.Size(102, 13);
			this.lblJunkdensity.TabIndex = 24;
			this.lblJunkdensity.Text = "Junk Track density:";
			// 
			// lblELdensity
			// 
			this.lblELdensity.AutoSize = true;
			this.lblELdensity.Location = new System.Drawing.Point(8, 108);
			this.lblELdensity.Name = "lblELdensity";
			this.lblELdensity.Size = new System.Drawing.Size(92, 13);
			this.lblELdensity.TabIndex = 23;
			this.lblELdensity.Text = "EL-Track density:";
			// 
			// lblHMdensity
			// 
			this.lblHMdensity.AutoSize = true;
			this.lblHMdensity.Location = new System.Drawing.Point(8, 88);
			this.lblHMdensity.Name = "lblHMdensity";
			this.lblHMdensity.Size = new System.Drawing.Size(96, 13);
			this.lblHMdensity.TabIndex = 22;
			this.lblHMdensity.Text = "HM-Track density:";
			// 
			// flytxtSlopeX
			// 
			this.flytxtSlopeX.AutoSize = false;
			this.flytxtSlopeX.BackColor = System.Drawing.SystemColors.InactiveBorder;
			this.flytxtSlopeX.BorderStyle = System.Windows.Forms.BorderStyle.None;
			this.flytxtSlopeX.Location = new System.Drawing.Point(240, 128);
			this.flytxtSlopeX.Name = "flytxtSlopeX";
			this.flytxtSlopeX.Size = new System.Drawing.Size(56, 16);
			this.flytxtSlopeX.TabIndex = 21;
			this.flytxtSlopeX.Text = "flytxtSlopeX";
			this.flytxtSlopeX.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
			this.flytxtSlopeX.Visible = false;
			// 
			// lstSlopeX
			// 
			this.lstSlopeX.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
																						this.columnName,
																						this.columnValue});
			this.lstSlopeX.FullRowSelect = true;
			this.lstSlopeX.GridLines = true;
			this.lstSlopeX.HeaderStyle = System.Windows.Forms.ColumnHeaderStyle.Nonclickable;
			this.lstSlopeX.Location = new System.Drawing.Point(176, 64);
			this.lstSlopeX.MultiSelect = false;
			this.lstSlopeX.Name = "lstSlopeX";
			this.lstSlopeX.Size = new System.Drawing.Size(124, 80);
			this.lstSlopeX.Sorting = System.Windows.Forms.SortOrder.Ascending;
			this.lstSlopeX.TabIndex = 20;
			this.lstSlopeX.View = System.Windows.Forms.View.Details;
			this.lstSlopeX.SelectedIndexChanged += new System.EventHandler(this.ParamList_SelectedIndexChanged);
			// 
			// columnName
			// 
			this.columnName.Text = "Name";
			// 
			// columnValue
			// 
			this.columnValue.Text = "Value";
			this.columnValue.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
			// 
			// cmbSlopeXDistrib
			// 
			this.cmbSlopeXDistrib.Location = new System.Drawing.Point(176, 32);
			this.cmbSlopeXDistrib.Name = "cmbSlopeXDistrib";
			this.cmbSlopeXDistrib.Size = new System.Drawing.Size(124, 21);
			this.cmbSlopeXDistrib.Sorted = true;
			this.cmbSlopeXDistrib.TabIndex = 18;
			this.cmbSlopeXDistrib.SelectedIndexChanged += new System.EventHandler(this.cmbSlopeXDistrib_SelectedIndexChanged);
			// 
			// txtHMTracks
			// 
			this.txtHMTracks.Location = new System.Drawing.Point(80, 16);
			this.txtHMTracks.Name = "txtHMTracks";
			this.txtHMTracks.Size = new System.Drawing.Size(64, 20);
			this.txtHMTracks.TabIndex = 15;
			this.txtHMTracks.Text = "";
			this.txtHMTracks.TextChanged += new System.EventHandler(this.txtHMTracks_TextChanged);
			// 
			// label30
			// 
			this.label30.AutoSize = true;
			this.label30.Location = new System.Drawing.Point(8, 16);
			this.label30.Name = "label30";
			this.label30.Size = new System.Drawing.Size(63, 13);
			this.label30.TabIndex = 16;
			this.label30.Text = "HM-Tracks:";
			// 
			// txtELTracks
			// 
			this.txtELTracks.Location = new System.Drawing.Point(80, 40);
			this.txtELTracks.Name = "txtELTracks";
			this.txtELTracks.Size = new System.Drawing.Size(64, 20);
			this.txtELTracks.TabIndex = 11;
			this.txtELTracks.Text = "";
			this.txtELTracks.TextChanged += new System.EventHandler(this.txtELTracks_TextChanged);
			// 
			// txtJunkTracks
			// 
			this.txtJunkTracks.Location = new System.Drawing.Point(80, 64);
			this.txtJunkTracks.Name = "txtJunkTracks";
			this.txtJunkTracks.Size = new System.Drawing.Size(64, 20);
			this.txtJunkTracks.TabIndex = 13;
			this.txtJunkTracks.Text = "";
			this.txtJunkTracks.TextChanged += new System.EventHandler(this.txtJunkTracks_TextChanged);
			// 
			// label31
			// 
			this.label31.AutoSize = true;
			this.label31.Location = new System.Drawing.Point(8, 64);
			this.label31.Name = "label31";
			this.label31.Size = new System.Drawing.Size(68, 13);
			this.label31.TabIndex = 14;
			this.label31.Text = "Junk Tracks:";
			// 
			// label32
			// 
			this.label32.AutoSize = true;
			this.label32.Location = new System.Drawing.Point(8, 40);
			this.label32.Name = "label32";
			this.label32.Size = new System.Drawing.Size(59, 13);
			this.label32.TabIndex = 12;
			this.label32.Text = "EL-Tracks:";
			// 
			// label34
			// 
			this.label34.AutoSize = true;
			this.label34.Location = new System.Drawing.Point(176, 16);
			this.label34.Name = "label34";
			this.label34.Size = new System.Drawing.Size(102, 13);
			this.label34.TabIndex = 19;
			this.label34.Text = "SlopeX distribution:";
			// 
			// txtConfigName
			// 
			this.txtConfigName.Location = new System.Drawing.Point(368, 13);
			this.txtConfigName.Name = "txtConfigName";
			this.txtConfigName.Size = new System.Drawing.Size(352, 20);
			this.txtConfigName.TabIndex = 11;
			this.txtConfigName.Text = "";
			// 
			// label35
			// 
			this.label35.AutoSize = true;
			this.label35.Location = new System.Drawing.Point(264, 16);
			this.label35.Name = "label35";
			this.label35.Size = new System.Drawing.Size(108, 13);
			this.label35.TabIndex = 10;
			this.label35.Text = "Configuration Name:";
			// 
			// cmdCancel
			// 
			this.cmdCancel.DialogResult = System.Windows.Forms.DialogResult.Cancel;
			this.cmdCancel.Location = new System.Drawing.Point(816, 280);
			this.cmdCancel.Name = "cmdCancel";
			this.cmdCancel.Size = new System.Drawing.Size(56, 24);
			this.cmdCancel.TabIndex = 18;
			this.cmdCancel.Text = "Cancel";
			// 
			// cmdOk
			// 
			this.cmdOk.DialogResult = System.Windows.Forms.DialogResult.OK;
			this.cmdOk.Location = new System.Drawing.Point(760, 280);
			this.cmdOk.Name = "cmdOk";
			this.cmdOk.Size = new System.Drawing.Size(56, 24);
			this.cmdOk.TabIndex = 17;
			this.cmdOk.Text = "OK";
			this.cmdOk.Click += new System.EventHandler(this.cmdOk_Click);
			// 
			// cmdDefault
			// 
			this.cmdDefault.Location = new System.Drawing.Point(648, 280);
			this.cmdDefault.Name = "cmdDefault";
			this.cmdDefault.Size = new System.Drawing.Size(56, 24);
			this.cmdDefault.TabIndex = 16;
			this.cmdDefault.Text = "Default";
			this.cmdDefault.Click += new System.EventHandler(this.cmdDefault_Click);
			// 
			// groupBox5
			// 
			this.groupBox5.Controls.AddRange(new System.Windows.Forms.Control[] {
																					this.label50,
																					this.txtSCoeffMax,
																					this.txtLongAlignMin,
																					this.label37,
																					this.txtLongAlignMax,
																					this.label49,
																					this.label43,
																					this.label48,
																					this.txtSShiftMax,
																					this.txtSShiftMin,
																					this.txtOutDMin,
																					this.label46,
																					this.txtOutDMax,
																					this.label47,
																					this.txtDiagMin,
																					this.label42,
																					this.txtDiagMax,
																					this.label45,
																					this.txtCoordAlignMin,
																					this.label44,
																					this.label33,
																					this.label38,
																					this.label24,
																					this.txtCoordAlignMax,
																					this.label36,
																					this.txtSCoeffMin,
																					this.label23});
			this.groupBox5.Location = new System.Drawing.Point(8, 360);
			this.groupBox5.Name = "groupBox5";
			this.groupBox5.Size = new System.Drawing.Size(864, 96);
			this.groupBox5.TabIndex = 19;
			this.groupBox5.TabStop = false;
			this.groupBox5.Text = "Affine";
			// 
			// label50
			// 
			this.label50.AutoSize = true;
			this.label50.Location = new System.Drawing.Point(281, 20);
			this.label50.Name = "label50";
			this.label50.Size = new System.Drawing.Size(102, 13);
			this.label50.TabIndex = 44;
			this.label50.Text = "Coordinates: Matrix";
			// 
			// txtLongAlignMin
			// 
			this.txtLongAlignMin.Location = new System.Drawing.Point(224, 64);
			this.txtLongAlignMin.Name = "txtLongAlignMin";
			this.txtLongAlignMin.Size = new System.Drawing.Size(48, 20);
			this.txtLongAlignMin.TabIndex = 42;
			this.txtLongAlignMin.Text = "";
			// 
			// label37
			// 
			this.label37.AutoSize = true;
			this.label37.Location = new System.Drawing.Point(144, 64);
			this.label37.Name = "label37";
			this.label37.Size = new System.Drawing.Size(83, 13);
			this.label37.TabIndex = 43;
			this.label37.Text = "Long Min Term:";
			// 
			// txtLongAlignMax
			// 
			this.txtLongAlignMax.Location = new System.Drawing.Point(224, 40);
			this.txtLongAlignMax.Name = "txtLongAlignMax";
			this.txtLongAlignMax.Size = new System.Drawing.Size(48, 20);
			this.txtLongAlignMax.TabIndex = 40;
			this.txtLongAlignMax.Text = "";
			// 
			// label49
			// 
			this.label49.AutoSize = true;
			this.label49.Location = new System.Drawing.Point(144, 40);
			this.label49.Name = "label49";
			this.label49.Size = new System.Drawing.Size(86, 13);
			this.label49.TabIndex = 41;
			this.label49.Text = "Long Max Term:";
			// 
			// label43
			// 
			this.label43.AutoSize = true;
			this.label43.Location = new System.Drawing.Point(680, 40);
			this.label43.Name = "label43";
			this.label43.Size = new System.Drawing.Size(54, 13);
			this.label43.TabIndex = 36;
			this.label43.Text = "Max Shift:";
			// 
			// label48
			// 
			this.label48.AutoSize = true;
			this.label48.Location = new System.Drawing.Point(680, 64);
			this.label48.Name = "label48";
			this.label48.Size = new System.Drawing.Size(51, 13);
			this.label48.TabIndex = 39;
			this.label48.Text = "Min Shift:";
			// 
			// txtSShiftMax
			// 
			this.txtSShiftMax.Location = new System.Drawing.Point(736, 40);
			this.txtSShiftMax.Name = "txtSShiftMax";
			this.txtSShiftMax.Size = new System.Drawing.Size(48, 20);
			this.txtSShiftMax.TabIndex = 35;
			this.txtSShiftMax.Text = "";
			// 
			// txtSShiftMin
			// 
			this.txtSShiftMin.Location = new System.Drawing.Point(736, 64);
			this.txtSShiftMin.Name = "txtSShiftMin";
			this.txtSShiftMin.Size = new System.Drawing.Size(48, 20);
			this.txtSShiftMin.TabIndex = 38;
			this.txtSShiftMin.Text = "";
			// 
			// txtOutDMin
			// 
			this.txtOutDMin.Location = new System.Drawing.Point(504, 64);
			this.txtOutDMin.Name = "txtOutDMin";
			this.txtOutDMin.Size = new System.Drawing.Size(48, 20);
			this.txtOutDMin.TabIndex = 33;
			this.txtOutDMin.Text = "";
			// 
			// label46
			// 
			this.label46.AutoSize = true;
			this.label46.Location = new System.Drawing.Point(424, 64);
			this.label46.Name = "label46";
			this.label46.Size = new System.Drawing.Size(85, 13);
			this.label46.TabIndex = 34;
			this.label46.Text = "OutD Min Term:";
			// 
			// txtOutDMax
			// 
			this.txtOutDMax.Location = new System.Drawing.Point(504, 40);
			this.txtOutDMax.Name = "txtOutDMax";
			this.txtOutDMax.Size = new System.Drawing.Size(48, 20);
			this.txtOutDMax.TabIndex = 31;
			this.txtOutDMax.Text = "";
			// 
			// label47
			// 
			this.label47.AutoSize = true;
			this.label47.Location = new System.Drawing.Point(424, 40);
			this.label47.Name = "label47";
			this.label47.Size = new System.Drawing.Size(88, 13);
			this.label47.TabIndex = 32;
			this.label47.Text = "OutD Max Term:";
			// 
			// txtDiagMin
			// 
			this.txtDiagMin.Location = new System.Drawing.Point(368, 64);
			this.txtDiagMin.Name = "txtDiagMin";
			this.txtDiagMin.Size = new System.Drawing.Size(48, 20);
			this.txtDiagMin.TabIndex = 29;
			this.txtDiagMin.Text = "";
			// 
			// label42
			// 
			this.label42.AutoSize = true;
			this.label42.Location = new System.Drawing.Point(280, 64);
			this.label42.Name = "label42";
			this.label42.Size = new System.Drawing.Size(81, 13);
			this.label42.TabIndex = 30;
			this.label42.Text = "Diag Min Term:";
			// 
			// txtDiagMax
			// 
			this.txtDiagMax.Location = new System.Drawing.Point(368, 40);
			this.txtDiagMax.Name = "txtDiagMax";
			this.txtDiagMax.Size = new System.Drawing.Size(48, 20);
			this.txtDiagMax.TabIndex = 27;
			this.txtDiagMax.Text = "";
			// 
			// label45
			// 
			this.label45.AutoSize = true;
			this.label45.Location = new System.Drawing.Point(280, 40);
			this.label45.Name = "label45";
			this.label45.Size = new System.Drawing.Size(85, 13);
			this.label45.TabIndex = 28;
			this.label45.Text = "Diag Max Term:";
			// 
			// txtCoordAlignMin
			// 
			this.txtCoordAlignMin.Location = new System.Drawing.Point(88, 64);
			this.txtCoordAlignMin.Name = "txtCoordAlignMin";
			this.txtCoordAlignMin.Size = new System.Drawing.Size(48, 20);
			this.txtCoordAlignMin.TabIndex = 22;
			this.txtCoordAlignMin.Text = "";
			// 
			// label44
			// 
			this.label44.AutoSize = true;
			this.label44.Location = new System.Drawing.Point(8, 64);
			this.label44.Name = "label44";
			this.label44.Size = new System.Drawing.Size(83, 13);
			this.label44.TabIndex = 23;
			this.label44.Text = "Align Min Term:";
			// 
			// label33
			// 
			this.label33.AutoSize = true;
			this.label33.Location = new System.Drawing.Point(560, 20);
			this.label33.Name = "label33";
			this.label33.Size = new System.Drawing.Size(128, 13);
			this.label33.TabIndex = 19;
			this.label33.Text = "Slopes: Linear Distortion";
			// 
			// label38
			// 
			this.label38.AutoSize = true;
			this.label38.Location = new System.Drawing.Point(8, 20);
			this.label38.Name = "label38";
			this.label38.TabIndex = 18;
			this.label38.Text = "Coordinates: Shifts";
			// 
			// txtCoordAlignMax
			// 
			this.txtCoordAlignMax.Location = new System.Drawing.Point(88, 40);
			this.txtCoordAlignMax.Name = "txtCoordAlignMax";
			this.txtCoordAlignMax.Size = new System.Drawing.Size(48, 20);
			this.txtCoordAlignMax.TabIndex = 9;
			this.txtCoordAlignMax.Text = "";
			// 
			// label23
			// 
			this.label23.AutoSize = true;
			this.label23.Location = new System.Drawing.Point(8, 40);
			this.label23.Name = "label23";
			this.label23.Size = new System.Drawing.Size(86, 13);
			this.label23.TabIndex = 10;
			this.label23.Text = "Align Max Term:";
			// 
			// cmdHelp
			// 
			this.cmdHelp.Location = new System.Drawing.Point(704, 280);
			this.cmdHelp.Name = "cmdHelp";
			this.cmdHelp.Size = new System.Drawing.Size(56, 24);
			this.cmdHelp.TabIndex = 20;
			this.cmdHelp.Text = "Help";
			this.cmdHelp.Click += new System.EventHandler(this.cmdHelp_Click);
			// 
			// groupBox6
			// 
			this.groupBox6.Controls.AddRange(new System.Windows.Forms.Control[] {
																					this.chkPrimaryTrack,
																					this.txtOutgoingTracks,
																					this.label51,
																					this.txtLocalVertexDepth,
																					this.label19});
			this.groupBox6.Location = new System.Drawing.Point(8, 272);
			this.groupBox6.Name = "groupBox6";
			this.groupBox6.Size = new System.Drawing.Size(248, 88);
			this.groupBox6.TabIndex = 21;
			this.groupBox6.TabStop = false;
			this.groupBox6.Text = "Event";
			// 
			// txtOutgoingTracks
			// 
			this.txtOutgoingTracks.Location = new System.Drawing.Point(122, 40);
			this.txtOutgoingTracks.Name = "txtOutgoingTracks";
			this.txtOutgoingTracks.Size = new System.Drawing.Size(66, 20);
			this.txtOutgoingTracks.TabIndex = 36;
			this.txtOutgoingTracks.Text = "";
			// 
			// label51
			// 
			this.label51.AutoSize = true;
			this.label51.Location = new System.Drawing.Point(10, 42);
			this.label51.Name = "label51";
			this.label51.Size = new System.Drawing.Size(90, 13);
			this.label51.TabIndex = 35;
			this.label51.Text = "Outgoing Tracks:";
			// 
			// txtLocalVertexDepth
			// 
			this.txtLocalVertexDepth.Location = new System.Drawing.Point(122, 14);
			this.txtLocalVertexDepth.Name = "txtLocalVertexDepth";
			this.txtLocalVertexDepth.Size = new System.Drawing.Size(66, 20);
			this.txtLocalVertexDepth.TabIndex = 33;
			this.txtLocalVertexDepth.Text = "";
			// 
			// label19
			// 
			this.label19.AutoSize = true;
			this.label19.Location = new System.Drawing.Point(10, 16);
			this.label19.Name = "label19";
			this.label19.TabIndex = 34;
			this.label19.Text = "Local vertex depth:";
			// 
			// chkPrimaryTrack
			// 
			this.chkPrimaryTrack.Location = new System.Drawing.Point(16, 64);
			this.chkPrimaryTrack.Name = "chkPrimaryTrack";
			this.chkPrimaryTrack.Size = new System.Drawing.Size(112, 16);
			this.chkPrimaryTrack.TabIndex = 37;
			this.chkPrimaryTrack.Text = "Add primary track";
			// 
			// frmParameters
			// 
			this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
			this.ClientSize = new System.Drawing.Size(880, 462);
			this.Controls.AddRange(new System.Windows.Forms.Control[] {
																		  this.groupBox6,
																		  this.groupBox5,
																		  this.cmdCancel,
																		  this.cmdOk,
																		  this.cmdDefault,
																		  this.txtConfigName,
																		  this.label35,
																		  this.groupBox4,
																		  this.groupBox3,
																		  this.groupBox2,
																		  this.groupBox1,
																		  this.cmdHelp});
			this.Name = "frmParameters";
			this.Text = "Edit Parameters";
			this.Load += new System.EventHandler(this.frmParameters_Load);
			this.groupBox1.ResumeLayout(false);
			this.groupBox2.ResumeLayout(false);
			this.groupBox3.ResumeLayout(false);
			this.groupBox4.ResumeLayout(false);
			this.groupBox5.ResumeLayout(false);
			this.groupBox6.ResumeLayout(false);
			this.ResumeLayout(false);

		}
		#endregion

		private void cmdDefault_Click(object sender, System.EventArgs e)
		{
			VolumeGeneration.VolumeGenerator vgen = new VolumeGeneration.VolumeGenerator();
			VConfig = (VolumeGeneration.Configuration)vgen.Config;
			//Assegnazione
			txtConfigName.Text = VConfig.Name;

			txtOBVMaxX.Text = Convert.ToString(VConfig.GeoPar.OutBoundsVolume.MaxX);
			txtOBVMinX.Text = Convert.ToString(VConfig.GeoPar.OutBoundsVolume.MinX);
			txtOBVMaxY.Text = Convert.ToString(VConfig.GeoPar.OutBoundsVolume.MaxY);
			txtOBVMinY.Text = Convert.ToString(VConfig.GeoPar.OutBoundsVolume.MinY);
			txtOBVMaxZ.Text = Convert.ToString(VConfig.GeoPar.OutBoundsVolume.MaxZ);
			txtOBVMinZ.Text = Convert.ToString(VConfig.GeoPar.OutBoundsVolume.MinZ);
			txtVMaxX.Text = Convert.ToString(VConfig.GeoPar.Volume.MaxX);
			txtVMinX.Text = Convert.ToString(VConfig.GeoPar.Volume.MinX);
			txtVMaxY.Text = Convert.ToString(VConfig.GeoPar.Volume.MaxY);
			txtVMinY.Text = Convert.ToString(VConfig.GeoPar.Volume.MinY);
			txtVMaxZ.Text = Convert.ToString(VConfig.GeoPar.Volume.MaxZ);
			txtVMinZ.Text = Convert.ToString(VConfig.GeoPar.Volume.MinZ);
			txtTracking.Text = Convert.ToString(VConfig.GeoPar.TrackingThickness);
			txtNonTracking.Text = Convert.ToString(VConfig.GeoPar.NotTrackingThickness);
			txtMostUpstreamPlate.Text = Convert.ToString(VConfig.GeoPar.MostUpstreamPlane);

			txtLocalVertexDepth.Text = Convert.ToString(VConfig.EvPar.LocalVertexDepth);
			txtOutgoingTracks.Text = Convert.ToString(VConfig.EvPar.OutgoingTracks);
			chkPrimaryTrack.Checked = VConfig.EvPar.PrimaryTrack;
			
			txtCoordErrX.Text = Convert.ToString(VConfig.ErrPar.CoordinateErrors.X);
			txtCoordErrY.Text = Convert.ToString(VConfig.ErrPar.CoordinateErrors.Y);
			//txtCoordAlignX.Text = Convert.ToString(VConfig.ErrPar.CoordinateAlignment.X);
			//txtCoordAlignY.Text = Convert.ToString(VConfig.ErrPar.CoordinateAlignment.Y);
			//txtSlopeAlignX.Text = Convert.ToString(VConfig.ErrPar.SlopeAlignment.X);
			//txtSlopeAlignY.Text = Convert.ToString(VConfig.ErrPar.SlopeAlignment.Y);
			txtSlopeErrX.Text = Convert.ToString(VConfig.ErrPar.SlopeErrors.X);
			txtSlopeErrY.Text = Convert.ToString(VConfig.ErrPar.SlopeErrors.Y);
			txtTFE.Text = Convert.ToString(VConfig.ErrPar.TrackFindingEfficiency);

			txtMinimumEnergyLoss.Text = Convert.ToString(VConfig.KinePar.MinimumEnergyForLoss);
			txtRadiationLength.Text = Convert.ToString(VConfig.KinePar.RadiationLength);

			txtDiagMax.Text = Convert.ToString(VConfig.AffPar.DiagMaxTerm);
			txtDiagMin.Text = Convert.ToString(VConfig.AffPar.DiagMinTerm);
			txtOutDMax.Text = Convert.ToString(VConfig.AffPar.OutDiagMaxTerm);
			txtOutDMin.Text = Convert.ToString(VConfig.AffPar.OutDiagMinTerm);
			txtCoordAlignMax.Text = Convert.ToString(VConfig.AffPar.AlignMaxShift);
			txtCoordAlignMin.Text = Convert.ToString(VConfig.AffPar.AlignMinShift);
			txtSShiftMax.Text = Convert.ToString(VConfig.AffPar.SlopeMaxShift);
			txtSShiftMin.Text = Convert.ToString(VConfig.AffPar.SlopeMinShift);
			txtSCoeffMax.Text = Convert.ToString(VConfig.AffPar.SlopeMaxCoeff);
			txtSCoeffMin.Text = Convert.ToString(VConfig.AffPar.SlopeMinCoeff);
			txtLongAlignMax.Text = Convert.ToString(VConfig.AffPar.LongAlignMaxShift);
			txtLongAlignMin.Text = Convert.ToString(VConfig.AffPar.LongAlignMinShift);

			txtHMTracks.Text = Convert.ToString(VConfig.HighMomentumTracks);
			txtELTracks.Text = Convert.ToString(VConfig.EnergyLossTracks);
			txtJunkTracks.Text = Convert.ToString(VConfig.JunkTracks);
			cmbSlopeXDistrib.SelectedIndex = (int)VConfig.XSlopesDistrib;
			cmbSlopeYDistrib.SelectedIndex = (int)VConfig.YSlopesDistrib;
			cmbMomentumDistrib.SelectedIndex = (int)VConfig.MomentumDistrib;

			

		}

		private void cmdOk_Click(object sender, System.EventArgs e)
		{
			VConfig = new Configuration(txtConfigName.Text);

			//Assegnazione
			VConfig.GeoPar.OutBoundsVolume.MaxX = Convert.ToDouble(txtOBVMaxX.Text);
			VConfig.GeoPar.OutBoundsVolume.MinX = Convert.ToDouble(txtOBVMinX.Text);
			VConfig.GeoPar.OutBoundsVolume.MaxY = Convert.ToDouble(txtOBVMaxY.Text);
			VConfig.GeoPar.OutBoundsVolume.MinY = Convert.ToDouble(txtOBVMinY.Text);
			VConfig.GeoPar.OutBoundsVolume.MaxZ = Convert.ToDouble(txtOBVMaxZ.Text);
			VConfig.GeoPar.OutBoundsVolume.MinZ = Convert.ToDouble(txtOBVMinZ.Text);
			VConfig.GeoPar.Volume.MaxX = Convert.ToDouble(txtVMaxX.Text);
			VConfig.GeoPar.Volume.MinX = Convert.ToDouble(txtVMinX.Text);
			VConfig.GeoPar.Volume.MaxY = Convert.ToDouble(txtVMaxY.Text);
			VConfig.GeoPar.Volume.MinY = Convert.ToDouble(txtVMinY.Text);
			VConfig.GeoPar.Volume.MaxZ = Convert.ToDouble(txtVMaxZ.Text);
			VConfig.GeoPar.Volume.MinZ = Convert.ToDouble(txtVMinZ.Text);
			VConfig.GeoPar.TrackingThickness = Convert.ToDouble(txtTracking.Text);
			VConfig.GeoPar.NotTrackingThickness = Convert.ToDouble(txtNonTracking.Text);
			VConfig.GeoPar.MostUpstreamPlane = Convert.ToInt32(txtMostUpstreamPlate.Text);

			VConfig.ErrPar.CoordinateErrors.X = Convert.ToDouble(txtCoordErrX.Text);
			VConfig.ErrPar.CoordinateErrors.Y = Convert.ToDouble(txtCoordErrY.Text);
			//VConfig.ErrPar.CoordinateAlignment.X = Convert.ToDouble(txtCoordAlignX.Text);
			//VConfig.ErrPar.CoordinateAlignment.Y = Convert.ToDouble(txtCoordAlignY.Text);
			//VConfig.ErrPar.SlopeAlignment.X = Convert.ToDouble(txtSlopeAlignX.Text);
			//VConfig.ErrPar.SlopeAlignment.Y = Convert.ToDouble(txtSlopeAlignY.Text);
			VConfig.ErrPar.SlopeErrors.X = Convert.ToDouble(txtSlopeErrX.Text);
			VConfig.ErrPar.SlopeErrors.Y = Convert.ToDouble(txtSlopeErrY.Text);
			VConfig.ErrPar.TrackFindingEfficiency = Convert.ToDouble(txtTFE.Text);

			VConfig.EvPar.LocalVertexDepth = Convert.ToDouble(txtLocalVertexDepth.Text);
			VConfig.EvPar.OutgoingTracks = Convert.ToInt32(txtOutgoingTracks.Text);
			VConfig.EvPar.PrimaryTrack = chkPrimaryTrack.Checked;

			VConfig.KinePar.MinimumEnergyForLoss = Convert.ToDouble(txtMinimumEnergyLoss.Text);
			VConfig.KinePar.RadiationLength = Convert.ToDouble(txtRadiationLength.Text);

			VConfig.AffPar.DiagMaxTerm = Convert.ToDouble (txtDiagMax.Text);
			VConfig.AffPar.DiagMinTerm = Convert.ToDouble (txtDiagMin.Text);
			VConfig.AffPar.OutDiagMaxTerm =  Convert.ToDouble (txtOutDMax.Text);
			VConfig.AffPar.OutDiagMinTerm = Convert.ToDouble (txtOutDMin.Text);
			VConfig.AffPar.AlignMaxShift =  Convert.ToDouble (txtCoordAlignMax.Text);
			VConfig.AffPar.AlignMinShift =  Convert.ToDouble (txtCoordAlignMin.Text);
			VConfig.AffPar.SlopeMaxShift =  Convert.ToDouble (txtSShiftMax.Text);
			VConfig.AffPar.SlopeMinShift = Convert.ToDouble ( txtSShiftMin.Text);
			VConfig.AffPar.SlopeMaxCoeff =  Convert.ToDouble (txtSCoeffMax.Text);
			VConfig.AffPar.SlopeMinCoeff =  Convert.ToDouble (txtSCoeffMin.Text);
			VConfig.AffPar.LongAlignMaxShift = Convert.ToDouble(txtLongAlignMax.Text);
			VConfig.AffPar.LongAlignMinShift = Convert.ToDouble(txtLongAlignMin.Text);

			VConfig.HighMomentumTracks = Convert.ToInt32(txtHMTracks.Text);
			VConfig.EnergyLossTracks = Convert.ToInt32(txtELTracks.Text);
			VConfig.JunkTracks = Convert.ToInt32(txtJunkTracks.Text);
			VConfig.XSlopesDistrib= (VolumeGeneration.Distribution)cmbSlopeXDistrib.SelectedIndex;
			VConfig.YSlopesDistrib= (VolumeGeneration.Distribution)cmbSlopeYDistrib.SelectedIndex;
			VConfig.MomentumDistrib= (VolumeGeneration.Distribution)cmbMomentumDistrib.SelectedIndex;

			DialogResult = DialogResult.OK;
			Close();

		}

		private void frmParameters_Load(object sender, System.EventArgs e)
		{
			//Assegnazione
			txtConfigName.Text = VConfig.Name;

			txtOBVMaxX.Text = Convert.ToString(VConfig.GeoPar.OutBoundsVolume.MaxX);
			txtOBVMinX.Text = Convert.ToString(VConfig.GeoPar.OutBoundsVolume.MinX);
			txtOBVMaxY.Text = Convert.ToString(VConfig.GeoPar.OutBoundsVolume.MaxY);
			txtOBVMinY.Text = Convert.ToString(VConfig.GeoPar.OutBoundsVolume.MinY);
			txtOBVMaxZ.Text = Convert.ToString(VConfig.GeoPar.OutBoundsVolume.MaxZ);
			txtOBVMinZ.Text = Convert.ToString(VConfig.GeoPar.OutBoundsVolume.MinZ);
			txtVMaxX.Text = Convert.ToString(VConfig.GeoPar.Volume.MaxX);
			txtVMinX.Text = Convert.ToString(VConfig.GeoPar.Volume.MinX);
			txtVMaxY.Text = Convert.ToString(VConfig.GeoPar.Volume.MaxY);
			txtVMinY.Text = Convert.ToString(VConfig.GeoPar.Volume.MinY);
			txtVMaxZ.Text = Convert.ToString(VConfig.GeoPar.Volume.MaxZ);
			txtVMinZ.Text = Convert.ToString(VConfig.GeoPar.Volume.MinZ);
			txtTracking.Text = Convert.ToString(VConfig.GeoPar.TrackingThickness);
			txtNonTracking.Text = Convert.ToString(VConfig.GeoPar.NotTrackingThickness);
			txtMostUpstreamPlate.Text = Convert.ToString(VConfig.GeoPar.MostUpstreamPlane);

			txtCoordErrX.Text = Convert.ToString(VConfig.ErrPar.CoordinateErrors.X);
			txtCoordErrY.Text = Convert.ToString(VConfig.ErrPar.CoordinateErrors.Y);
			//txtCoordAlignX.Text = Convert.ToString(VConfig.ErrPar.CoordinateAlignment.X);
			//txtCoordAlignY.Text = Convert.ToString(VConfig.ErrPar.CoordinateAlignment.Y);
			//txtSlopeAlignX.Text = Convert.ToString(VConfig.ErrPar.SlopeAlignment.X);
			//txtSlopeAlignY.Text = Convert.ToString(VConfig.ErrPar.SlopeAlignment.Y);
			txtSlopeErrX.Text = Convert.ToString(VConfig.ErrPar.SlopeErrors.X);
			txtSlopeErrY.Text = Convert.ToString(VConfig.ErrPar.SlopeErrors.Y);
			txtTFE.Text = Convert.ToString(VConfig.ErrPar.TrackFindingEfficiency);

			txtLocalVertexDepth.Text = Convert.ToString(VConfig.EvPar.LocalVertexDepth);
			txtOutgoingTracks.Text = Convert.ToString(VConfig.EvPar.OutgoingTracks);
			chkPrimaryTrack.Checked = VConfig.EvPar.PrimaryTrack;

			txtMinimumEnergyLoss.Text = Convert.ToString(VConfig.KinePar.MinimumEnergyForLoss);
			txtRadiationLength.Text = Convert.ToString(VConfig.KinePar.RadiationLength);
		
			txtDiagMax.Text = Convert.ToString(VConfig.AffPar.DiagMaxTerm);
			txtDiagMin.Text = Convert.ToString(VConfig.AffPar.DiagMinTerm);
			txtOutDMax.Text = Convert.ToString(VConfig.AffPar.OutDiagMaxTerm);
			txtOutDMin.Text = Convert.ToString(VConfig.AffPar.OutDiagMinTerm);
			txtCoordAlignMax.Text = Convert.ToString(VConfig.AffPar.AlignMaxShift);
			txtCoordAlignMin.Text = Convert.ToString(VConfig.AffPar.AlignMinShift);
			txtSShiftMax.Text = Convert.ToString(VConfig.AffPar.SlopeMaxShift);
			txtSShiftMin.Text = Convert.ToString(VConfig.AffPar.SlopeMinShift);
			txtSCoeffMax.Text = Convert.ToString(VConfig.AffPar.SlopeMaxCoeff);
			txtSCoeffMin.Text = Convert.ToString(VConfig.AffPar.SlopeMinCoeff);
			txtLongAlignMax.Text = Convert.ToString(VConfig.AffPar.LongAlignMaxShift);
			txtLongAlignMin.Text = Convert.ToString(VConfig.AffPar.LongAlignMinShift);

			txtHMTracks.Text = Convert.ToString(VConfig.HighMomentumTracks);
			txtELTracks.Text = Convert.ToString(VConfig.EnergyLossTracks);
			txtJunkTracks.Text = Convert.ToString(VConfig.JunkTracks);

			cmbSlopeXDistrib.Items.Add(VolumeGeneration.Distribution.Flat);
			cmbSlopeXDistrib.Items.Add(VolumeGeneration.Distribution.Gaussian);
			cmbSlopeXDistrib.Items.Add(VolumeGeneration.Distribution.Custom);
			cmbSlopeXDistrib.Items.Add(VolumeGeneration.Distribution.SingleValue);
			cmbSlopeXDistrib.SelectedIndex = (int)VConfig.XSlopesDistrib;

			cmbSlopeYDistrib.Items.Add(VolumeGeneration.Distribution.Flat);
			cmbSlopeYDistrib.Items.Add(VolumeGeneration.Distribution.Gaussian);
			cmbSlopeYDistrib.Items.Add(VolumeGeneration.Distribution.Custom);
			cmbSlopeYDistrib.Items.Add(VolumeGeneration.Distribution.SingleValue);
			cmbSlopeYDistrib.SelectedIndex = (int)VConfig.YSlopesDistrib;

			cmbMomentumDistrib.Items.Add(VolumeGeneration.Distribution.Flat);
			cmbMomentumDistrib.Items.Add(VolumeGeneration.Distribution.Gaussian);
			cmbMomentumDistrib.Items.Add(VolumeGeneration.Distribution.Custom);
			cmbMomentumDistrib.Items.Add(VolumeGeneration.Distribution.SingleValue);
			cmbMomentumDistrib.SelectedIndex = (int)VConfig.MomentumDistrib;
			
		}

		private void txtOBVMaxX_TextChanged(object sender, System.EventArgs e)
		{
			try
			{
				double den = ( (Convert.ToDouble(txtOBVMaxX.Text)-Convert.ToDouble(txtOBVMinX.Text))*
					(Convert.ToDouble(txtOBVMaxY.Text)-Convert.ToDouble(txtOBVMinY.Text)) )/1000000;
				lblHMdensity.Text = "HM-Track density: " + (Convert.ToDouble(txtHMTracks.Text)/den).ToString("#.##") +
					" tr/mm^2";
				lblELdensity.Text = "EL-Track density: " + (Convert.ToDouble(txtELTracks.Text)/den).ToString("#.##") +
					" tr/mm^2";
				lblJunkdensity.Text = "Junk Track density: " + (Convert.ToDouble(txtJunkTracks.Text)/den).ToString("#.##") +
					" tr/mm^2";
			}
			catch
			{
			
			}
		}

		private void txtOBVMinX_TextChanged(object sender, System.EventArgs e)
		{
			try
			{
				double den = ( (Convert.ToDouble(txtOBVMaxX.Text)-Convert.ToDouble(txtOBVMinX.Text))*
					(Convert.ToDouble(txtOBVMaxY.Text)-Convert.ToDouble(txtOBVMinY.Text)) )/1000000;
				lblHMdensity.Text = "HM-Track density: " + (Convert.ToDouble(txtHMTracks.Text)/den).ToString("#.##") +
					" tr/mm^2";
				lblELdensity.Text = "EL-Track density: " + (Convert.ToDouble(txtELTracks.Text)/den).ToString("#.##") +
					" tr/mm^2";
				lblJunkdensity.Text = "Junk Track density: " + (Convert.ToDouble(txtJunkTracks.Text)/den).ToString("#.##") +
					" tr/mm^2";
			}
			catch
			{
			
			}
		
		}

		private void txtOBVMaxY_TextChanged(object sender, System.EventArgs e)
		{
			try
			{
				double den = ( (Convert.ToDouble(txtOBVMaxX.Text)-Convert.ToDouble(txtOBVMinX.Text))*
					(Convert.ToDouble(txtOBVMaxY.Text)-Convert.ToDouble(txtOBVMinY.Text)) )/1000000;
				lblHMdensity.Text = "HM-Track density: " + (Convert.ToDouble(txtHMTracks.Text)/den).ToString("#.##") +
					" tr/mm^2";
				lblELdensity.Text = "EL-Track density: " + (Convert.ToDouble(txtELTracks.Text)/den).ToString("#.##") +
					" tr/mm^2";
				lblJunkdensity.Text = "Junk Track density: " + (Convert.ToDouble(txtJunkTracks.Text)/den).ToString("#.##") +
					" tr/mm^2";
			}
			catch
			{
			
			}

		}

		private void txtOBVMinY_TextChanged(object sender, System.EventArgs e)
		{
			try
			{
				double den = ( (Convert.ToDouble(txtOBVMaxX.Text)-Convert.ToDouble(txtOBVMinX.Text))*
					(Convert.ToDouble(txtOBVMaxY.Text)-Convert.ToDouble(txtOBVMinY.Text)) )/1000000;
				lblHMdensity.Text = "HM-Track density: " + (Convert.ToDouble(txtHMTracks.Text)/den).ToString("#.##") +
					" tr/mm^2";
				lblELdensity.Text = "EL-Track density: " + (Convert.ToDouble(txtELTracks.Text)/den).ToString("#.##") +
					" tr/mm^2";
				lblJunkdensity.Text = "Junk Track density: " + (Convert.ToDouble(txtJunkTracks.Text)/den).ToString("#.##") +
					" tr/mm^2";
			}
			catch
			{
			
			}

		}

		private void txtHMTracks_TextChanged(object sender, System.EventArgs e)
		{
			try
			{
				double den = ( (Convert.ToDouble(txtOBVMaxX.Text)-Convert.ToDouble(txtOBVMinX.Text))*
					(Convert.ToDouble(txtOBVMaxY.Text)-Convert.ToDouble(txtOBVMinY.Text)) )/1000000;
				lblHMdensity.Text = "HM-Track density: " + (Convert.ToDouble(txtHMTracks.Text)/den).ToString("#.##") +
					" tr/mm^2";
			}
			catch
			{
			
			}

		}

		private void txtELTracks_TextChanged(object sender, System.EventArgs e)
		{
			try
			{
				double den = ( (Convert.ToDouble(txtOBVMaxX.Text)-Convert.ToDouble(txtOBVMinX.Text))*
					(Convert.ToDouble(txtOBVMaxY.Text)-Convert.ToDouble(txtOBVMinY.Text)) )/1000000;
				lblELdensity.Text = "EL-Track density: " + (Convert.ToDouble(txtELTracks.Text)/den).ToString("#.##") +
					" tr/mm^2";
			}
			catch
			{
			
			}

		}

		private void txtJunkTracks_TextChanged(object sender, System.EventArgs e)
		{
			try
			{
				double den = ( (Convert.ToDouble(txtOBVMaxX.Text)-Convert.ToDouble(txtOBVMinX.Text))*
					(Convert.ToDouble(txtOBVMaxY.Text)-Convert.ToDouble(txtOBVMinY.Text)) )/1000000;
				lblJunkdensity.Text = "Junk Track density: " + (Convert.ToDouble(txtJunkTracks.Text)/den).ToString("#.##") +
					" tr/mm^2";
			}
			catch
			{
			
			}

		}

		private void ParamList_SelectedIndexChanged(object sender, System.EventArgs e)
		{
		
		}

		private void cmbSlopeXDistrib_SelectedIndexChanged(object sender, System.EventArgs e)
		{
			VConfig.XSlopesDistrib = (VolumeGeneration.Distribution)cmbSlopeXDistrib.SelectedIndex;
			int i;
			lstSlopeX.Items.Clear();
			for (i = 0; i < VConfig.XSlopesDistribParameters.Length; i++)
			{
				int index = lstSlopeX.Items.Add("Par #" + i).Index;
				lstSlopeX.Items[index].SubItems.Add(VConfig.XSlopesDistribParameters[i].ToString());
			}
	
		}

		private void cmbSlopeYDistrib_SelectedIndexChanged(object sender, System.EventArgs e)
		{
			VConfig.YSlopesDistrib = (VolumeGeneration.Distribution)cmbSlopeYDistrib.SelectedIndex;
			int i;
			lstSlopeY.Items.Clear();
			for (i = 0; i < VConfig.YSlopesDistribParameters.Length; i++)
			{
				int index = lstSlopeY.Items.Add("Par #" + i).Index;
				lstSlopeY.Items[index].SubItems.Add(VConfig.YSlopesDistribParameters[i].ToString());
			}
		
		}

		private void cmbMomentumDistrib_SelectedIndexChanged(object sender, System.EventArgs e)
		{
			VConfig.MomentumDistrib = (VolumeGeneration.Distribution)cmbMomentumDistrib.SelectedIndex;
			int i;
			lstMomentum.Items.Clear();
			for (i = 0; i < VConfig.MomentumDistribParameters.Length; i++)
			{
				int index = lstMomentum.Items.Add("Par #" + i).Index;
				lstMomentum.Items[index].SubItems.Add(VConfig.MomentumDistribParameters[i].ToString());
			}

		}

		private void cmdHelp_Click(object sender, System.EventArgs e)
		{
			string chmpath = System.Diagnostics.Process.GetCurrentProcess().MainModule.FileName;
			chmpath = chmpath.Remove(chmpath.Length - 3, 3) + "chm";
			System.Windows.Forms.Help.ShowHelp(this, chmpath, "VolGenConfig.htm");

		}
	}
}
