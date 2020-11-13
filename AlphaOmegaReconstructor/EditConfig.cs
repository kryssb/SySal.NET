using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;
using SySal;
using SySal.TotalScan;
using NumericalTools;

namespace SySal.Processing.AlphaOmegaReconstruction
{
	/// <summary>
	/// Summary description for frmAORecEditConfig.
	/// </summary>
	internal class frmAORecEditConfig : System.Windows.Forms.Form
    {
		private System.Windows.Forms.Label label1;
		private System.Windows.Forms.Label label2;
		private System.Windows.Forms.Label label3;
		private System.Windows.Forms.Label label4;
		private System.Windows.Forms.ComboBox cmbPrescanMode;
		private System.Windows.Forms.TextBox txtLeverArm;
        private System.Windows.Forms.TextBox txtZoneWidth;
		private System.Windows.Forms.Label label5;
		private System.Windows.Forms.Label label6;
		private System.Windows.Forms.Label label7;
		private System.Windows.Forms.Label label8;
		private System.Windows.Forms.TextBox txtPosTol;
		private System.Windows.Forms.TextBox txtSlopeTol;
		private System.Windows.Forms.TextBox txtInitialPosTol;
		private System.Windows.Forms.TextBox txtInitialSlopeTol;
		private System.Windows.Forms.Label label10;
		private System.Windows.Forms.Label label11;
		private System.Windows.Forms.Label label12;
		private System.Windows.Forms.Label label9;
		private System.Windows.Forms.Label label13;
		private System.Windows.Forms.Label label14;
		private System.Windows.Forms.TextBox txtMaxIterNum;
		private System.Windows.Forms.TextBox txtMaxMissSeg;
		private System.Windows.Forms.Label label16;
		private System.Windows.Forms.TextBox txtConfigName;
		private System.Windows.Forms.Button cmdDefault;
		private System.Windows.Forms.Button cmdOk;
		private System.Windows.Forms.Button cmdCancel;
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;
		private System.Windows.Forms.TextBox txtBeamSY;
		private System.Windows.Forms.TextBox txtBeamSX;
		private System.Windows.Forms.TextBox txtBeamWid;
		private System.Windows.Forms.TextBox txtLocCellSize;
		private System.Windows.Forms.CheckBox chkAlignOnLink;
		private System.Windows.Forms.TextBox txtExtents;
		private System.Windows.Forms.GroupBox groupBox3;
		private System.Windows.Forms.Label label17;
		private System.Windows.Forms.TextBox txtRiskFactor;
		private System.Windows.Forms.TextBox txtSlopesCellSizeX;
		private System.Windows.Forms.Label label18;
		private System.Windows.Forms.TextBox txtSlopesCellSizeY;
		private System.Windows.Forms.Label label19;
		private System.Windows.Forms.TextBox txtMaxShiftX;
		private System.Windows.Forms.Label label20;
		private System.Windows.Forms.TextBox txtMaxShiftY;
		private System.Windows.Forms.Label label21;
		private System.Windows.Forms.Button cmdHelp;
		private System.Windows.Forms.TextBox txtSlopeIncrement;
		private System.Windows.Forms.TextBox txtPosIncrement;
		private System.Windows.Forms.Label label29;
		private System.Windows.Forms.Label label30;
		private System.Windows.Forms.CheckBox chkCorrectSlopes;
		private System.Windows.Forms.CheckBox chkZfixed;
		private System.Windows.Forms.GroupBox groupBox5;
		private System.Windows.Forms.Label label33;
		private System.Windows.Forms.TextBox txtMinTracksPairs;
		private System.Windows.Forms.Label label34;
		private System.Windows.Forms.TextBox txtMinSegNumber;
		private System.Windows.Forms.CheckBox chkUpdateTrans;
		private System.Windows.Forms.RadioButton radKalman;
		private System.Windows.Forms.RadioButton radTrackFit;
		private System.Windows.Forms.TextBox txtFittingSegments;
		private System.Windows.Forms.Label label15;
		private System.Windows.Forms.Label label39;
		private System.Windows.Forms.TextBox txtMinKalman;
		private System.Windows.Forms.Label label40;
		private System.Windows.Forms.TextBox txtMinimumCritical;
		private System.Windows.Forms.CheckBox chkKinkDetection;
		private System.Windows.Forms.GroupBox groupBox7;
		private System.Windows.Forms.Label label41;
		private System.Windows.Forms.Label label42;
		private System.Windows.Forms.TextBox txtKinkMinSlopeDiff;
		private System.Windows.Forms.TextBox txtKinkMinSeg;
		private System.Windows.Forms.Label label43;
		private System.Windows.Forms.TextBox txtKinkFactor;
		private System.Windows.Forms.TextBox txtKinkFilterThreshold;
		private System.Windows.Forms.Label label44;
		private System.Windows.Forms.Label label45;
		private System.Windows.Forms.ComboBox cmbFilterLength;
		private System.Windows.Forms.TabControl VtxTab;
		private System.Windows.Forms.TabPage tabPage1;
		private System.Windows.Forms.TabPage tabPage2;
		private System.Windows.Forms.TextBox txtMaximumZ;
		private System.Windows.Forms.CheckBox chkUseCells;
		private System.Windows.Forms.GroupBox groupBox6;
		private System.Windows.Forms.TextBox txtMatrix;
		private System.Windows.Forms.Label label38;
		private System.Windows.Forms.TextBox txtZCellsSize;
		private System.Windows.Forms.Label label37;
		private System.Windows.Forms.TextBox txtYCellsSize;
		private System.Windows.Forms.TextBox txtXCellsSize;
		private System.Windows.Forms.Label label36;
		private System.Windows.Forms.Label label35;
		private System.Windows.Forms.Label label32;
		private System.Windows.Forms.CheckBox chkTopologyKink;
		private System.Windows.Forms.CheckBox chkTopologyLambda;
		private System.Windows.Forms.CheckBox chkTopologyY;
		private System.Windows.Forms.CheckBox chkTopologyX;
		private System.Windows.Forms.CheckBox chkTopologyV;
		private System.Windows.Forms.TextBox txtMinVertexTrackSegments;
		private System.Windows.Forms.Label label31;
		private System.Windows.Forms.TextBox txtStartingClusterToleranceLong;
		private System.Windows.Forms.TextBox txtMinimumZ;
		private System.Windows.Forms.Label label22;
		private System.Windows.Forms.Label label24;
		private System.Windows.Forms.TextBox txtMaximumClusterToleranceLong;
		private System.Windows.Forms.TextBox txtCrossTolerance;
		private System.Windows.Forms.Label label25;
		private System.Windows.Forms.Label label26;
		private System.Windows.Forms.Label label23;
		private System.Windows.Forms.TextBox txtMaximumClusterToleranceTrans;
		private System.Windows.Forms.TextBox txtStartingClusterToleranceTrans;
		private System.Windows.Forms.Label label27;
		private System.Windows.Forms.Label label28;
		private System.Windows.Forms.Label label46;
		private System.Windows.Forms.Label label47;
		private System.Windows.Forms.Label label48;
		private System.Windows.Forms.Label label49;
		private System.Windows.Forms.TabPage tabPage3;
		private System.Windows.Forms.Label label50;
		private System.Windows.Forms.TextBox txtGVtxMinCount;
		private System.Windows.Forms.TextBox txtGVtxRadius;
		private System.Windows.Forms.TextBox txtGVtxMaxExt;
		private System.Windows.Forms.Label label51;
		private System.Windows.Forms.TextBox txtGVtxMaxSlopeDivergence;
		private System.Windows.Forms.GroupBox groupBox4;
		private System.Windows.Forms.CheckBox VtxFitWeightEnableCheck;
		private System.Windows.Forms.Label label52;
		private System.Windows.Forms.TextBox VtxFitWeightTolText;
		private System.Windows.Forms.Label label53;
		private System.Windows.Forms.TextBox VtxFitWeightZStepText;
		private System.Windows.Forms.Label label54;
		private System.Windows.Forms.TextBox VtxFitWeightXYStepText;
        private TextBox txtRelinkAperture;
        private Label label55;
        private TextBox txtRelinkDeltaZ;
        private Label label56;
        private CheckBox chkRelinkEnable;
        private TextBox txtGVtxFilter;
        private Label label57;
        private TextBox txtTrackFilter;
        private Label label58;
        private TextBox txtExtraTrackIters;
        private Label label59;
        private TextBox txtCleaningError;
        private Label label61;
        private TextBox txtCleaningChi2Limit;
        private Label label60;
        private TabControl TkTab;
        private TabPage tabPage6;
        private TabPage tabPage7;
        private TabPage tabPage8;
        private TabPage tabPage9;
        private Label label63;
        private Label label64;
        private Label label65;
        private Label label66;
        private TextBox txtTrackAlignMinTrackSegments;
        private TextBox txtTrackAlignTranslationStep;
        private TextBox txtTrackAlignTranslationSweep;
        private TextBox txtTrackAlignRotationStep;
        private TextBox txtTrackAlignOptAcceptance;
        private TextBox txtTrackAlignRotationSweep;
        private Label label67;
        private Label label68;
        private Label label62;
        private Label label69;
        private TextBox txtTrackAlignMinLayerSegments;
        private CheckBox chkIgnoreBadLayers;

		//Local Variables
		public AlphaOmegaReconstruction.Configuration AOConfig;

		/// <summary>
		/// Form to edit configuration
		/// </summary>
		public frmAORecEditConfig()
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
            this.txtMaxShiftY = new System.Windows.Forms.TextBox();
            this.label21 = new System.Windows.Forms.Label();
            this.txtMaxShiftX = new System.Windows.Forms.TextBox();
            this.label20 = new System.Windows.Forms.Label();
            this.txtExtents = new System.Windows.Forms.TextBox();
            this.txtZoneWidth = new System.Windows.Forms.TextBox();
            this.txtLeverArm = new System.Windows.Forms.TextBox();
            this.cmbPrescanMode = new System.Windows.Forms.ComboBox();
            this.label4 = new System.Windows.Forms.Label();
            this.label3 = new System.Windows.Forms.Label();
            this.label2 = new System.Windows.Forms.Label();
            this.label1 = new System.Windows.Forms.Label();
            this.txtCleaningChi2Limit = new System.Windows.Forms.TextBox();
            this.label60 = new System.Windows.Forms.Label();
            this.txtCleaningError = new System.Windows.Forms.TextBox();
            this.label61 = new System.Windows.Forms.Label();
            this.txtExtraTrackIters = new System.Windows.Forms.TextBox();
            this.label59 = new System.Windows.Forms.Label();
            this.txtTrackFilter = new System.Windows.Forms.TextBox();
            this.label58 = new System.Windows.Forms.Label();
            this.txtMinimumCritical = new System.Windows.Forms.TextBox();
            this.label40 = new System.Windows.Forms.Label();
            this.txtMinKalman = new System.Windows.Forms.TextBox();
            this.label39 = new System.Windows.Forms.Label();
            this.txtFittingSegments = new System.Windows.Forms.TextBox();
            this.radKalman = new System.Windows.Forms.RadioButton();
            this.chkZfixed = new System.Windows.Forms.CheckBox();
            this.chkCorrectSlopes = new System.Windows.Forms.CheckBox();
            this.txtSlopeIncrement = new System.Windows.Forms.TextBox();
            this.txtPosIncrement = new System.Windows.Forms.TextBox();
            this.label29 = new System.Windows.Forms.Label();
            this.label30 = new System.Windows.Forms.Label();
            this.txtMaxMissSeg = new System.Windows.Forms.TextBox();
            this.txtMaxIterNum = new System.Windows.Forms.TextBox();
            this.label14 = new System.Windows.Forms.Label();
            this.label13 = new System.Windows.Forms.Label();
            this.chkAlignOnLink = new System.Windows.Forms.CheckBox();
            this.txtBeamSY = new System.Windows.Forms.TextBox();
            this.txtBeamSX = new System.Windows.Forms.TextBox();
            this.txtBeamWid = new System.Windows.Forms.TextBox();
            this.txtLocCellSize = new System.Windows.Forms.TextBox();
            this.label10 = new System.Windows.Forms.Label();
            this.label11 = new System.Windows.Forms.Label();
            this.label12 = new System.Windows.Forms.Label();
            this.txtInitialSlopeTol = new System.Windows.Forms.TextBox();
            this.txtInitialPosTol = new System.Windows.Forms.TextBox();
            this.txtSlopeTol = new System.Windows.Forms.TextBox();
            this.txtPosTol = new System.Windows.Forms.TextBox();
            this.label7 = new System.Windows.Forms.Label();
            this.label8 = new System.Windows.Forms.Label();
            this.label6 = new System.Windows.Forms.Label();
            this.label5 = new System.Windows.Forms.Label();
            this.label9 = new System.Windows.Forms.Label();
            this.label15 = new System.Windows.Forms.Label();
            this.radTrackFit = new System.Windows.Forms.RadioButton();
            this.label16 = new System.Windows.Forms.Label();
            this.txtConfigName = new System.Windows.Forms.TextBox();
            this.cmdDefault = new System.Windows.Forms.Button();
            this.cmdOk = new System.Windows.Forms.Button();
            this.cmdCancel = new System.Windows.Forms.Button();
            this.groupBox3 = new System.Windows.Forms.GroupBox();
            this.txtSlopesCellSizeY = new System.Windows.Forms.TextBox();
            this.label19 = new System.Windows.Forms.Label();
            this.txtSlopesCellSizeX = new System.Windows.Forms.TextBox();
            this.label18 = new System.Windows.Forms.Label();
            this.txtRiskFactor = new System.Windows.Forms.TextBox();
            this.label17 = new System.Windows.Forms.Label();
            this.cmdHelp = new System.Windows.Forms.Button();
            this.chkUpdateTrans = new System.Windows.Forms.CheckBox();
            this.groupBox5 = new System.Windows.Forms.GroupBox();
            this.chkRelinkEnable = new System.Windows.Forms.CheckBox();
            this.txtRelinkDeltaZ = new System.Windows.Forms.TextBox();
            this.txtRelinkAperture = new System.Windows.Forms.TextBox();
            this.label56 = new System.Windows.Forms.Label();
            this.txtMinSegNumber = new System.Windows.Forms.TextBox();
            this.label55 = new System.Windows.Forms.Label();
            this.txtMinTracksPairs = new System.Windows.Forms.TextBox();
            this.label33 = new System.Windows.Forms.Label();
            this.label34 = new System.Windows.Forms.Label();
            this.chkKinkDetection = new System.Windows.Forms.CheckBox();
            this.groupBox7 = new System.Windows.Forms.GroupBox();
            this.label45 = new System.Windows.Forms.Label();
            this.cmbFilterLength = new System.Windows.Forms.ComboBox();
            this.txtKinkFilterThreshold = new System.Windows.Forms.TextBox();
            this.label44 = new System.Windows.Forms.Label();
            this.txtKinkFactor = new System.Windows.Forms.TextBox();
            this.label43 = new System.Windows.Forms.Label();
            this.txtKinkMinSlopeDiff = new System.Windows.Forms.TextBox();
            this.label42 = new System.Windows.Forms.Label();
            this.txtKinkMinSeg = new System.Windows.Forms.TextBox();
            this.label41 = new System.Windows.Forms.Label();
            this.VtxTab = new System.Windows.Forms.TabControl();
            this.tabPage3 = new System.Windows.Forms.TabPage();
            this.label50 = new System.Windows.Forms.Label();
            this.tabPage1 = new System.Windows.Forms.TabPage();
            this.txtMaximumZ = new System.Windows.Forms.TextBox();
            this.chkUseCells = new System.Windows.Forms.CheckBox();
            this.groupBox6 = new System.Windows.Forms.GroupBox();
            this.txtMatrix = new System.Windows.Forms.TextBox();
            this.label38 = new System.Windows.Forms.Label();
            this.txtZCellsSize = new System.Windows.Forms.TextBox();
            this.label37 = new System.Windows.Forms.Label();
            this.txtYCellsSize = new System.Windows.Forms.TextBox();
            this.txtXCellsSize = new System.Windows.Forms.TextBox();
            this.label36 = new System.Windows.Forms.Label();
            this.label35 = new System.Windows.Forms.Label();
            this.label32 = new System.Windows.Forms.Label();
            this.chkTopologyKink = new System.Windows.Forms.CheckBox();
            this.chkTopologyLambda = new System.Windows.Forms.CheckBox();
            this.chkTopologyY = new System.Windows.Forms.CheckBox();
            this.chkTopologyX = new System.Windows.Forms.CheckBox();
            this.chkTopologyV = new System.Windows.Forms.CheckBox();
            this.txtMinVertexTrackSegments = new System.Windows.Forms.TextBox();
            this.label31 = new System.Windows.Forms.Label();
            this.txtStartingClusterToleranceLong = new System.Windows.Forms.TextBox();
            this.txtMinimumZ = new System.Windows.Forms.TextBox();
            this.label22 = new System.Windows.Forms.Label();
            this.label24 = new System.Windows.Forms.Label();
            this.txtMaximumClusterToleranceLong = new System.Windows.Forms.TextBox();
            this.txtCrossTolerance = new System.Windows.Forms.TextBox();
            this.label25 = new System.Windows.Forms.Label();
            this.label26 = new System.Windows.Forms.Label();
            this.label23 = new System.Windows.Forms.Label();
            this.txtMaximumClusterToleranceTrans = new System.Windows.Forms.TextBox();
            this.txtStartingClusterToleranceTrans = new System.Windows.Forms.TextBox();
            this.label27 = new System.Windows.Forms.Label();
            this.label28 = new System.Windows.Forms.Label();
            this.tabPage2 = new System.Windows.Forms.TabPage();
            this.txtGVtxFilter = new System.Windows.Forms.TextBox();
            this.txtGVtxMinCount = new System.Windows.Forms.TextBox();
            this.label57 = new System.Windows.Forms.Label();
            this.label46 = new System.Windows.Forms.Label();
            this.txtGVtxMaxSlopeDivergence = new System.Windows.Forms.TextBox();
            this.txtGVtxRadius = new System.Windows.Forms.TextBox();
            this.label47 = new System.Windows.Forms.Label();
            this.txtGVtxMaxExt = new System.Windows.Forms.TextBox();
            this.label48 = new System.Windows.Forms.Label();
            this.label49 = new System.Windows.Forms.Label();
            this.label51 = new System.Windows.Forms.Label();
            this.groupBox4 = new System.Windows.Forms.GroupBox();
            this.VtxFitWeightZStepText = new System.Windows.Forms.TextBox();
            this.label54 = new System.Windows.Forms.Label();
            this.VtxFitWeightXYStepText = new System.Windows.Forms.TextBox();
            this.label53 = new System.Windows.Forms.Label();
            this.VtxFitWeightTolText = new System.Windows.Forms.TextBox();
            this.label52 = new System.Windows.Forms.Label();
            this.VtxFitWeightEnableCheck = new System.Windows.Forms.CheckBox();
            this.TkTab = new System.Windows.Forms.TabControl();
            this.tabPage6 = new System.Windows.Forms.TabPage();
            this.tabPage7 = new System.Windows.Forms.TabPage();
            this.tabPage8 = new System.Windows.Forms.TabPage();
            this.label69 = new System.Windows.Forms.Label();
            this.txtTrackAlignMinLayerSegments = new System.Windows.Forms.TextBox();
            this.label63 = new System.Windows.Forms.Label();
            this.label64 = new System.Windows.Forms.Label();
            this.label65 = new System.Windows.Forms.Label();
            this.label66 = new System.Windows.Forms.Label();
            this.txtTrackAlignMinTrackSegments = new System.Windows.Forms.TextBox();
            this.txtTrackAlignTranslationStep = new System.Windows.Forms.TextBox();
            this.txtTrackAlignTranslationSweep = new System.Windows.Forms.TextBox();
            this.txtTrackAlignRotationStep = new System.Windows.Forms.TextBox();
            this.txtTrackAlignOptAcceptance = new System.Windows.Forms.TextBox();
            this.txtTrackAlignRotationSweep = new System.Windows.Forms.TextBox();
            this.label67 = new System.Windows.Forms.Label();
            this.label68 = new System.Windows.Forms.Label();
            this.tabPage9 = new System.Windows.Forms.TabPage();
            this.label62 = new System.Windows.Forms.Label();
            this.chkIgnoreBadLayers = new System.Windows.Forms.CheckBox();
            this.groupBox3.SuspendLayout();
            this.groupBox5.SuspendLayout();
            this.groupBox7.SuspendLayout();
            this.VtxTab.SuspendLayout();
            this.tabPage3.SuspendLayout();
            this.tabPage1.SuspendLayout();
            this.groupBox6.SuspendLayout();
            this.tabPage2.SuspendLayout();
            this.groupBox4.SuspendLayout();
            this.TkTab.SuspendLayout();
            this.tabPage6.SuspendLayout();
            this.tabPage7.SuspendLayout();
            this.tabPage8.SuspendLayout();
            this.tabPage9.SuspendLayout();
            this.SuspendLayout();
            // 
            // txtMaxShiftY
            // 
            this.txtMaxShiftY.Location = new System.Drawing.Point(90, 138);
            this.txtMaxShiftY.Name = "txtMaxShiftY";
            this.txtMaxShiftY.Size = new System.Drawing.Size(88, 20);
            this.txtMaxShiftY.TabIndex = 11;
            // 
            // label21
            // 
            this.label21.AutoSize = true;
            this.label21.Location = new System.Drawing.Point(18, 140);
            this.label21.Name = "label21";
            this.label21.Size = new System.Drawing.Size(64, 13);
            this.label21.TabIndex = 10;
            this.label21.Text = "Max Shift Y:";
            // 
            // txtMaxShiftX
            // 
            this.txtMaxShiftX.Location = new System.Drawing.Point(90, 114);
            this.txtMaxShiftX.Name = "txtMaxShiftX";
            this.txtMaxShiftX.Size = new System.Drawing.Size(88, 20);
            this.txtMaxShiftX.TabIndex = 9;
            // 
            // label20
            // 
            this.label20.AutoSize = true;
            this.label20.Location = new System.Drawing.Point(18, 117);
            this.label20.Name = "label20";
            this.label20.Size = new System.Drawing.Size(64, 13);
            this.label20.TabIndex = 8;
            this.label20.Text = "Max Shift X:";
            // 
            // txtExtents
            // 
            this.txtExtents.Location = new System.Drawing.Point(90, 90);
            this.txtExtents.Name = "txtExtents";
            this.txtExtents.Size = new System.Drawing.Size(88, 20);
            this.txtExtents.TabIndex = 7;
            // 
            // txtZoneWidth
            // 
            this.txtZoneWidth.Location = new System.Drawing.Point(90, 66);
            this.txtZoneWidth.Name = "txtZoneWidth";
            this.txtZoneWidth.Size = new System.Drawing.Size(88, 20);
            this.txtZoneWidth.TabIndex = 6;
            // 
            // txtLeverArm
            // 
            this.txtLeverArm.Location = new System.Drawing.Point(90, 42);
            this.txtLeverArm.Name = "txtLeverArm";
            this.txtLeverArm.Size = new System.Drawing.Size(88, 20);
            this.txtLeverArm.TabIndex = 5;
            // 
            // cmbPrescanMode
            // 
            this.cmbPrescanMode.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cmbPrescanMode.Location = new System.Drawing.Point(90, 18);
            this.cmbPrescanMode.Name = "cmbPrescanMode";
            this.cmbPrescanMode.Size = new System.Drawing.Size(88, 21);
            this.cmbPrescanMode.TabIndex = 4;
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(18, 18);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(37, 13);
            this.label4.TabIndex = 3;
            this.label4.Text = "Mode:";
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(18, 94);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(45, 13);
            this.label3.TabIndex = 2;
            this.label3.Text = "Extents:";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(18, 70);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(66, 13);
            this.label2.TabIndex = 1;
            this.label2.Text = "Zone Width:";
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(18, 44);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(58, 13);
            this.label1.TabIndex = 0;
            this.label1.Text = "Lever Arm:";
            // 
            // txtCleaningChi2Limit
            // 
            this.txtCleaningChi2Limit.Location = new System.Drawing.Point(316, 196);
            this.txtCleaningChi2Limit.Name = "txtCleaningChi2Limit";
            this.txtCleaningChi2Limit.Size = new System.Drawing.Size(31, 20);
            this.txtCleaningChi2Limit.TabIndex = 74;
            // 
            // label60
            // 
            this.label60.AutoSize = true;
            this.label60.Location = new System.Drawing.Point(178, 198);
            this.label60.Name = "label60";
            this.label60.Size = new System.Drawing.Size(96, 13);
            this.label60.TabIndex = 73;
            this.label60.Text = "Cleaning Chi² Limit:";
            // 
            // txtCleaningError
            // 
            this.txtCleaningError.Location = new System.Drawing.Point(316, 175);
            this.txtCleaningError.Name = "txtCleaningError";
            this.txtCleaningError.Size = new System.Drawing.Size(31, 20);
            this.txtCleaningError.TabIndex = 72;
            // 
            // label61
            // 
            this.label61.AutoSize = true;
            this.label61.Location = new System.Drawing.Point(178, 177);
            this.label61.Name = "label61";
            this.label61.Size = new System.Drawing.Size(76, 13);
            this.label61.TabIndex = 71;
            this.label61.Text = "Cleaning Error:";
            // 
            // txtExtraTrackIters
            // 
            this.txtExtraTrackIters.Location = new System.Drawing.Point(316, 154);
            this.txtExtraTrackIters.Name = "txtExtraTrackIters";
            this.txtExtraTrackIters.Size = new System.Drawing.Size(31, 20);
            this.txtExtraTrackIters.TabIndex = 29;
            // 
            // label59
            // 
            this.label59.AutoSize = true;
            this.label59.Location = new System.Drawing.Point(178, 156);
            this.label59.Name = "label59";
            this.label59.Size = new System.Drawing.Size(116, 13);
            this.label59.TabIndex = 28;
            this.label59.Text = "Extra Tracking Passes:";
            // 
            // txtTrackFilter
            // 
            this.txtTrackFilter.Location = new System.Drawing.Point(255, 220);
            this.txtTrackFilter.Name = "txtTrackFilter";
            this.txtTrackFilter.Size = new System.Drawing.Size(93, 20);
            this.txtTrackFilter.TabIndex = 68;
            // 
            // label58
            // 
            this.label58.AutoSize = true;
            this.label58.Location = new System.Drawing.Point(179, 220);
            this.label58.Name = "label58";
            this.label58.Size = new System.Drawing.Size(73, 13);
            this.label58.TabIndex = 67;
            this.label58.Text = "Filter Function";
            // 
            // txtMinimumCritical
            // 
            this.txtMinimumCritical.Location = new System.Drawing.Point(108, 220);
            this.txtMinimumCritical.Name = "txtMinimumCritical";
            this.txtMinimumCritical.Size = new System.Drawing.Size(32, 20);
            this.txtMinimumCritical.TabIndex = 42;
            // 
            // label40
            // 
            this.label40.Location = new System.Drawing.Point(14, 223);
            this.label40.Name = "label40";
            this.label40.Size = new System.Drawing.Size(90, 16);
            this.label40.TabIndex = 41;
            this.label40.Text = "Min Segments";
            // 
            // txtMinKalman
            // 
            this.txtMinKalman.Location = new System.Drawing.Point(233, 251);
            this.txtMinKalman.Name = "txtMinKalman";
            this.txtMinKalman.Size = new System.Drawing.Size(48, 20);
            this.txtMinKalman.TabIndex = 39;
            // 
            // label39
            // 
            this.label39.AutoSize = true;
            this.label39.Location = new System.Drawing.Point(108, 254);
            this.label39.Name = "label39";
            this.label39.Size = new System.Drawing.Size(115, 13);
            this.label39.TabIndex = 40;
            this.label39.Text = "Min Kalman Segments:";
            // 
            // txtFittingSegments
            // 
            this.txtFittingSegments.Location = new System.Drawing.Point(233, 275);
            this.txtFittingSegments.Name = "txtFittingSegments";
            this.txtFittingSegments.Size = new System.Drawing.Size(48, 20);
            this.txtFittingSegments.TabIndex = 37;
            // 
            // radKalman
            // 
            this.radKalman.Location = new System.Drawing.Point(20, 248);
            this.radKalman.Name = "radKalman";
            this.radKalman.Size = new System.Drawing.Size(104, 24);
            this.radKalman.TabIndex = 35;
            this.radKalman.Text = "Kalman Filter";
            // 
            // chkZfixed
            // 
            this.chkZfixed.Location = new System.Drawing.Point(15, 197);
            this.chkZfixed.Name = "chkZfixed";
            this.chkZfixed.Size = new System.Drawing.Size(157, 16);
            this.chkZfixed.TabIndex = 34;
            this.chkZfixed.Text = "Freeze Z nominal position";
            // 
            // chkCorrectSlopes
            // 
            this.chkCorrectSlopes.Location = new System.Drawing.Point(15, 165);
            this.chkCorrectSlopes.Name = "chkCorrectSlopes";
            this.chkCorrectSlopes.Size = new System.Drawing.Size(141, 32);
            this.chkCorrectSlopes.TabIndex = 33;
            this.chkCorrectSlopes.Text = "Correct Slopes during Alignment";
            // 
            // txtSlopeIncrement
            // 
            this.txtSlopeIncrement.Location = new System.Drawing.Point(108, 130);
            this.txtSlopeIncrement.Name = "txtSlopeIncrement";
            this.txtSlopeIncrement.Size = new System.Drawing.Size(64, 20);
            this.txtSlopeIncrement.TabIndex = 32;
            // 
            // txtPosIncrement
            // 
            this.txtPosIncrement.Location = new System.Drawing.Point(108, 106);
            this.txtPosIncrement.Name = "txtPosIncrement";
            this.txtPosIncrement.Size = new System.Drawing.Size(64, 20);
            this.txtPosIncrement.TabIndex = 31;
            // 
            // label29
            // 
            this.label29.AutoSize = true;
            this.label29.Location = new System.Drawing.Point(11, 132);
            this.label29.Name = "label29";
            this.label29.Size = new System.Drawing.Size(87, 13);
            this.label29.TabIndex = 30;
            this.label29.Text = "Slope Increment:";
            // 
            // label30
            // 
            this.label30.AutoSize = true;
            this.label30.Location = new System.Drawing.Point(11, 108);
            this.label30.Name = "label30";
            this.label30.Size = new System.Drawing.Size(86, 13);
            this.label30.TabIndex = 29;
            this.label30.Text = "Posit. Increment:";
            // 
            // txtMaxMissSeg
            // 
            this.txtMaxMissSeg.Location = new System.Drawing.Point(316, 132);
            this.txtMaxMissSeg.Name = "txtMaxMissSeg";
            this.txtMaxMissSeg.Size = new System.Drawing.Size(31, 20);
            this.txtMaxMissSeg.TabIndex = 27;
            // 
            // txtMaxIterNum
            // 
            this.txtMaxIterNum.Location = new System.Drawing.Point(283, 108);
            this.txtMaxIterNum.Name = "txtMaxIterNum";
            this.txtMaxIterNum.Size = new System.Drawing.Size(64, 20);
            this.txtMaxIterNum.TabIndex = 26;
            // 
            // label14
            // 
            this.label14.AutoSize = true;
            this.label14.Location = new System.Drawing.Point(178, 134);
            this.label14.Name = "label14";
            this.label14.Size = new System.Drawing.Size(118, 13);
            this.label14.TabIndex = 24;
            this.label14.Text = "Max Missing Segments:";
            // 
            // label13
            // 
            this.label13.AutoSize = true;
            this.label13.Location = new System.Drawing.Point(178, 113);
            this.label13.Name = "label13";
            this.label13.Size = new System.Drawing.Size(76, 13);
            this.label13.TabIndex = 23;
            this.label13.Text = "Max Iterations:";
            // 
            // chkAlignOnLink
            // 
            this.chkAlignOnLink.Location = new System.Drawing.Point(15, 149);
            this.chkAlignOnLink.Name = "chkAlignOnLink";
            this.chkAlignOnLink.Size = new System.Drawing.Size(104, 17);
            this.chkAlignOnLink.TabIndex = 21;
            this.chkAlignOnLink.Text = "Align On Linked";
            // 
            // txtBeamSY
            // 
            this.txtBeamSY.Location = new System.Drawing.Point(283, 83);
            this.txtBeamSY.Name = "txtBeamSY";
            this.txtBeamSY.Size = new System.Drawing.Size(64, 20);
            this.txtBeamSY.TabIndex = 19;
            // 
            // txtBeamSX
            // 
            this.txtBeamSX.Location = new System.Drawing.Point(283, 59);
            this.txtBeamSX.Name = "txtBeamSX";
            this.txtBeamSX.Size = new System.Drawing.Size(64, 20);
            this.txtBeamSX.TabIndex = 18;
            // 
            // txtBeamWid
            // 
            this.txtBeamWid.Location = new System.Drawing.Point(283, 35);
            this.txtBeamWid.Name = "txtBeamWid";
            this.txtBeamWid.Size = new System.Drawing.Size(64, 20);
            this.txtBeamWid.TabIndex = 17;
            // 
            // txtLocCellSize
            // 
            this.txtLocCellSize.Location = new System.Drawing.Point(283, 11);
            this.txtLocCellSize.Name = "txtLocCellSize";
            this.txtLocCellSize.Size = new System.Drawing.Size(64, 20);
            this.txtLocCellSize.TabIndex = 16;
            // 
            // label10
            // 
            this.label10.AutoSize = true;
            this.label10.Location = new System.Drawing.Point(180, 63);
            this.label10.Name = "label10";
            this.label10.Size = new System.Drawing.Size(103, 13);
            this.label10.TabIndex = 14;
            this.label10.Text = "Align Beam Slope X:";
            // 
            // label11
            // 
            this.label11.AutoSize = true;
            this.label11.Location = new System.Drawing.Point(180, 39);
            this.label11.Name = "label11";
            this.label11.Size = new System.Drawing.Size(94, 13);
            this.label11.TabIndex = 13;
            this.label11.Text = "Align Beam Width:";
            // 
            // label12
            // 
            this.label12.AutoSize = true;
            this.label12.Location = new System.Drawing.Point(180, 15);
            this.label12.Name = "label12";
            this.label12.Size = new System.Drawing.Size(89, 13);
            this.label12.TabIndex = 12;
            this.label12.Text = "Locality Cell Size:";
            // 
            // txtInitialSlopeTol
            // 
            this.txtInitialSlopeTol.Location = new System.Drawing.Point(108, 82);
            this.txtInitialSlopeTol.Name = "txtInitialSlopeTol";
            this.txtInitialSlopeTol.Size = new System.Drawing.Size(64, 20);
            this.txtInitialSlopeTol.TabIndex = 11;
            // 
            // txtInitialPosTol
            // 
            this.txtInitialPosTol.Location = new System.Drawing.Point(108, 58);
            this.txtInitialPosTol.Name = "txtInitialPosTol";
            this.txtInitialPosTol.Size = new System.Drawing.Size(64, 20);
            this.txtInitialPosTol.TabIndex = 10;
            // 
            // txtSlopeTol
            // 
            this.txtSlopeTol.Location = new System.Drawing.Point(108, 34);
            this.txtSlopeTol.Name = "txtSlopeTol";
            this.txtSlopeTol.Size = new System.Drawing.Size(64, 20);
            this.txtSlopeTol.TabIndex = 9;
            // 
            // txtPosTol
            // 
            this.txtPosTol.Location = new System.Drawing.Point(108, 10);
            this.txtPosTol.Name = "txtPosTol";
            this.txtPosTol.Size = new System.Drawing.Size(64, 20);
            this.txtPosTol.TabIndex = 8;
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.Location = new System.Drawing.Point(14, 83);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(94, 13);
            this.label7.TabIndex = 7;
            this.label7.Text = "Initial Slope Toler.:";
            // 
            // label8
            // 
            this.label8.AutoSize = true;
            this.label8.Location = new System.Drawing.Point(14, 59);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(93, 13);
            this.label8.TabIndex = 6;
            this.label8.Text = "Initial Posit. Toler.:";
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Location = new System.Drawing.Point(12, 35);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(88, 13);
            this.label6.TabIndex = 5;
            this.label6.Text = "Slope Tolerance:";
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(12, 11);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(98, 13);
            this.label5.TabIndex = 4;
            this.label5.Text = "Position Tolerance:";
            // 
            // label9
            // 
            this.label9.AutoSize = true;
            this.label9.Location = new System.Drawing.Point(180, 85);
            this.label9.Name = "label9";
            this.label9.Size = new System.Drawing.Size(103, 13);
            this.label9.TabIndex = 20;
            this.label9.Text = "Align Beam Slope Y:";
            // 
            // label15
            // 
            this.label15.AutoSize = true;
            this.label15.Location = new System.Drawing.Point(108, 279);
            this.label15.Name = "label15";
            this.label15.Size = new System.Drawing.Size(88, 13);
            this.label15.TabIndex = 38;
            this.label15.Text = "Fitting Segments:";
            // 
            // radTrackFit
            // 
            this.radTrackFit.Location = new System.Drawing.Point(20, 278);
            this.radTrackFit.Name = "radTrackFit";
            this.radTrackFit.Size = new System.Drawing.Size(96, 16);
            this.radTrackFit.TabIndex = 36;
            this.radTrackFit.Text = "Track Fit";
            // 
            // label16
            // 
            this.label16.AutoSize = true;
            this.label16.Location = new System.Drawing.Point(8, 8);
            this.label16.Name = "label16";
            this.label16.Size = new System.Drawing.Size(103, 13);
            this.label16.TabIndex = 5;
            this.label16.Text = "Configuration Name:";
            // 
            // txtConfigName
            // 
            this.txtConfigName.Location = new System.Drawing.Point(112, 8);
            this.txtConfigName.Name = "txtConfigName";
            this.txtConfigName.Size = new System.Drawing.Size(440, 20);
            this.txtConfigName.TabIndex = 9;
            // 
            // cmdDefault
            // 
            this.cmdDefault.Location = new System.Drawing.Point(203, 392);
            this.cmdDefault.Name = "cmdDefault";
            this.cmdDefault.Size = new System.Drawing.Size(88, 24);
            this.cmdDefault.TabIndex = 10;
            this.cmdDefault.Text = "Default";
            this.cmdDefault.Click += new System.EventHandler(this.cmdDefault_Click);
            // 
            // cmdOk
            // 
            this.cmdOk.DialogResult = System.Windows.Forms.DialogResult.OK;
            this.cmdOk.Location = new System.Drawing.Point(748, 392);
            this.cmdOk.Name = "cmdOk";
            this.cmdOk.Size = new System.Drawing.Size(88, 24);
            this.cmdOk.TabIndex = 14;
            this.cmdOk.Text = "OK";
            this.cmdOk.Click += new System.EventHandler(this.cmdOk_Click);
            // 
            // cmdCancel
            // 
            this.cmdCancel.DialogResult = System.Windows.Forms.DialogResult.Cancel;
            this.cmdCancel.Location = new System.Drawing.Point(11, 392);
            this.cmdCancel.Name = "cmdCancel";
            this.cmdCancel.Size = new System.Drawing.Size(88, 24);
            this.cmdCancel.TabIndex = 15;
            this.cmdCancel.Text = "Cancel";
            // 
            // groupBox3
            // 
            this.groupBox3.Controls.Add(this.txtSlopesCellSizeY);
            this.groupBox3.Controls.Add(this.label19);
            this.groupBox3.Controls.Add(this.txtSlopesCellSizeX);
            this.groupBox3.Controls.Add(this.label18);
            this.groupBox3.Controls.Add(this.txtRiskFactor);
            this.groupBox3.Controls.Add(this.label17);
            this.groupBox3.Location = new System.Drawing.Point(147, 191);
            this.groupBox3.Name = "groupBox3";
            this.groupBox3.Size = new System.Drawing.Size(184, 88);
            this.groupBox3.TabIndex = 16;
            this.groupBox3.TabStop = false;
            this.groupBox3.Text = "Low Momentum Tracks";
            // 
            // txtSlopesCellSizeY
            // 
            this.txtSlopesCellSizeY.Location = new System.Drawing.Point(106, 64);
            this.txtSlopesCellSizeY.Name = "txtSlopesCellSizeY";
            this.txtSlopesCellSizeY.Size = new System.Drawing.Size(64, 20);
            this.txtSlopesCellSizeY.TabIndex = 5;
            // 
            // label19
            // 
            this.label19.AutoSize = true;
            this.label19.Location = new System.Drawing.Point(10, 68);
            this.label19.Name = "label19";
            this.label19.Size = new System.Drawing.Size(95, 13);
            this.label19.TabIndex = 4;
            this.label19.Text = "Slopes Cell Size Y:";
            // 
            // txtSlopesCellSizeX
            // 
            this.txtSlopesCellSizeX.Location = new System.Drawing.Point(106, 40);
            this.txtSlopesCellSizeX.Name = "txtSlopesCellSizeX";
            this.txtSlopesCellSizeX.Size = new System.Drawing.Size(64, 20);
            this.txtSlopesCellSizeX.TabIndex = 3;
            // 
            // label18
            // 
            this.label18.AutoSize = true;
            this.label18.Location = new System.Drawing.Point(10, 44);
            this.label18.Name = "label18";
            this.label18.Size = new System.Drawing.Size(95, 13);
            this.label18.TabIndex = 2;
            this.label18.Text = "Slopes Cell Size X:";
            // 
            // txtRiskFactor
            // 
            this.txtRiskFactor.Location = new System.Drawing.Point(106, 16);
            this.txtRiskFactor.Name = "txtRiskFactor";
            this.txtRiskFactor.Size = new System.Drawing.Size(64, 20);
            this.txtRiskFactor.TabIndex = 1;
            // 
            // label17
            // 
            this.label17.AutoSize = true;
            this.label17.Location = new System.Drawing.Point(12, 19);
            this.label17.Name = "label17";
            this.label17.Size = new System.Drawing.Size(64, 13);
            this.label17.TabIndex = 0;
            this.label17.Text = "Risk Factor:";
            // 
            // cmdHelp
            // 
            this.cmdHelp.Location = new System.Drawing.Point(105, 392);
            this.cmdHelp.Name = "cmdHelp";
            this.cmdHelp.Size = new System.Drawing.Size(88, 24);
            this.cmdHelp.TabIndex = 18;
            this.cmdHelp.Text = "Help";
            this.cmdHelp.Click += new System.EventHandler(this.cmdHelp_Click);
            // 
            // chkUpdateTrans
            // 
            this.chkUpdateTrans.Location = new System.Drawing.Point(19, 12);
            this.chkUpdateTrans.Name = "chkUpdateTrans";
            this.chkUpdateTrans.Size = new System.Drawing.Size(105, 42);
            this.chkUpdateTrans.TabIndex = 20;
            this.chkUpdateTrans.Text = "Updating Transformations";
            // 
            // groupBox5
            // 
            this.groupBox5.Controls.Add(this.chkRelinkEnable);
            this.groupBox5.Controls.Add(this.txtRelinkDeltaZ);
            this.groupBox5.Controls.Add(this.txtRelinkAperture);
            this.groupBox5.Controls.Add(this.label56);
            this.groupBox5.Controls.Add(this.txtMinSegNumber);
            this.groupBox5.Controls.Add(this.label55);
            this.groupBox5.Controls.Add(this.txtMinTracksPairs);
            this.groupBox5.Controls.Add(this.label33);
            this.groupBox5.Controls.Add(this.label34);
            this.groupBox5.Location = new System.Drawing.Point(19, 52);
            this.groupBox5.Name = "groupBox5";
            this.groupBox5.Size = new System.Drawing.Size(113, 227);
            this.groupBox5.TabIndex = 21;
            this.groupBox5.TabStop = false;
            // 
            // chkRelinkEnable
            // 
            this.chkRelinkEnable.Location = new System.Drawing.Point(4, 102);
            this.chkRelinkEnable.Name = "chkRelinkEnable";
            this.chkRelinkEnable.Size = new System.Drawing.Size(105, 27);
            this.chkRelinkEnable.TabIndex = 31;
            this.chkRelinkEnable.Text = "Enable Relink";
            // 
            // txtRelinkDeltaZ
            // 
            this.txtRelinkDeltaZ.Location = new System.Drawing.Point(8, 189);
            this.txtRelinkDeltaZ.Name = "txtRelinkDeltaZ";
            this.txtRelinkDeltaZ.Size = new System.Drawing.Size(64, 20);
            this.txtRelinkDeltaZ.TabIndex = 30;
            // 
            // txtRelinkAperture
            // 
            this.txtRelinkAperture.Location = new System.Drawing.Point(8, 149);
            this.txtRelinkAperture.Name = "txtRelinkAperture";
            this.txtRelinkAperture.Size = new System.Drawing.Size(64, 20);
            this.txtRelinkAperture.TabIndex = 28;
            // 
            // label56
            // 
            this.label56.AutoSize = true;
            this.label56.Location = new System.Drawing.Point(4, 172);
            this.label56.Name = "label56";
            this.label56.Size = new System.Drawing.Size(72, 13);
            this.label56.TabIndex = 29;
            this.label56.Text = "Relink DeltaZ";
            // 
            // txtMinSegNumber
            // 
            this.txtMinSegNumber.Location = new System.Drawing.Point(8, 76);
            this.txtMinSegNumber.Name = "txtMinSegNumber";
            this.txtMinSegNumber.Size = new System.Drawing.Size(64, 20);
            this.txtMinSegNumber.TabIndex = 3;
            // 
            // label55
            // 
            this.label55.AutoSize = true;
            this.label55.Location = new System.Drawing.Point(4, 132);
            this.label55.Name = "label55";
            this.label55.Size = new System.Drawing.Size(80, 13);
            this.label55.TabIndex = 27;
            this.label55.Text = "Relink Aperture";
            // 
            // txtMinTracksPairs
            // 
            this.txtMinTracksPairs.Location = new System.Drawing.Point(8, 32);
            this.txtMinTracksPairs.Name = "txtMinTracksPairs";
            this.txtMinTracksPairs.Size = new System.Drawing.Size(64, 20);
            this.txtMinTracksPairs.TabIndex = 1;
            // 
            // label33
            // 
            this.label33.AutoSize = true;
            this.label33.Location = new System.Drawing.Point(4, 15);
            this.label33.Name = "label33";
            this.label33.Size = new System.Drawing.Size(89, 13);
            this.label33.TabIndex = 0;
            this.label33.Text = "Min Tracks Pairs:";
            // 
            // label34
            // 
            this.label34.AutoSize = true;
            this.label34.Location = new System.Drawing.Point(4, 57);
            this.label34.Name = "label34";
            this.label34.Size = new System.Drawing.Size(102, 13);
            this.label34.TabIndex = 2;
            this.label34.Text = "Min Segments Num:";
            // 
            // chkKinkDetection
            // 
            this.chkKinkDetection.Location = new System.Drawing.Point(147, 25);
            this.chkKinkDetection.Name = "chkKinkDetection";
            this.chkKinkDetection.Size = new System.Drawing.Size(128, 16);
            this.chkKinkDetection.TabIndex = 22;
            this.chkKinkDetection.Text = "Kink Detection";
            // 
            // groupBox7
            // 
            this.groupBox7.Controls.Add(this.label45);
            this.groupBox7.Controls.Add(this.cmbFilterLength);
            this.groupBox7.Controls.Add(this.txtKinkFilterThreshold);
            this.groupBox7.Controls.Add(this.label44);
            this.groupBox7.Controls.Add(this.txtKinkFactor);
            this.groupBox7.Controls.Add(this.label43);
            this.groupBox7.Controls.Add(this.txtKinkMinSlopeDiff);
            this.groupBox7.Controls.Add(this.label42);
            this.groupBox7.Controls.Add(this.txtKinkMinSeg);
            this.groupBox7.Controls.Add(this.label41);
            this.groupBox7.Location = new System.Drawing.Point(147, 52);
            this.groupBox7.Name = "groupBox7";
            this.groupBox7.Size = new System.Drawing.Size(184, 133);
            this.groupBox7.TabIndex = 23;
            this.groupBox7.TabStop = false;
            // 
            // label45
            // 
            this.label45.AutoSize = true;
            this.label45.Location = new System.Drawing.Point(5, 101);
            this.label45.Name = "label45";
            this.label45.Size = new System.Drawing.Size(68, 13);
            this.label45.TabIndex = 13;
            this.label45.Text = "Filter Length:";
            // 
            // cmbFilterLength
            // 
            this.cmbFilterLength.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cmbFilterLength.Location = new System.Drawing.Point(117, 100);
            this.cmbFilterLength.Name = "cmbFilterLength";
            this.cmbFilterLength.Size = new System.Drawing.Size(53, 21);
            this.cmbFilterLength.TabIndex = 12;
            // 
            // txtKinkFilterThreshold
            // 
            this.txtKinkFilterThreshold.Location = new System.Drawing.Point(118, 78);
            this.txtKinkFilterThreshold.Name = "txtKinkFilterThreshold";
            this.txtKinkFilterThreshold.Size = new System.Drawing.Size(52, 20);
            this.txtKinkFilterThreshold.TabIndex = 11;
            // 
            // label44
            // 
            this.label44.AutoSize = true;
            this.label44.Location = new System.Drawing.Point(4, 81);
            this.label44.Name = "label44";
            this.label44.Size = new System.Drawing.Size(82, 13);
            this.label44.TabIndex = 10;
            this.label44.Text = "Filter Threshold:";
            // 
            // txtKinkFactor
            // 
            this.txtKinkFactor.Location = new System.Drawing.Point(124, 56);
            this.txtKinkFactor.Name = "txtKinkFactor";
            this.txtKinkFactor.Size = new System.Drawing.Size(46, 20);
            this.txtKinkFactor.TabIndex = 9;
            // 
            // label43
            // 
            this.label43.AutoSize = true;
            this.label43.Location = new System.Drawing.Point(2, 56);
            this.label43.Name = "label43";
            this.label43.Size = new System.Drawing.Size(78, 13);
            this.label43.TabIndex = 8;
            this.label43.Text = "Kink Fit Factor:";
            // 
            // txtKinkMinSlopeDiff
            // 
            this.txtKinkMinSlopeDiff.Location = new System.Drawing.Point(124, 33);
            this.txtKinkMinSlopeDiff.Name = "txtKinkMinSlopeDiff";
            this.txtKinkMinSlopeDiff.Size = new System.Drawing.Size(46, 20);
            this.txtKinkMinSlopeDiff.TabIndex = 7;
            // 
            // label42
            // 
            this.label42.AutoSize = true;
            this.label42.Location = new System.Drawing.Point(2, 34);
            this.label42.Name = "label42";
            this.label42.Size = new System.Drawing.Size(85, 13);
            this.label42.TabIndex = 6;
            this.label42.Text = "Min Slope Differ:";
            // 
            // txtKinkMinSeg
            // 
            this.txtKinkMinSeg.Location = new System.Drawing.Point(124, 9);
            this.txtKinkMinSeg.Name = "txtKinkMinSeg";
            this.txtKinkMinSeg.Size = new System.Drawing.Size(46, 20);
            this.txtKinkMinSeg.TabIndex = 5;
            // 
            // label41
            // 
            this.label41.AutoSize = true;
            this.label41.Location = new System.Drawing.Point(2, 13);
            this.label41.Name = "label41";
            this.label41.Size = new System.Drawing.Size(102, 13);
            this.label41.TabIndex = 4;
            this.label41.Text = "Min Segments Num:";
            // 
            // VtxTab
            // 
            this.VtxTab.Controls.Add(this.tabPage3);
            this.VtxTab.Controls.Add(this.tabPage1);
            this.VtxTab.Controls.Add(this.tabPage2);
            this.VtxTab.Location = new System.Drawing.Point(396, 50);
            this.VtxTab.Name = "VtxTab";
            this.VtxTab.SelectedIndex = 0;
            this.VtxTab.Size = new System.Drawing.Size(440, 253);
            this.VtxTab.TabIndex = 24;
            // 
            // tabPage3
            // 
            this.tabPage3.Controls.Add(this.label50);
            this.tabPage3.Location = new System.Drawing.Point(4, 22);
            this.tabPage3.Name = "tabPage3";
            this.tabPage3.Size = new System.Drawing.Size(432, 227);
            this.tabPage3.TabIndex = 2;
            this.tabPage3.Text = "None";
            // 
            // label50
            // 
            this.label50.Location = new System.Drawing.Point(64, 64);
            this.label50.Name = "label50";
            this.label50.Size = new System.Drawing.Size(320, 16);
            this.label50.TabIndex = 0;
            this.label50.Text = "No parameters. Select a vertex algorithm to set its parameters.";
            // 
            // tabPage1
            // 
            this.tabPage1.Controls.Add(this.txtMaximumZ);
            this.tabPage1.Controls.Add(this.chkUseCells);
            this.tabPage1.Controls.Add(this.groupBox6);
            this.tabPage1.Controls.Add(this.label32);
            this.tabPage1.Controls.Add(this.chkTopologyKink);
            this.tabPage1.Controls.Add(this.chkTopologyLambda);
            this.tabPage1.Controls.Add(this.chkTopologyY);
            this.tabPage1.Controls.Add(this.chkTopologyX);
            this.tabPage1.Controls.Add(this.chkTopologyV);
            this.tabPage1.Controls.Add(this.txtMinVertexTrackSegments);
            this.tabPage1.Controls.Add(this.label31);
            this.tabPage1.Controls.Add(this.txtStartingClusterToleranceLong);
            this.tabPage1.Controls.Add(this.txtMinimumZ);
            this.tabPage1.Controls.Add(this.label22);
            this.tabPage1.Controls.Add(this.label24);
            this.tabPage1.Controls.Add(this.txtMaximumClusterToleranceLong);
            this.tabPage1.Controls.Add(this.txtCrossTolerance);
            this.tabPage1.Controls.Add(this.label25);
            this.tabPage1.Controls.Add(this.label26);
            this.tabPage1.Controls.Add(this.label23);
            this.tabPage1.Controls.Add(this.txtMaximumClusterToleranceTrans);
            this.tabPage1.Controls.Add(this.txtStartingClusterToleranceTrans);
            this.tabPage1.Controls.Add(this.label27);
            this.tabPage1.Controls.Add(this.label28);
            this.tabPage1.Location = new System.Drawing.Point(4, 22);
            this.tabPage1.Name = "tabPage1";
            this.tabPage1.Size = new System.Drawing.Size(432, 227);
            this.tabPage1.TabIndex = 0;
            this.tabPage1.Text = "Pair-Based";
            // 
            // txtMaximumZ
            // 
            this.txtMaximumZ.Location = new System.Drawing.Point(70, 112);
            this.txtMaximumZ.Name = "txtMaximumZ";
            this.txtMaximumZ.Size = new System.Drawing.Size(64, 20);
            this.txtMaximumZ.TabIndex = 54;
            // 
            // chkUseCells
            // 
            this.chkUseCells.Location = new System.Drawing.Point(308, 8);
            this.chkUseCells.Name = "chkUseCells";
            this.chkUseCells.Size = new System.Drawing.Size(80, 16);
            this.chkUseCells.TabIndex = 66;
            this.chkUseCells.Text = "Use Cells";
            // 
            // groupBox6
            // 
            this.groupBox6.Controls.Add(this.txtMatrix);
            this.groupBox6.Controls.Add(this.label38);
            this.groupBox6.Controls.Add(this.txtZCellsSize);
            this.groupBox6.Controls.Add(this.label37);
            this.groupBox6.Controls.Add(this.txtYCellsSize);
            this.groupBox6.Controls.Add(this.txtXCellsSize);
            this.groupBox6.Controls.Add(this.label36);
            this.groupBox6.Controls.Add(this.label35);
            this.groupBox6.Location = new System.Drawing.Point(308, 32);
            this.groupBox6.Name = "groupBox6";
            this.groupBox6.Size = new System.Drawing.Size(112, 128);
            this.groupBox6.TabIndex = 65;
            this.groupBox6.TabStop = false;
            this.groupBox6.Text = "Cells";
            // 
            // txtMatrix
            // 
            this.txtMatrix.Location = new System.Drawing.Point(56, 94);
            this.txtMatrix.Name = "txtMatrix";
            this.txtMatrix.Size = new System.Drawing.Size(42, 20);
            this.txtMatrix.TabIndex = 7;
            // 
            // label38
            // 
            this.label38.AutoSize = true;
            this.label38.Location = new System.Drawing.Point(10, 71);
            this.label38.Name = "label38";
            this.label38.Size = new System.Drawing.Size(40, 13);
            this.label38.TabIndex = 6;
            this.label38.Text = "Z Size:";
            // 
            // txtZCellsSize
            // 
            this.txtZCellsSize.Location = new System.Drawing.Point(55, 68);
            this.txtZCellsSize.Name = "txtZCellsSize";
            this.txtZCellsSize.Size = new System.Drawing.Size(42, 20);
            this.txtZCellsSize.TabIndex = 5;
            // 
            // label37
            // 
            this.label37.AutoSize = true;
            this.label37.Location = new System.Drawing.Point(9, 99);
            this.label37.Name = "label37";
            this.label37.Size = new System.Drawing.Size(38, 13);
            this.label37.TabIndex = 4;
            this.label37.Text = "Matrix:";
            // 
            // txtYCellsSize
            // 
            this.txtYCellsSize.Location = new System.Drawing.Point(55, 43);
            this.txtYCellsSize.Name = "txtYCellsSize";
            this.txtYCellsSize.Size = new System.Drawing.Size(42, 20);
            this.txtYCellsSize.TabIndex = 3;
            // 
            // txtXCellsSize
            // 
            this.txtXCellsSize.Location = new System.Drawing.Point(56, 16);
            this.txtXCellsSize.Name = "txtXCellsSize";
            this.txtXCellsSize.Size = new System.Drawing.Size(42, 20);
            this.txtXCellsSize.TabIndex = 2;
            // 
            // label36
            // 
            this.label36.AutoSize = true;
            this.label36.Location = new System.Drawing.Point(11, 46);
            this.label36.Name = "label36";
            this.label36.Size = new System.Drawing.Size(40, 13);
            this.label36.TabIndex = 1;
            this.label36.Text = "Y Size:";
            // 
            // label35
            // 
            this.label35.AutoSize = true;
            this.label35.Location = new System.Drawing.Point(11, 22);
            this.label35.Name = "label35";
            this.label35.Size = new System.Drawing.Size(40, 13);
            this.label35.TabIndex = 0;
            this.label35.Text = "X Size:";
            // 
            // label32
            // 
            this.label32.Location = new System.Drawing.Point(5, 8);
            this.label32.Name = "label32";
            this.label32.Size = new System.Drawing.Size(63, 26);
            this.label32.TabIndex = 64;
            this.label32.Text = "Intersection Topology";
            // 
            // chkTopologyKink
            // 
            this.chkTopologyKink.Location = new System.Drawing.Point(9, 40);
            this.chkTopologyKink.Name = "chkTopologyKink";
            this.chkTopologyKink.Size = new System.Drawing.Size(70, 16);
            this.chkTopologyKink.TabIndex = 63;
            this.chkTopologyKink.Text = "Kink";
            // 
            // chkTopologyLambda
            // 
            this.chkTopologyLambda.Location = new System.Drawing.Point(8, 104);
            this.chkTopologyLambda.Name = "chkTopologyLambda";
            this.chkTopologyLambda.Size = new System.Drawing.Size(64, 16);
            this.chkTopologyLambda.TabIndex = 62;
            this.chkTopologyLambda.Text = "Lambda";
            // 
            // chkTopologyY
            // 
            this.chkTopologyY.Location = new System.Drawing.Point(8, 88);
            this.chkTopologyY.Name = "chkTopologyY";
            this.chkTopologyY.Size = new System.Drawing.Size(30, 16);
            this.chkTopologyY.TabIndex = 61;
            this.chkTopologyY.Text = "Y";
            // 
            // chkTopologyX
            // 
            this.chkTopologyX.Location = new System.Drawing.Point(8, 72);
            this.chkTopologyX.Name = "chkTopologyX";
            this.chkTopologyX.Size = new System.Drawing.Size(30, 16);
            this.chkTopologyX.TabIndex = 60;
            this.chkTopologyX.Text = "X";
            // 
            // chkTopologyV
            // 
            this.chkTopologyV.Location = new System.Drawing.Point(8, 56);
            this.chkTopologyV.Name = "chkTopologyV";
            this.chkTopologyV.Size = new System.Drawing.Size(30, 16);
            this.chkTopologyV.TabIndex = 59;
            this.chkTopologyV.Text = "V";
            // 
            // txtMinVertexTrackSegments
            // 
            this.txtMinVertexTrackSegments.Location = new System.Drawing.Point(236, 8);
            this.txtMinVertexTrackSegments.Name = "txtMinVertexTrackSegments";
            this.txtMinVertexTrackSegments.Size = new System.Drawing.Size(64, 20);
            this.txtMinVertexTrackSegments.TabIndex = 57;
            // 
            // label31
            // 
            this.label31.AutoSize = true;
            this.label31.Location = new System.Drawing.Point(116, 8);
            this.label31.Name = "label31";
            this.label31.Size = new System.Drawing.Size(117, 13);
            this.label31.TabIndex = 58;
            this.label31.Text = "Min Segments Number:";
            // 
            // txtStartingClusterToleranceLong
            // 
            this.txtStartingClusterToleranceLong.Location = new System.Drawing.Point(236, 88);
            this.txtStartingClusterToleranceLong.Name = "txtStartingClusterToleranceLong";
            this.txtStartingClusterToleranceLong.Size = new System.Drawing.Size(64, 20);
            this.txtStartingClusterToleranceLong.TabIndex = 56;
            // 
            // txtMinimumZ
            // 
            this.txtMinimumZ.Location = new System.Drawing.Point(70, 136);
            this.txtMinimumZ.Name = "txtMinimumZ";
            this.txtMinimumZ.Size = new System.Drawing.Size(64, 20);
            this.txtMinimumZ.TabIndex = 55;
            // 
            // label22
            // 
            this.label22.AutoSize = true;
            this.label22.Location = new System.Drawing.Point(136, 88);
            this.label22.Name = "label22";
            this.label22.Size = new System.Drawing.Size(87, 13);
            this.label22.TabIndex = 53;
            this.label22.Text = "Init. Longitudinal:";
            // 
            // label24
            // 
            this.label24.AutoSize = true;
            this.label24.Location = new System.Drawing.Point(6, 120);
            this.label24.Name = "label24";
            this.label24.Size = new System.Drawing.Size(64, 13);
            this.label24.TabIndex = 51;
            this.label24.Text = "Maximum Z:";
            // 
            // txtMaximumClusterToleranceLong
            // 
            this.txtMaximumClusterToleranceLong.Location = new System.Drawing.Point(236, 136);
            this.txtMaximumClusterToleranceLong.Name = "txtMaximumClusterToleranceLong";
            this.txtMaximumClusterToleranceLong.Size = new System.Drawing.Size(64, 20);
            this.txtMaximumClusterToleranceLong.TabIndex = 50;
            // 
            // txtCrossTolerance
            // 
            this.txtCrossTolerance.Location = new System.Drawing.Point(236, 40);
            this.txtCrossTolerance.Name = "txtCrossTolerance";
            this.txtCrossTolerance.Size = new System.Drawing.Size(64, 20);
            this.txtCrossTolerance.TabIndex = 47;
            // 
            // label25
            // 
            this.label25.AutoSize = true;
            this.label25.Location = new System.Drawing.Point(140, 136);
            this.label25.Name = "label25";
            this.label25.Size = new System.Drawing.Size(93, 13);
            this.label25.TabIndex = 46;
            this.label25.Text = "Max. Longitudinal:";
            // 
            // label26
            // 
            this.label26.AutoSize = true;
            this.label26.Location = new System.Drawing.Point(136, 112);
            this.label26.Name = "label26";
            this.label26.Size = new System.Drawing.Size(89, 13);
            this.label26.TabIndex = 45;
            this.label26.Text = "Max. Transverse:";
            // 
            // label23
            // 
            this.label23.AutoSize = true;
            this.label23.Location = new System.Drawing.Point(6, 144);
            this.label23.Name = "label23";
            this.label23.Size = new System.Drawing.Size(61, 13);
            this.label23.TabIndex = 52;
            this.label23.Text = "Minimum Z:";
            // 
            // txtMaximumClusterToleranceTrans
            // 
            this.txtMaximumClusterToleranceTrans.Location = new System.Drawing.Point(236, 112);
            this.txtMaximumClusterToleranceTrans.Name = "txtMaximumClusterToleranceTrans";
            this.txtMaximumClusterToleranceTrans.Size = new System.Drawing.Size(64, 20);
            this.txtMaximumClusterToleranceTrans.TabIndex = 49;
            // 
            // txtStartingClusterToleranceTrans
            // 
            this.txtStartingClusterToleranceTrans.Location = new System.Drawing.Point(236, 64);
            this.txtStartingClusterToleranceTrans.Name = "txtStartingClusterToleranceTrans";
            this.txtStartingClusterToleranceTrans.Size = new System.Drawing.Size(64, 20);
            this.txtStartingClusterToleranceTrans.TabIndex = 48;
            // 
            // label27
            // 
            this.label27.AutoSize = true;
            this.label27.Location = new System.Drawing.Point(136, 64);
            this.label27.Name = "label27";
            this.label27.Size = new System.Drawing.Size(83, 13);
            this.label27.TabIndex = 44;
            this.label27.Text = "Init. Transverse:";
            // 
            // label28
            // 
            this.label28.AutoSize = true;
            this.label28.Location = new System.Drawing.Point(128, 40);
            this.label28.Name = "label28";
            this.label28.Size = new System.Drawing.Size(93, 13);
            this.label28.TabIndex = 43;
            this.label28.Text = "Closest Approach:";
            // 
            // tabPage2
            // 
            this.tabPage2.Controls.Add(this.txtGVtxFilter);
            this.tabPage2.Controls.Add(this.txtGVtxMinCount);
            this.tabPage2.Controls.Add(this.label57);
            this.tabPage2.Controls.Add(this.label46);
            this.tabPage2.Controls.Add(this.txtGVtxMaxSlopeDivergence);
            this.tabPage2.Controls.Add(this.txtGVtxRadius);
            this.tabPage2.Controls.Add(this.label47);
            this.tabPage2.Controls.Add(this.txtGVtxMaxExt);
            this.tabPage2.Controls.Add(this.label48);
            this.tabPage2.Controls.Add(this.label49);
            this.tabPage2.Location = new System.Drawing.Point(4, 22);
            this.tabPage2.Name = "tabPage2";
            this.tabPage2.Size = new System.Drawing.Size(432, 227);
            this.tabPage2.TabIndex = 1;
            this.tabPage2.Text = "Global";
            // 
            // txtGVtxFilter
            // 
            this.txtGVtxFilter.Location = new System.Drawing.Point(144, 113);
            this.txtGVtxFilter.Name = "txtGVtxFilter";
            this.txtGVtxFilter.Size = new System.Drawing.Size(276, 20);
            this.txtGVtxFilter.TabIndex = 66;
            // 
            // txtGVtxMinCount
            // 
            this.txtGVtxMinCount.Location = new System.Drawing.Point(144, 16);
            this.txtGVtxMinCount.Name = "txtGVtxMinCount";
            this.txtGVtxMinCount.Size = new System.Drawing.Size(64, 20);
            this.txtGVtxMinCount.TabIndex = 65;
            // 
            // label57
            // 
            this.label57.AutoSize = true;
            this.label57.Location = new System.Drawing.Point(8, 113);
            this.label57.Name = "label57";
            this.label57.Size = new System.Drawing.Size(70, 13);
            this.label57.TabIndex = 65;
            this.label57.Text = "Filter function";
            // 
            // label46
            // 
            this.label46.AutoSize = true;
            this.label46.Location = new System.Drawing.Point(8, 16);
            this.label46.Name = "label46";
            this.label46.Size = new System.Drawing.Size(117, 13);
            this.label46.TabIndex = 66;
            this.label46.Text = "Min Segments Number:";
            // 
            // txtGVtxMaxSlopeDivergence
            // 
            this.txtGVtxMaxSlopeDivergence.Location = new System.Drawing.Point(144, 88);
            this.txtGVtxMaxSlopeDivergence.Name = "txtGVtxMaxSlopeDivergence";
            this.txtGVtxMaxSlopeDivergence.Size = new System.Drawing.Size(64, 20);
            this.txtGVtxMaxSlopeDivergence.TabIndex = 64;
            // 
            // txtGVtxRadius
            // 
            this.txtGVtxRadius.Location = new System.Drawing.Point(144, 40);
            this.txtGVtxRadius.Name = "txtGVtxRadius";
            this.txtGVtxRadius.Size = new System.Drawing.Size(64, 20);
            this.txtGVtxRadius.TabIndex = 62;
            // 
            // label47
            // 
            this.label47.AutoSize = true;
            this.label47.Location = new System.Drawing.Point(8, 88);
            this.label47.Name = "label47";
            this.label47.Size = new System.Drawing.Size(111, 13);
            this.label47.TabIndex = 61;
            this.label47.Text = "Max slope divergence";
            // 
            // txtGVtxMaxExt
            // 
            this.txtGVtxMaxExt.Location = new System.Drawing.Point(144, 64);
            this.txtGVtxMaxExt.Name = "txtGVtxMaxExt";
            this.txtGVtxMaxExt.Size = new System.Drawing.Size(64, 20);
            this.txtGVtxMaxExt.TabIndex = 63;
            // 
            // label48
            // 
            this.label48.AutoSize = true;
            this.label48.Location = new System.Drawing.Point(8, 64);
            this.label48.Name = "label48";
            this.label48.Size = new System.Drawing.Size(91, 13);
            this.label48.TabIndex = 60;
            this.label48.Text = "Max Extrapolation";
            // 
            // label49
            // 
            this.label49.AutoSize = true;
            this.label49.Location = new System.Drawing.Point(8, 41);
            this.label49.Name = "label49";
            this.label49.Size = new System.Drawing.Size(73, 13);
            this.label49.TabIndex = 59;
            this.label49.Text = "Vertex Radius";
            // 
            // label51
            // 
            this.label51.Location = new System.Drawing.Point(393, 31);
            this.label51.Name = "label51";
            this.label51.Size = new System.Drawing.Size(176, 16);
            this.label51.TabIndex = 25;
            this.label51.Text = "Vertex reconstruction";
            // 
            // groupBox4
            // 
            this.groupBox4.Controls.Add(this.VtxFitWeightZStepText);
            this.groupBox4.Controls.Add(this.label54);
            this.groupBox4.Controls.Add(this.VtxFitWeightXYStepText);
            this.groupBox4.Controls.Add(this.label53);
            this.groupBox4.Controls.Add(this.VtxFitWeightTolText);
            this.groupBox4.Controls.Add(this.label52);
            this.groupBox4.Controls.Add(this.VtxFitWeightEnableCheck);
            this.groupBox4.Location = new System.Drawing.Point(400, 306);
            this.groupBox4.Name = "groupBox4";
            this.groupBox4.Size = new System.Drawing.Size(264, 80);
            this.groupBox4.TabIndex = 26;
            this.groupBox4.TabStop = false;
            this.groupBox4.Text = "Weighted Vertex Fit";
            // 
            // VtxFitWeightZStepText
            // 
            this.VtxFitWeightZStepText.Location = new System.Drawing.Point(216, 48);
            this.VtxFitWeightZStepText.Name = "VtxFitWeightZStepText";
            this.VtxFitWeightZStepText.Size = new System.Drawing.Size(40, 20);
            this.VtxFitWeightZStepText.TabIndex = 6;
            // 
            // label54
            // 
            this.label54.Location = new System.Drawing.Point(152, 48);
            this.label54.Name = "label54";
            this.label54.Size = new System.Drawing.Size(56, 16);
            this.label54.TabIndex = 5;
            this.label54.Text = "Z Step";
            // 
            // VtxFitWeightXYStepText
            // 
            this.VtxFitWeightXYStepText.Location = new System.Drawing.Point(72, 48);
            this.VtxFitWeightXYStepText.Name = "VtxFitWeightXYStepText";
            this.VtxFitWeightXYStepText.Size = new System.Drawing.Size(40, 20);
            this.VtxFitWeightXYStepText.TabIndex = 4;
            // 
            // label53
            // 
            this.label53.Location = new System.Drawing.Point(8, 48);
            this.label53.Name = "label53";
            this.label53.Size = new System.Drawing.Size(56, 16);
            this.label53.TabIndex = 3;
            this.label53.Text = "XY Step";
            // 
            // VtxFitWeightTolText
            // 
            this.VtxFitWeightTolText.Location = new System.Drawing.Point(216, 16);
            this.VtxFitWeightTolText.Name = "VtxFitWeightTolText";
            this.VtxFitWeightTolText.Size = new System.Drawing.Size(40, 20);
            this.VtxFitWeightTolText.TabIndex = 2;
            // 
            // label52
            // 
            this.label52.Location = new System.Drawing.Point(128, 16);
            this.label52.Name = "label52";
            this.label52.Size = new System.Drawing.Size(72, 16);
            this.label52.TabIndex = 1;
            this.label52.Text = "Tolerance";
            // 
            // VtxFitWeightEnableCheck
            // 
            this.VtxFitWeightEnableCheck.Location = new System.Drawing.Point(8, 16);
            this.VtxFitWeightEnableCheck.Name = "VtxFitWeightEnableCheck";
            this.VtxFitWeightEnableCheck.Size = new System.Drawing.Size(64, 16);
            this.VtxFitWeightEnableCheck.TabIndex = 0;
            this.VtxFitWeightEnableCheck.Text = "Enable";
            // 
            // TkTab
            // 
            this.TkTab.Controls.Add(this.tabPage6);
            this.TkTab.Controls.Add(this.tabPage7);
            this.TkTab.Controls.Add(this.tabPage8);
            this.TkTab.Controls.Add(this.tabPage9);
            this.TkTab.Location = new System.Drawing.Point(12, 50);
            this.TkTab.Name = "TkTab";
            this.TkTab.SelectedIndex = 0;
            this.TkTab.Size = new System.Drawing.Size(376, 336);
            this.TkTab.TabIndex = 27;
            // 
            // tabPage6
            // 
            this.tabPage6.BackColor = System.Drawing.SystemColors.Control;
            this.tabPage6.Controls.Add(this.chkIgnoreBadLayers);
            this.tabPage6.Controls.Add(this.txtMaxShiftY);
            this.tabPage6.Controls.Add(this.label4);
            this.tabPage6.Controls.Add(this.label21);
            this.tabPage6.Controls.Add(this.label1);
            this.tabPage6.Controls.Add(this.txtMaxShiftX);
            this.tabPage6.Controls.Add(this.label2);
            this.tabPage6.Controls.Add(this.label20);
            this.tabPage6.Controls.Add(this.label3);
            this.tabPage6.Controls.Add(this.txtExtents);
            this.tabPage6.Controls.Add(this.cmbPrescanMode);
            this.tabPage6.Controls.Add(this.txtZoneWidth);
            this.tabPage6.Controls.Add(this.txtLeverArm);
            this.tabPage6.Location = new System.Drawing.Point(4, 22);
            this.tabPage6.Name = "tabPage6";
            this.tabPage6.Size = new System.Drawing.Size(368, 310);
            this.tabPage6.TabIndex = 0;
            this.tabPage6.Text = "Prescan";
            // 
            // tabPage7
            // 
            this.tabPage7.BackColor = System.Drawing.SystemColors.Control;
            this.tabPage7.Controls.Add(this.txtCleaningChi2Limit);
            this.tabPage7.Controls.Add(this.label5);
            this.tabPage7.Controls.Add(this.label60);
            this.tabPage7.Controls.Add(this.radTrackFit);
            this.tabPage7.Controls.Add(this.txtCleaningError);
            this.tabPage7.Controls.Add(this.label15);
            this.tabPage7.Controls.Add(this.label61);
            this.tabPage7.Controls.Add(this.label9);
            this.tabPage7.Controls.Add(this.txtExtraTrackIters);
            this.tabPage7.Controls.Add(this.label6);
            this.tabPage7.Controls.Add(this.label59);
            this.tabPage7.Controls.Add(this.label8);
            this.tabPage7.Controls.Add(this.txtTrackFilter);
            this.tabPage7.Controls.Add(this.label7);
            this.tabPage7.Controls.Add(this.label58);
            this.tabPage7.Controls.Add(this.txtPosTol);
            this.tabPage7.Controls.Add(this.txtMinimumCritical);
            this.tabPage7.Controls.Add(this.txtSlopeTol);
            this.tabPage7.Controls.Add(this.label40);
            this.tabPage7.Controls.Add(this.txtInitialPosTol);
            this.tabPage7.Controls.Add(this.txtMinKalman);
            this.tabPage7.Controls.Add(this.txtInitialSlopeTol);
            this.tabPage7.Controls.Add(this.label39);
            this.tabPage7.Controls.Add(this.label12);
            this.tabPage7.Controls.Add(this.txtFittingSegments);
            this.tabPage7.Controls.Add(this.label11);
            this.tabPage7.Controls.Add(this.radKalman);
            this.tabPage7.Controls.Add(this.label10);
            this.tabPage7.Controls.Add(this.chkZfixed);
            this.tabPage7.Controls.Add(this.txtLocCellSize);
            this.tabPage7.Controls.Add(this.chkCorrectSlopes);
            this.tabPage7.Controls.Add(this.txtBeamWid);
            this.tabPage7.Controls.Add(this.txtSlopeIncrement);
            this.tabPage7.Controls.Add(this.txtBeamSX);
            this.tabPage7.Controls.Add(this.txtPosIncrement);
            this.tabPage7.Controls.Add(this.txtBeamSY);
            this.tabPage7.Controls.Add(this.label29);
            this.tabPage7.Controls.Add(this.chkAlignOnLink);
            this.tabPage7.Controls.Add(this.label30);
            this.tabPage7.Controls.Add(this.label13);
            this.tabPage7.Controls.Add(this.txtMaxMissSeg);
            this.tabPage7.Controls.Add(this.label14);
            this.tabPage7.Controls.Add(this.txtMaxIterNum);
            this.tabPage7.Location = new System.Drawing.Point(4, 22);
            this.tabPage7.Name = "tabPage7";
            this.tabPage7.Size = new System.Drawing.Size(368, 310);
            this.tabPage7.TabIndex = 1;
            this.tabPage7.Text = "Track Linking";
            // 
            // tabPage8
            // 
            this.tabPage8.BackColor = System.Drawing.SystemColors.Control;
            this.tabPage8.Controls.Add(this.label69);
            this.tabPage8.Controls.Add(this.txtTrackAlignMinLayerSegments);
            this.tabPage8.Controls.Add(this.label63);
            this.tabPage8.Controls.Add(this.label64);
            this.tabPage8.Controls.Add(this.label65);
            this.tabPage8.Controls.Add(this.label66);
            this.tabPage8.Controls.Add(this.txtTrackAlignMinTrackSegments);
            this.tabPage8.Controls.Add(this.txtTrackAlignTranslationStep);
            this.tabPage8.Controls.Add(this.txtTrackAlignTranslationSweep);
            this.tabPage8.Controls.Add(this.txtTrackAlignRotationStep);
            this.tabPage8.Controls.Add(this.txtTrackAlignOptAcceptance);
            this.tabPage8.Controls.Add(this.txtTrackAlignRotationSweep);
            this.tabPage8.Controls.Add(this.label67);
            this.tabPage8.Controls.Add(this.label68);
            this.tabPage8.Location = new System.Drawing.Point(4, 22);
            this.tabPage8.Name = "tabPage8";
            this.tabPage8.Size = new System.Drawing.Size(368, 310);
            this.tabPage8.TabIndex = 2;
            this.tabPage8.Text = "Alignment by tracks";
            // 
            // label69
            // 
            this.label69.AutoSize = true;
            this.label69.Location = new System.Drawing.Point(14, 157);
            this.label69.Name = "label69";
            this.label69.Size = new System.Drawing.Size(157, 13);
            this.label69.TabIndex = 45;
            this.label69.Text = "Min matching segments in layer:";
            // 
            // txtTrackAlignMinLayerSegments
            // 
            this.txtTrackAlignMinLayerSegments.Location = new System.Drawing.Point(185, 156);
            this.txtTrackAlignMinLayerSegments.Name = "txtTrackAlignMinLayerSegments";
            this.txtTrackAlignMinLayerSegments.Size = new System.Drawing.Size(64, 20);
            this.txtTrackAlignMinLayerSegments.TabIndex = 46;
            // 
            // label63
            // 
            this.label63.AutoSize = true;
            this.label63.Location = new System.Drawing.Point(14, 12);
            this.label63.Name = "label63";
            this.label63.Size = new System.Drawing.Size(161, 13);
            this.label63.TabIndex = 33;
            this.label63.Text = "Min segments in alignment track:";
            // 
            // label64
            // 
            this.label64.AutoSize = true;
            this.label64.Location = new System.Drawing.Point(14, 36);
            this.label64.Name = "label64";
            this.label64.Size = new System.Drawing.Size(125, 13);
            this.label64.TabIndex = 34;
            this.label64.Text = "Translation step (micron):";
            // 
            // label65
            // 
            this.label65.AutoSize = true;
            this.label65.Location = new System.Drawing.Point(14, 60);
            this.label65.Name = "label65";
            this.label65.Size = new System.Drawing.Size(136, 13);
            this.label65.TabIndex = 35;
            this.label65.Text = "Translation sweep (micron):";
            // 
            // label66
            // 
            this.label66.AutoSize = true;
            this.label66.Location = new System.Drawing.Point(14, 84);
            this.label66.Name = "label66";
            this.label66.Size = new System.Drawing.Size(97, 13);
            this.label66.TabIndex = 36;
            this.label66.Text = "Rotation step (rad):";
            // 
            // txtTrackAlignMinTrackSegments
            // 
            this.txtTrackAlignMinTrackSegments.Location = new System.Drawing.Point(185, 11);
            this.txtTrackAlignMinTrackSegments.Name = "txtTrackAlignMinTrackSegments";
            this.txtTrackAlignMinTrackSegments.Size = new System.Drawing.Size(64, 20);
            this.txtTrackAlignMinTrackSegments.TabIndex = 37;
            // 
            // txtTrackAlignTranslationStep
            // 
            this.txtTrackAlignTranslationStep.Location = new System.Drawing.Point(185, 35);
            this.txtTrackAlignTranslationStep.Name = "txtTrackAlignTranslationStep";
            this.txtTrackAlignTranslationStep.Size = new System.Drawing.Size(64, 20);
            this.txtTrackAlignTranslationStep.TabIndex = 38;
            // 
            // txtTrackAlignTranslationSweep
            // 
            this.txtTrackAlignTranslationSweep.Location = new System.Drawing.Point(185, 59);
            this.txtTrackAlignTranslationSweep.Name = "txtTrackAlignTranslationSweep";
            this.txtTrackAlignTranslationSweep.Size = new System.Drawing.Size(64, 20);
            this.txtTrackAlignTranslationSweep.TabIndex = 39;
            // 
            // txtTrackAlignRotationStep
            // 
            this.txtTrackAlignRotationStep.Location = new System.Drawing.Point(185, 83);
            this.txtTrackAlignRotationStep.Name = "txtTrackAlignRotationStep";
            this.txtTrackAlignRotationStep.Size = new System.Drawing.Size(64, 20);
            this.txtTrackAlignRotationStep.TabIndex = 40;
            // 
            // txtTrackAlignOptAcceptance
            // 
            this.txtTrackAlignOptAcceptance.Location = new System.Drawing.Point(185, 131);
            this.txtTrackAlignOptAcceptance.Name = "txtTrackAlignOptAcceptance";
            this.txtTrackAlignOptAcceptance.Size = new System.Drawing.Size(64, 20);
            this.txtTrackAlignOptAcceptance.TabIndex = 44;
            // 
            // txtTrackAlignRotationSweep
            // 
            this.txtTrackAlignRotationSweep.Location = new System.Drawing.Point(185, 107);
            this.txtTrackAlignRotationSweep.Name = "txtTrackAlignRotationSweep";
            this.txtTrackAlignRotationSweep.Size = new System.Drawing.Size(64, 20);
            this.txtTrackAlignRotationSweep.TabIndex = 43;
            // 
            // label67
            // 
            this.label67.AutoSize = true;
            this.label67.Location = new System.Drawing.Point(14, 133);
            this.label67.Name = "label67";
            this.label67.Size = new System.Drawing.Size(107, 13);
            this.label67.TabIndex = 42;
            this.label67.Text = "Position acceptance:";
            // 
            // label68
            // 
            this.label68.AutoSize = true;
            this.label68.Location = new System.Drawing.Point(14, 109);
            this.label68.Name = "label68";
            this.label68.Size = new System.Drawing.Size(108, 13);
            this.label68.TabIndex = 41;
            this.label68.Text = "Rotation sweep (rad):";
            // 
            // tabPage9
            // 
            this.tabPage9.BackColor = System.Drawing.SystemColors.Control;
            this.tabPage9.Controls.Add(this.groupBox3);
            this.tabPage9.Controls.Add(this.chkUpdateTrans);
            this.tabPage9.Controls.Add(this.groupBox5);
            this.tabPage9.Controls.Add(this.groupBox7);
            this.tabPage9.Controls.Add(this.chkKinkDetection);
            this.tabPage9.Location = new System.Drawing.Point(4, 22);
            this.tabPage9.Name = "tabPage9";
            this.tabPage9.Size = new System.Drawing.Size(368, 310);
            this.tabPage9.TabIndex = 3;
            this.tabPage9.Text = "Extras";
            // 
            // label62
            // 
            this.label62.Location = new System.Drawing.Point(9, 31);
            this.label62.Name = "label62";
            this.label62.Size = new System.Drawing.Size(176, 16);
            this.label62.TabIndex = 28;
            this.label62.Text = "Tracking";
            // 
            // chkIgnoreBadLayers
            // 
            this.chkIgnoreBadLayers.Location = new System.Drawing.Point(21, 167);
            this.chkIgnoreBadLayers.Name = "chkIgnoreBadLayers";
            this.chkIgnoreBadLayers.Size = new System.Drawing.Size(177, 33);
            this.chkIgnoreBadLayers.TabIndex = 29;
            this.chkIgnoreBadLayers.Text = "Ignore unaligned layers";
            // 
            // frmAORecEditConfig
            // 
            this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
            this.ClientSize = new System.Drawing.Size(846, 428);
            this.Controls.Add(this.label62);
            this.Controls.Add(this.TkTab);
            this.Controls.Add(this.groupBox4);
            this.Controls.Add(this.label51);
            this.Controls.Add(this.cmdHelp);
            this.Controls.Add(this.cmdCancel);
            this.Controls.Add(this.cmdOk);
            this.Controls.Add(this.cmdDefault);
            this.Controls.Add(this.txtConfigName);
            this.Controls.Add(this.label16);
            this.Controls.Add(this.VtxTab);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
            this.Name = "frmAORecEditConfig";
            this.Text = "Configuration Manager";
            this.Load += new System.EventHandler(this.frmAORecEditConfig_Load);
            this.groupBox3.ResumeLayout(false);
            this.groupBox3.PerformLayout();
            this.groupBox5.ResumeLayout(false);
            this.groupBox5.PerformLayout();
            this.groupBox7.ResumeLayout(false);
            this.groupBox7.PerformLayout();
            this.VtxTab.ResumeLayout(false);
            this.tabPage3.ResumeLayout(false);
            this.tabPage1.ResumeLayout(false);
            this.tabPage1.PerformLayout();
            this.groupBox6.ResumeLayout(false);
            this.groupBox6.PerformLayout();
            this.tabPage2.ResumeLayout(false);
            this.tabPage2.PerformLayout();
            this.groupBox4.ResumeLayout(false);
            this.groupBox4.PerformLayout();
            this.TkTab.ResumeLayout(false);
            this.tabPage6.ResumeLayout(false);
            this.tabPage6.PerformLayout();
            this.tabPage7.ResumeLayout(false);
            this.tabPage7.PerformLayout();
            this.tabPage8.ResumeLayout(false);
            this.tabPage8.PerformLayout();
            this.tabPage9.ResumeLayout(false);
            this.ResumeLayout(false);
            this.PerformLayout();

		}
		#endregion

/*		private void cmdInput_Click(object sender, System.EventArgs e)
		{
			OpenFileDialog of = new OpenFileDialog();
			of.CheckFileExists = true;
			of.Filter = "xml files (*.xml)|*.xml|All files (*.*)|*.*";
			of.FilterIndex = 0;
			if (of.ShowDialog() == DialogResult.OK)
				txtConfigFile.Text = of.FileName;

		}

		private void cmdLoad_Click(object sender, System.EventArgs e)
		{
			System.IO.FileStream f = new System.IO.FileStream(txtConfigFile.Text, System.IO.FileMode.Open, System.IO.FileAccess.Read);
			System.Runtime.Serialization.Formatters.Soap.SoapFormatter fmt = new System.Runtime.Serialization.Formatters.Soap.SoapFormatter();
			AlphaOmegaReconstruction.Configuration AOConfig = (AlphaOmegaReconstruction.Configuration)fmt.Deserialize(f);
			f.Close();

			//Assegnazione
			txtBeamSX.Text = Convert.ToString(AOConfig.AlignBeamSlope.X);
			txtBeamSY.Text = Convert.ToString(AOConfig.AlignBeamSlope.Y);
			txtBeamWid.Text = Convert.ToString(AOConfig.AlignBeamWidth);
			chkAlignOnLink.Checked = AOConfig.AlignOnLinked;
			txtPosTol.Text = Convert.ToString(AOConfig.D_Pos);
			txtSlopeTol.Text = Convert.ToString(AOConfig.D_Slope);
			txtExtents.Text = Convert.ToString(AOConfig.Extents);
			chkFinalAlign.Checked = AOConfig.FinalAlignment_Enable;
			txtInitialPosTol.Text = Convert.ToString(AOConfig.Initial_D_Pos);
			txtInitialSlopeTol.Text = Convert.ToString(AOConfig.Initial_D_Slope);
			txtLeverArm.Text = Convert.ToString(AOConfig.LeverArm);
			txtLocCellSize.Text = Convert.ToString(AOConfig.LocalityCellSize);
			txtMaxIterNum.Text = Convert.ToString(AOConfig.MaxIters);
			txtMaxMissSeg.Text = Convert.ToString(AOConfig.MaxMissingSegments);
			txtSegMaxRemeas.Text = Convert.ToString(AOConfig.MaxRemeasuresInSegment);
			txtZoneWidth.Text = Convert.ToString(AOConfig.ZoneWidth);


		}
*/
		private void cmdOk_Click(object sender, System.EventArgs e)
		{
            try
            {
                //object [] constructorparams = new object [1];
                /*AlphaOmegaReconstruction.Configuration*/
                AOConfig = new Configuration(txtConfigName.Text);
                //Assegnazione
                switch (VtxTab.SelectedIndex)
                {
                    case 1: AOConfig.VtxAlgorithm = VertexAlgorithm.PairBased; break;
                    case 2: AOConfig.VtxAlgorithm = VertexAlgorithm.Global; break;
                    default: AOConfig.VtxAlgorithm = VertexAlgorithm.None; break;
                }
                AOConfig.MinVertexTracksSegments = Convert.ToInt32(txtMinVertexTrackSegments.Text);
                AOConfig.TopologyKink = chkTopologyKink.Checked;
                AOConfig.TopologyLambda = chkTopologyLambda.Checked;
                AOConfig.TopologyV = chkTopologyV.Checked;
                AOConfig.TopologyX = chkTopologyX.Checked;
                AOConfig.TopologyY = chkTopologyY.Checked;
                AOConfig.AlignBeamSlope.X = Convert.ToDouble(txtBeamSX.Text);
                AOConfig.AlignBeamSlope.Y = Convert.ToDouble(txtBeamSY.Text);
                AOConfig.AlignBeamWidth = Convert.ToDouble(txtBeamWid.Text);
                AOConfig.AlignOnLinked = chkAlignOnLink.Checked;
                AOConfig.D_Pos = Convert.ToDouble(txtPosTol.Text);
                AOConfig.D_Slope = Convert.ToDouble(txtSlopeTol.Text);
                AOConfig.D_PosIncrement = Convert.ToDouble(txtPosIncrement.Text);
                AOConfig.D_SlopeIncrement = Convert.ToDouble(txtSlopeIncrement.Text);
                AOConfig.Extents = Convert.ToDouble(txtExtents.Text);
                AOConfig.CorrectSlopesAlign = chkCorrectSlopes.Checked;
                AOConfig.FreezeZ = chkZfixed.Checked;
                AOConfig.Initial_D_Pos = Convert.ToDouble(txtInitialPosTol.Text);
                AOConfig.Initial_D_Slope = Convert.ToDouble(txtInitialSlopeTol.Text);
                AOConfig.LeverArm = Convert.ToDouble(txtLeverArm.Text);
                AOConfig.LocalityCellSize = Convert.ToDouble(txtLocCellSize.Text);
                AOConfig.MaxIters = Convert.ToInt32(txtMaxIterNum.Text);
                AOConfig.ExtraTrackingPasses = Convert.ToInt32(txtExtraTrackIters.Text);
                AOConfig.TrackCleanError = Convert.ToDouble(txtCleaningError.Text);
                AOConfig.TrackCleanChi2Limit = Convert.ToDouble(txtCleaningChi2Limit.Text);
                AOConfig.MaxMissingSegments = Convert.ToInt32(txtMaxMissSeg.Text);
                AOConfig.TrackFilter = (txtTrackFilter.Text.Trim().Length > 0) ? txtTrackFilter.Text.Trim() : null;
                AOConfig.ZoneWidth = Convert.ToDouble(txtZoneWidth.Text);
                AOConfig.PrescanMode = (PrescanModeValue)cmbPrescanMode.SelectedIndex;
                AOConfig.RiskFactor = Convert.ToDouble(txtRiskFactor.Text);
                AOConfig.SlopesCellSize.X = Convert.ToDouble(txtSlopesCellSizeX.Text);
                AOConfig.SlopesCellSize.Y = Convert.ToDouble(txtSlopesCellSizeY.Text);
                AOConfig.MaximumShift.X = Convert.ToDouble(txtMaxShiftX.Text);
                AOConfig.MaximumShift.Y = Convert.ToDouble(txtMaxShiftY.Text);
                AOConfig.CrossTolerance = Convert.ToDouble(txtCrossTolerance.Text);
                AOConfig.MinimumZ = Convert.ToDouble(txtMinimumZ.Text);
                AOConfig.MaximumZ = Convert.ToDouble(txtMaximumZ.Text);
                AOConfig.StartingClusterToleranceLong = Convert.ToDouble(txtStartingClusterToleranceLong.Text);
                AOConfig.StartingClusterToleranceTrans = Convert.ToDouble(txtStartingClusterToleranceTrans.Text);
                AOConfig.MaximumClusterToleranceLong = Convert.ToDouble(txtMaximumClusterToleranceLong.Text);
                AOConfig.MaximumClusterToleranceTrans = Convert.ToDouble(txtMaximumClusterToleranceTrans.Text);
                AOConfig.MinimumSegmentsNumber = Convert.ToInt16(txtMinSegNumber.Text);
                AOConfig.MinimumTracksPairs = Convert.ToInt16(txtMinTracksPairs.Text);
                AOConfig.UpdateTransformations = chkUpdateTrans.Checked;

                AOConfig.UseCells = chkUseCells.Checked;
                AOConfig.Matrix = Convert.ToDouble(txtMatrix.Text);
                AOConfig.XCellSize = Convert.ToDouble(txtXCellsSize.Text);
                AOConfig.YCellSize = Convert.ToDouble(txtYCellsSize.Text);
                AOConfig.ZCellSize = Convert.ToDouble(txtZCellsSize.Text);

                AOConfig.KalmanFilter = radKalman.Checked;
                AOConfig.FittingTracks = Convert.ToInt16(txtFittingSegments.Text);
                AOConfig.MinKalman = Convert.ToInt16(txtMinKalman.Text);
                AOConfig.MinimumCritical = Convert.ToInt16(txtMinimumCritical.Text);

                AOConfig.KinkDetection = chkKinkDetection.Checked;
                AOConfig.KinkMinimumSegments = Convert.ToInt32(txtKinkMinSeg.Text);
                AOConfig.KinkFactor = Convert.ToDouble(txtKinkFactor.Text);
                AOConfig.KinkMinimumDeltaS = Convert.ToDouble(txtKinkMinSlopeDiff.Text);
                AOConfig.FilterThreshold = Convert.ToDouble(txtKinkFilterThreshold.Text);
                AOConfig.FilterLength = Convert.ToInt32(cmbFilterLength.Text);

                AOConfig.GVtxMaxExt = Convert.ToDouble(txtGVtxMaxExt.Text);
                AOConfig.GVtxRadius = Convert.ToDouble(txtGVtxRadius.Text);
                AOConfig.GVtxMaxSlopeDivergence = Convert.ToDouble(txtGVtxMaxSlopeDivergence.Text);
                AOConfig.GVtxMinCount = Convert.ToInt32(txtGVtxMinCount.Text);
                AOConfig.GVtxFilter = (txtGVtxFilter.Text.Trim().Length > 0) ? txtGVtxFilter.Text.Trim() : null;

                AOConfig.VtxFitWeightEnable = VtxFitWeightEnableCheck.Checked;
                AOConfig.VtxFitWeightOptStepXY = Convert.ToDouble(VtxFitWeightXYStepText.Text);
                AOConfig.VtxFitWeightOptStepZ = Convert.ToDouble(VtxFitWeightZStepText.Text);
                AOConfig.VtxFitWeightTol = Convert.ToDouble(VtxFitWeightTolText.Text);

                AOConfig.RelinkAperture = Convert.ToDouble(txtRelinkAperture.Text);
                AOConfig.RelinkDeltaZ = Convert.ToDouble(txtRelinkDeltaZ.Text);
                AOConfig.RelinkEnable = chkRelinkEnable.Checked;

                AOConfig.TrackAlignMinLayerSegments = (int)Convert.ToUInt32(txtTrackAlignMinLayerSegments.Text);
                AOConfig.TrackAlignMinTrackSegments = (int)Convert.ToUInt32(txtTrackAlignMinTrackSegments.Text);
                AOConfig.TrackAlignOptAcceptance = Convert.ToDouble(txtTrackAlignOptAcceptance.Text);
                AOConfig.TrackAlignRotationStep = Convert.ToDouble(txtTrackAlignRotationStep.Text);
                AOConfig.TrackAlignRotationSweep = Convert.ToDouble(txtTrackAlignRotationSweep.Text);
                AOConfig.TrackAlignTranslationStep = Convert.ToDouble(txtTrackAlignTranslationStep.Text);
                AOConfig.TrackAlignTranslationSweep = Convert.ToDouble(txtTrackAlignTranslationSweep.Text);
                AOConfig.IgnoreBadLayers = chkIgnoreBadLayers.Checked;                

                DialogResult = DialogResult.OK;
                Close();
            }
            catch (Exception x)
            {
                MessageBox.Show(x.ToString(), "Input error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }

		}

		private void cmdDefault_Click(object sender, System.EventArgs e)
		{
			AlphaOmegaReconstruction.AlphaOmegaReconstructor aorec = new AlphaOmegaReconstruction.AlphaOmegaReconstructor();
			AOConfig = (AlphaOmegaReconstruction.Configuration)aorec.Config;
			//Assegnazione

			switch (AOConfig.VtxAlgorithm)
			{
				case VertexAlgorithm.PairBased:		VtxTab.SelectedIndex = 1; break;
				case VertexAlgorithm.Global:		VtxTab.SelectedIndex = 2; break;
				default:							VtxTab.SelectedIndex = 0; break;
			}

			VtxFitWeightEnableCheck.Enabled = AOConfig.VtxFitWeightEnable;
			VtxFitWeightXYStepText.Text = AOConfig.VtxFitWeightOptStepXY.ToString();
			VtxFitWeightZStepText.Text = AOConfig.VtxFitWeightOptStepZ.ToString();
			VtxFitWeightTolText.Text = AOConfig.VtxFitWeightTol.ToString();

			txtGVtxMinCount.Text = AOConfig.GVtxMinCount.ToString();
			txtGVtxMaxExt.Text = AOConfig.GVtxMaxExt.ToString();
			txtGVtxRadius.Text = AOConfig.GVtxRadius.ToString();
			txtGVtxMaxSlopeDivergence.Text = AOConfig.GVtxMaxSlopeDivergence.ToString();

			txtConfigName.Text = AOConfig.Name;
			txtMinVertexTrackSegments.Text= Convert.ToString(AOConfig.MinVertexTracksSegments); 
			chkTopologyKink.Checked = AOConfig.TopologyKink;
			chkTopologyLambda.Checked = AOConfig.TopologyLambda;
			chkTopologyV.Checked = AOConfig.TopologyV;
			chkTopologyX.Checked = AOConfig.TopologyX;
			chkTopologyY.Checked = AOConfig.TopologyY;
			txtBeamSX.Text = Convert.ToString(AOConfig.AlignBeamSlope.X);
			txtBeamSY.Text = Convert.ToString(AOConfig.AlignBeamSlope.Y);
			txtBeamWid.Text = Convert.ToString(AOConfig.AlignBeamWidth);
			chkAlignOnLink.Checked = AOConfig.AlignOnLinked;
			txtPosTol.Text = Convert.ToString(AOConfig.D_Pos);
			txtSlopeTol.Text = Convert.ToString(AOConfig.D_Slope);
			txtPosIncrement.Text = Convert.ToString(AOConfig.D_PosIncrement);
			txtSlopeIncrement.Text = Convert.ToString(AOConfig.D_SlopeIncrement);
			txtExtents.Text = Convert.ToString(AOConfig.Extents);
			chkCorrectSlopes.Checked = AOConfig.CorrectSlopesAlign;
			chkZfixed.Checked = AOConfig.FreezeZ;
			txtInitialPosTol.Text = Convert.ToString(AOConfig.Initial_D_Pos);
			txtInitialSlopeTol.Text = Convert.ToString(AOConfig.Initial_D_Slope);
			txtLeverArm.Text = Convert.ToString(AOConfig.LeverArm);
			txtLocCellSize.Text = Convert.ToString(AOConfig.LocalityCellSize);
			txtMaxIterNum.Text = Convert.ToString(AOConfig.MaxIters);
            txtExtraTrackIters.Text = Convert.ToString(AOConfig.ExtraTrackingPasses);
            txtCleaningError.Text = Convert.ToString(AOConfig.TrackCleanError);
            txtCleaningChi2Limit.Text = Convert.ToString(AOConfig.TrackCleanChi2Limit);
			txtMaxMissSeg.Text = Convert.ToString(AOConfig.MaxMissingSegments);
			txtZoneWidth.Text = Convert.ToString(AOConfig.ZoneWidth);
			cmbPrescanMode.SelectedItem = AOConfig.PrescanMode;
			txtRiskFactor.Text = Convert.ToString(AOConfig.RiskFactor);
			txtSlopesCellSizeX.Text = Convert.ToString(AOConfig.SlopesCellSize.X);
			txtSlopesCellSizeY.Text = Convert.ToString(AOConfig.SlopesCellSize.Y);
			txtMaxShiftX.Text = Convert.ToString(AOConfig.MaximumShift.X);
			txtMaxShiftY.Text = Convert.ToString(AOConfig.MaximumShift.Y);
			txtCrossTolerance.Text = Convert.ToString(AOConfig.CrossTolerance);
			txtMinimumZ.Text = Convert.ToString(AOConfig.MinimumZ);
			txtMaximumZ.Text = Convert.ToString(AOConfig.MaximumZ);
			txtStartingClusterToleranceLong.Text = Convert.ToString(AOConfig.StartingClusterToleranceLong);
			txtStartingClusterToleranceTrans.Text = Convert.ToString(AOConfig.StartingClusterToleranceTrans);
			txtMaximumClusterToleranceLong.Text = Convert.ToString(AOConfig.MaximumClusterToleranceLong);
			txtMaximumClusterToleranceTrans.Text = Convert.ToString(AOConfig.MaximumClusterToleranceTrans);
			txtMinSegNumber.Text = Convert.ToString(AOConfig.MinimumSegmentsNumber);
			txtMinTracksPairs.Text = Convert.ToString(AOConfig.MinimumTracksPairs);
			chkUpdateTrans.Checked = AOConfig.UpdateTransformations;

			chkUseCells.Checked = AOConfig.UseCells;
			txtMatrix.Text = Convert.ToString(AOConfig.Matrix);
			txtXCellsSize.Text = Convert.ToString(AOConfig.XCellSize);
			txtYCellsSize.Text = Convert.ToString(AOConfig.YCellSize);
			txtZCellsSize.Text = Convert.ToString(AOConfig.ZCellSize);

			radKalman.Checked = AOConfig.KalmanFilter;
			radTrackFit.Checked = !AOConfig.KalmanFilter;
			txtFittingSegments.Text = Convert.ToString(AOConfig.FittingTracks);
			txtMinKalman.Text = Convert.ToString(AOConfig.MinKalman);
			txtMinimumCritical.Text = Convert.ToString(AOConfig.MinimumCritical);

			chkKinkDetection.Checked = AOConfig.KinkDetection;
			txtKinkMinSeg.Text = Convert.ToString(AOConfig.KinkMinimumSegments);
			txtKinkFactor.Text = Convert.ToString(AOConfig.KinkFactor);
			txtKinkMinSlopeDiff.Text = Convert.ToString(AOConfig.KinkMinimumDeltaS);
			txtKinkFilterThreshold.Text = Convert.ToString(AOConfig.FilterThreshold);
			cmbFilterLength.Text = Convert.ToString(AOConfig.FilterLength);

            txtRelinkAperture.Text = Convert.ToString(AOConfig.RelinkAperture);
            txtRelinkDeltaZ.Text = Convert.ToString(AOConfig.RelinkDeltaZ);
            chkRelinkEnable.Checked = AOConfig.RelinkEnable;

            txtTrackAlignMinLayerSegments.Text = Convert.ToString(AOConfig.TrackAlignMinLayerSegments);
            txtTrackAlignMinTrackSegments.Text = Convert.ToString(AOConfig.TrackAlignMinTrackSegments);
            txtTrackAlignOptAcceptance.Text = Convert.ToString(AOConfig.TrackAlignOptAcceptance);
            txtTrackAlignRotationStep.Text = Convert.ToString(AOConfig.TrackAlignRotationStep);
            txtTrackAlignRotationSweep.Text = Convert.ToString(AOConfig.TrackAlignRotationSweep);
            txtTrackAlignTranslationStep.Text = Convert.ToString(AOConfig.TrackAlignTranslationStep);
            txtTrackAlignTranslationSweep.Text = Convert.ToString(AOConfig.TrackAlignTranslationSweep);
            chkIgnoreBadLayers.Checked = false;
		}

		private void frmAORecEditConfig_Load(object sender, System.EventArgs e)
		{			
			txtConfigName.Text = AOConfig.Name;
			switch (AOConfig.VtxAlgorithm)
			{
				case VertexAlgorithm.PairBased:		VtxTab.SelectedIndex = 1; break;
				case VertexAlgorithm.Global:		VtxTab.SelectedIndex = 2; break;
				default:							VtxTab.SelectedIndex = 0; break;
			}

			VtxFitWeightEnableCheck.Checked = AOConfig.VtxFitWeightEnable;
			VtxFitWeightXYStepText.Text = AOConfig.VtxFitWeightOptStepXY.ToString();
			VtxFitWeightZStepText.Text = AOConfig.VtxFitWeightOptStepZ.ToString();
			VtxFitWeightTolText.Text = AOConfig.VtxFitWeightTol.ToString();

			txtGVtxMinCount.Text = AOConfig.GVtxMinCount.ToString();
			txtGVtxMaxExt.Text = AOConfig.GVtxMaxExt.ToString();
			txtGVtxRadius.Text = AOConfig.GVtxRadius.ToString();
			txtGVtxMaxSlopeDivergence.Text = AOConfig.GVtxMaxSlopeDivergence.ToString();
            txtGVtxFilter.Text = (AOConfig.GVtxFilter == null) ? "" : AOConfig.GVtxFilter;

			txtMinVertexTrackSegments.Text= Convert.ToString(AOConfig.MinVertexTracksSegments); 
			chkTopologyKink.Checked = AOConfig.TopologyKink;
			chkTopologyLambda.Checked = AOConfig.TopologyLambda;
			chkTopologyV.Checked = AOConfig.TopologyV;
			chkTopologyX.Checked = AOConfig.TopologyX;
			chkTopologyY.Checked = AOConfig.TopologyY;
			txtBeamSX.Text = Convert.ToString(AOConfig.AlignBeamSlope.X);
			txtBeamSY.Text = Convert.ToString(AOConfig.AlignBeamSlope.Y);
			txtBeamWid.Text = Convert.ToString(AOConfig.AlignBeamWidth);
			chkAlignOnLink.Checked = AOConfig.AlignOnLinked;
			txtPosTol.Text = Convert.ToString(AOConfig.D_Pos);
			txtSlopeTol.Text = Convert.ToString(AOConfig.D_Slope);
			txtPosIncrement.Text = Convert.ToString(AOConfig.D_PosIncrement);
			txtSlopeIncrement.Text = Convert.ToString(AOConfig.D_SlopeIncrement);
			txtExtents.Text = Convert.ToString(AOConfig.Extents);
			chkCorrectSlopes.Checked = AOConfig.CorrectSlopesAlign;
			chkZfixed.Checked = AOConfig.FreezeZ;
			txtInitialPosTol.Text = Convert.ToString(AOConfig.Initial_D_Pos);
			txtInitialSlopeTol.Text = Convert.ToString(AOConfig.Initial_D_Slope);
			txtLeverArm.Text = Convert.ToString(AOConfig.LeverArm);
			txtLocCellSize.Text = Convert.ToString(AOConfig.LocalityCellSize);
			txtMaxIterNum.Text = Convert.ToString(AOConfig.MaxIters);
            txtExtraTrackIters.Text = Convert.ToString(AOConfig.ExtraTrackingPasses);            
            txtCleaningError.Text = Convert.ToString(AOConfig.TrackCleanError);
            txtCleaningChi2Limit.Text = Convert.ToString(AOConfig.TrackCleanChi2Limit);
			txtMaxMissSeg.Text = Convert.ToString(AOConfig.MaxMissingSegments);
            txtTrackFilter.Text = (AOConfig.TrackFilter == null) ? "" : AOConfig.TrackFilter;
			txtZoneWidth.Text = Convert.ToString(AOConfig.ZoneWidth);
			cmbPrescanMode.Items.Add(PrescanModeValue.None);
			cmbPrescanMode.Items.Add(PrescanModeValue.Translation);
			cmbPrescanMode.Items.Add(PrescanModeValue.Affine);
			cmbPrescanMode.Items.Add(PrescanModeValue.Rototranslation);
			cmbFilterLength.Items.Add(6);
			cmbFilterLength.Items.Add(10);
			cmbFilterLength.Items.Add(14);
			cmbFilterLength.SelectedIndex = 0;
			cmbPrescanMode.SelectedIndex = (int)AOConfig.PrescanMode;
			txtRiskFactor.Text = Convert.ToString(AOConfig.RiskFactor);
			txtSlopesCellSizeX.Text = Convert.ToString(AOConfig.SlopesCellSize.X);
			txtSlopesCellSizeY.Text = Convert.ToString(AOConfig.SlopesCellSize.Y);
			txtMaxShiftX.Text = Convert.ToString(AOConfig.MaximumShift.X);
			txtMaxShiftY.Text = Convert.ToString(AOConfig.MaximumShift.Y);
			txtCrossTolerance.Text = Convert.ToString(AOConfig.CrossTolerance);
			txtMinimumZ.Text = Convert.ToString(AOConfig.MinimumZ);
			txtMaximumZ.Text = Convert.ToString(AOConfig.MaximumZ);
			txtStartingClusterToleranceLong.Text = Convert.ToString(AOConfig.StartingClusterToleranceLong);
			txtStartingClusterToleranceTrans.Text = Convert.ToString(AOConfig.StartingClusterToleranceTrans);
			txtMaximumClusterToleranceLong.Text = Convert.ToString(AOConfig.MaximumClusterToleranceLong);
			txtMaximumClusterToleranceTrans.Text = Convert.ToString(AOConfig.MaximumClusterToleranceTrans);
			txtMinSegNumber.Text = Convert.ToString(AOConfig.MinimumSegmentsNumber);
			txtMinTracksPairs.Text = Convert.ToString(AOConfig.MinimumTracksPairs);
			chkUpdateTrans.Checked = AOConfig.UpdateTransformations;

			chkUseCells.Checked = AOConfig.UseCells;
			txtMatrix.Text = Convert.ToString(AOConfig.Matrix);
			txtXCellsSize.Text = Convert.ToString(AOConfig.XCellSize);
			txtYCellsSize.Text = Convert.ToString(AOConfig.YCellSize);
			txtZCellsSize.Text = Convert.ToString(AOConfig.ZCellSize);

			radKalman.Checked = AOConfig.KalmanFilter;
			radTrackFit.Checked = !AOConfig.KalmanFilter;
			txtFittingSegments.Text = Convert.ToString(AOConfig.FittingTracks);
			txtMinKalman.Text = Convert.ToString(AOConfig.MinKalman);
			txtMinimumCritical.Text = Convert.ToString(AOConfig.MinimumCritical);
		
			chkKinkDetection.Checked = AOConfig.KinkDetection;
			txtKinkMinSeg.Text = Convert.ToString(AOConfig.KinkMinimumSegments);
			txtKinkFactor.Text = Convert.ToString(AOConfig.KinkFactor);
			txtKinkMinSlopeDiff.Text = Convert.ToString(AOConfig.KinkMinimumDeltaS);
			txtKinkFilterThreshold.Text = Convert.ToString(AOConfig.FilterThreshold);
			cmbFilterLength.Text = Convert.ToString(AOConfig.FilterLength);

            txtRelinkAperture.Text = Convert.ToString(AOConfig.RelinkAperture);
            txtRelinkDeltaZ.Text = Convert.ToString(AOConfig.RelinkDeltaZ);
            chkRelinkEnable.Checked = AOConfig.RelinkEnable;

            txtTrackAlignMinLayerSegments.Text = Convert.ToString(AOConfig.TrackAlignMinLayerSegments);
            txtTrackAlignMinTrackSegments.Text = Convert.ToString(AOConfig.TrackAlignMinTrackSegments);
            txtTrackAlignOptAcceptance.Text = Convert.ToString(AOConfig.TrackAlignOptAcceptance);
            txtTrackAlignRotationStep.Text = Convert.ToString(AOConfig.TrackAlignRotationStep);
            txtTrackAlignRotationSweep.Text = Convert.ToString(AOConfig.TrackAlignRotationSweep);
            txtTrackAlignTranslationStep.Text = Convert.ToString(AOConfig.TrackAlignTranslationStep);
            txtTrackAlignTranslationSweep.Text = Convert.ToString(AOConfig.TrackAlignTranslationSweep);

            chkIgnoreBadLayers.Checked = AOConfig.IgnoreBadLayers;
		}

		private void cmdHelp_Click(object sender, System.EventArgs e)
		{
			string chmpath = System.Diagnostics.Process.GetCurrentProcess().MainModule.FileName;
			chmpath = chmpath.Remove(chmpath.Length - 3, 3) + "chm";
			System.Windows.Forms.Help.ShowHelp(this, chmpath, "RecConfig.htm");

		}

	}
}
