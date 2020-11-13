namespace SySal.Executables.NExTScanner
{
    partial class FullSpeedAcquisitionForm
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.components = new System.ComponentModel.Container();
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(FullSpeedAcquisitionForm));
            System.Windows.Forms.ListViewItem listViewItem1 = new System.Windows.Forms.ListViewItem(new string[] {
            "ClusterChainer",
            ""}, -1);
            System.Windows.Forms.ListViewItem listViewItem2 = new System.Windows.Forms.ListViewItem(new string[] {
            "Tracker",
            ""}, -1);
            System.Windows.Forms.ListViewItem listViewItem3 = new System.Windows.Forms.ListViewItem(new string[] {
            "Cleaner",
            ""}, -1);
            System.Windows.Forms.ListViewItem listViewItem4 = new System.Windows.Forms.ListViewItem(new string[] {
            "Linker",
            ""}, -1);
            System.Windows.Forms.ListViewItem listViewItem5 = new System.Windows.Forms.ListViewItem(new string[] {
            "Selector",
            ""}, -1);
            this.label1 = new System.Windows.Forms.Label();
            this.txtLayers = new System.Windows.Forms.TextBox();
            this.label2 = new System.Windows.Forms.Label();
            this.txtPitch = new System.Windows.Forms.TextBox();
            this.label3 = new System.Windows.Forms.Label();
            this.txtZSweep = new System.Windows.Forms.TextBox();
            this.label4 = new System.Windows.Forms.Label();
            this.txtClusterThreshold = new System.Windows.Forms.TextBox();
            this.label5 = new System.Windows.Forms.Label();
            this.txtMinValidLayers = new System.Windows.Forms.TextBox();
            this.label6 = new System.Windows.Forms.Label();
            this.txtFocusSweep = new System.Windows.Forms.TextBox();
            this.chkTop = new System.Windows.Forms.CheckBox();
            this.chkBottom = new System.Windows.Forms.CheckBox();
            this.label7 = new System.Windows.Forms.Label();
            this.txtBaseThickness = new System.Windows.Forms.TextBox();
            this.label8 = new System.Windows.Forms.Label();
            this.txtFOVSize = new System.Windows.Forms.TextBox();
            this.label9 = new System.Windows.Forms.Label();
            this.txtViewOverlap = new System.Windows.Forms.TextBox();
            this.rdMoveX = new System.Windows.Forms.RadioButton();
            this.rdMoveY = new System.Windows.Forms.RadioButton();
            this.label10 = new System.Windows.Forms.Label();
            this.txtXYSpeed = new System.Windows.Forms.TextBox();
            this.label11 = new System.Windows.Forms.Label();
            this.txtXYAcceleration = new System.Windows.Forms.TextBox();
            this.label12 = new System.Windows.Forms.Label();
            this.txtZAcceleration = new System.Windows.Forms.TextBox();
            this.label13 = new System.Windows.Forms.Label();
            this.txtZSpeed = new System.Windows.Forms.TextBox();
            this.label14 = new System.Windows.Forms.Label();
            this.txtSummaryFile = new System.Windows.Forms.TextBox();
            this.label15 = new System.Windows.Forms.Label();
            this.txtMinX = new System.Windows.Forms.TextBox();
            this.txtMinY = new System.Windows.Forms.TextBox();
            this.label16 = new System.Windows.Forms.Label();
            this.txtMaxY = new System.Windows.Forms.TextBox();
            this.txtMaxX = new System.Windows.Forms.TextBox();
            this.btnFromHere = new SySal.SySalNExTControls.SySalButton();
            this.btnToHere = new SySal.SySalNExTControls.SySalButton();
            this.panel1 = new System.Windows.Forms.Panel();
            this.txtCurrentConfiguration = new System.Windows.Forms.TextBox();
            this.btnMakeCurrent = new SySal.SySalNExTControls.SySalButton();
            this.btnDel = new SySal.SySalNExTControls.SySalButton();
            this.btnLoad = new SySal.SySalNExTControls.SySalButton();
            this.txtNewName = new System.Windows.Forms.TextBox();
            this.btnDuplicate = new SySal.SySalNExTControls.SySalButton();
            this.lbConfigurations = new System.Windows.Forms.ListBox();
            this.btnNew = new SySal.SySalNExTControls.SySalButton();
            this.panel2 = new System.Windows.Forms.Panel();
            this.btnStart = new SySal.SySalNExTControls.SySalButton();
            this.btnStop = new SySal.SySalNExTControls.SySalButton();
            this.pbScanProgress = new SySal.SySalNExTControls.SySalProgressBar();
            this.label17 = new System.Windows.Forms.Label();
            this.txtSlowdownTime = new System.Windows.Forms.TextBox();
            this.label18 = new System.Windows.Forms.Label();
            this.txtPositionTolerance = new System.Windows.Forms.TextBox();
            this.btnExit = new SySal.SySalNExTControls.SySalButton();
            this.label19 = new System.Windows.Forms.Label();
            this.txtContinuousMotionFraction = new System.Windows.Forms.TextBox();
            this.label20 = new System.Windows.Forms.Label();
            this.txtEmuThickness = new System.Windows.Forms.TextBox();
            this.txtStatus = new System.Windows.Forms.TextBox();
            this.txtFPS = new System.Windows.Forms.TextBox();
            this.btnMeasureFramesPerSecond = new SySal.SySalNExTControls.SySalButton();
            this.label21 = new System.Windows.Forms.Label();
            this.label22 = new System.Windows.Forms.Label();
            this.pbQueueLength = new SySal.SySalNExTControls.SySalProgressBar();
            this.btnContinuousMotionFraction = new SySal.SySalNExTControls.SySalButton();
            this.sdlgZTest = new System.Windows.Forms.SaveFileDialog();
            this.label23 = new System.Windows.Forms.Label();
            this.txtMotionLatency = new System.Windows.Forms.TextBox();
            this.chkStageSimOnly = new System.Windows.Forms.CheckBox();
            this.label24 = new System.Windows.Forms.Label();
            this.pbWriteQueueLength = new SySal.SySalNExTControls.SySalProgressBar();
            this.chkEnableGPUTest = new System.Windows.Forms.CheckBox();
            this.btnEmptyFocusMap = new SySal.SySalNExTControls.SySalButton();
            this.label25 = new System.Windows.Forms.Label();
            this.txtZDataExpirationMin = new System.Windows.Forms.TextBox();
            this.label26 = new System.Windows.Forms.Label();
            this.txtMaxViewsStrip = new System.Windows.Forms.TextBox();
            this.btnDumpZMap = new SySal.SySalNExTControls.SySalButton();
            this.chkDumpFocusInfo = new System.Windows.Forms.CheckBox();
            this.chkWriteClusterFiles = new System.Windows.Forms.CheckBox();
            this.sdlgZDump = new System.Windows.Forms.SaveFileDialog();
            this.label27 = new System.Windows.Forms.Label();
            this.txtTransfMaxY = new System.Windows.Forms.TextBox();
            this.txtTransfMaxX = new System.Windows.Forms.TextBox();
            this.label28 = new System.Windows.Forms.Label();
            this.txtTransfMinY = new System.Windows.Forms.TextBox();
            this.txtTransfMinX = new System.Windows.Forms.TextBox();
            this.lvPostProcessing = new System.Windows.Forms.ListView();
            this.columnHeader1 = ((System.Windows.Forms.ColumnHeader)(new System.Windows.Forms.ColumnHeader()));
            this.columnHeader2 = ((System.Windows.Forms.ColumnHeader)(new System.Windows.Forms.ColumnHeader()));
            this.btnStepReset = new SySal.SySalNExTControls.SySalButton();
            this.btnStepPaste = new SySal.SySalNExTControls.SySalButton();
            this.FullSpeedAcqToolTip = new System.Windows.Forms.ToolTip(this.components);
            this.btnStepSave = new SySal.SySalNExTControls.SySalButton();
            this.btnStepCopy = new SySal.SySalNExTControls.SySalButton();
            this.btnRecover = new SySal.SySalNExTControls.SySalButton();
            this.btnDebug = new SySal.SySalNExTControls.SySalButton();
            this.odlgRecoverZone = new System.Windows.Forms.OpenFileDialog();
            this.SuspendLayout();
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.BackColor = System.Drawing.Color.Transparent;
            this.label1.Location = new System.Drawing.Point(454, 36);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(55, 21);
            this.label1.TabIndex = 3;
            this.label1.Text = "Layers";
            // 
            // txtLayers
            // 
            this.txtLayers.BackColor = System.Drawing.Color.GhostWhite;
            this.txtLayers.Location = new System.Drawing.Point(601, 33);
            this.txtLayers.Name = "txtLayers";
            this.txtLayers.Size = new System.Drawing.Size(51, 29);
            this.txtLayers.TabIndex = 4;
            this.txtLayers.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.txtLayers.Leave += new System.EventHandler(this.OnLayersLeave);
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.BackColor = System.Drawing.Color.Transparent;
            this.label2.Location = new System.Drawing.Point(454, 71);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(44, 21);
            this.label2.TabIndex = 5;
            this.label2.Text = "Pitch";
            // 
            // txtPitch
            // 
            this.txtPitch.BackColor = System.Drawing.Color.GhostWhite;
            this.txtPitch.Location = new System.Drawing.Point(601, 68);
            this.txtPitch.Name = "txtPitch";
            this.txtPitch.Size = new System.Drawing.Size(51, 29);
            this.txtPitch.TabIndex = 6;
            this.txtPitch.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.txtPitch.Leave += new System.EventHandler(this.OnPitchLeave);
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.BackColor = System.Drawing.Color.Transparent;
            this.label3.Location = new System.Drawing.Point(454, 106);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(67, 21);
            this.label3.TabIndex = 7;
            this.label3.Text = "Z sweep";
            // 
            // txtZSweep
            // 
            this.txtZSweep.BackColor = System.Drawing.Color.LightGray;
            this.txtZSweep.Location = new System.Drawing.Point(601, 103);
            this.txtZSweep.Name = "txtZSweep";
            this.txtZSweep.ReadOnly = true;
            this.txtZSweep.Size = new System.Drawing.Size(51, 29);
            this.txtZSweep.TabIndex = 8;
            this.txtZSweep.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.BackColor = System.Drawing.Color.Transparent;
            this.label4.Location = new System.Drawing.Point(454, 141);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(129, 21);
            this.label4.TabIndex = 9;
            this.label4.Text = "Cluster threshold";
            // 
            // txtClusterThreshold
            // 
            this.txtClusterThreshold.BackColor = System.Drawing.Color.GhostWhite;
            this.txtClusterThreshold.Location = new System.Drawing.Point(601, 138);
            this.txtClusterThreshold.Name = "txtClusterThreshold";
            this.txtClusterThreshold.Size = new System.Drawing.Size(51, 29);
            this.txtClusterThreshold.TabIndex = 10;
            this.txtClusterThreshold.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.txtClusterThreshold.Leave += new System.EventHandler(this.OnClusterThresholdLeave);
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.BackColor = System.Drawing.Color.Transparent;
            this.label5.Location = new System.Drawing.Point(454, 177);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(122, 21);
            this.label5.TabIndex = 11;
            this.label5.Text = "Min. valid layers";
            // 
            // txtMinValidLayers
            // 
            this.txtMinValidLayers.BackColor = System.Drawing.Color.GhostWhite;
            this.txtMinValidLayers.Location = new System.Drawing.Point(601, 174);
            this.txtMinValidLayers.Name = "txtMinValidLayers";
            this.txtMinValidLayers.Size = new System.Drawing.Size(51, 29);
            this.txtMinValidLayers.TabIndex = 12;
            this.txtMinValidLayers.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.txtMinValidLayers.Leave += new System.EventHandler(this.OnMinValidLayersLeave);
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.BackColor = System.Drawing.Color.Transparent;
            this.label6.Location = new System.Drawing.Point(454, 213);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(98, 21);
            this.label6.TabIndex = 13;
            this.label6.Text = "Focus sweep";
            // 
            // txtFocusSweep
            // 
            this.txtFocusSweep.BackColor = System.Drawing.Color.GhostWhite;
            this.txtFocusSweep.Location = new System.Drawing.Point(601, 210);
            this.txtFocusSweep.Name = "txtFocusSweep";
            this.txtFocusSweep.Size = new System.Drawing.Size(51, 29);
            this.txtFocusSweep.TabIndex = 14;
            this.txtFocusSweep.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.txtFocusSweep.Leave += new System.EventHandler(this.OnFocusSweepLeave);
            // 
            // chkTop
            // 
            this.chkTop.AutoSize = true;
            this.chkTop.BackColor = System.Drawing.Color.Transparent;
            this.chkTop.Location = new System.Drawing.Point(458, 246);
            this.chkTop.Name = "chkTop";
            this.chkTop.Size = new System.Drawing.Size(55, 25);
            this.chkTop.TabIndex = 15;
            this.chkTop.Text = "Top";
            this.chkTop.UseVisualStyleBackColor = false;
            this.chkTop.CheckedChanged += new System.EventHandler(this.OnTopChecked);
            // 
            // chkBottom
            // 
            this.chkBottom.AutoSize = true;
            this.chkBottom.BackColor = System.Drawing.Color.Transparent;
            this.chkBottom.Location = new System.Drawing.Point(532, 246);
            this.chkBottom.Name = "chkBottom";
            this.chkBottom.Size = new System.Drawing.Size(80, 25);
            this.chkBottom.TabIndex = 16;
            this.chkBottom.Text = "Bottom";
            this.chkBottom.UseVisualStyleBackColor = false;
            this.chkBottom.CheckedChanged += new System.EventHandler(this.OnBottomChecked);
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.BackColor = System.Drawing.Color.Transparent;
            this.label7.Location = new System.Drawing.Point(454, 282);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(110, 21);
            this.label7.TabIndex = 17;
            this.label7.Text = "Base thickness";
            // 
            // txtBaseThickness
            // 
            this.txtBaseThickness.BackColor = System.Drawing.Color.GhostWhite;
            this.txtBaseThickness.Location = new System.Drawing.Point(601, 279);
            this.txtBaseThickness.Name = "txtBaseThickness";
            this.txtBaseThickness.Size = new System.Drawing.Size(51, 29);
            this.txtBaseThickness.TabIndex = 18;
            this.txtBaseThickness.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.txtBaseThickness.Leave += new System.EventHandler(this.OnBaseThicknessLeave);
            // 
            // label8
            // 
            this.label8.AutoSize = true;
            this.label8.BackColor = System.Drawing.Color.Transparent;
            this.label8.Location = new System.Drawing.Point(672, 35);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(70, 21);
            this.label8.TabIndex = 19;
            this.label8.Text = "FOV size";
            // 
            // txtFOVSize
            // 
            this.txtFOVSize.BackColor = System.Drawing.Color.LightGray;
            this.txtFOVSize.Location = new System.Drawing.Point(750, 32);
            this.txtFOVSize.Name = "txtFOVSize";
            this.txtFOVSize.ReadOnly = true;
            this.txtFOVSize.Size = new System.Drawing.Size(120, 29);
            this.txtFOVSize.TabIndex = 20;
            this.txtFOVSize.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label9
            // 
            this.label9.AutoSize = true;
            this.label9.BackColor = System.Drawing.Color.Transparent;
            this.label9.Location = new System.Drawing.Point(670, 70);
            this.label9.Name = "label9";
            this.label9.Size = new System.Drawing.Size(100, 21);
            this.label9.TabIndex = 21;
            this.label9.Text = "View overlap";
            // 
            // txtViewOverlap
            // 
            this.txtViewOverlap.BackColor = System.Drawing.Color.GhostWhite;
            this.txtViewOverlap.Location = new System.Drawing.Point(817, 67);
            this.txtViewOverlap.Name = "txtViewOverlap";
            this.txtViewOverlap.Size = new System.Drawing.Size(51, 29);
            this.txtViewOverlap.TabIndex = 22;
            this.txtViewOverlap.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.txtViewOverlap.Leave += new System.EventHandler(this.OnViewOverlapLeave);
            // 
            // rdMoveX
            // 
            this.rdMoveX.AutoSize = true;
            this.rdMoveX.BackColor = System.Drawing.Color.Transparent;
            this.rdMoveX.Location = new System.Drawing.Point(673, 136);
            this.rdMoveX.Name = "rdMoveX";
            this.rdMoveX.Size = new System.Drawing.Size(80, 25);
            this.rdMoveX.TabIndex = 24;
            this.rdMoveX.TabStop = true;
            this.rdMoveX.Text = "Move X";
            this.rdMoveX.UseVisualStyleBackColor = false;
            this.rdMoveX.CheckedChanged += new System.EventHandler(this.OnMoveXChecked);
            // 
            // rdMoveY
            // 
            this.rdMoveY.AutoSize = true;
            this.rdMoveY.BackColor = System.Drawing.Color.Transparent;
            this.rdMoveY.Location = new System.Drawing.Point(759, 136);
            this.rdMoveY.Name = "rdMoveY";
            this.rdMoveY.Size = new System.Drawing.Size(80, 25);
            this.rdMoveY.TabIndex = 25;
            this.rdMoveY.TabStop = true;
            this.rdMoveY.Text = "Move Y";
            this.rdMoveY.UseVisualStyleBackColor = false;
            this.rdMoveY.CheckedChanged += new System.EventHandler(this.OnMoveYChecked);
            // 
            // label10
            // 
            this.label10.AutoSize = true;
            this.label10.BackColor = System.Drawing.Color.Transparent;
            this.label10.Location = new System.Drawing.Point(670, 176);
            this.label10.Name = "label10";
            this.label10.Size = new System.Drawing.Size(73, 21);
            this.label10.TabIndex = 26;
            this.label10.Text = "XY speed";
            // 
            // txtXYSpeed
            // 
            this.txtXYSpeed.BackColor = System.Drawing.Color.GhostWhite;
            this.txtXYSpeed.Location = new System.Drawing.Point(795, 173);
            this.txtXYSpeed.Name = "txtXYSpeed";
            this.txtXYSpeed.Size = new System.Drawing.Size(73, 29);
            this.txtXYSpeed.TabIndex = 27;
            this.txtXYSpeed.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.txtXYSpeed.Leave += new System.EventHandler(this.OnXYSpeedLeave);
            // 
            // label11
            // 
            this.label11.AutoSize = true;
            this.label11.BackColor = System.Drawing.Color.Transparent;
            this.label11.Location = new System.Drawing.Point(670, 211);
            this.label11.Name = "label11";
            this.label11.Size = new System.Drawing.Size(115, 21);
            this.label11.TabIndex = 28;
            this.label11.Text = "XY acceleration";
            // 
            // txtXYAcceleration
            // 
            this.txtXYAcceleration.BackColor = System.Drawing.Color.GhostWhite;
            this.txtXYAcceleration.Location = new System.Drawing.Point(795, 208);
            this.txtXYAcceleration.Name = "txtXYAcceleration";
            this.txtXYAcceleration.Size = new System.Drawing.Size(73, 29);
            this.txtXYAcceleration.TabIndex = 29;
            this.txtXYAcceleration.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.txtXYAcceleration.Leave += new System.EventHandler(this.OnXYAccelLeave);
            // 
            // label12
            // 
            this.label12.AutoSize = true;
            this.label12.BackColor = System.Drawing.Color.Transparent;
            this.label12.Location = new System.Drawing.Point(670, 281);
            this.label12.Name = "label12";
            this.label12.Size = new System.Drawing.Size(106, 21);
            this.label12.TabIndex = 32;
            this.label12.Text = "Z acceleration";
            // 
            // txtZAcceleration
            // 
            this.txtZAcceleration.BackColor = System.Drawing.Color.GhostWhite;
            this.txtZAcceleration.Location = new System.Drawing.Point(795, 278);
            this.txtZAcceleration.Name = "txtZAcceleration";
            this.txtZAcceleration.Size = new System.Drawing.Size(73, 29);
            this.txtZAcceleration.TabIndex = 33;
            this.txtZAcceleration.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.txtZAcceleration.Leave += new System.EventHandler(this.OnZAccelLeave);
            // 
            // label13
            // 
            this.label13.AutoSize = true;
            this.label13.BackColor = System.Drawing.Color.Transparent;
            this.label13.Location = new System.Drawing.Point(670, 246);
            this.label13.Name = "label13";
            this.label13.Size = new System.Drawing.Size(64, 21);
            this.label13.TabIndex = 30;
            this.label13.Text = "Z speed";
            // 
            // txtZSpeed
            // 
            this.txtZSpeed.BackColor = System.Drawing.Color.GhostWhite;
            this.txtZSpeed.Location = new System.Drawing.Point(795, 243);
            this.txtZSpeed.Name = "txtZSpeed";
            this.txtZSpeed.Size = new System.Drawing.Size(73, 29);
            this.txtZSpeed.TabIndex = 31;
            this.txtZSpeed.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.txtZSpeed.Leave += new System.EventHandler(this.OnZSpeedLeave);
            // 
            // label14
            // 
            this.label14.AutoSize = true;
            this.label14.BackColor = System.Drawing.Color.Transparent;
            this.label14.Location = new System.Drawing.Point(17, 571);
            this.label14.Name = "label14";
            this.label14.Size = new System.Drawing.Size(94, 21);
            this.label14.TabIndex = 34;
            this.label14.Text = "Output path";
            // 
            // txtSummaryFile
            // 
            this.txtSummaryFile.BackColor = System.Drawing.Color.GhostWhite;
            this.txtSummaryFile.Location = new System.Drawing.Point(129, 568);
            this.txtSummaryFile.Name = "txtSummaryFile";
            this.txtSummaryFile.Size = new System.Drawing.Size(689, 29);
            this.txtSummaryFile.TabIndex = 108;
            this.txtSummaryFile.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.FullSpeedAcqToolTip.SetToolTip(this.txtSummaryFile, resources.GetString("txtSummaryFile.ToolTip"));
            // 
            // label15
            // 
            this.label15.AutoSize = true;
            this.label15.BackColor = System.Drawing.Color.Transparent;
            this.label15.Location = new System.Drawing.Point(17, 500);
            this.label15.Name = "label15";
            this.label15.Size = new System.Drawing.Size(114, 21);
            this.label15.TabIndex = 36;
            this.label15.Text = "MinX,Y/MaxX,Y";
            // 
            // txtMinX
            // 
            this.txtMinX.BackColor = System.Drawing.Color.GhostWhite;
            this.txtMinX.Location = new System.Drawing.Point(231, 497);
            this.txtMinX.Name = "txtMinX";
            this.txtMinX.Size = new System.Drawing.Size(66, 29);
            this.txtMinX.TabIndex = 101;
            this.txtMinX.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.txtMinX.Leave += new System.EventHandler(this.OnMinXLeave);
            // 
            // txtMinY
            // 
            this.txtMinY.BackColor = System.Drawing.Color.GhostWhite;
            this.txtMinY.Location = new System.Drawing.Point(303, 497);
            this.txtMinY.Name = "txtMinY";
            this.txtMinY.Size = new System.Drawing.Size(66, 29);
            this.txtMinY.TabIndex = 102;
            this.txtMinY.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.txtMinY.Leave += new System.EventHandler(this.OnMinYLeave);
            // 
            // label16
            // 
            this.label16.AutoSize = true;
            this.label16.BackColor = System.Drawing.Color.Transparent;
            this.label16.Location = new System.Drawing.Point(379, 500);
            this.label16.Name = "label16";
            this.label16.Size = new System.Drawing.Size(16, 21);
            this.label16.TabIndex = 39;
            this.label16.Text = "/";
            // 
            // txtMaxY
            // 
            this.txtMaxY.BackColor = System.Drawing.Color.GhostWhite;
            this.txtMaxY.Location = new System.Drawing.Point(481, 497);
            this.txtMaxY.Name = "txtMaxY";
            this.txtMaxY.Size = new System.Drawing.Size(66, 29);
            this.txtMaxY.TabIndex = 104;
            this.txtMaxY.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.txtMaxY.Leave += new System.EventHandler(this.OnMaxYLeave);
            // 
            // txtMaxX
            // 
            this.txtMaxX.BackColor = System.Drawing.Color.GhostWhite;
            this.txtMaxX.Location = new System.Drawing.Point(407, 497);
            this.txtMaxX.Name = "txtMaxX";
            this.txtMaxX.Size = new System.Drawing.Size(66, 29);
            this.txtMaxX.TabIndex = 103;
            this.txtMaxX.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.txtMaxX.Leave += new System.EventHandler(this.OnMaxXLeave);
            // 
            // btnFromHere
            // 
            this.btnFromHere.AutoSize = true;
            this.btnFromHere.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnFromHere.BackColor = System.Drawing.Color.Transparent;
            this.btnFromHere.FocusedColor = System.Drawing.Color.Navy;
            this.btnFromHere.Font = new System.Drawing.Font("Segoe UI", 14.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btnFromHere.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnFromHere.Location = new System.Drawing.Point(181, 497);
            this.btnFromHere.Margin = new System.Windows.Forms.Padding(6);
            this.btnFromHere.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnFromHere.Name = "btnFromHere";
            this.btnFromHere.Size = new System.Drawing.Size(41, 29);
            this.btnFromHere.TabIndex = 100;
            this.btnFromHere.Text = "!XY!";
            this.btnFromHere.Click += new System.EventHandler(this.btnFromHere_Click);
            // 
            // btnToHere
            // 
            this.btnToHere.AutoSize = true;
            this.btnToHere.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnToHere.BackColor = System.Drawing.Color.Transparent;
            this.btnToHere.FocusedColor = System.Drawing.Color.Navy;
            this.btnToHere.Font = new System.Drawing.Font("Segoe UI", 14.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btnToHere.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnToHere.Location = new System.Drawing.Point(548, 497);
            this.btnToHere.Margin = new System.Windows.Forms.Padding(6);
            this.btnToHere.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnToHere.Name = "btnToHere";
            this.btnToHere.Size = new System.Drawing.Size(41, 29);
            this.btnToHere.TabIndex = 105;
            this.btnToHere.Text = "!XY!";
            this.btnToHere.Click += new System.EventHandler(this.btnToHere_Click);
            // 
            // panel1
            // 
            this.panel1.BorderStyle = System.Windows.Forms.BorderStyle.Fixed3D;
            this.panel1.Location = new System.Drawing.Point(406, 34);
            this.panel1.Name = "panel1";
            this.panel1.Size = new System.Drawing.Size(10, 440);
            this.panel1.TabIndex = 52;
            // 
            // txtCurrentConfiguration
            // 
            this.txtCurrentConfiguration.BackColor = System.Drawing.Color.LightGray;
            this.txtCurrentConfiguration.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtCurrentConfiguration.ForeColor = System.Drawing.Color.Navy;
            this.txtCurrentConfiguration.Location = new System.Drawing.Point(272, 166);
            this.txtCurrentConfiguration.Name = "txtCurrentConfiguration";
            this.txtCurrentConfiguration.ReadOnly = true;
            this.txtCurrentConfiguration.Size = new System.Drawing.Size(121, 25);
            this.txtCurrentConfiguration.TabIndex = 50;
            // 
            // btnMakeCurrent
            // 
            this.btnMakeCurrent.AutoSize = true;
            this.btnMakeCurrent.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnMakeCurrent.BackColor = System.Drawing.Color.Transparent;
            this.btnMakeCurrent.FocusedColor = System.Drawing.Color.Navy;
            this.btnMakeCurrent.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.btnMakeCurrent.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnMakeCurrent.Location = new System.Drawing.Point(272, 132);
            this.btnMakeCurrent.Margin = new System.Windows.Forms.Padding(6);
            this.btnMakeCurrent.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnMakeCurrent.Name = "btnMakeCurrent";
            this.btnMakeCurrent.Size = new System.Drawing.Size(103, 25);
            this.btnMakeCurrent.TabIndex = 49;
            this.btnMakeCurrent.Text = "Make current";
            this.btnMakeCurrent.Click += new System.EventHandler(this.btnMakeCurrent_Click);
            // 
            // btnDel
            // 
            this.btnDel.AutoSize = true;
            this.btnDel.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnDel.BackColor = System.Drawing.Color.Transparent;
            this.btnDel.FocusedColor = System.Drawing.Color.Navy;
            this.btnDel.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.btnDel.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnDel.Location = new System.Drawing.Point(272, 200);
            this.btnDel.Margin = new System.Windows.Forms.Padding(6);
            this.btnDel.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnDel.Name = "btnDel";
            this.btnDel.Size = new System.Drawing.Size(54, 25);
            this.btnDel.TabIndex = 51;
            this.btnDel.Text = "Delete";
            this.btnDel.Click += new System.EventHandler(this.btnDel_Click);
            // 
            // btnLoad
            // 
            this.btnLoad.AutoSize = true;
            this.btnLoad.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnLoad.BackColor = System.Drawing.Color.Transparent;
            this.btnLoad.FocusedColor = System.Drawing.Color.Navy;
            this.btnLoad.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.btnLoad.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnLoad.Location = new System.Drawing.Point(272, 104);
            this.btnLoad.Margin = new System.Windows.Forms.Padding(6);
            this.btnLoad.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnLoad.Name = "btnLoad";
            this.btnLoad.Size = new System.Drawing.Size(42, 25);
            this.btnLoad.TabIndex = 48;
            this.btnLoad.Text = "Load";
            this.btnLoad.Click += new System.EventHandler(this.btnLoad_Click);
            // 
            // txtNewName
            // 
            this.txtNewName.BackColor = System.Drawing.Color.GhostWhite;
            this.txtNewName.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtNewName.ForeColor = System.Drawing.Color.DodgerBlue;
            this.txtNewName.Location = new System.Drawing.Point(272, 70);
            this.txtNewName.Name = "txtNewName";
            this.txtNewName.Size = new System.Drawing.Size(121, 25);
            this.txtNewName.TabIndex = 47;
            // 
            // btnDuplicate
            // 
            this.btnDuplicate.AutoSize = true;
            this.btnDuplicate.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnDuplicate.BackColor = System.Drawing.Color.Transparent;
            this.btnDuplicate.FocusedColor = System.Drawing.Color.Navy;
            this.btnDuplicate.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.btnDuplicate.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnDuplicate.Location = new System.Drawing.Point(324, 36);
            this.btnDuplicate.Margin = new System.Windows.Forms.Padding(6);
            this.btnDuplicate.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnDuplicate.Name = "btnDuplicate";
            this.btnDuplicate.Size = new System.Drawing.Size(76, 25);
            this.btnDuplicate.TabIndex = 46;
            this.btnDuplicate.Text = "Duplicate";
            this.btnDuplicate.Click += new System.EventHandler(this.btnDuplicate_Click);
            // 
            // lbConfigurations
            // 
            this.lbConfigurations.BackColor = System.Drawing.Color.WhiteSmoke;
            this.lbConfigurations.ForeColor = System.Drawing.Color.DodgerBlue;
            this.lbConfigurations.FormattingEnabled = true;
            this.lbConfigurations.ItemHeight = 21;
            this.lbConfigurations.Location = new System.Drawing.Point(12, 36);
            this.lbConfigurations.Name = "lbConfigurations";
            this.lbConfigurations.Size = new System.Drawing.Size(251, 193);
            this.lbConfigurations.Sorted = true;
            this.lbConfigurations.TabIndex = 44;
            // 
            // btnNew
            // 
            this.btnNew.AutoSize = true;
            this.btnNew.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnNew.BackColor = System.Drawing.Color.Transparent;
            this.btnNew.FocusedColor = System.Drawing.Color.Navy;
            this.btnNew.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.btnNew.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnNew.Location = new System.Drawing.Point(272, 36);
            this.btnNew.Margin = new System.Windows.Forms.Padding(6);
            this.btnNew.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnNew.Name = "btnNew";
            this.btnNew.Size = new System.Drawing.Size(40, 25);
            this.btnNew.TabIndex = 45;
            this.btnNew.Text = "New";
            this.btnNew.Click += new System.EventHandler(this.btnNew_Click);
            // 
            // panel2
            // 
            this.panel2.BorderStyle = System.Windows.Forms.BorderStyle.Fixed3D;
            this.panel2.Location = new System.Drawing.Point(13, 480);
            this.panel2.Name = "panel2";
            this.panel2.Size = new System.Drawing.Size(878, 10);
            this.panel2.TabIndex = 53;
            // 
            // btnStart
            // 
            this.btnStart.AutoSize = true;
            this.btnStart.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnStart.BackColor = System.Drawing.Color.Transparent;
            this.btnStart.FocusedColor = System.Drawing.Color.Navy;
            this.btnStart.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.btnStart.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnStart.Location = new System.Drawing.Point(827, 572);
            this.btnStart.Margin = new System.Windows.Forms.Padding(6);
            this.btnStart.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnStart.Name = "btnStart";
            this.btnStart.Size = new System.Drawing.Size(41, 25);
            this.btnStart.TabIndex = 109;
            this.btnStart.Text = "Start";
            this.btnStart.Click += new System.EventHandler(this.btnStart_Click);
            // 
            // btnStop
            // 
            this.btnStop.AutoSize = true;
            this.btnStop.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnStop.BackColor = System.Drawing.Color.Transparent;
            this.btnStop.FocusedColor = System.Drawing.Color.Navy;
            this.btnStop.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.btnStop.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnStop.Location = new System.Drawing.Point(827, 611);
            this.btnStop.Margin = new System.Windows.Forms.Padding(6);
            this.btnStop.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnStop.Name = "btnStop";
            this.btnStop.Size = new System.Drawing.Size(41, 25);
            this.btnStop.TabIndex = 110;
            this.btnStop.Text = "Stop";
            this.btnStop.Click += new System.EventHandler(this.btnStop_Click);
            // 
            // pbScanProgress
            // 
            this.pbScanProgress.Direction = SySal.SySalNExTControls.SySalProgressBarDirection.LeftToRight;
            this.pbScanProgress.EmptyGradientColors = new System.Drawing.Color[] {
        System.Drawing.Color.Lavender,
        System.Drawing.Color.Azure,
        System.Drawing.Color.Lavender};
            this.pbScanProgress.EmptyGradientStops = new double[] {
        0.5D};
            this.pbScanProgress.FillGradientColors = new System.Drawing.Color[] {
        System.Drawing.Color.PowderBlue,
        System.Drawing.Color.DodgerBlue,
        System.Drawing.Color.PowderBlue};
            this.pbScanProgress.FillGradientStops = new double[] {
        0.5D};
            this.pbScanProgress.Font = new System.Drawing.Font("Segoe UI", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.pbScanProgress.ForeColor = System.Drawing.Color.White;
            this.pbScanProgress.Location = new System.Drawing.Point(129, 609);
            this.pbScanProgress.Name = "pbScanProgress";
            this.pbScanProgress.Size = new System.Drawing.Size(689, 18);
            this.pbScanProgress.TabIndex = 56;
            // 
            // label17
            // 
            this.label17.AutoSize = true;
            this.label17.BackColor = System.Drawing.Color.Transparent;
            this.label17.Location = new System.Drawing.Point(670, 317);
            this.label17.Name = "label17";
            this.label17.Size = new System.Drawing.Size(118, 21);
            this.label17.TabIndex = 34;
            this.label17.Text = "Slowdown time";
            // 
            // txtSlowdownTime
            // 
            this.txtSlowdownTime.BackColor = System.Drawing.Color.GhostWhite;
            this.txtSlowdownTime.Location = new System.Drawing.Point(817, 314);
            this.txtSlowdownTime.Name = "txtSlowdownTime";
            this.txtSlowdownTime.Size = new System.Drawing.Size(51, 29);
            this.txtSlowdownTime.TabIndex = 35;
            this.txtSlowdownTime.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.txtSlowdownTime.Leave += new System.EventHandler(this.OnSlowdownTimeLeave);
            // 
            // label18
            // 
            this.label18.AutoSize = true;
            this.label18.BackColor = System.Drawing.Color.Transparent;
            this.label18.Location = new System.Drawing.Point(670, 352);
            this.label18.Name = "label18";
            this.label18.Size = new System.Drawing.Size(134, 21);
            this.label18.TabIndex = 36;
            this.label18.Text = "Position tolerance";
            // 
            // txtPositionTolerance
            // 
            this.txtPositionTolerance.BackColor = System.Drawing.Color.GhostWhite;
            this.txtPositionTolerance.Location = new System.Drawing.Point(817, 349);
            this.txtPositionTolerance.Name = "txtPositionTolerance";
            this.txtPositionTolerance.Size = new System.Drawing.Size(51, 29);
            this.txtPositionTolerance.TabIndex = 37;
            this.txtPositionTolerance.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.txtPositionTolerance.Leave += new System.EventHandler(this.OnPosTolLeave);
            // 
            // btnExit
            // 
            this.btnExit.AutoSize = true;
            this.btnExit.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnExit.BackColor = System.Drawing.Color.Transparent;
            this.btnExit.FocusedColor = System.Drawing.Color.Navy;
            this.btnExit.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.btnExit.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnExit.Location = new System.Drawing.Point(829, 706);
            this.btnExit.Margin = new System.Windows.Forms.Padding(6);
            this.btnExit.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnExit.Name = "btnExit";
            this.btnExit.Size = new System.Drawing.Size(32, 25);
            this.btnExit.TabIndex = 57;
            this.btnExit.Text = "Exit";
            this.btnExit.Click += new System.EventHandler(this.btnExit_Click);
            // 
            // label19
            // 
            this.label19.AutoSize = true;
            this.label19.BackColor = System.Drawing.Color.Transparent;
            this.label19.Location = new System.Drawing.Point(670, 105);
            this.label19.Name = "label19";
            this.label19.Size = new System.Drawing.Size(141, 21);
            this.label19.TabIndex = 23;
            this.label19.Text = "Cont. Motion Fract.";
            // 
            // txtContinuousMotionFraction
            // 
            this.txtContinuousMotionFraction.BackColor = System.Drawing.Color.GhostWhite;
            this.txtContinuousMotionFraction.Location = new System.Drawing.Point(817, 102);
            this.txtContinuousMotionFraction.Name = "txtContinuousMotionFraction";
            this.txtContinuousMotionFraction.Size = new System.Drawing.Size(51, 29);
            this.txtContinuousMotionFraction.TabIndex = 24;
            this.txtContinuousMotionFraction.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.txtContinuousMotionFraction.Leave += new System.EventHandler(this.OnContinuousMotionFractionLeave);
            // 
            // label20
            // 
            this.label20.AutoSize = true;
            this.label20.BackColor = System.Drawing.Color.Transparent;
            this.label20.Location = new System.Drawing.Point(454, 317);
            this.label20.Name = "label20";
            this.label20.Size = new System.Drawing.Size(142, 21);
            this.label20.TabIndex = 19;
            this.label20.Text = "Emulsion thickness";
            // 
            // txtEmuThickness
            // 
            this.txtEmuThickness.BackColor = System.Drawing.Color.GhostWhite;
            this.txtEmuThickness.Location = new System.Drawing.Point(601, 314);
            this.txtEmuThickness.Name = "txtEmuThickness";
            this.txtEmuThickness.Size = new System.Drawing.Size(51, 29);
            this.txtEmuThickness.TabIndex = 20;
            this.txtEmuThickness.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.txtEmuThickness.Leave += new System.EventHandler(this.OnEmuThicknessLeave);
            // 
            // txtStatus
            // 
            this.txtStatus.BackColor = System.Drawing.Color.LightGray;
            this.txtStatus.Location = new System.Drawing.Point(12, 705);
            this.txtStatus.Name = "txtStatus";
            this.txtStatus.ReadOnly = true;
            this.txtStatus.Size = new System.Drawing.Size(808, 29);
            this.txtStatus.TabIndex = 58;
            this.txtStatus.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // txtFPS
            // 
            this.txtFPS.BackColor = System.Drawing.Color.GhostWhite;
            this.txtFPS.Location = new System.Drawing.Point(601, 349);
            this.txtFPS.Name = "txtFPS";
            this.txtFPS.Size = new System.Drawing.Size(51, 29);
            this.txtFPS.TabIndex = 22;
            this.txtFPS.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.txtFPS.Leave += new System.EventHandler(this.OnLeaveFPS);
            // 
            // btnMeasureFramesPerSecond
            // 
            this.btnMeasureFramesPerSecond.AutoSize = true;
            this.btnMeasureFramesPerSecond.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnMeasureFramesPerSecond.BackColor = System.Drawing.Color.Transparent;
            this.btnMeasureFramesPerSecond.FocusedColor = System.Drawing.Color.Navy;
            this.btnMeasureFramesPerSecond.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.btnMeasureFramesPerSecond.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnMeasureFramesPerSecond.Location = new System.Drawing.Point(454, 349);
            this.btnMeasureFramesPerSecond.Margin = new System.Windows.Forms.Padding(6);
            this.btnMeasureFramesPerSecond.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnMeasureFramesPerSecond.Name = "btnMeasureFramesPerSecond";
            this.btnMeasureFramesPerSecond.Size = new System.Drawing.Size(144, 25);
            this.btnMeasureFramesPerSecond.TabIndex = 21;
            this.btnMeasureFramesPerSecond.Text = "Frames per second";
            this.btnMeasureFramesPerSecond.Click += new System.EventHandler(this.btnMeasureFramesPerSecond_Click);
            // 
            // label21
            // 
            this.label21.AutoSize = true;
            this.label21.BackColor = System.Drawing.Color.Transparent;
            this.label21.Location = new System.Drawing.Point(17, 606);
            this.label21.Name = "label21";
            this.label21.Size = new System.Drawing.Size(71, 21);
            this.label21.TabIndex = 60;
            this.label21.Text = "Progress";
            // 
            // label22
            // 
            this.label22.AutoSize = true;
            this.label22.BackColor = System.Drawing.Color.Transparent;
            this.label22.Location = new System.Drawing.Point(17, 637);
            this.label22.Name = "label22";
            this.label22.Size = new System.Drawing.Size(110, 21);
            this.label22.TabIndex = 62;
            this.label22.Text = "Process queue";
            // 
            // pbQueueLength
            // 
            this.pbQueueLength.Direction = SySal.SySalNExTControls.SySalProgressBarDirection.LeftToRight;
            this.pbQueueLength.EmptyGradientColors = new System.Drawing.Color[] {
        System.Drawing.Color.DimGray,
        System.Drawing.Color.Silver,
        System.Drawing.Color.DimGray};
            this.pbQueueLength.EmptyGradientStops = new double[] {
        0.5D};
            this.pbQueueLength.FillGradientColors = new System.Drawing.Color[] {
        System.Drawing.Color.GreenYellow,
        System.Drawing.Color.Yellow,
        System.Drawing.Color.OrangeRed};
            this.pbQueueLength.FillGradientStops = new double[] {
        0.6D};
            this.pbQueueLength.Font = new System.Drawing.Font("Segoe UI", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.pbQueueLength.ForeColor = System.Drawing.Color.White;
            this.pbQueueLength.GradientDirection = SySal.SySalNExTControls.SySalProgressBarDirection.LeftToRight;
            this.pbQueueLength.Location = new System.Drawing.Point(129, 640);
            this.pbQueueLength.Name = "pbQueueLength";
            this.pbQueueLength.Size = new System.Drawing.Size(689, 18);
            this.pbQueueLength.TabIndex = 61;
            // 
            // btnContinuousMotionFraction
            // 
            this.btnContinuousMotionFraction.AutoSize = true;
            this.btnContinuousMotionFraction.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnContinuousMotionFraction.BackColor = System.Drawing.Color.Transparent;
            this.btnContinuousMotionFraction.FocusedColor = System.Drawing.Color.Navy;
            this.btnContinuousMotionFraction.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.btnContinuousMotionFraction.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnContinuousMotionFraction.Location = new System.Drawing.Point(454, 446);
            this.btnContinuousMotionFraction.Margin = new System.Windows.Forms.Padding(6);
            this.btnContinuousMotionFraction.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnContinuousMotionFraction.Name = "btnContinuousMotionFraction";
            this.btnContinuousMotionFraction.Size = new System.Drawing.Size(143, 25);
            this.btnContinuousMotionFraction.TabIndex = 63;
            this.btnContinuousMotionFraction.Text = "Cont. Motion fract.";
            this.btnContinuousMotionFraction.Click += new System.EventHandler(this.btnContinuousMotionFraction_Click);
            // 
            // sdlgZTest
            // 
            this.sdlgZTest.Filter = "Text files (*.txt)|*.txt|All files (*.*)|*.*";
            this.sdlgZTest.Title = "Select file for Z test output";
            // 
            // label23
            // 
            this.label23.AutoSize = true;
            this.label23.BackColor = System.Drawing.Color.Transparent;
            this.label23.Location = new System.Drawing.Point(670, 387);
            this.label23.Name = "label23";
            this.label23.Size = new System.Drawing.Size(113, 21);
            this.label23.TabIndex = 38;
            this.label23.Text = "Motion latency";
            // 
            // txtMotionLatency
            // 
            this.txtMotionLatency.BackColor = System.Drawing.Color.GhostWhite;
            this.txtMotionLatency.Location = new System.Drawing.Point(817, 384);
            this.txtMotionLatency.Name = "txtMotionLatency";
            this.txtMotionLatency.Size = new System.Drawing.Size(51, 29);
            this.txtMotionLatency.TabIndex = 39;
            this.txtMotionLatency.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.txtMotionLatency.Leave += new System.EventHandler(this.OnMotionLatencyLeave);
            // 
            // chkStageSimOnly
            // 
            this.chkStageSimOnly.AutoSize = true;
            this.chkStageSimOnly.BackColor = System.Drawing.Color.Transparent;
            this.chkStageSimOnly.Location = new System.Drawing.Point(212, 388);
            this.chkStageSimOnly.Name = "chkStageSimOnly";
            this.chkStageSimOnly.Size = new System.Drawing.Size(179, 25);
            this.chkStageSimOnly.TabIndex = 66;
            this.chkStageSimOnly.Text = "Stage simulation only";
            this.chkStageSimOnly.UseVisualStyleBackColor = false;
            // 
            // label24
            // 
            this.label24.AutoSize = true;
            this.label24.BackColor = System.Drawing.Color.Transparent;
            this.label24.Location = new System.Drawing.Point(17, 668);
            this.label24.Name = "label24";
            this.label24.Size = new System.Drawing.Size(101, 21);
            this.label24.TabIndex = 68;
            this.label24.Text = "Writer queue";
            // 
            // pbWriteQueueLength
            // 
            this.pbWriteQueueLength.Direction = SySal.SySalNExTControls.SySalProgressBarDirection.LeftToRight;
            this.pbWriteQueueLength.EmptyGradientColors = new System.Drawing.Color[] {
        System.Drawing.Color.DimGray,
        System.Drawing.Color.Silver,
        System.Drawing.Color.DimGray};
            this.pbWriteQueueLength.EmptyGradientStops = new double[] {
        0.5D};
            this.pbWriteQueueLength.FillGradientColors = new System.Drawing.Color[] {
        System.Drawing.Color.GreenYellow,
        System.Drawing.Color.Yellow,
        System.Drawing.Color.OrangeRed};
            this.pbWriteQueueLength.FillGradientStops = new double[] {
        0.6D};
            this.pbWriteQueueLength.Font = new System.Drawing.Font("Segoe UI", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.pbWriteQueueLength.ForeColor = System.Drawing.Color.White;
            this.pbWriteQueueLength.GradientDirection = SySal.SySalNExTControls.SySalProgressBarDirection.LeftToRight;
            this.pbWriteQueueLength.Location = new System.Drawing.Point(129, 671);
            this.pbWriteQueueLength.Name = "pbWriteQueueLength";
            this.pbWriteQueueLength.Size = new System.Drawing.Size(689, 18);
            this.pbWriteQueueLength.TabIndex = 67;
            // 
            // chkEnableGPUTest
            // 
            this.chkEnableGPUTest.AutoSize = true;
            this.chkEnableGPUTest.BackColor = System.Drawing.Color.Transparent;
            this.chkEnableGPUTest.Location = new System.Drawing.Point(12, 388);
            this.chkEnableGPUTest.Name = "chkEnableGPUTest";
            this.chkEnableGPUTest.Size = new System.Drawing.Size(161, 25);
            this.chkEnableGPUTest.TabIndex = 69;
            this.chkEnableGPUTest.Text = "Enable GPU testing";
            this.chkEnableGPUTest.UseVisualStyleBackColor = false;
            // 
            // btnEmptyFocusMap
            // 
            this.btnEmptyFocusMap.AutoSize = true;
            this.btnEmptyFocusMap.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnEmptyFocusMap.BackColor = System.Drawing.Color.Transparent;
            this.btnEmptyFocusMap.FocusedColor = System.Drawing.Color.Navy;
            this.btnEmptyFocusMap.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.btnEmptyFocusMap.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnEmptyFocusMap.Location = new System.Drawing.Point(737, 501);
            this.btnEmptyFocusMap.Margin = new System.Windows.Forms.Padding(6);
            this.btnEmptyFocusMap.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnEmptyFocusMap.Name = "btnEmptyFocusMap";
            this.btnEmptyFocusMap.Size = new System.Drawing.Size(136, 25);
            this.btnEmptyFocusMap.TabIndex = 107;
            this.btnEmptyFocusMap.Text = "Empty Focus Map";
            this.btnEmptyFocusMap.Click += new System.EventHandler(this.btnEmptyFocusMap_Click);
            // 
            // label25
            // 
            this.label25.AutoSize = true;
            this.label25.BackColor = System.Drawing.Color.Transparent;
            this.label25.Location = new System.Drawing.Point(670, 422);
            this.label25.Name = "label25";
            this.label25.Size = new System.Drawing.Size(135, 21);
            this.label25.TabIndex = 40;
            this.label25.Text = "Z data expir. (min)";
            // 
            // txtZDataExpirationMin
            // 
            this.txtZDataExpirationMin.BackColor = System.Drawing.Color.GhostWhite;
            this.txtZDataExpirationMin.Location = new System.Drawing.Point(817, 419);
            this.txtZDataExpirationMin.Name = "txtZDataExpirationMin";
            this.txtZDataExpirationMin.Size = new System.Drawing.Size(51, 29);
            this.txtZDataExpirationMin.TabIndex = 41;
            this.txtZDataExpirationMin.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.txtZDataExpirationMin.Leave += new System.EventHandler(this.OnZDataExpirMinLeave);
            // 
            // label26
            // 
            this.label26.AutoSize = true;
            this.label26.BackColor = System.Drawing.Color.Transparent;
            this.label26.Location = new System.Drawing.Point(454, 387);
            this.label26.Name = "label26";
            this.label26.Size = new System.Drawing.Size(122, 21);
            this.label26.TabIndex = 23;
            this.label26.Text = "Max. views/strip";
            // 
            // txtMaxViewsStrip
            // 
            this.txtMaxViewsStrip.BackColor = System.Drawing.Color.GhostWhite;
            this.txtMaxViewsStrip.Location = new System.Drawing.Point(601, 384);
            this.txtMaxViewsStrip.Name = "txtMaxViewsStrip";
            this.txtMaxViewsStrip.Size = new System.Drawing.Size(51, 29);
            this.txtMaxViewsStrip.TabIndex = 24;
            this.txtMaxViewsStrip.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.txtMaxViewsStrip.Leave += new System.EventHandler(this.OnMaxViewsPerStripLeave);
            // 
            // btnDumpZMap
            // 
            this.btnDumpZMap.AutoSize = true;
            this.btnDumpZMap.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnDumpZMap.BackColor = System.Drawing.Color.Transparent;
            this.btnDumpZMap.FocusedColor = System.Drawing.Color.Navy;
            this.btnDumpZMap.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.btnDumpZMap.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnDumpZMap.Location = new System.Drawing.Point(596, 501);
            this.btnDumpZMap.Margin = new System.Windows.Forms.Padding(6);
            this.btnDumpZMap.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnDumpZMap.Name = "btnDumpZMap";
            this.btnDumpZMap.Size = new System.Drawing.Size(103, 25);
            this.btnDumpZMap.TabIndex = 106;
            this.btnDumpZMap.Text = "Dump Z Map";
            this.btnDumpZMap.Click += new System.EventHandler(this.btnDumpZMap_Click);
            // 
            // chkDumpFocusInfo
            // 
            this.chkDumpFocusInfo.AutoSize = true;
            this.chkDumpFocusInfo.BackColor = System.Drawing.Color.Transparent;
            this.chkDumpFocusInfo.Location = new System.Drawing.Point(212, 423);
            this.chkDumpFocusInfo.Name = "chkDumpFocusInfo";
            this.chkDumpFocusInfo.Size = new System.Drawing.Size(144, 25);
            this.chkDumpFocusInfo.TabIndex = 68;
            this.chkDumpFocusInfo.Text = "Dump focus info";
            this.chkDumpFocusInfo.UseVisualStyleBackColor = false;
            // 
            // chkWriteClusterFiles
            // 
            this.chkWriteClusterFiles.AutoSize = true;
            this.chkWriteClusterFiles.BackColor = System.Drawing.Color.Transparent;
            this.chkWriteClusterFiles.Checked = true;
            this.chkWriteClusterFiles.CheckState = System.Windows.Forms.CheckState.Checked;
            this.chkWriteClusterFiles.Location = new System.Drawing.Point(12, 423);
            this.chkWriteClusterFiles.Name = "chkWriteClusterFiles";
            this.chkWriteClusterFiles.Size = new System.Drawing.Size(149, 25);
            this.chkWriteClusterFiles.TabIndex = 67;
            this.chkWriteClusterFiles.Text = "Write cluster files";
            this.chkWriteClusterFiles.UseVisualStyleBackColor = false;
            // 
            // sdlgZDump
            // 
            this.sdlgZDump.Filter = "Text files (*.txt)|*.txt|All files (*.*)|*.*";
            this.sdlgZDump.Title = "Select file for Z Map output";
            // 
            // label27
            // 
            this.label27.AutoSize = true;
            this.label27.BackColor = System.Drawing.Color.Transparent;
            this.label27.Location = new System.Drawing.Point(17, 533);
            this.label27.Name = "label27";
            this.label27.Size = new System.Drawing.Size(207, 21);
            this.label27.TabIndex = 111;
            this.label27.Text = "Transformed MinX,Y/MaxX,Y";
            // 
            // txtTransfMaxY
            // 
            this.txtTransfMaxY.BackColor = System.Drawing.Color.GhostWhite;
            this.txtTransfMaxY.Location = new System.Drawing.Point(481, 533);
            this.txtTransfMaxY.Name = "txtTransfMaxY";
            this.txtTransfMaxY.ReadOnly = true;
            this.txtTransfMaxY.Size = new System.Drawing.Size(66, 29);
            this.txtTransfMaxY.TabIndex = 116;
            this.txtTransfMaxY.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // txtTransfMaxX
            // 
            this.txtTransfMaxX.BackColor = System.Drawing.Color.GhostWhite;
            this.txtTransfMaxX.Location = new System.Drawing.Point(407, 533);
            this.txtTransfMaxX.Name = "txtTransfMaxX";
            this.txtTransfMaxX.ReadOnly = true;
            this.txtTransfMaxX.Size = new System.Drawing.Size(66, 29);
            this.txtTransfMaxX.TabIndex = 115;
            this.txtTransfMaxX.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label28
            // 
            this.label28.AutoSize = true;
            this.label28.BackColor = System.Drawing.Color.Transparent;
            this.label28.Location = new System.Drawing.Point(379, 536);
            this.label28.Name = "label28";
            this.label28.Size = new System.Drawing.Size(16, 21);
            this.label28.TabIndex = 112;
            this.label28.Text = "/";
            // 
            // txtTransfMinY
            // 
            this.txtTransfMinY.BackColor = System.Drawing.Color.GhostWhite;
            this.txtTransfMinY.Location = new System.Drawing.Point(303, 533);
            this.txtTransfMinY.Name = "txtTransfMinY";
            this.txtTransfMinY.ReadOnly = true;
            this.txtTransfMinY.Size = new System.Drawing.Size(66, 29);
            this.txtTransfMinY.TabIndex = 114;
            this.txtTransfMinY.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // txtTransfMinX
            // 
            this.txtTransfMinX.BackColor = System.Drawing.Color.GhostWhite;
            this.txtTransfMinX.Location = new System.Drawing.Point(231, 533);
            this.txtTransfMinX.Name = "txtTransfMinX";
            this.txtTransfMinX.ReadOnly = true;
            this.txtTransfMinX.Size = new System.Drawing.Size(66, 29);
            this.txtTransfMinX.TabIndex = 113;
            this.txtTransfMinX.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // lvPostProcessing
            // 
            this.lvPostProcessing.BackColor = System.Drawing.Color.WhiteSmoke;
            this.lvPostProcessing.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.columnHeader1,
            this.columnHeader2});
            this.lvPostProcessing.Font = new System.Drawing.Font("Segoe UI", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lvPostProcessing.ForeColor = System.Drawing.Color.DodgerBlue;
            this.lvPostProcessing.FullRowSelect = true;
            this.lvPostProcessing.GridLines = true;
            this.lvPostProcessing.HeaderStyle = System.Windows.Forms.ColumnHeaderStyle.Nonclickable;
            this.lvPostProcessing.HideSelection = false;
            this.lvPostProcessing.Items.AddRange(new System.Windows.Forms.ListViewItem[] {
            listViewItem1,
            listViewItem2,
            listViewItem3,
            listViewItem4,
            listViewItem5});
            this.lvPostProcessing.Location = new System.Drawing.Point(10, 246);
            this.lvPostProcessing.MultiSelect = false;
            this.lvPostProcessing.Name = "lvPostProcessing";
            this.lvPostProcessing.Size = new System.Drawing.Size(335, 127);
            this.lvPostProcessing.TabIndex = 117;
            this.lvPostProcessing.UseCompatibleStateImageBehavior = false;
            this.lvPostProcessing.View = System.Windows.Forms.View.Details;
            // 
            // columnHeader1
            // 
            this.columnHeader1.Text = "Step";
            this.columnHeader1.Width = 137;
            // 
            // columnHeader2
            // 
            this.columnHeader2.Text = "Settings";
            this.columnHeader2.Width = 193;
            // 
            // btnStepReset
            // 
            this.btnStepReset.AutoSize = true;
            this.btnStepReset.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnStepReset.BackColor = System.Drawing.Color.Transparent;
            this.btnStepReset.FocusedColor = System.Drawing.Color.Navy;
            this.btnStepReset.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.btnStepReset.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnStepReset.Location = new System.Drawing.Point(352, 317);
            this.btnStepReset.Margin = new System.Windows.Forms.Padding(6);
            this.btnStepReset.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnStepReset.Name = "btnStepReset";
            this.btnStepReset.Size = new System.Drawing.Size(47, 25);
            this.btnStepReset.TabIndex = 120;
            this.btnStepReset.Text = "Reset";
            this.FullSpeedAcqToolTip.SetToolTip(this.btnStepReset, "Resets the settings of the selected processing step,\r\nactually disabling that ste" +
        "p.");
            this.btnStepReset.Click += new System.EventHandler(this.btnStepReset_Click);
            // 
            // btnStepPaste
            // 
            this.btnStepPaste.AutoSize = true;
            this.btnStepPaste.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnStepPaste.BackColor = System.Drawing.Color.Transparent;
            this.btnStepPaste.FocusedColor = System.Drawing.Color.Navy;
            this.btnStepPaste.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.btnStepPaste.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnStepPaste.Location = new System.Drawing.Point(352, 287);
            this.btnStepPaste.Margin = new System.Windows.Forms.Padding(6);
            this.btnStepPaste.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnStepPaste.Name = "btnStepPaste";
            this.btnStepPaste.Size = new System.Drawing.Size(46, 25);
            this.btnStepPaste.TabIndex = 121;
            this.btnStepPaste.Text = "Paste";
            this.FullSpeedAcqToolTip.SetToolTip(this.btnStepPaste, "Copies text from the clipboard into the settings for this step. \r\nTo import text " +
        "from a file, open it with Notepad or another\r\ntext editor, then copy the text an" +
        "d paste it here using this \r\nbutton.");
            this.btnStepPaste.Click += new System.EventHandler(this.btnStepPaste_Click);
            // 
            // btnStepSave
            // 
            this.btnStepSave.AutoSize = true;
            this.btnStepSave.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnStepSave.BackColor = System.Drawing.Color.Transparent;
            this.btnStepSave.FocusedColor = System.Drawing.Color.Navy;
            this.btnStepSave.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.btnStepSave.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnStepSave.Location = new System.Drawing.Point(352, 348);
            this.btnStepSave.Margin = new System.Windows.Forms.Padding(6);
            this.btnStepSave.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnStepSave.Name = "btnStepSave";
            this.btnStepSave.Size = new System.Drawing.Size(40, 25);
            this.btnStepSave.TabIndex = 122;
            this.btnStepSave.Text = "Save";
            this.FullSpeedAcqToolTip.SetToolTip(this.btnStepSave, "Saves the set of processing steps.\r\nThey will appear the next time this dialog is" +
        " opened.");
            this.btnStepSave.Click += new System.EventHandler(this.btnStepSave_Click);
            // 
            // btnStepCopy
            // 
            this.btnStepCopy.AutoSize = true;
            this.btnStepCopy.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnStepCopy.BackColor = System.Drawing.Color.Transparent;
            this.btnStepCopy.FocusedColor = System.Drawing.Color.Navy;
            this.btnStepCopy.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.btnStepCopy.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnStepCopy.Location = new System.Drawing.Point(352, 255);
            this.btnStepCopy.Margin = new System.Windows.Forms.Padding(6);
            this.btnStepCopy.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnStepCopy.Name = "btnStepCopy";
            this.btnStepCopy.Size = new System.Drawing.Size(44, 25);
            this.btnStepCopy.TabIndex = 123;
            this.btnStepCopy.Text = "Copy";
            this.FullSpeedAcqToolTip.SetToolTip(this.btnStepCopy, "Save the text of settings for the selected step into the clipboard.\r\nYou can then" +
        " paste this text into Notepad or another\r\ntext editor to edit it.");
            this.btnStepCopy.Click += new System.EventHandler(this.btnStepCopy_Click);
            // 
            // btnRecover
            // 
            this.btnRecover.AutoSize = true;
            this.btnRecover.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnRecover.BackColor = System.Drawing.Color.Transparent;
            this.btnRecover.FocusedColor = System.Drawing.Color.Navy;
            this.btnRecover.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.btnRecover.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnRecover.Location = new System.Drawing.Point(827, 668);
            this.btnRecover.Margin = new System.Windows.Forms.Padding(6);
            this.btnRecover.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnRecover.Name = "btnRecover";
            this.btnRecover.Size = new System.Drawing.Size(65, 25);
            this.btnRecover.TabIndex = 124;
            this.btnRecover.Text = "Recover";
            this.btnRecover.Click += new System.EventHandler(this.btnRecover_Click);
            // 
            // btnDebug
            // 
            this.btnDebug.AutoSize = true;
            this.btnDebug.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnDebug.BackColor = System.Drawing.Color.Transparent;
            this.btnDebug.FocusedColor = System.Drawing.Color.Navy;
            this.btnDebug.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.btnDebug.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnDebug.Location = new System.Drawing.Point(827, 640);
            this.btnDebug.Margin = new System.Windows.Forms.Padding(6);
            this.btnDebug.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnDebug.Name = "btnDebug";
            this.btnDebug.Size = new System.Drawing.Size(56, 25);
            this.btnDebug.TabIndex = 125;
            this.btnDebug.Text = "Debug";
            this.btnDebug.Click += new System.EventHandler(this.btnDebug_Click);
            // 
            // odlgRecoverZone
            // 
            this.odlgRecoverZone.Filter = "SCAN files (*." + QuasiStaticAcquisition.TypeStringScan + ")|*." + QuasiStaticAcquisition.TypeStringScan + "|All files (*.*)|*.*";
            this.odlgRecoverZone.Title = "Select SCAN file of zone to recover";
            // 
            // FullSpeedAcquisitionForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(9F, 21F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(899, 746);
            this.Controls.Add(this.btnDebug);
            this.Controls.Add(this.btnRecover);
            this.Controls.Add(this.btnStepCopy);
            this.Controls.Add(this.btnStepSave);
            this.Controls.Add(this.btnStepPaste);
            this.Controls.Add(this.btnStepReset);
            this.Controls.Add(this.lvPostProcessing);
            this.Controls.Add(this.txtTransfMaxY);
            this.Controls.Add(this.txtTransfMaxX);
            this.Controls.Add(this.label28);
            this.Controls.Add(this.txtTransfMinY);
            this.Controls.Add(this.txtTransfMinX);
            this.Controls.Add(this.label27);
            this.Controls.Add(this.chkWriteClusterFiles);
            this.Controls.Add(this.chkDumpFocusInfo);
            this.Controls.Add(this.btnDumpZMap);
            this.Controls.Add(this.label26);
            this.Controls.Add(this.txtMaxViewsStrip);
            this.Controls.Add(this.label25);
            this.Controls.Add(this.txtZDataExpirationMin);
            this.Controls.Add(this.btnEmptyFocusMap);
            this.Controls.Add(this.chkEnableGPUTest);
            this.Controls.Add(this.label24);
            this.Controls.Add(this.pbWriteQueueLength);
            this.Controls.Add(this.chkStageSimOnly);
            this.Controls.Add(this.label23);
            this.Controls.Add(this.txtMotionLatency);
            this.Controls.Add(this.btnContinuousMotionFraction);
            this.Controls.Add(this.label22);
            this.Controls.Add(this.pbQueueLength);
            this.Controls.Add(this.label21);
            this.Controls.Add(this.btnMeasureFramesPerSecond);
            this.Controls.Add(this.txtFPS);
            this.Controls.Add(this.txtStatus);
            this.Controls.Add(this.label20);
            this.Controls.Add(this.txtEmuThickness);
            this.Controls.Add(this.label19);
            this.Controls.Add(this.txtContinuousMotionFraction);
            this.Controls.Add(this.btnExit);
            this.Controls.Add(this.label18);
            this.Controls.Add(this.txtPositionTolerance);
            this.Controls.Add(this.label17);
            this.Controls.Add(this.txtSlowdownTime);
            this.Controls.Add(this.pbScanProgress);
            this.Controls.Add(this.btnStop);
            this.Controls.Add(this.btnStart);
            this.Controls.Add(this.panel2);
            this.Controls.Add(this.panel1);
            this.Controls.Add(this.txtCurrentConfiguration);
            this.Controls.Add(this.btnMakeCurrent);
            this.Controls.Add(this.btnDel);
            this.Controls.Add(this.btnLoad);
            this.Controls.Add(this.txtNewName);
            this.Controls.Add(this.btnDuplicate);
            this.Controls.Add(this.lbConfigurations);
            this.Controls.Add(this.btnNew);
            this.Controls.Add(this.btnToHere);
            this.Controls.Add(this.btnFromHere);
            this.Controls.Add(this.txtMaxY);
            this.Controls.Add(this.txtMaxX);
            this.Controls.Add(this.label16);
            this.Controls.Add(this.txtMinY);
            this.Controls.Add(this.label15);
            this.Controls.Add(this.txtMinX);
            this.Controls.Add(this.txtSummaryFile);
            this.Controls.Add(this.label14);
            this.Controls.Add(this.txtLayers);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.label12);
            this.Controls.Add(this.txtZAcceleration);
            this.Controls.Add(this.label13);
            this.Controls.Add(this.txtZSpeed);
            this.Controls.Add(this.label11);
            this.Controls.Add(this.txtXYAcceleration);
            this.Controls.Add(this.label10);
            this.Controls.Add(this.txtXYSpeed);
            this.Controls.Add(this.rdMoveY);
            this.Controls.Add(this.rdMoveX);
            this.Controls.Add(this.label9);
            this.Controls.Add(this.txtViewOverlap);
            this.Controls.Add(this.label8);
            this.Controls.Add(this.txtFOVSize);
            this.Controls.Add(this.label7);
            this.Controls.Add(this.txtBaseThickness);
            this.Controls.Add(this.chkBottom);
            this.Controls.Add(this.chkTop);
            this.Controls.Add(this.label6);
            this.Controls.Add(this.txtFocusSweep);
            this.Controls.Add(this.label5);
            this.Controls.Add(this.txtMinValidLayers);
            this.Controls.Add(this.label4);
            this.Controls.Add(this.txtClusterThreshold);
            this.Controls.Add(this.label3);
            this.Controls.Add(this.txtZSweep);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.txtPitch);
            this.DialogCaption = "Full-speed Acquisition";
            this.Name = "FullSpeedAcquisitionForm";
            this.NoCloseButton = true;
            this.Load += new System.EventHandler(this.OnLoad);
            this.Controls.SetChildIndex(this.txtPitch, 0);
            this.Controls.SetChildIndex(this.label2, 0);
            this.Controls.SetChildIndex(this.txtZSweep, 0);
            this.Controls.SetChildIndex(this.label3, 0);
            this.Controls.SetChildIndex(this.txtClusterThreshold, 0);
            this.Controls.SetChildIndex(this.label4, 0);
            this.Controls.SetChildIndex(this.txtMinValidLayers, 0);
            this.Controls.SetChildIndex(this.label5, 0);
            this.Controls.SetChildIndex(this.txtFocusSweep, 0);
            this.Controls.SetChildIndex(this.label6, 0);
            this.Controls.SetChildIndex(this.chkTop, 0);
            this.Controls.SetChildIndex(this.chkBottom, 0);
            this.Controls.SetChildIndex(this.txtBaseThickness, 0);
            this.Controls.SetChildIndex(this.label7, 0);
            this.Controls.SetChildIndex(this.txtFOVSize, 0);
            this.Controls.SetChildIndex(this.label8, 0);
            this.Controls.SetChildIndex(this.txtViewOverlap, 0);
            this.Controls.SetChildIndex(this.label9, 0);
            this.Controls.SetChildIndex(this.rdMoveX, 0);
            this.Controls.SetChildIndex(this.rdMoveY, 0);
            this.Controls.SetChildIndex(this.txtXYSpeed, 0);
            this.Controls.SetChildIndex(this.label10, 0);
            this.Controls.SetChildIndex(this.txtXYAcceleration, 0);
            this.Controls.SetChildIndex(this.label11, 0);
            this.Controls.SetChildIndex(this.txtZSpeed, 0);
            this.Controls.SetChildIndex(this.label13, 0);
            this.Controls.SetChildIndex(this.txtZAcceleration, 0);
            this.Controls.SetChildIndex(this.label12, 0);
            this.Controls.SetChildIndex(this.label1, 0);
            this.Controls.SetChildIndex(this.txtLayers, 0);
            this.Controls.SetChildIndex(this.label14, 0);
            this.Controls.SetChildIndex(this.txtSummaryFile, 0);
            this.Controls.SetChildIndex(this.txtMinX, 0);
            this.Controls.SetChildIndex(this.label15, 0);
            this.Controls.SetChildIndex(this.txtMinY, 0);
            this.Controls.SetChildIndex(this.label16, 0);
            this.Controls.SetChildIndex(this.txtMaxX, 0);
            this.Controls.SetChildIndex(this.txtMaxY, 0);
            this.Controls.SetChildIndex(this.btnFromHere, 0);
            this.Controls.SetChildIndex(this.btnToHere, 0);
            this.Controls.SetChildIndex(this.btnNew, 0);
            this.Controls.SetChildIndex(this.lbConfigurations, 0);
            this.Controls.SetChildIndex(this.btnDuplicate, 0);
            this.Controls.SetChildIndex(this.txtNewName, 0);
            this.Controls.SetChildIndex(this.btnLoad, 0);
            this.Controls.SetChildIndex(this.btnDel, 0);
            this.Controls.SetChildIndex(this.btnMakeCurrent, 0);
            this.Controls.SetChildIndex(this.txtCurrentConfiguration, 0);
            this.Controls.SetChildIndex(this.panel1, 0);
            this.Controls.SetChildIndex(this.panel2, 0);
            this.Controls.SetChildIndex(this.btnStart, 0);
            this.Controls.SetChildIndex(this.btnStop, 0);
            this.Controls.SetChildIndex(this.pbScanProgress, 0);
            this.Controls.SetChildIndex(this.txtSlowdownTime, 0);
            this.Controls.SetChildIndex(this.label17, 0);
            this.Controls.SetChildIndex(this.txtPositionTolerance, 0);
            this.Controls.SetChildIndex(this.label18, 0);
            this.Controls.SetChildIndex(this.btnExit, 0);
            this.Controls.SetChildIndex(this.txtContinuousMotionFraction, 0);
            this.Controls.SetChildIndex(this.label19, 0);
            this.Controls.SetChildIndex(this.txtEmuThickness, 0);
            this.Controls.SetChildIndex(this.label20, 0);
            this.Controls.SetChildIndex(this.txtStatus, 0);
            this.Controls.SetChildIndex(this.txtFPS, 0);
            this.Controls.SetChildIndex(this.btnMeasureFramesPerSecond, 0);
            this.Controls.SetChildIndex(this.label21, 0);
            this.Controls.SetChildIndex(this.pbQueueLength, 0);
            this.Controls.SetChildIndex(this.label22, 0);
            this.Controls.SetChildIndex(this.btnContinuousMotionFraction, 0);
            this.Controls.SetChildIndex(this.txtMotionLatency, 0);
            this.Controls.SetChildIndex(this.label23, 0);
            this.Controls.SetChildIndex(this.chkStageSimOnly, 0);
            this.Controls.SetChildIndex(this.pbWriteQueueLength, 0);
            this.Controls.SetChildIndex(this.label24, 0);
            this.Controls.SetChildIndex(this.chkEnableGPUTest, 0);
            this.Controls.SetChildIndex(this.btnEmptyFocusMap, 0);
            this.Controls.SetChildIndex(this.txtZDataExpirationMin, 0);
            this.Controls.SetChildIndex(this.label25, 0);
            this.Controls.SetChildIndex(this.txtMaxViewsStrip, 0);
            this.Controls.SetChildIndex(this.label26, 0);
            this.Controls.SetChildIndex(this.btnDumpZMap, 0);
            this.Controls.SetChildIndex(this.chkDumpFocusInfo, 0);
            this.Controls.SetChildIndex(this.chkWriteClusterFiles, 0);
            this.Controls.SetChildIndex(this.label27, 0);
            this.Controls.SetChildIndex(this.txtTransfMinX, 0);
            this.Controls.SetChildIndex(this.txtTransfMinY, 0);
            this.Controls.SetChildIndex(this.label28, 0);
            this.Controls.SetChildIndex(this.txtTransfMaxX, 0);
            this.Controls.SetChildIndex(this.txtTransfMaxY, 0);
            this.Controls.SetChildIndex(this.lvPostProcessing, 0);
            this.Controls.SetChildIndex(this.btnStepReset, 0);
            this.Controls.SetChildIndex(this.btnStepPaste, 0);
            this.Controls.SetChildIndex(this.btnStepSave, 0);
            this.Controls.SetChildIndex(this.btnStepCopy, 0);
            this.Controls.SetChildIndex(this.btnRecover, 0);
            this.Controls.SetChildIndex(this.btnDebug, 0);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TextBox txtLayers;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.TextBox txtPitch;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.TextBox txtZSweep;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.TextBox txtClusterThreshold;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.TextBox txtMinValidLayers;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.TextBox txtFocusSweep;
        private System.Windows.Forms.CheckBox chkTop;
        private System.Windows.Forms.CheckBox chkBottom;
        private System.Windows.Forms.Label label7;
        private System.Windows.Forms.TextBox txtBaseThickness;
        private System.Windows.Forms.Label label8;
        private System.Windows.Forms.TextBox txtFOVSize;
        private System.Windows.Forms.Label label9;
        private System.Windows.Forms.TextBox txtViewOverlap;
        private System.Windows.Forms.RadioButton rdMoveX;
        private System.Windows.Forms.RadioButton rdMoveY;
        private System.Windows.Forms.Label label10;
        private System.Windows.Forms.TextBox txtXYSpeed;
        private System.Windows.Forms.Label label11;
        private System.Windows.Forms.TextBox txtXYAcceleration;
        private System.Windows.Forms.Label label12;
        private System.Windows.Forms.TextBox txtZAcceleration;
        private System.Windows.Forms.Label label13;
        private System.Windows.Forms.TextBox txtZSpeed;
        private System.Windows.Forms.Label label14;
        private System.Windows.Forms.TextBox txtSummaryFile;
        private System.Windows.Forms.Label label15;
        private System.Windows.Forms.TextBox txtMinX;
        private System.Windows.Forms.TextBox txtMinY;
        private System.Windows.Forms.Label label16;
        private System.Windows.Forms.TextBox txtMaxY;
        private System.Windows.Forms.TextBox txtMaxX;
        private SySalNExTControls.SySalButton btnFromHere;
        private SySalNExTControls.SySalButton btnToHere;
        private System.Windows.Forms.Panel panel1;
        private System.Windows.Forms.TextBox txtCurrentConfiguration;
        private SySalNExTControls.SySalButton btnMakeCurrent;
        private SySalNExTControls.SySalButton btnDel;
        private SySalNExTControls.SySalButton btnLoad;
        private System.Windows.Forms.TextBox txtNewName;
        private SySalNExTControls.SySalButton btnDuplicate;
        private System.Windows.Forms.ListBox lbConfigurations;
        private SySalNExTControls.SySalButton btnNew;
        private System.Windows.Forms.Panel panel2;
        private SySalNExTControls.SySalButton btnStart;
        private SySalNExTControls.SySalButton btnStop;
        private SySalNExTControls.SySalProgressBar pbScanProgress;
        private System.Windows.Forms.Label label17;
        private System.Windows.Forms.TextBox txtSlowdownTime;
        private System.Windows.Forms.Label label18;
        private System.Windows.Forms.TextBox txtPositionTolerance;
        private SySalNExTControls.SySalButton btnExit;
        private System.Windows.Forms.Label label19;
        private System.Windows.Forms.TextBox txtContinuousMotionFraction;
        private System.Windows.Forms.Label label20;
        private System.Windows.Forms.TextBox txtEmuThickness;
        private System.Windows.Forms.TextBox txtStatus;
        private System.Windows.Forms.TextBox txtFPS;
        private SySalNExTControls.SySalButton btnMeasureFramesPerSecond;
        private System.Windows.Forms.Label label21;
        private System.Windows.Forms.Label label22;
        private SySalNExTControls.SySalProgressBar pbQueueLength;
        private SySalNExTControls.SySalButton btnContinuousMotionFraction;
        private System.Windows.Forms.SaveFileDialog sdlgZTest;
        private System.Windows.Forms.Label label23;
        private System.Windows.Forms.TextBox txtMotionLatency;
        private System.Windows.Forms.CheckBox chkStageSimOnly;
        private System.Windows.Forms.Label label24;
        private SySalNExTControls.SySalProgressBar pbWriteQueueLength;
        private System.Windows.Forms.CheckBox chkEnableGPUTest;
        private SySalNExTControls.SySalButton btnEmptyFocusMap;
        private System.Windows.Forms.Label label25;
        private System.Windows.Forms.TextBox txtZDataExpirationMin;
        private System.Windows.Forms.Label label26;
        private System.Windows.Forms.TextBox txtMaxViewsStrip;
        private SySalNExTControls.SySalButton btnDumpZMap;
        private System.Windows.Forms.CheckBox chkDumpFocusInfo;
        private System.Windows.Forms.CheckBox chkWriteClusterFiles;
        private System.Windows.Forms.SaveFileDialog sdlgZDump;
        private System.Windows.Forms.Label label27;
        private System.Windows.Forms.TextBox txtTransfMaxY;
        private System.Windows.Forms.TextBox txtTransfMaxX;
        private System.Windows.Forms.Label label28;
        private System.Windows.Forms.TextBox txtTransfMinY;
        private System.Windows.Forms.TextBox txtTransfMinX;
        private System.Windows.Forms.ListView lvPostProcessing;
        private System.Windows.Forms.ColumnHeader columnHeader1;
        private System.Windows.Forms.ColumnHeader columnHeader2;
        private SySalNExTControls.SySalButton btnStepReset;
        private SySalNExTControls.SySalButton btnStepPaste;
        private System.Windows.Forms.ToolTip FullSpeedAcqToolTip;
        private SySalNExTControls.SySalButton btnStepSave;
        private SySalNExTControls.SySalButton btnStepCopy;
        private SySalNExTControls.SySalButton btnRecover;
        private SySalNExTControls.SySalButton btnDebug;
        private System.Windows.Forms.OpenFileDialog odlgRecoverZone;
    }
}