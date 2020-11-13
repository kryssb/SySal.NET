namespace SySal.DAQSystem.Drivers.PredictionScan3Driver
{
    partial class CSToBrickForm
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
            this.gdiDisplay1 = new GDI3D.Control.GDIDisplay();
            this.btnXY = new System.Windows.Forms.Button();
            this.btnXZ = new System.Windows.Forms.Button();
            this.btnYZ = new System.Windows.Forms.Button();
            this.btnSetFocus = new System.Windows.Forms.Button();
            this.btnSave = new System.Windows.Forms.Button();
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.txtMaxDeltaSlope = new System.Windows.Forms.TextBox();
            this.label4 = new System.Windows.Forms.Label();
            this.txtMaxDeltaPos = new System.Windows.Forms.TextBox();
            this.label3 = new System.Windows.Forms.Label();
            this.txtMaxSigma = new System.Windows.Forms.TextBox();
            this.label2 = new System.Windows.Forms.Label();
            this.txtMinGrains = new System.Windows.Forms.TextBox();
            this.label1 = new System.Windows.Forms.Label();
            this.btnAcceptCand = new System.Windows.Forms.Button();
            this.label5 = new System.Windows.Forms.Label();
            this.txtTrackInfo = new System.Windows.Forms.TextBox();
            this.groupBox2 = new System.Windows.Forms.GroupBox();
            this.btnRemoveTrack = new System.Windows.Forms.Button();
            this.lvAcceptedTracks = new System.Windows.Forms.ListView();
            this.columnHeader1 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader2 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader3 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader4 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader5 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader6 = new System.Windows.Forms.ColumnHeader();
            this.btnZoomIn = new System.Windows.Forms.Button();
            this.btnZoomOut = new System.Windows.Forms.Button();
            this.btnExportSBInit = new System.Windows.Forms.Button();
            this.cmbPlates = new System.Windows.Forms.ComboBox();
            this.btnExportTSInit = new System.Windows.Forms.Button();
            this.chkTSUseSlope = new System.Windows.Forms.CheckBox();
            this.label6 = new System.Windows.Forms.Label();
            this.txtDownstreamPlates = new System.Windows.Forms.TextBox();
            this.txtUpstreamPlates = new System.Windows.Forms.TextBox();
            this.label7 = new System.Windows.Forms.Label();
            this.txtTSWidth = new System.Windows.Forms.TextBox();
            this.label8 = new System.Windows.Forms.Label();
            this.txtTSHeight = new System.Windows.Forms.TextBox();
            this.label9 = new System.Windows.Forms.Label();
            this.btnVertex = new System.Windows.Forms.Button();
            this.txtVtxX = new System.Windows.Forms.TextBox();
            this.label10 = new System.Windows.Forms.Label();
            this.txtVtxY = new System.Windows.Forms.TextBox();
            this.label11 = new System.Windows.Forms.Label();
            this.txtVtxZ = new System.Windows.Forms.TextBox();
            this.label12 = new System.Windows.Forms.Label();
            this.lvIPs = new System.Windows.Forms.ListView();
            this.columnHeader7 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader8 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader9 = new System.Windows.Forms.ColumnHeader();
            this.txtVtxDownstreamPlate = new System.Windows.Forms.TextBox();
            this.label13 = new System.Windows.Forms.Label();
            this.txtVtxDepth = new System.Windows.Forms.TextBox();
            this.label14 = new System.Windows.Forms.Label();
            this.btnAddVertexPos = new System.Windows.Forms.Button();
            this.btnTrackFollowAutoStart = new System.Windows.Forms.Button();
            this.btnSetPathsAndCSBrick = new System.Windows.Forms.Button();
            this.groupBox1.SuspendLayout();
            this.groupBox2.SuspendLayout();
            this.SuspendLayout();
            // 
            // gdiDisplay1
            // 
            this.gdiDisplay1.Alpha = 1;
            this.gdiDisplay1.AutoRender = true;
            this.gdiDisplay1.BackColor = System.Drawing.Color.Black;
            this.gdiDisplay1.BorderWidth = 1;
            this.gdiDisplay1.ClickSelect = null;
            this.gdiDisplay1.Distance = 1000000;
            this.gdiDisplay1.DoubleClickSelect = null;
            this.gdiDisplay1.Infinity = true;
            this.gdiDisplay1.LabelFontName = "Arial";
            this.gdiDisplay1.LabelFontSize = 12;
            this.gdiDisplay1.LineWidth = 1;
            this.gdiDisplay1.Location = new System.Drawing.Point(108, 12);
            this.gdiDisplay1.MouseMode = GDI3D.Control.MouseMotion.Rotate;
            this.gdiDisplay1.MouseMultiplier = 0.01;
            this.gdiDisplay1.Name = "gdiDisplay1";
            this.gdiDisplay1.NextClickSetsCenter = false;
            this.gdiDisplay1.PointSize = 5;
            this.gdiDisplay1.Size = new System.Drawing.Size(576, 433);
            this.gdiDisplay1.TabIndex = 0;
            this.gdiDisplay1.Zoom = 0.0025;
            // 
            // btnXY
            // 
            this.btnXY.Location = new System.Drawing.Point(6, 12);
            this.btnXY.Name = "btnXY";
            this.btnXY.Size = new System.Drawing.Size(96, 24);
            this.btnXY.TabIndex = 1;
            this.btnXY.Text = "XY View";
            this.btnXY.UseVisualStyleBackColor = true;
            this.btnXY.Click += new System.EventHandler(this.btnXY_Click);
            // 
            // btnXZ
            // 
            this.btnXZ.Location = new System.Drawing.Point(6, 42);
            this.btnXZ.Name = "btnXZ";
            this.btnXZ.Size = new System.Drawing.Size(96, 24);
            this.btnXZ.TabIndex = 2;
            this.btnXZ.Text = "XZ View";
            this.btnXZ.UseVisualStyleBackColor = true;
            this.btnXZ.Click += new System.EventHandler(this.btnXZ_Click);
            // 
            // btnYZ
            // 
            this.btnYZ.Location = new System.Drawing.Point(6, 72);
            this.btnYZ.Name = "btnYZ";
            this.btnYZ.Size = new System.Drawing.Size(96, 24);
            this.btnYZ.TabIndex = 3;
            this.btnYZ.Text = "YZ View";
            this.btnYZ.UseVisualStyleBackColor = true;
            this.btnYZ.Click += new System.EventHandler(this.btnYZ_Click);
            // 
            // btnSetFocus
            // 
            this.btnSetFocus.Location = new System.Drawing.Point(6, 102);
            this.btnSetFocus.Name = "btnSetFocus";
            this.btnSetFocus.Size = new System.Drawing.Size(96, 24);
            this.btnSetFocus.TabIndex = 4;
            this.btnSetFocus.Text = "Set Focus";
            this.btnSetFocus.UseVisualStyleBackColor = true;
            this.btnSetFocus.Click += new System.EventHandler(this.btnSetFocus_Click);
            // 
            // btnSave
            // 
            this.btnSave.Location = new System.Drawing.Point(6, 197);
            this.btnSave.Name = "btnSave";
            this.btnSave.Size = new System.Drawing.Size(96, 24);
            this.btnSave.TabIndex = 5;
            this.btnSave.Text = "Save Plot";
            this.btnSave.UseVisualStyleBackColor = true;
            this.btnSave.Click += new System.EventHandler(this.btnSave_Click);
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.txtMaxDeltaSlope);
            this.groupBox1.Controls.Add(this.label4);
            this.groupBox1.Controls.Add(this.txtMaxDeltaPos);
            this.groupBox1.Controls.Add(this.label3);
            this.groupBox1.Controls.Add(this.txtMaxSigma);
            this.groupBox1.Controls.Add(this.label2);
            this.groupBox1.Controls.Add(this.txtMinGrains);
            this.groupBox1.Controls.Add(this.label1);
            this.groupBox1.Location = new System.Drawing.Point(693, 5);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(311, 77);
            this.groupBox1.TabIndex = 6;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Selections";
            // 
            // txtMaxDeltaSlope
            // 
            this.txtMaxDeltaSlope.Location = new System.Drawing.Point(243, 45);
            this.txtMaxDeltaSlope.Name = "txtMaxDeltaSlope";
            this.txtMaxDeltaSlope.Size = new System.Drawing.Size(49, 20);
            this.txtMaxDeltaSlope.TabIndex = 7;
            this.txtMaxDeltaSlope.Leave += new System.EventHandler(this.OnMaxDeltaSlopeLeave);
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(165, 47);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(78, 13);
            this.label4.TabIndex = 6;
            this.label4.Text = "Max deltaslope";
            // 
            // txtMaxDeltaPos
            // 
            this.txtMaxDeltaPos.Location = new System.Drawing.Point(243, 20);
            this.txtMaxDeltaPos.Name = "txtMaxDeltaPos";
            this.txtMaxDeltaPos.Size = new System.Drawing.Size(49, 20);
            this.txtMaxDeltaPos.TabIndex = 5;
            this.txtMaxDeltaPos.Leave += new System.EventHandler(this.OnMaxDeltaPosLeave);
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(165, 22);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(70, 13);
            this.label3.TabIndex = 4;
            this.label3.Text = "Max deltapos";
            // 
            // txtMaxSigma
            // 
            this.txtMaxSigma.Location = new System.Drawing.Point(91, 46);
            this.txtMaxSigma.Name = "txtMaxSigma";
            this.txtMaxSigma.Size = new System.Drawing.Size(49, 20);
            this.txtMaxSigma.TabIndex = 3;
            this.txtMaxSigma.Leave += new System.EventHandler(this.OnMaxSigmaLeave);
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(13, 48);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(57, 13);
            this.label2.TabIndex = 2;
            this.label2.Text = "Max sigma";
            // 
            // txtMinGrains
            // 
            this.txtMinGrains.Location = new System.Drawing.Point(91, 20);
            this.txtMinGrains.Name = "txtMinGrains";
            this.txtMinGrains.Size = new System.Drawing.Size(49, 20);
            this.txtMinGrains.TabIndex = 1;
            this.txtMinGrains.Leave += new System.EventHandler(this.OnMinGrainsLeave);
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(13, 22);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(55, 13);
            this.label1.TabIndex = 0;
            this.label1.Text = "Min grains";
            // 
            // btnAcceptCand
            // 
            this.btnAcceptCand.Location = new System.Drawing.Point(693, 93);
            this.btnAcceptCand.Name = "btnAcceptCand";
            this.btnAcceptCand.Size = new System.Drawing.Size(140, 24);
            this.btnAcceptCand.TabIndex = 7;
            this.btnAcceptCand.Text = "Accept track";
            this.btnAcceptCand.UseVisualStyleBackColor = true;
            this.btnAcceptCand.Click += new System.EventHandler(this.btnAcceptCand_Click);
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(866, 98);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(78, 13);
            this.label5.TabIndex = 8;
            this.label5.Text = "Project to plate";
            // 
            // txtTrackInfo
            // 
            this.txtTrackInfo.Location = new System.Drawing.Point(695, 126);
            this.txtTrackInfo.Multiline = true;
            this.txtTrackInfo.Name = "txtTrackInfo";
            this.txtTrackInfo.ReadOnly = true;
            this.txtTrackInfo.Size = new System.Drawing.Size(309, 91);
            this.txtTrackInfo.TabIndex = 10;
            // 
            // groupBox2
            // 
            this.groupBox2.Controls.Add(this.btnRemoveTrack);
            this.groupBox2.Controls.Add(this.lvAcceptedTracks);
            this.groupBox2.Location = new System.Drawing.Point(693, 223);
            this.groupBox2.Name = "groupBox2";
            this.groupBox2.Size = new System.Drawing.Size(311, 222);
            this.groupBox2.TabIndex = 11;
            this.groupBox2.TabStop = false;
            this.groupBox2.Text = "Accepted tracks";
            // 
            // btnRemoveTrack
            // 
            this.btnRemoveTrack.Location = new System.Drawing.Point(11, 184);
            this.btnRemoveTrack.Name = "btnRemoveTrack";
            this.btnRemoveTrack.Size = new System.Drawing.Size(289, 24);
            this.btnRemoveTrack.TabIndex = 12;
            this.btnRemoveTrack.Text = "Remove track";
            this.btnRemoveTrack.UseVisualStyleBackColor = true;
            this.btnRemoveTrack.Click += new System.EventHandler(this.btnRemoveTrack_Click);
            // 
            // lvAcceptedTracks
            // 
            this.lvAcceptedTracks.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.columnHeader1,
            this.columnHeader2,
            this.columnHeader3,
            this.columnHeader4,
            this.columnHeader5,
            this.columnHeader6});
            this.lvAcceptedTracks.FullRowSelect = true;
            this.lvAcceptedTracks.GridLines = true;
            this.lvAcceptedTracks.Location = new System.Drawing.Point(11, 22);
            this.lvAcceptedTracks.Name = "lvAcceptedTracks";
            this.lvAcceptedTracks.Size = new System.Drawing.Size(289, 156);
            this.lvAcceptedTracks.TabIndex = 0;
            this.lvAcceptedTracks.UseCompatibleStateImageBehavior = false;
            this.lvAcceptedTracks.View = System.Windows.Forms.View.Details;
            this.lvAcceptedTracks.SelectedIndexChanged += new System.EventHandler(this.OnSelAcceptTrackChanged);
            // 
            // columnHeader1
            // 
            this.columnHeader1.Text = "Path";
            this.columnHeader1.Width = 20;
            // 
            // columnHeader2
            // 
            this.columnHeader2.Text = "PX";
            this.columnHeader2.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // columnHeader3
            // 
            this.columnHeader3.Text = "PY";
            this.columnHeader3.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // columnHeader4
            // 
            this.columnHeader4.Text = "SX";
            this.columnHeader4.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.columnHeader4.Width = 50;
            // 
            // columnHeader5
            // 
            this.columnHeader5.Text = "SY";
            this.columnHeader5.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.columnHeader5.Width = 50;
            // 
            // columnHeader6
            // 
            this.columnHeader6.Text = "Plate";
            this.columnHeader6.Width = 40;
            // 
            // btnZoomIn
            // 
            this.btnZoomIn.Location = new System.Drawing.Point(6, 135);
            this.btnZoomIn.Name = "btnZoomIn";
            this.btnZoomIn.Size = new System.Drawing.Size(38, 24);
            this.btnZoomIn.TabIndex = 12;
            this.btnZoomIn.Text = "+";
            this.btnZoomIn.UseVisualStyleBackColor = true;
            this.btnZoomIn.Click += new System.EventHandler(this.btnZoomIn_Click);
            // 
            // btnZoomOut
            // 
            this.btnZoomOut.Location = new System.Drawing.Point(64, 135);
            this.btnZoomOut.Name = "btnZoomOut";
            this.btnZoomOut.Size = new System.Drawing.Size(38, 24);
            this.btnZoomOut.TabIndex = 13;
            this.btnZoomOut.Text = "-";
            this.btnZoomOut.UseVisualStyleBackColor = true;
            this.btnZoomOut.Click += new System.EventHandler(this.btnZoomOut_Click);
            // 
            // btnExportSBInit
            // 
            this.btnExportSBInit.Location = new System.Drawing.Point(693, 451);
            this.btnExportSBInit.Name = "btnExportSBInit";
            this.btnExportSBInit.Size = new System.Drawing.Size(158, 24);
            this.btnExportSBInit.TabIndex = 14;
            this.btnExportSBInit.Text = "Export to SB init file";
            this.btnExportSBInit.UseVisualStyleBackColor = true;
            this.btnExportSBInit.Click += new System.EventHandler(this.btnExportSBInit_Click);
            // 
            // cmbPlates
            // 
            this.cmbPlates.DropDownHeight = 200;
            this.cmbPlates.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cmbPlates.FormattingEnabled = true;
            this.cmbPlates.IntegralHeight = false;
            this.cmbPlates.Location = new System.Drawing.Point(954, 93);
            this.cmbPlates.Name = "cmbPlates";
            this.cmbPlates.Size = new System.Drawing.Size(50, 21);
            this.cmbPlates.TabIndex = 15;
            this.cmbPlates.SelectedIndexChanged += new System.EventHandler(this.OnProjPlateChanged);
            // 
            // btnExportTSInit
            // 
            this.btnExportTSInit.Location = new System.Drawing.Point(693, 476);
            this.btnExportTSInit.Name = "btnExportTSInit";
            this.btnExportTSInit.Size = new System.Drawing.Size(158, 24);
            this.btnExportTSInit.TabIndex = 16;
            this.btnExportTSInit.Text = "Export to TS init file";
            this.btnExportTSInit.UseVisualStyleBackColor = true;
            this.btnExportTSInit.Click += new System.EventHandler(this.btnExportTSInit_Click);
            // 
            // chkTSUseSlope
            // 
            this.chkTSUseSlope.AutoSize = true;
            this.chkTSUseSlope.Checked = true;
            this.chkTSUseSlope.CheckState = System.Windows.Forms.CheckState.Checked;
            this.chkTSUseSlope.Location = new System.Drawing.Point(861, 481);
            this.chkTSUseSlope.Name = "chkTSUseSlope";
            this.chkTSUseSlope.Size = new System.Drawing.Size(105, 17);
            this.chkTSUseSlope.TabIndex = 17;
            this.chkTSUseSlope.Text = "Use track slopes";
            this.chkTSUseSlope.UseVisualStyleBackColor = true;
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Location = new System.Drawing.Point(694, 504);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(97, 13);
            this.label6.TabIndex = 18;
            this.label6.Text = "Downstream plates";
            // 
            // txtDownstreamPlates
            // 
            this.txtDownstreamPlates.Location = new System.Drawing.Point(797, 502);
            this.txtDownstreamPlates.Name = "txtDownstreamPlates";
            this.txtDownstreamPlates.Size = new System.Drawing.Size(55, 20);
            this.txtDownstreamPlates.TabIndex = 19;
            this.txtDownstreamPlates.Leave += new System.EventHandler(this.OnDownstreamPlatesLeave);
            // 
            // txtUpstreamPlates
            // 
            this.txtUpstreamPlates.Location = new System.Drawing.Point(796, 527);
            this.txtUpstreamPlates.Name = "txtUpstreamPlates";
            this.txtUpstreamPlates.Size = new System.Drawing.Size(55, 20);
            this.txtUpstreamPlates.TabIndex = 21;
            this.txtUpstreamPlates.Leave += new System.EventHandler(this.OnUpstreamPlatesLeave);
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.Location = new System.Drawing.Point(694, 530);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(83, 13);
            this.label7.TabIndex = 20;
            this.label7.Text = "Upstream plates";
            // 
            // txtTSWidth
            // 
            this.txtTSWidth.Location = new System.Drawing.Point(899, 503);
            this.txtTSWidth.Name = "txtTSWidth";
            this.txtTSWidth.Size = new System.Drawing.Size(55, 20);
            this.txtTSWidth.TabIndex = 23;
            this.txtTSWidth.Leave += new System.EventHandler(this.OnTSWidthLeave);
            // 
            // label8
            // 
            this.label8.AutoSize = true;
            this.label8.Location = new System.Drawing.Point(858, 506);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(35, 13);
            this.label8.TabIndex = 22;
            this.label8.Text = "Width";
            // 
            // txtTSHeight
            // 
            this.txtTSHeight.Location = new System.Drawing.Point(899, 527);
            this.txtTSHeight.Name = "txtTSHeight";
            this.txtTSHeight.Size = new System.Drawing.Size(55, 20);
            this.txtTSHeight.TabIndex = 25;
            this.txtTSHeight.Leave += new System.EventHandler(this.OnTSHeightLeave);
            // 
            // label9
            // 
            this.label9.AutoSize = true;
            this.label9.Location = new System.Drawing.Point(858, 530);
            this.label9.Name = "label9";
            this.label9.Size = new System.Drawing.Size(38, 13);
            this.label9.TabIndex = 24;
            this.label9.Text = "Height";
            // 
            // btnVertex
            // 
            this.btnVertex.Location = new System.Drawing.Point(104, 451);
            this.btnVertex.Name = "btnVertex";
            this.btnVertex.Size = new System.Drawing.Size(104, 24);
            this.btnVertex.TabIndex = 26;
            this.btnVertex.Text = "Test vertex";
            this.btnVertex.UseVisualStyleBackColor = true;
            this.btnVertex.Click += new System.EventHandler(this.btnVertex_Click);
            // 
            // txtVtxX
            // 
            this.txtVtxX.Location = new System.Drawing.Point(235, 454);
            this.txtVtxX.Name = "txtVtxX";
            this.txtVtxX.ReadOnly = true;
            this.txtVtxX.Size = new System.Drawing.Size(55, 20);
            this.txtVtxX.TabIndex = 28;
            // 
            // label10
            // 
            this.label10.AutoSize = true;
            this.label10.Location = new System.Drawing.Point(215, 456);
            this.label10.Name = "label10";
            this.label10.Size = new System.Drawing.Size(14, 13);
            this.label10.TabIndex = 27;
            this.label10.Text = "X";
            // 
            // txtVtxY
            // 
            this.txtVtxY.Location = new System.Drawing.Point(319, 454);
            this.txtVtxY.Name = "txtVtxY";
            this.txtVtxY.ReadOnly = true;
            this.txtVtxY.Size = new System.Drawing.Size(55, 20);
            this.txtVtxY.TabIndex = 30;
            // 
            // label11
            // 
            this.label11.AutoSize = true;
            this.label11.Location = new System.Drawing.Point(299, 456);
            this.label11.Name = "label11";
            this.label11.Size = new System.Drawing.Size(14, 13);
            this.label11.TabIndex = 29;
            this.label11.Text = "Y";
            // 
            // txtVtxZ
            // 
            this.txtVtxZ.Location = new System.Drawing.Point(398, 454);
            this.txtVtxZ.Name = "txtVtxZ";
            this.txtVtxZ.ReadOnly = true;
            this.txtVtxZ.Size = new System.Drawing.Size(55, 20);
            this.txtVtxZ.TabIndex = 32;
            // 
            // label12
            // 
            this.label12.AutoSize = true;
            this.label12.Location = new System.Drawing.Point(378, 456);
            this.label12.Name = "label12";
            this.label12.Size = new System.Drawing.Size(14, 13);
            this.label12.TabIndex = 31;
            this.label12.Text = "Z";
            // 
            // lvIPs
            // 
            this.lvIPs.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.columnHeader7,
            this.columnHeader8,
            this.columnHeader9});
            this.lvIPs.FullRowSelect = true;
            this.lvIPs.GridLines = true;
            this.lvIPs.Location = new System.Drawing.Point(494, 451);
            this.lvIPs.Name = "lvIPs";
            this.lvIPs.Size = new System.Drawing.Size(190, 127);
            this.lvIPs.TabIndex = 33;
            this.lvIPs.UseCompatibleStateImageBehavior = false;
            this.lvIPs.View = System.Windows.Forms.View.Details;
            // 
            // columnHeader7
            // 
            this.columnHeader7.Text = "Track#";
            // 
            // columnHeader8
            // 
            this.columnHeader8.Text = "IP";
            this.columnHeader8.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // columnHeader9
            // 
            this.columnHeader9.Text = "D_IP";
            this.columnHeader9.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // txtVtxDownstreamPlate
            // 
            this.txtVtxDownstreamPlate.Location = new System.Drawing.Point(235, 480);
            this.txtVtxDownstreamPlate.Name = "txtVtxDownstreamPlate";
            this.txtVtxDownstreamPlate.ReadOnly = true;
            this.txtVtxDownstreamPlate.Size = new System.Drawing.Size(55, 20);
            this.txtVtxDownstreamPlate.TabIndex = 35;
            // 
            // label13
            // 
            this.label13.AutoSize = true;
            this.label13.Location = new System.Drawing.Point(137, 483);
            this.label13.Name = "label13";
            this.label13.Size = new System.Drawing.Size(92, 13);
            this.label13.TabIndex = 34;
            this.label13.Text = "Downstream plate";
            // 
            // txtVtxDepth
            // 
            this.txtVtxDepth.Location = new System.Drawing.Point(398, 480);
            this.txtVtxDepth.Name = "txtVtxDepth";
            this.txtVtxDepth.ReadOnly = true;
            this.txtVtxDepth.Size = new System.Drawing.Size(55, 20);
            this.txtVtxDepth.TabIndex = 37;
            // 
            // label14
            // 
            this.label14.AutoSize = true;
            this.label14.Location = new System.Drawing.Point(356, 483);
            this.label14.Name = "label14";
            this.label14.Size = new System.Drawing.Size(36, 13);
            this.label14.TabIndex = 36;
            this.label14.Text = "Depth";
            // 
            // btnAddVertexPos
            // 
            this.btnAddVertexPos.Location = new System.Drawing.Point(104, 518);
            this.btnAddVertexPos.Name = "btnAddVertexPos";
            this.btnAddVertexPos.Size = new System.Drawing.Size(209, 24);
            this.btnAddVertexPos.TabIndex = 38;
            this.btnAddVertexPos.Text = "Add vertex position as track";
            this.btnAddVertexPos.UseVisualStyleBackColor = true;
            this.btnAddVertexPos.Click += new System.EventHandler(this.btnAddVertexPos_Click);
            // 
            // btnTrackFollowAutoStart
            // 
            this.btnTrackFollowAutoStart.Location = new System.Drawing.Point(857, 451);
            this.btnTrackFollowAutoStart.Name = "btnTrackFollowAutoStart";
            this.btnTrackFollowAutoStart.Size = new System.Drawing.Size(147, 24);
            this.btnTrackFollowAutoStart.TabIndex = 39;
            this.btnTrackFollowAutoStart.Text = "Enqueue TrackFollow";
            this.btnTrackFollowAutoStart.UseVisualStyleBackColor = true;
            this.btnTrackFollowAutoStart.Click += new System.EventHandler(this.btnTrackFollowAutoStart_Click);
            // 
            // btnSetPathsAndCSBrick
            // 
            this.btnSetPathsAndCSBrick.Location = new System.Drawing.Point(694, 553);
            this.btnSetPathsAndCSBrick.Name = "btnSetPathsAndCSBrick";
            this.btnSetPathsAndCSBrick.Size = new System.Drawing.Size(310, 24);
            this.btnSetPathsAndCSBrick.TabIndex = 40;
            this.btnSetPathsAndCSBrick.Text = "Set paths in CS-Brick connection and close operation";
            this.btnSetPathsAndCSBrick.UseVisualStyleBackColor = true;
            this.btnSetPathsAndCSBrick.Click += new System.EventHandler(this.btnSetPathsAndCSBrick_Click);
            // 
            // CSToBrickForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1017, 590);
            this.Controls.Add(this.btnSetPathsAndCSBrick);
            this.Controls.Add(this.btnTrackFollowAutoStart);
            this.Controls.Add(this.btnAddVertexPos);
            this.Controls.Add(this.txtVtxDepth);
            this.Controls.Add(this.label14);
            this.Controls.Add(this.txtVtxDownstreamPlate);
            this.Controls.Add(this.label13);
            this.Controls.Add(this.lvIPs);
            this.Controls.Add(this.txtVtxZ);
            this.Controls.Add(this.label12);
            this.Controls.Add(this.txtVtxY);
            this.Controls.Add(this.label11);
            this.Controls.Add(this.txtVtxX);
            this.Controls.Add(this.label10);
            this.Controls.Add(this.btnVertex);
            this.Controls.Add(this.txtTSHeight);
            this.Controls.Add(this.label9);
            this.Controls.Add(this.txtTSWidth);
            this.Controls.Add(this.label8);
            this.Controls.Add(this.txtUpstreamPlates);
            this.Controls.Add(this.label7);
            this.Controls.Add(this.txtDownstreamPlates);
            this.Controls.Add(this.label6);
            this.Controls.Add(this.chkTSUseSlope);
            this.Controls.Add(this.btnExportTSInit);
            this.Controls.Add(this.cmbPlates);
            this.Controls.Add(this.btnExportSBInit);
            this.Controls.Add(this.btnZoomOut);
            this.Controls.Add(this.btnZoomIn);
            this.Controls.Add(this.groupBox2);
            this.Controls.Add(this.txtTrackInfo);
            this.Controls.Add(this.label5);
            this.Controls.Add(this.btnAcceptCand);
            this.Controls.Add(this.groupBox1);
            this.Controls.Add(this.btnSave);
            this.Controls.Add(this.btnSetFocus);
            this.Controls.Add(this.btnYZ);
            this.Controls.Add(this.btnXZ);
            this.Controls.Add(this.btnXY);
            this.Controls.Add(this.gdiDisplay1);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedSingle;
            this.Name = "CSToBrickForm";
            this.Text = "CS-Brick Connection Assistant";
            this.Load += new System.EventHandler(this.OnLoad);
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            this.groupBox2.ResumeLayout(false);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private GDI3D.Control.GDIDisplay gdiDisplay1;
        private System.Windows.Forms.Button btnXY;
        private System.Windows.Forms.Button btnXZ;
        private System.Windows.Forms.Button btnYZ;
        private System.Windows.Forms.Button btnSetFocus;
        private System.Windows.Forms.Button btnSave;
        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.TextBox txtMaxDeltaSlope;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.TextBox txtMaxDeltaPos;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.TextBox txtMaxSigma;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.TextBox txtMinGrains;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Button btnAcceptCand;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.TextBox txtTrackInfo;
        private System.Windows.Forms.GroupBox groupBox2;
        private System.Windows.Forms.Button btnRemoveTrack;
        private System.Windows.Forms.ListView lvAcceptedTracks;
        private System.Windows.Forms.Button btnZoomIn;
        private System.Windows.Forms.Button btnZoomOut;
        private System.Windows.Forms.ColumnHeader columnHeader1;
        private System.Windows.Forms.ColumnHeader columnHeader2;
        private System.Windows.Forms.ColumnHeader columnHeader3;
        private System.Windows.Forms.ColumnHeader columnHeader4;
        private System.Windows.Forms.ColumnHeader columnHeader5;
        private System.Windows.Forms.ColumnHeader columnHeader6;
        private System.Windows.Forms.Button btnExportSBInit;
        private System.Windows.Forms.ComboBox cmbPlates;
        private System.Windows.Forms.Button btnExportTSInit;
        private System.Windows.Forms.CheckBox chkTSUseSlope;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.TextBox txtDownstreamPlates;
        private System.Windows.Forms.TextBox txtUpstreamPlates;
        private System.Windows.Forms.Label label7;
        private System.Windows.Forms.TextBox txtTSWidth;
        private System.Windows.Forms.Label label8;
        private System.Windows.Forms.TextBox txtTSHeight;
        private System.Windows.Forms.Label label9;
        private System.Windows.Forms.Button btnVertex;
        private System.Windows.Forms.TextBox txtVtxX;
        private System.Windows.Forms.Label label10;
        private System.Windows.Forms.TextBox txtVtxY;
        private System.Windows.Forms.Label label11;
        private System.Windows.Forms.TextBox txtVtxZ;
        private System.Windows.Forms.Label label12;
        private System.Windows.Forms.ListView lvIPs;
        private System.Windows.Forms.ColumnHeader columnHeader7;
        private System.Windows.Forms.ColumnHeader columnHeader8;
        private System.Windows.Forms.ColumnHeader columnHeader9;
        private System.Windows.Forms.TextBox txtVtxDownstreamPlate;
        private System.Windows.Forms.Label label13;
        private System.Windows.Forms.TextBox txtVtxDepth;
        private System.Windows.Forms.Label label14;
        private System.Windows.Forms.Button btnAddVertexPos;
        private System.Windows.Forms.Button btnTrackFollowAutoStart;
        private System.Windows.Forms.Button btnSetPathsAndCSBrick;
    }
}