namespace SySal.Executables.OperaLabManager
{
    partial class MainForm
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
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(MainForm));
            this.lvBricks = new System.Windows.Forms.ListView();
            this.columnHeader1 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader2 = new System.Windows.Forms.ColumnHeader();
            this.label1 = new System.Windows.Forms.Label();
            this.label2 = new System.Windows.Forms.Label();
            this.lvOps = new System.Windows.Forms.ListView();
            this.columnHeader3 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader4 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader5 = new System.Windows.Forms.ColumnHeader();
            this.txtNotes = new System.Windows.Forms.TextBox();
            this.label3 = new System.Windows.Forms.Label();
            this.chkToDo = new System.Windows.Forms.CheckBox();
            this.btnReceiveBrick = new System.Windows.Forms.Button();
            this.btnRefresh = new System.Windows.Forms.Button();
            this.txtRecvBrickId = new System.Windows.Forms.TextBox();
            this.btnCloseBrick = new System.Windows.Forms.Button();
            this.btnAddManualOp = new System.Windows.Forms.Button();
            this.label4 = new System.Windows.Forms.Label();
            this.cmbOpType = new System.Windows.Forms.ComboBox();
            this.btnImportAutoOp = new System.Windows.Forms.Button();
            this.btnPubBrick = new System.Windows.Forms.Button();
            this.btnUpdateNotes = new System.Windows.Forms.Button();
            this.btnAddLabManagerSupport = new System.Windows.Forms.Button();
            this.lvFiles = new System.Windows.Forms.ListView();
            this.imgFiles = new System.Windows.Forms.ImageList(this.components);
            this.label5 = new System.Windows.Forms.Label();
            this.btnImportFile = new System.Windows.Forms.Button();
            this.btnSaveFile = new System.Windows.Forms.Button();
            this.btnRemoveFile = new System.Windows.Forms.Button();
            this.ofnSelectImportFile = new System.Windows.Forms.OpenFileDialog();
            this.sfSelectSaveFile = new System.Windows.Forms.SaveFileDialog();
            this.pcBox = new System.Windows.Forms.PictureBox();
            this.rtbText = new System.Windows.Forms.RichTextBox();
            this.gdiDisp = new GDI3D.Control.GDIDisplay();
            this.btnZoomIn = new System.Windows.Forms.Button();
            this.btnZoomOut = new System.Windows.Forms.Button();
            this.btnInfinity = new System.Windows.Forms.Button();
            this.btnDeleteOp = new System.Windows.Forms.Button();
            this.txtFilter = new System.Windows.Forms.TextBox();
            this.label6 = new System.Windows.Forms.Label();
            this.lvSecondBrick = new System.Windows.Forms.ListView();
            this.NumberOfBrick = new System.Windows.Forms.ColumnHeader();
            this.columnHeader6 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader7 = new System.Windows.Forms.ColumnHeader();
            this.txtFilterEvent = new System.Windows.Forms.TextBox();
            this.btnSearchEvent = new System.Windows.Forms.Button();
            this.btnSearchStatus = new System.Windows.Forms.Button();
            this.txtStatus = new System.Windows.Forms.TextBox();
            this.box2008 = new System.Windows.Forms.CheckBox();
            this.box2009 = new System.Windows.Forms.CheckBox();
            this.box2010 = new System.Windows.Forms.CheckBox();
            this.box2011 = new System.Windows.Forms.CheckBox();
            this.boxAll = new System.Windows.Forms.CheckBox();
            this.txtResultStatus = new System.Windows.Forms.TextBox();
            this.txtType = new System.Windows.Forms.TextBox();
            ((System.ComponentModel.ISupportInitialize)(this.pcBox)).BeginInit();
            this.SuspendLayout();
            // 
            // lvBricks
            // 
            this.lvBricks.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.columnHeader1,
            this.columnHeader2});
            this.lvBricks.FullRowSelect = true;
            this.lvBricks.GridLines = true;
            this.lvBricks.HideSelection = false;
            this.lvBricks.Location = new System.Drawing.Point(15, 156);
            this.lvBricks.MultiSelect = false;
            this.lvBricks.Name = "lvBricks";
            this.lvBricks.Size = new System.Drawing.Size(274, 175);
            this.lvBricks.TabIndex = 0;
            this.lvBricks.UseCompatibleStateImageBehavior = false;
            this.lvBricks.View = System.Windows.Forms.View.Details;
            this.lvBricks.SelectedIndexChanged += new System.EventHandler(this.OnSelBrick);
            // 
            // columnHeader1
            // 
            this.columnHeader1.Text = "Brick";
            this.columnHeader1.Width = 73;
            // 
            // columnHeader2
            // 
            this.columnHeader2.Text = "Status";
            this.columnHeader2.Width = 179;
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label1.Location = new System.Drawing.Point(12, 9);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(42, 13);
            this.label1.TabIndex = 1;
            this.label1.Text = "Bricks";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label2.Location = new System.Drawing.Point(307, 9);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(171, 13);
            this.label2.TabIndex = 3;
            this.label2.Text = "Operations for selected brick";
            // 
            // lvOps
            // 
            this.lvOps.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.columnHeader3,
            this.columnHeader4,
            this.columnHeader5});
            this.lvOps.FullRowSelect = true;
            this.lvOps.GridLines = true;
            this.lvOps.HideSelection = false;
            this.lvOps.Location = new System.Drawing.Point(310, 32);
            this.lvOps.MultiSelect = false;
            this.lvOps.Name = "lvOps";
            this.lvOps.Size = new System.Drawing.Size(346, 195);
            this.lvOps.TabIndex = 4;
            this.lvOps.UseCompatibleStateImageBehavior = false;
            this.lvOps.View = System.Windows.Forms.View.Details;
            this.lvOps.SelectedIndexChanged += new System.EventHandler(this.OnSelOp);
            // 
            // columnHeader3
            // 
            this.columnHeader3.Text = "Type";
            this.columnHeader3.Width = 73;
            // 
            // columnHeader4
            // 
            this.columnHeader4.Text = "Process Operation Id";
            this.columnHeader4.Width = 179;
            // 
            // columnHeader5
            // 
            this.columnHeader5.Text = "Notes";
            this.columnHeader5.Width = 68;
            // 
            // txtNotes
            // 
            this.txtNotes.Location = new System.Drawing.Point(310, 259);
            this.txtNotes.Multiline = true;
            this.txtNotes.Name = "txtNotes";
            this.txtNotes.ScrollBars = System.Windows.Forms.ScrollBars.Both;
            this.txtNotes.Size = new System.Drawing.Size(636, 242);
            this.txtNotes.TabIndex = 5;
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label3.Location = new System.Drawing.Point(307, 243);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(168, 13);
            this.label3.TabIndex = 6;
            this.label3.Text = "Notes for selected operation";
            // 
            // chkToDo
            // 
            this.chkToDo.AutoSize = true;
            this.chkToDo.Checked = true;
            this.chkToDo.CheckState = System.Windows.Forms.CheckState.Checked;
            this.chkToDo.Location = new System.Drawing.Point(77, 8);
            this.chkToDo.Name = "chkToDo";
            this.chkToDo.Size = new System.Drawing.Size(76, 17);
            this.chkToDo.TabIndex = 7;
            this.chkToDo.Text = "To do only";
            this.chkToDo.UseVisualStyleBackColor = true;
            // 
            // btnReceiveBrick
            // 
            this.btnReceiveBrick.Location = new System.Drawing.Point(12, 495);
            this.btnReceiveBrick.Name = "btnReceiveBrick";
            this.btnReceiveBrick.Size = new System.Drawing.Size(95, 22);
            this.btnReceiveBrick.TabIndex = 9;
            this.btnReceiveBrick.Text = "Receive Brick";
            this.btnReceiveBrick.UseVisualStyleBackColor = true;
            this.btnReceiveBrick.Click += new System.EventHandler(this.btnReceiveBrick_Click);
            // 
            // btnRefresh
            // 
            this.btnRefresh.Location = new System.Drawing.Point(227, 26);
            this.btnRefresh.Name = "btnRefresh";
            this.btnRefresh.Size = new System.Drawing.Size(63, 22);
            this.btnRefresh.TabIndex = 10;
            this.btnRefresh.Text = "Refresh";
            this.btnRefresh.UseVisualStyleBackColor = true;
            this.btnRefresh.Click += new System.EventHandler(this.btnRefresh_Click);
            // 
            // txtRecvBrickId
            // 
            this.txtRecvBrickId.Location = new System.Drawing.Point(129, 495);
            this.txtRecvBrickId.Name = "txtRecvBrickId";
            this.txtRecvBrickId.Size = new System.Drawing.Size(94, 20);
            this.txtRecvBrickId.TabIndex = 11;
            // 
            // btnCloseBrick
            // 
            this.btnCloseBrick.Location = new System.Drawing.Point(12, 361);
            this.btnCloseBrick.Name = "btnCloseBrick";
            this.btnCloseBrick.Size = new System.Drawing.Size(123, 22);
            this.btnCloseBrick.TabIndex = 12;
            this.btnCloseBrick.Text = "Close Selected Brick";
            this.btnCloseBrick.UseVisualStyleBackColor = true;
            this.btnCloseBrick.Click += new System.EventHandler(this.btnCloseBrick_Click);
            // 
            // btnAddManualOp
            // 
            this.btnAddManualOp.Location = new System.Drawing.Point(12, 393);
            this.btnAddManualOp.Name = "btnAddManualOp";
            this.btnAddManualOp.Size = new System.Drawing.Size(274, 22);
            this.btnAddManualOp.TabIndex = 13;
            this.btnAddManualOp.Text = "Add manual operation for selected brick";
            this.btnAddManualOp.UseVisualStyleBackColor = true;
            this.btnAddManualOp.Click += new System.EventHandler(this.btnAddManualOp_Click);
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label4.Location = new System.Drawing.Point(12, 424);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(31, 13);
            this.label4.TabIndex = 15;
            this.label4.Text = "Type";
            // 
            // cmbOpType
            // 
            this.cmbOpType.FormattingEnabled = true;
            this.cmbOpType.Location = new System.Drawing.Point(64, 421);
            this.cmbOpType.Name = "cmbOpType";
            this.cmbOpType.Size = new System.Drawing.Size(159, 21);
            this.cmbOpType.TabIndex = 16;
            // 
            // btnImportAutoOp
            // 
            this.btnImportAutoOp.Location = new System.Drawing.Point(12, 458);
            this.btnImportAutoOp.Name = "btnImportAutoOp";
            this.btnImportAutoOp.Size = new System.Drawing.Size(274, 22);
            this.btnImportAutoOp.TabIndex = 17;
            this.btnImportAutoOp.Text = "Import managed operation(s) for selected brick";
            this.btnImportAutoOp.UseVisualStyleBackColor = true;
            this.btnImportAutoOp.Click += new System.EventHandler(this.btnImportAutoOp_Click);
            // 
            // btnPubBrick
            // 
            this.btnPubBrick.Location = new System.Drawing.Point(163, 361);
            this.btnPubBrick.Name = "btnPubBrick";
            this.btnPubBrick.Size = new System.Drawing.Size(123, 22);
            this.btnPubBrick.TabIndex = 18;
            this.btnPubBrick.Text = "Publish Selected Brick";
            this.btnPubBrick.UseVisualStyleBackColor = true;
            this.btnPubBrick.Click += new System.EventHandler(this.btnPubBrick_Click);
            // 
            // btnUpdateNotes
            // 
            this.btnUpdateNotes.Location = new System.Drawing.Point(753, 231);
            this.btnUpdateNotes.Name = "btnUpdateNotes";
            this.btnUpdateNotes.Size = new System.Drawing.Size(193, 22);
            this.btnUpdateNotes.TabIndex = 19;
            this.btnUpdateNotes.Text = "Update notes for selected operation";
            this.btnUpdateNotes.UseVisualStyleBackColor = true;
            this.btnUpdateNotes.Click += new System.EventHandler(this.btnUpdateNotes_Click);
            // 
            // btnAddLabManagerSupport
            // 
            this.btnAddLabManagerSupport.Location = new System.Drawing.Point(809, 6);
            this.btnAddLabManagerSupport.Name = "btnAddLabManagerSupport";
            this.btnAddLabManagerSupport.Size = new System.Drawing.Size(139, 61);
            this.btnAddLabManagerSupport.TabIndex = 20;
            this.btnAddLabManagerSupport.Text = "Add Lab Manager support to existing DB\r\n(Administrators Only)";
            this.btnAddLabManagerSupport.UseVisualStyleBackColor = true;
            this.btnAddLabManagerSupport.Click += new System.EventHandler(this.btnAddLabManagerSupport_Click);
            // 
            // lvFiles
            // 
            this.lvFiles.LargeImageList = this.imgFiles;
            this.lvFiles.Location = new System.Drawing.Point(15, 563);
            this.lvFiles.MultiSelect = false;
            this.lvFiles.Name = "lvFiles";
            this.lvFiles.Size = new System.Drawing.Size(414, 135);
            this.lvFiles.TabIndex = 21;
            this.lvFiles.UseCompatibleStateImageBehavior = false;
            this.lvFiles.SelectedIndexChanged += new System.EventHandler(this.OnFileSel);
            // 
            // imgFiles
            // 
            this.imgFiles.ImageStream = ((System.Windows.Forms.ImageListStreamer)(resources.GetObject("imgFiles.ImageStream")));
            this.imgFiles.TransparentColor = System.Drawing.Color.Transparent;
            this.imgFiles.Images.SetKeyName(0, "text.png");
            this.imgFiles.Images.SetKeyName(1, "x3l.png");
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label5.Location = new System.Drawing.Point(9, 530);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(136, 13);
            this.label5.TabIndex = 22;
            this.label5.Text = "Files for selected brick";
            // 
            // btnImportFile
            // 
            this.btnImportFile.Location = new System.Drawing.Point(435, 517);
            this.btnImportFile.Name = "btnImportFile";
            this.btnImportFile.Size = new System.Drawing.Size(76, 22);
            this.btnImportFile.TabIndex = 23;
            this.btnImportFile.Text = "Import";
            this.btnImportFile.UseVisualStyleBackColor = true;
            this.btnImportFile.Click += new System.EventHandler(this.btnImportFile_Click);
            // 
            // btnSaveFile
            // 
            this.btnSaveFile.Location = new System.Drawing.Point(435, 545);
            this.btnSaveFile.Name = "btnSaveFile";
            this.btnSaveFile.Size = new System.Drawing.Size(76, 22);
            this.btnSaveFile.TabIndex = 24;
            this.btnSaveFile.Text = "Save";
            this.btnSaveFile.UseVisualStyleBackColor = true;
            this.btnSaveFile.Click += new System.EventHandler(this.btnSaveFile_Click);
            // 
            // btnRemoveFile
            // 
            this.btnRemoveFile.Location = new System.Drawing.Point(435, 676);
            this.btnRemoveFile.Name = "btnRemoveFile";
            this.btnRemoveFile.Size = new System.Drawing.Size(76, 22);
            this.btnRemoveFile.TabIndex = 25;
            this.btnRemoveFile.Text = "Remove";
            this.btnRemoveFile.UseVisualStyleBackColor = true;
            this.btnRemoveFile.Click += new System.EventHandler(this.btnRemoveFile_Click);
            // 
            // ofnSelectImportFile
            // 
            this.ofnSelectImportFile.Filter = "Text files (*.txt)|*.txt|Image files|*.gif;*.png;*.jpeg;*.jpg;*.emf|3D Scene file" +
                "s|*.x3l";
            this.ofnSelectImportFile.ShowReadOnly = true;
            this.ofnSelectImportFile.Title = "Select file to import";
            // 
            // sfSelectSaveFile
            // 
            this.sfSelectSaveFile.Title = "Select path to save file";
            // 
            // pcBox
            // 
            this.pcBox.Location = new System.Drawing.Point(557, 517);
            this.pcBox.Name = "pcBox";
            this.pcBox.Size = new System.Drawing.Size(388, 181);
            this.pcBox.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.pcBox.TabIndex = 26;
            this.pcBox.TabStop = false;
            this.pcBox.Visible = false;
            // 
            // rtbText
            // 
            this.rtbText.Location = new System.Drawing.Point(557, 517);
            this.rtbText.Name = "rtbText";
            this.rtbText.ReadOnly = true;
            this.rtbText.Size = new System.Drawing.Size(388, 181);
            this.rtbText.TabIndex = 27;
            this.rtbText.Text = "";
            this.rtbText.Visible = false;
            // 
            // gdiDisp
            // 
            this.gdiDisp.Alpha = 1;
            this.gdiDisp.AutoRender = true;
            this.gdiDisp.BackColor = System.Drawing.Color.Black;
            this.gdiDisp.BorderWidth = 1;
            this.gdiDisp.ClickSelect = null;
            this.gdiDisp.Distance = 10;
            this.gdiDisp.DoubleClickSelect = null;
            this.gdiDisp.Infinity = false;
            this.gdiDisp.LabelFontName = "Arial";
            this.gdiDisp.LabelFontSize = 12;
            this.gdiDisp.LineWidth = 2;
            this.gdiDisp.Location = new System.Drawing.Point(557, 517);
            this.gdiDisp.MouseMode = GDI3D.Control.MouseMotion.Rotate;
            this.gdiDisp.MouseMultiplier = 0.01;
            this.gdiDisp.Name = "gdiDisp";
            this.gdiDisp.NextClickSetsCenter = false;
            this.gdiDisp.PointSize = 5;
            this.gdiDisp.Size = new System.Drawing.Size(390, 181);
            this.gdiDisp.TabIndex = 28;
            this.gdiDisp.Visible = false;
            this.gdiDisp.Zoom = 100;
            // 
            // btnZoomIn
            // 
            this.btnZoomIn.Location = new System.Drawing.Point(521, 517);
            this.btnZoomIn.Name = "btnZoomIn";
            this.btnZoomIn.Size = new System.Drawing.Size(30, 22);
            this.btnZoomIn.TabIndex = 29;
            this.btnZoomIn.Text = "+";
            this.btnZoomIn.UseVisualStyleBackColor = true;
            this.btnZoomIn.Visible = false;
            this.btnZoomIn.Click += new System.EventHandler(this.btnZoomIn_Click);
            // 
            // btnZoomOut
            // 
            this.btnZoomOut.Location = new System.Drawing.Point(521, 545);
            this.btnZoomOut.Name = "btnZoomOut";
            this.btnZoomOut.Size = new System.Drawing.Size(30, 22);
            this.btnZoomOut.TabIndex = 30;
            this.btnZoomOut.Text = "-";
            this.btnZoomOut.UseVisualStyleBackColor = true;
            this.btnZoomOut.Visible = false;
            this.btnZoomOut.Click += new System.EventHandler(this.btnZoomOut_Click);
            // 
            // btnInfinity
            // 
            this.btnInfinity.Font = new System.Drawing.Font("Symbol", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(2)));
            this.btnInfinity.Location = new System.Drawing.Point(521, 573);
            this.btnInfinity.Name = "btnInfinity";
            this.btnInfinity.Size = new System.Drawing.Size(30, 22);
            this.btnInfinity.TabIndex = 31;
            this.btnInfinity.Text = "¥";
            this.btnInfinity.UseVisualStyleBackColor = true;
            this.btnInfinity.Visible = false;
            this.btnInfinity.Click += new System.EventHandler(this.btnInfinity_Click);
            // 
            // btnDeleteOp
            // 
            this.btnDeleteOp.Location = new System.Drawing.Point(685, 8);
            this.btnDeleteOp.Name = "btnDeleteOp";
            this.btnDeleteOp.Size = new System.Drawing.Size(109, 59);
            this.btnDeleteOp.TabIndex = 32;
            this.btnDeleteOp.Text = "Delete operation record";
            this.btnDeleteOp.UseVisualStyleBackColor = true;
            this.btnDeleteOp.Click += new System.EventHandler(this.btnDeleteOp_Click);
            // 
            // txtFilter
            // 
            this.txtFilter.Location = new System.Drawing.Point(129, 26);
            this.txtFilter.Name = "txtFilter";
            this.txtFilter.Size = new System.Drawing.Size(88, 20);
            this.txtFilter.TabIndex = 33;
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label6.Location = new System.Drawing.Point(12, 29);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(117, 13);
            this.label6.TabIndex = 34;
            this.label6.Text = "Filter ID_EVENTBRICK";
            // 
            // lvSecondBrick
            // 
            this.lvSecondBrick.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.NumberOfBrick,
            this.columnHeader6,
            this.columnHeader7});
            this.lvSecondBrick.GridLines = true;
            this.lvSecondBrick.Location = new System.Drawing.Point(685, 123);
            this.lvSecondBrick.Name = "lvSecondBrick";
            this.lvSecondBrick.Size = new System.Drawing.Size(261, 102);
            this.lvSecondBrick.TabIndex = 35;
            this.lvSecondBrick.UseCompatibleStateImageBehavior = false;
            this.lvSecondBrick.View = System.Windows.Forms.View.Details;
            // 
            // NumberOfBrick
            // 
            this.NumberOfBrick.Text = "EVENT";
            this.NumberOfBrick.Width = 100;
            // 
            // columnHeader6
            // 
            this.columnHeader6.Text = "BRICK";
            // 
            // columnHeader7
            // 
            this.columnHeader7.Text = "STATUS";
            this.columnHeader7.Width = 100;
            // 
            // txtFilterEvent
            // 
            this.txtFilterEvent.Location = new System.Drawing.Point(131, 70);
            this.txtFilterEvent.Name = "txtFilterEvent";
            this.txtFilterEvent.Size = new System.Drawing.Size(86, 20);
            this.txtFilterEvent.TabIndex = 37;
            // 
            // btnSearchEvent
            // 
            this.btnSearchEvent.Location = new System.Drawing.Point(2, 68);
            this.btnSearchEvent.Name = "btnSearchEvent";
            this.btnSearchEvent.Size = new System.Drawing.Size(123, 23);
            this.btnSearchEvent.TabIndex = 38;
            this.btnSearchEvent.Text = "Search by Event";
            this.btnSearchEvent.UseVisualStyleBackColor = true;
            this.btnSearchEvent.Click += new System.EventHandler(this.btnSearchEvent_Click);
            // 
            // btnSearchStatus
            // 
            this.btnSearchStatus.Location = new System.Drawing.Point(2, 107);
            this.btnSearchStatus.Name = "btnSearchStatus";
            this.btnSearchStatus.Size = new System.Drawing.Size(123, 23);
            this.btnSearchStatus.TabIndex = 39;
            this.btnSearchStatus.Text = "Search by Status";
            this.btnSearchStatus.UseVisualStyleBackColor = true;
            this.btnSearchStatus.Click += new System.EventHandler(this.btnSearchStatus_Click);
            // 
            // txtStatus
            // 
            this.txtStatus.Location = new System.Drawing.Point(131, 109);
            this.txtStatus.Name = "txtStatus";
            this.txtStatus.Size = new System.Drawing.Size(86, 20);
            this.txtStatus.TabIndex = 40;
            // 
            // box2008
            // 
            this.box2008.AutoSize = true;
            this.box2008.Location = new System.Drawing.Point(227, 55);
            this.box2008.Name = "box2008";
            this.box2008.Size = new System.Drawing.Size(50, 17);
            this.box2008.TabIndex = 41;
            this.box2008.Text = "2008";
            this.box2008.UseVisualStyleBackColor = true;
            this.box2008.CheckedChanged += new System.EventHandler(this.box2008_CheckedChanged);
            // 
            // box2009
            // 
            this.box2009.AutoSize = true;
            this.box2009.Location = new System.Drawing.Point(227, 73);
            this.box2009.Name = "box2009";
            this.box2009.Size = new System.Drawing.Size(50, 17);
            this.box2009.TabIndex = 42;
            this.box2009.Text = "2009";
            this.box2009.UseVisualStyleBackColor = true;
            this.box2009.CheckedChanged += new System.EventHandler(this.box2009_CheckedChanged);
            // 
            // box2010
            // 
            this.box2010.AutoSize = true;
            this.box2010.Location = new System.Drawing.Point(227, 93);
            this.box2010.Name = "box2010";
            this.box2010.Size = new System.Drawing.Size(50, 17);
            this.box2010.TabIndex = 43;
            this.box2010.Text = "2010";
            this.box2010.UseVisualStyleBackColor = true;
            this.box2010.CheckedChanged += new System.EventHandler(this.box2010_CheckedChanged);
            // 
            // box2011
            // 
            this.box2011.AutoSize = true;
            this.box2011.Location = new System.Drawing.Point(226, 112);
            this.box2011.Name = "box2011";
            this.box2011.Size = new System.Drawing.Size(50, 17);
            this.box2011.TabIndex = 44;
            this.box2011.Text = "2011";
            this.box2011.UseVisualStyleBackColor = true;
            this.box2011.CheckedChanged += new System.EventHandler(this.box2011_CheckedChanged);
            // 
            // boxAll
            // 
            this.boxAll.AutoSize = true;
            this.boxAll.Checked = true;
            this.boxAll.CheckState = System.Windows.Forms.CheckState.Checked;
            this.boxAll.Location = new System.Drawing.Point(226, 133);
            this.boxAll.Name = "boxAll";
            this.boxAll.Size = new System.Drawing.Size(37, 17);
            this.boxAll.TabIndex = 45;
            this.boxAll.Text = "All";
            this.boxAll.UseVisualStyleBackColor = true;
            // 
            // txtResultStatus
            // 
            this.txtResultStatus.Location = new System.Drawing.Point(774, 91);
            this.txtResultStatus.Name = "txtResultStatus";
            this.txtResultStatus.Size = new System.Drawing.Size(170, 20);
            this.txtResultStatus.TabIndex = 46;
            // 
            // txtType
            // 
            this.txtType.Location = new System.Drawing.Point(687, 90);
            this.txtType.Name = "txtType";
            this.txtType.Size = new System.Drawing.Size(66, 20);
            this.txtType.TabIndex = 47;
            // 
            // MainForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(959, 710);
            this.Controls.Add(this.txtType);
            this.Controls.Add(this.txtResultStatus);
            this.Controls.Add(this.boxAll);
            this.Controls.Add(this.box2011);
            this.Controls.Add(this.box2010);
            this.Controls.Add(this.box2009);
            this.Controls.Add(this.box2008);
            this.Controls.Add(this.txtStatus);
            this.Controls.Add(this.btnSearchStatus);
            this.Controls.Add(this.btnSearchEvent);
            this.Controls.Add(this.txtFilterEvent);
            this.Controls.Add(this.lvSecondBrick);
            this.Controls.Add(this.label6);
            this.Controls.Add(this.txtFilter);
            this.Controls.Add(this.btnDeleteOp);
            this.Controls.Add(this.btnInfinity);
            this.Controls.Add(this.btnZoomOut);
            this.Controls.Add(this.btnZoomIn);
            this.Controls.Add(this.gdiDisp);
            this.Controls.Add(this.pcBox);
            this.Controls.Add(this.btnRemoveFile);
            this.Controls.Add(this.btnSaveFile);
            this.Controls.Add(this.btnImportFile);
            this.Controls.Add(this.label5);
            this.Controls.Add(this.lvFiles);
            this.Controls.Add(this.btnAddLabManagerSupport);
            this.Controls.Add(this.btnUpdateNotes);
            this.Controls.Add(this.btnPubBrick);
            this.Controls.Add(this.btnImportAutoOp);
            this.Controls.Add(this.cmbOpType);
            this.Controls.Add(this.label4);
            this.Controls.Add(this.btnAddManualOp);
            this.Controls.Add(this.btnCloseBrick);
            this.Controls.Add(this.txtRecvBrickId);
            this.Controls.Add(this.btnRefresh);
            this.Controls.Add(this.btnReceiveBrick);
            this.Controls.Add(this.chkToDo);
            this.Controls.Add(this.label3);
            this.Controls.Add(this.txtNotes);
            this.Controls.Add(this.lvOps);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.lvBricks);
            this.Controls.Add(this.rtbText);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedSingle;
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.Name = "MainForm";
            this.Text = "Opera Lab Manager";
            this.Load += new System.EventHandler(this.MainForm_Load);
            ((System.ComponentModel.ISupportInitialize)(this.pcBox)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.ListView lvBricks;
        private System.Windows.Forms.ColumnHeader columnHeader1;
        private System.Windows.Forms.ColumnHeader columnHeader2;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.ListView lvOps;
        private System.Windows.Forms.ColumnHeader columnHeader3;
        private System.Windows.Forms.ColumnHeader columnHeader4;
        private System.Windows.Forms.ColumnHeader columnHeader5;
        private System.Windows.Forms.TextBox txtNotes;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.CheckBox chkToDo;
        private System.Windows.Forms.Button btnReceiveBrick;
        private System.Windows.Forms.Button btnRefresh;
        private System.Windows.Forms.TextBox txtRecvBrickId;
        private System.Windows.Forms.Button btnCloseBrick;
        private System.Windows.Forms.Button btnAddManualOp;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.ComboBox cmbOpType;
        private System.Windows.Forms.Button btnImportAutoOp;
        private System.Windows.Forms.Button btnPubBrick;
        private System.Windows.Forms.Button btnUpdateNotes;
        private System.Windows.Forms.Button btnAddLabManagerSupport;
        private System.Windows.Forms.ListView lvFiles;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.Button btnImportFile;
        private System.Windows.Forms.Button btnSaveFile;
        private System.Windows.Forms.Button btnRemoveFile;
        private System.Windows.Forms.OpenFileDialog ofnSelectImportFile;
        private System.Windows.Forms.ImageList imgFiles;
        private System.Windows.Forms.SaveFileDialog sfSelectSaveFile;
        private System.Windows.Forms.PictureBox pcBox;
        private System.Windows.Forms.RichTextBox rtbText;
        private GDI3D.Control.GDIDisplay gdiDisp;
        private System.Windows.Forms.Button btnZoomIn;
        private System.Windows.Forms.Button btnZoomOut;
        private System.Windows.Forms.Button btnInfinity;
        private System.Windows.Forms.Button btnDeleteOp;
        private System.Windows.Forms.TextBox txtFilter;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.ListView lvSecondBrick;
        private System.Windows.Forms.ColumnHeader NumberOfBrick;
        private System.Windows.Forms.ColumnHeader columnHeader6;
        private System.Windows.Forms.ColumnHeader columnHeader7;
        private System.Windows.Forms.TextBox txtFilterEvent;
        private System.Windows.Forms.Button btnSearchEvent;
        private System.Windows.Forms.Button btnSearchStatus;
        private System.Windows.Forms.TextBox txtStatus;
        private System.Windows.Forms.CheckBox box2008;
        private System.Windows.Forms.CheckBox box2009;
        private System.Windows.Forms.CheckBox box2010;
        private System.Windows.Forms.CheckBox box2011;
        private System.Windows.Forms.CheckBox boxAll;
        private System.Windows.Forms.TextBox txtResultStatus;
        private System.Windows.Forms.TextBox txtType;
    }
}

