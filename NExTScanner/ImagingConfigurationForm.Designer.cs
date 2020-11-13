namespace SySal.Executables.NExTScanner
{
    partial class ImagingConfigurationForm
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
            this.ImagingConfigToolTip = new System.Windows.Forms.ToolTip(this.components);
            this.lbConfigurations = new System.Windows.Forms.ListBox();
            this.btnNew = new SySal.SySalNExTControls.SySalButton();
            this.btnDuplicate = new SySal.SySalNExTControls.SySalButton();
            this.btnLoad = new SySal.SySalNExTControls.SySalButton();
            this.btnDel = new SySal.SySalNExTControls.SySalButton();
            this.btnMakeCurrent = new SySal.SySalNExTControls.SySalButton();
            this.txtCurrentConfiguration = new System.Windows.Forms.TextBox();
            this.txtWidth = new System.Windows.Forms.TextBox();
            this.txtHeight = new System.Windows.Forms.TextBox();
            this.btnEmptyImage = new SySal.SySalNExTControls.SySalButton();
            this.txtEmptyImage = new System.Windows.Forms.TextBox();
            this.btnViewEmptyImage = new SySal.SySalNExTControls.SySalButton();
            this.btnViewThresholdImage = new SySal.SySalNExTControls.SySalButton();
            this.txtThresholdImage = new System.Windows.Forms.TextBox();
            this.btnThresholdImage = new SySal.SySalNExTControls.SySalButton();
            this.txtGreyLevelTargetMedian = new System.Windows.Forms.TextBox();
            this.txtMaxClusters = new System.Windows.Forms.TextBox();
            this.txtMaxSegsLine = new System.Windows.Forms.TextBox();
            this.txtPixMicronY = new System.Windows.Forms.TextBox();
            this.txtPixMicronX = new System.Windows.Forms.TextBox();
            this.txtDmagDZ = new System.Windows.Forms.TextBox();
            this.txtZCurvature = new System.Windows.Forms.TextBox();
            this.txtXYCurvature = new System.Windows.Forms.TextBox();
            this.btnExit = new SySal.SySalNExTControls.SySalButton();
            this.btnPixelToMicron = new SySal.SySalNExTControls.SySalButton();
            this.btnComputeClusterMaps = new SySal.SySalNExTControls.SySalButton();
            this.txtSummaryFile = new System.Windows.Forms.TextBox();
            this.btnMakeChains = new SySal.SySalNExTControls.SySalButton();
            this.txtCMPosTol = new System.Windows.Forms.TextBox();
            this.txtCMMaxOffset = new System.Windows.Forms.TextBox();
            this.txtMinClusterArea = new System.Windows.Forms.TextBox();
            this.txtCMMinMatches = new System.Windows.Forms.TextBox();
            this.txtMinGrainVolume = new System.Windows.Forms.TextBox();
            this.btnViewAcquisition = new SySal.SySalNExTControls.SySalButton();
            this.txtNewName = new System.Windows.Forms.TextBox();
            this.label1 = new System.Windows.Forms.Label();
            this.label4 = new System.Windows.Forms.Label();
            this.label5 = new System.Windows.Forms.Label();
            this.panel1 = new System.Windows.Forms.Panel();
            this.label3 = new System.Windows.Forms.Label();
            this.OpenEmptyImagesDlg = new System.Windows.Forms.OpenFileDialog();
            this.label6 = new System.Windows.Forms.Label();
            this.label7 = new System.Windows.Forms.Label();
            this.label8 = new System.Windows.Forms.Label();
            this.label9 = new System.Windows.Forms.Label();
            this.label10 = new System.Windows.Forms.Label();
            this.chkOptimizeDMagDZ = new System.Windows.Forms.CheckBox();
            this.panel2 = new System.Windows.Forms.Panel();
            this.label2 = new System.Windows.Forms.Label();
            this.label11 = new System.Windows.Forms.Label();
            this.label12 = new System.Windows.Forms.Label();
            this.label13 = new System.Windows.Forms.Label();
            this.txtDmagDY = new System.Windows.Forms.TextBox();
            this.label14 = new System.Windows.Forms.Label();
            this.txtDmagDX = new System.Windows.Forms.TextBox();
            this.label15 = new System.Windows.Forms.Label();
            this.txtCameraRotation = new System.Windows.Forms.TextBox();
            this.label16 = new System.Windows.Forms.Label();
            this.txtXSlant = new System.Windows.Forms.TextBox();
            this.label17 = new System.Windows.Forms.Label();
            this.txtYSlant = new System.Windows.Forms.TextBox();
            this.SuspendLayout();
            // 
            // lbConfigurations
            // 
            this.lbConfigurations.BackColor = System.Drawing.Color.WhiteSmoke;
            this.lbConfigurations.ForeColor = System.Drawing.Color.DodgerBlue;
            this.lbConfigurations.FormattingEnabled = true;
            this.lbConfigurations.ItemHeight = 21;
            this.lbConfigurations.Location = new System.Drawing.Point(21, 35);
            this.lbConfigurations.Name = "lbConfigurations";
            this.lbConfigurations.Size = new System.Drawing.Size(310, 256);
            this.lbConfigurations.Sorted = true;
            this.lbConfigurations.TabIndex = 1;
            this.ImagingConfigToolTip.SetToolTip(this.lbConfigurations, "Select the imaging configuration to work with.");
            // 
            // btnNew
            // 
            this.btnNew.AutoSize = true;
            this.btnNew.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnNew.BackColor = System.Drawing.Color.Transparent;
            this.btnNew.FocusedColor = System.Drawing.Color.Navy;
            this.btnNew.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.btnNew.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnNew.Location = new System.Drawing.Point(340, 34);
            this.btnNew.Margin = new System.Windows.Forms.Padding(6);
            this.btnNew.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnNew.Name = "btnNew";
            this.btnNew.Size = new System.Drawing.Size(40, 25);
            this.btnNew.TabIndex = 2;
            this.btnNew.Text = "New";
            this.ImagingConfigToolTip.SetToolTip(this.btnNew, "Create a new empty configuration.\r\nThe name of the new configuration must be spec" +
        "ified in the textbox below.");
            this.btnNew.Click += new System.EventHandler(this.btnNew_Click);
            // 
            // btnDuplicate
            // 
            this.btnDuplicate.AutoSize = true;
            this.btnDuplicate.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnDuplicate.BackColor = System.Drawing.Color.Transparent;
            this.btnDuplicate.FocusedColor = System.Drawing.Color.Navy;
            this.btnDuplicate.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.btnDuplicate.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnDuplicate.Location = new System.Drawing.Point(392, 34);
            this.btnDuplicate.Margin = new System.Windows.Forms.Padding(6);
            this.btnDuplicate.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnDuplicate.Name = "btnDuplicate";
            this.btnDuplicate.Size = new System.Drawing.Size(76, 25);
            this.btnDuplicate.TabIndex = 3;
            this.btnDuplicate.Text = "Duplicate";
            this.ImagingConfigToolTip.SetToolTip(this.btnDuplicate, "Create a new configuration by duplicating an existing one.\r\nThe name of the new c" +
        "onfiguration must be specified in the textbox below.");
            this.btnDuplicate.Click += new System.EventHandler(this.btnDuplicate_Click);
            // 
            // btnLoad
            // 
            this.btnLoad.AutoSize = true;
            this.btnLoad.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnLoad.BackColor = System.Drawing.Color.Transparent;
            this.btnLoad.FocusedColor = System.Drawing.Color.Navy;
            this.btnLoad.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.btnLoad.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnLoad.Location = new System.Drawing.Point(340, 102);
            this.btnLoad.Margin = new System.Windows.Forms.Padding(6);
            this.btnLoad.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnLoad.Name = "btnLoad";
            this.btnLoad.Size = new System.Drawing.Size(42, 25);
            this.btnLoad.TabIndex = 5;
            this.btnLoad.Text = "Load";
            this.ImagingConfigToolTip.SetToolTip(this.btnLoad, "Load the currently selected configuration. This can become the currently used one" +
        " by clicking on \"Make Current\".");
            this.btnLoad.Click += new System.EventHandler(this.btnLoad_Click);
            // 
            // btnDel
            // 
            this.btnDel.AutoSize = true;
            this.btnDel.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnDel.BackColor = System.Drawing.Color.Transparent;
            this.btnDel.FocusedColor = System.Drawing.Color.Navy;
            this.btnDel.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.btnDel.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnDel.Location = new System.Drawing.Point(340, 198);
            this.btnDel.Margin = new System.Windows.Forms.Padding(6);
            this.btnDel.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnDel.Name = "btnDel";
            this.btnDel.Size = new System.Drawing.Size(54, 25);
            this.btnDel.TabIndex = 8;
            this.btnDel.Text = "Delete";
            this.ImagingConfigToolTip.SetToolTip(this.btnDel, "Delete the selected configuration.");
            this.btnDel.Click += new System.EventHandler(this.btnDel_Click);
            // 
            // btnMakeCurrent
            // 
            this.btnMakeCurrent.AutoSize = true;
            this.btnMakeCurrent.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnMakeCurrent.BackColor = System.Drawing.Color.Transparent;
            this.btnMakeCurrent.FocusedColor = System.Drawing.Color.Navy;
            this.btnMakeCurrent.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.btnMakeCurrent.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnMakeCurrent.Location = new System.Drawing.Point(340, 130);
            this.btnMakeCurrent.Margin = new System.Windows.Forms.Padding(6);
            this.btnMakeCurrent.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnMakeCurrent.Name = "btnMakeCurrent";
            this.btnMakeCurrent.Size = new System.Drawing.Size(103, 25);
            this.btnMakeCurrent.TabIndex = 6;
            this.btnMakeCurrent.Text = "Make current";
            this.ImagingConfigToolTip.SetToolTip(this.btnMakeCurrent, "Make the selected configuration the current one.");
            this.btnMakeCurrent.Click += new System.EventHandler(this.btnMakeCurrent_Click);
            // 
            // txtCurrentConfiguration
            // 
            this.txtCurrentConfiguration.BackColor = System.Drawing.Color.LightGray;
            this.txtCurrentConfiguration.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtCurrentConfiguration.ForeColor = System.Drawing.Color.Navy;
            this.txtCurrentConfiguration.Location = new System.Drawing.Point(340, 164);
            this.txtCurrentConfiguration.Name = "txtCurrentConfiguration";
            this.txtCurrentConfiguration.ReadOnly = true;
            this.txtCurrentConfiguration.Size = new System.Drawing.Size(121, 25);
            this.txtCurrentConfiguration.TabIndex = 7;
            this.ImagingConfigToolTip.SetToolTip(this.txtCurrentConfiguration, "The name of the configuration currently being used.");
            // 
            // txtWidth
            // 
            this.txtWidth.BackColor = System.Drawing.Color.GhostWhite;
            this.txtWidth.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtWidth.ForeColor = System.Drawing.Color.DodgerBlue;
            this.txtWidth.Location = new System.Drawing.Point(700, 34);
            this.txtWidth.Name = "txtWidth";
            this.txtWidth.Size = new System.Drawing.Size(53, 25);
            this.txtWidth.TabIndex = 11;
            this.txtWidth.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.ImagingConfigToolTip.SetToolTip(this.txtWidth, "Width of the image in pixels.");
            this.txtWidth.Leave += new System.EventHandler(this.OnWidthLeave);
            // 
            // txtHeight
            // 
            this.txtHeight.BackColor = System.Drawing.Color.GhostWhite;
            this.txtHeight.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtHeight.ForeColor = System.Drawing.Color.DodgerBlue;
            this.txtHeight.Location = new System.Drawing.Point(759, 34);
            this.txtHeight.Name = "txtHeight";
            this.txtHeight.Size = new System.Drawing.Size(53, 25);
            this.txtHeight.TabIndex = 12;
            this.txtHeight.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.ImagingConfigToolTip.SetToolTip(this.txtHeight, "Height of the image in pixels.");
            this.txtHeight.Leave += new System.EventHandler(this.OnHeightLeave);
            // 
            // btnEmptyImage
            // 
            this.btnEmptyImage.AutoSize = true;
            this.btnEmptyImage.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnEmptyImage.BackColor = System.Drawing.Color.Transparent;
            this.btnEmptyImage.FocusedColor = System.Drawing.Color.Navy;
            this.btnEmptyImage.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.btnEmptyImage.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnEmptyImage.Location = new System.Drawing.Point(490, 68);
            this.btnEmptyImage.Margin = new System.Windows.Forms.Padding(6);
            this.btnEmptyImage.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnEmptyImage.Name = "btnEmptyImage";
            this.btnEmptyImage.Size = new System.Drawing.Size(102, 25);
            this.btnEmptyImage.TabIndex = 13;
            this.btnEmptyImage.Text = "Empty image";
            this.ImagingConfigToolTip.SetToolTip(this.btnEmptyImage, "Work the empty image from images taken out of the emulsion.");
            this.btnEmptyImage.Click += new System.EventHandler(this.btnEmptyImage_Click);
            // 
            // txtEmptyImage
            // 
            this.txtEmptyImage.BackColor = System.Drawing.Color.GhostWhite;
            this.txtEmptyImage.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtEmptyImage.ForeColor = System.Drawing.Color.Navy;
            this.txtEmptyImage.Location = new System.Drawing.Point(627, 68);
            this.txtEmptyImage.Name = "txtEmptyImage";
            this.txtEmptyImage.Size = new System.Drawing.Size(126, 25);
            this.txtEmptyImage.TabIndex = 14;
            this.txtEmptyImage.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.ImagingConfigToolTip.SetToolTip(this.txtEmptyImage, "Text-encoding of the image that is used to compensate ADC pedestals and camera sp" +
        "ots.");
            this.txtEmptyImage.Leave += new System.EventHandler(this.OnEmptyImageLeave);
            // 
            // btnViewEmptyImage
            // 
            this.btnViewEmptyImage.AutoSize = true;
            this.btnViewEmptyImage.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnViewEmptyImage.FocusedColor = System.Drawing.Color.Navy;
            this.btnViewEmptyImage.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.btnViewEmptyImage.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnViewEmptyImage.Location = new System.Drawing.Point(770, 68);
            this.btnViewEmptyImage.Margin = new System.Windows.Forms.Padding(6);
            this.btnViewEmptyImage.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnViewEmptyImage.Name = "btnViewEmptyImage";
            this.btnViewEmptyImage.Size = new System.Drawing.Size(42, 25);
            this.btnViewEmptyImage.TabIndex = 15;
            this.btnViewEmptyImage.Text = "View";
            this.ImagingConfigToolTip.SetToolTip(this.btnViewEmptyImage, "Shows the empty image in a picture.");
            this.btnViewEmptyImage.Click += new System.EventHandler(this.btnViewEmptyImage_Click);
            // 
            // btnViewThresholdImage
            // 
            this.btnViewThresholdImage.AutoSize = true;
            this.btnViewThresholdImage.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnViewThresholdImage.FocusedColor = System.Drawing.Color.Navy;
            this.btnViewThresholdImage.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.btnViewThresholdImage.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnViewThresholdImage.Location = new System.Drawing.Point(770, 99);
            this.btnViewThresholdImage.Margin = new System.Windows.Forms.Padding(6);
            this.btnViewThresholdImage.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnViewThresholdImage.Name = "btnViewThresholdImage";
            this.btnViewThresholdImage.Size = new System.Drawing.Size(42, 25);
            this.btnViewThresholdImage.TabIndex = 18;
            this.btnViewThresholdImage.Text = "View";
            this.ImagingConfigToolTip.SetToolTip(this.btnViewThresholdImage, "Shows the threshold image in a picture.");
            this.btnViewThresholdImage.Click += new System.EventHandler(this.btnViewThresholdImage_Click);
            // 
            // txtThresholdImage
            // 
            this.txtThresholdImage.BackColor = System.Drawing.Color.GhostWhite;
            this.txtThresholdImage.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtThresholdImage.ForeColor = System.Drawing.Color.Navy;
            this.txtThresholdImage.Location = new System.Drawing.Point(627, 99);
            this.txtThresholdImage.Name = "txtThresholdImage";
            this.txtThresholdImage.Size = new System.Drawing.Size(126, 25);
            this.txtThresholdImage.TabIndex = 17;
            this.txtThresholdImage.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.ImagingConfigToolTip.SetToolTip(this.txtThresholdImage, "Text-encoding of the threshold image.");
            this.txtThresholdImage.Leave += new System.EventHandler(this.OnThresholdImageLeave);
            // 
            // btnThresholdImage
            // 
            this.btnThresholdImage.AutoSize = true;
            this.btnThresholdImage.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnThresholdImage.BackColor = System.Drawing.Color.Transparent;
            this.btnThresholdImage.FocusedColor = System.Drawing.Color.Navy;
            this.btnThresholdImage.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.btnThresholdImage.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnThresholdImage.Location = new System.Drawing.Point(490, 99);
            this.btnThresholdImage.Margin = new System.Windows.Forms.Padding(6);
            this.btnThresholdImage.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnThresholdImage.Name = "btnThresholdImage";
            this.btnThresholdImage.Size = new System.Drawing.Size(128, 25);
            this.btnThresholdImage.TabIndex = 16;
            this.btnThresholdImage.Text = "Threshold image";
            this.ImagingConfigToolTip.SetToolTip(this.btnThresholdImage, "Work out the threshold image from samples of images taken in emulsion.");
            this.btnThresholdImage.Click += new System.EventHandler(this.btnThresholdImage_Click);
            // 
            // txtGreyLevelTargetMedian
            // 
            this.txtGreyLevelTargetMedian.BackColor = System.Drawing.Color.GhostWhite;
            this.txtGreyLevelTargetMedian.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtGreyLevelTargetMedian.ForeColor = System.Drawing.Color.DodgerBlue;
            this.txtGreyLevelTargetMedian.Location = new System.Drawing.Point(759, 130);
            this.txtGreyLevelTargetMedian.Name = "txtGreyLevelTargetMedian";
            this.txtGreyLevelTargetMedian.Size = new System.Drawing.Size(53, 25);
            this.txtGreyLevelTargetMedian.TabIndex = 20;
            this.txtGreyLevelTargetMedian.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.ImagingConfigToolTip.SetToolTip(this.txtGreyLevelTargetMedian, "Sets the expected median of the grey level.\r\nThe real histogram is distorted to m" +
        "atch this expectation, thus compensating the illumination.");
            this.txtGreyLevelTargetMedian.Leave += new System.EventHandler(this.OnGreyLevelTargetMedianLeave);
            // 
            // txtMaxClusters
            // 
            this.txtMaxClusters.BackColor = System.Drawing.Color.GhostWhite;
            this.txtMaxClusters.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtMaxClusters.ForeColor = System.Drawing.Color.DodgerBlue;
            this.txtMaxClusters.Location = new System.Drawing.Point(759, 200);
            this.txtMaxClusters.Name = "txtMaxClusters";
            this.txtMaxClusters.Size = new System.Drawing.Size(53, 25);
            this.txtMaxClusters.TabIndex = 24;
            this.txtMaxClusters.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.ImagingConfigToolTip.SetToolTip(this.txtMaxClusters, "If the clustering algorithm needs a limit on the number of clusters in an image, " +
        "this setting is applied.\r\nSpecific algorithms may ignore this setting.");
            this.txtMaxClusters.Leave += new System.EventHandler(this.OnMaxClustersLeave);
            // 
            // txtMaxSegsLine
            // 
            this.txtMaxSegsLine.BackColor = System.Drawing.Color.GhostWhite;
            this.txtMaxSegsLine.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtMaxSegsLine.ForeColor = System.Drawing.Color.DodgerBlue;
            this.txtMaxSegsLine.Location = new System.Drawing.Point(759, 165);
            this.txtMaxSegsLine.Name = "txtMaxSegsLine";
            this.txtMaxSegsLine.Size = new System.Drawing.Size(53, 25);
            this.txtMaxSegsLine.TabIndex = 22;
            this.txtMaxSegsLine.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.ImagingConfigToolTip.SetToolTip(this.txtMaxSegsLine, "If the clustering algorithm needs a limit on the number of segments in a single l" +
        "ine, this setting is applied.\r\nSpecific algorithms may ignore this setting.");
            this.txtMaxSegsLine.Leave += new System.EventHandler(this.OnMaxSegsLeave);
            // 
            // txtPixMicronY
            // 
            this.txtPixMicronY.BackColor = System.Drawing.Color.GhostWhite;
            this.txtPixMicronY.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtPixMicronY.ForeColor = System.Drawing.Color.DodgerBlue;
            this.txtPixMicronY.Location = new System.Drawing.Point(287, 352);
            this.txtPixMicronY.Name = "txtPixMicronY";
            this.txtPixMicronY.Size = new System.Drawing.Size(53, 25);
            this.txtPixMicronY.TabIndex = 29;
            this.txtPixMicronY.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.ImagingConfigToolTip.SetToolTip(this.txtPixMicronY, "Pixel/micron conversion factor (Y axis).");
            this.txtPixMicronY.Leave += new System.EventHandler(this.OnPixMicronYLeave);
            // 
            // txtPixMicronX
            // 
            this.txtPixMicronX.BackColor = System.Drawing.Color.GhostWhite;
            this.txtPixMicronX.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtPixMicronX.ForeColor = System.Drawing.Color.DodgerBlue;
            this.txtPixMicronX.Location = new System.Drawing.Point(228, 352);
            this.txtPixMicronX.Name = "txtPixMicronX";
            this.txtPixMicronX.Size = new System.Drawing.Size(53, 25);
            this.txtPixMicronX.TabIndex = 28;
            this.txtPixMicronX.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.ImagingConfigToolTip.SetToolTip(this.txtPixMicronX, "Pixel/micron conversion factor (X axis).");
            this.txtPixMicronX.Leave += new System.EventHandler(this.OnPixMicronXLeave);
            // 
            // txtDmagDZ
            // 
            this.txtDmagDZ.BackColor = System.Drawing.Color.GhostWhite;
            this.txtDmagDZ.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtDmagDZ.ForeColor = System.Drawing.Color.DodgerBlue;
            this.txtDmagDZ.Location = new System.Drawing.Point(730, 450);
            this.txtDmagDZ.Name = "txtDmagDZ";
            this.txtDmagDZ.Size = new System.Drawing.Size(82, 25);
            this.txtDmagDZ.TabIndex = 31;
            this.txtDmagDZ.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.ImagingConfigToolTip.SetToolTip(this.txtDmagDZ, "Dependency of the magnification factor on Z");
            this.txtDmagDZ.Leave += new System.EventHandler(this.OnDmagDZLeave);
            // 
            // txtZCurvature
            // 
            this.txtZCurvature.BackColor = System.Drawing.Color.GhostWhite;
            this.txtZCurvature.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtZCurvature.ForeColor = System.Drawing.Color.DodgerBlue;
            this.txtZCurvature.Location = new System.Drawing.Point(730, 482);
            this.txtZCurvature.Name = "txtZCurvature";
            this.txtZCurvature.Size = new System.Drawing.Size(82, 25);
            this.txtZCurvature.TabIndex = 33;
            this.txtZCurvature.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.ImagingConfigToolTip.SetToolTip(this.txtZCurvature, "The Z dependency on the radius from the center");
            this.txtZCurvature.Leave += new System.EventHandler(this.OnZCurvLeave);
            // 
            // txtXYCurvature
            // 
            this.txtXYCurvature.BackColor = System.Drawing.Color.GhostWhite;
            this.txtXYCurvature.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtXYCurvature.ForeColor = System.Drawing.Color.DodgerBlue;
            this.txtXYCurvature.Location = new System.Drawing.Point(730, 514);
            this.txtXYCurvature.Name = "txtXYCurvature";
            this.txtXYCurvature.Size = new System.Drawing.Size(82, 25);
            this.txtXYCurvature.TabIndex = 35;
            this.txtXYCurvature.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.ImagingConfigToolTip.SetToolTip(this.txtXYCurvature, "The variation of the metric factor along the radius from the center");
            this.txtXYCurvature.Leave += new System.EventHandler(this.OnXYCurvLeave);
            // 
            // btnExit
            // 
            this.btnExit.AutoSize = true;
            this.btnExit.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnExit.BackColor = System.Drawing.Color.Transparent;
            this.btnExit.FocusedColor = System.Drawing.Color.Navy;
            this.btnExit.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.btnExit.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnExit.Location = new System.Drawing.Point(348, 268);
            this.btnExit.Margin = new System.Windows.Forms.Padding(6);
            this.btnExit.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnExit.Name = "btnExit";
            this.btnExit.Size = new System.Drawing.Size(32, 25);
            this.btnExit.TabIndex = 9;
            this.btnExit.Text = "Exit";
            this.ImagingConfigToolTip.SetToolTip(this.btnExit, "Exit this wizard.");
            this.btnExit.Click += new System.EventHandler(this.btnExit_Click);
            // 
            // btnPixelToMicron
            // 
            this.btnPixelToMicron.AutoSize = true;
            this.btnPixelToMicron.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnPixelToMicron.BackColor = System.Drawing.Color.Transparent;
            this.btnPixelToMicron.FocusedColor = System.Drawing.Color.Navy;
            this.btnPixelToMicron.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.btnPixelToMicron.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnPixelToMicron.Location = new System.Drawing.Point(19, 352);
            this.btnPixelToMicron.Margin = new System.Windows.Forms.Padding(6);
            this.btnPixelToMicron.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnPixelToMicron.Name = "btnPixelToMicron";
            this.btnPixelToMicron.Size = new System.Drawing.Size(169, 25);
            this.btnPixelToMicron.TabIndex = 27;
            this.btnPixelToMicron.Text = "Compute Pixel/micron";
            this.ImagingConfigToolTip.SetToolTip(this.btnPixelToMicron, "Compute the pixel/micron conversion factor.\r\nRequires a set of cluster maps taken" +
        " with constant X/Y stage step.");
            this.btnPixelToMicron.Click += new System.EventHandler(this.btnPixelToMicron_Click);
            // 
            // btnComputeClusterMaps
            // 
            this.btnComputeClusterMaps.AutoSize = true;
            this.btnComputeClusterMaps.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnComputeClusterMaps.BackColor = System.Drawing.Color.Transparent;
            this.btnComputeClusterMaps.FocusedColor = System.Drawing.Color.Navy;
            this.btnComputeClusterMaps.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.btnComputeClusterMaps.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnComputeClusterMaps.Location = new System.Drawing.Point(20, 321);
            this.btnComputeClusterMaps.Margin = new System.Windows.Forms.Padding(6);
            this.btnComputeClusterMaps.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnComputeClusterMaps.Name = "btnComputeClusterMaps";
            this.btnComputeClusterMaps.Size = new System.Drawing.Size(170, 25);
            this.btnComputeClusterMaps.TabIndex = 25;
            this.btnComputeClusterMaps.Text = "Compute cluster maps";
            this.ImagingConfigToolTip.SetToolTip(this.btnComputeClusterMaps, "Computes cluster maps with this configuration.\r\nCluster maps are needed for next " +
        "steps, which deal with optical corrections.");
            this.btnComputeClusterMaps.Click += new System.EventHandler(this.btnComputeClusterMaps_Click);
            // 
            // txtSummaryFile
            // 
            this.txtSummaryFile.BackColor = System.Drawing.Color.GhostWhite;
            this.txtSummaryFile.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtSummaryFile.ForeColor = System.Drawing.Color.DodgerBlue;
            this.txtSummaryFile.Location = new System.Drawing.Point(627, 234);
            this.txtSummaryFile.Name = "txtSummaryFile";
            this.txtSummaryFile.Size = new System.Drawing.Size(185, 25);
            this.txtSummaryFile.TabIndex = 26;
            this.txtSummaryFile.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.ImagingConfigToolTip.SetToolTip(this.txtSummaryFile, "Shows information about an acquisition");
            // 
            // btnMakeChains
            // 
            this.btnMakeChains.AutoSize = true;
            this.btnMakeChains.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnMakeChains.BackColor = System.Drawing.Color.Transparent;
            this.btnMakeChains.FocusedColor = System.Drawing.Color.Navy;
            this.btnMakeChains.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.btnMakeChains.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnMakeChains.Location = new System.Drawing.Point(245, 321);
            this.btnMakeChains.Margin = new System.Windows.Forms.Padding(6);
            this.btnMakeChains.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnMakeChains.Name = "btnMakeChains";
            this.btnMakeChains.Size = new System.Drawing.Size(95, 25);
            this.btnMakeChains.TabIndex = 38;
            this.btnMakeChains.Text = "Make grains";
            this.ImagingConfigToolTip.SetToolTip(this.btnMakeChains, "Makes grains by chaining consecutive grains together.\r\nSize selections are also u" +
        "sed.");
            this.btnMakeChains.Click += new System.EventHandler(this.btnMakeChains_Click);
            // 
            // txtCMPosTol
            // 
            this.txtCMPosTol.BackColor = System.Drawing.Color.GhostWhite;
            this.txtCMPosTol.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtCMPosTol.ForeColor = System.Drawing.Color.DodgerBlue;
            this.txtCMPosTol.Location = new System.Drawing.Point(287, 388);
            this.txtCMPosTol.Name = "txtCMPosTol";
            this.txtCMPosTol.Size = new System.Drawing.Size(53, 25);
            this.txtCMPosTol.TabIndex = 40;
            this.txtCMPosTol.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.ImagingConfigToolTip.SetToolTip(this.txtCMPosTol, "Two clusters are merged into one grain if they belong to neighouring layers and a" +
        "re within this tolerance in X/Y.");
            this.txtCMPosTol.Leave += new System.EventHandler(this.OnClusterMatchPosTolLeave);
            // 
            // txtCMMaxOffset
            // 
            this.txtCMMaxOffset.BackColor = System.Drawing.Color.GhostWhite;
            this.txtCMMaxOffset.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtCMMaxOffset.ForeColor = System.Drawing.Color.DodgerBlue;
            this.txtCMMaxOffset.Location = new System.Drawing.Point(287, 421);
            this.txtCMMaxOffset.Name = "txtCMMaxOffset";
            this.txtCMMaxOffset.Size = new System.Drawing.Size(53, 25);
            this.txtCMMaxOffset.TabIndex = 42;
            this.txtCMMaxOffset.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.ImagingConfigToolTip.SetToolTip(this.txtCMMaxOffset, "The maximum global offset between two layers of grains.");
            this.txtCMMaxOffset.Leave += new System.EventHandler(this.OnClusterMatchMaxOffsetLeave);
            // 
            // txtMinClusterArea
            // 
            this.txtMinClusterArea.BackColor = System.Drawing.Color.GhostWhite;
            this.txtMinClusterArea.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtMinClusterArea.ForeColor = System.Drawing.Color.DodgerBlue;
            this.txtMinClusterArea.Location = new System.Drawing.Point(287, 454);
            this.txtMinClusterArea.Name = "txtMinClusterArea";
            this.txtMinClusterArea.Size = new System.Drawing.Size(53, 25);
            this.txtMinClusterArea.TabIndex = 44;
            this.txtMinClusterArea.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.ImagingConfigToolTip.SetToolTip(this.txtMinClusterArea, "Minimum area of a cluster to be eligible to form grains.");
            this.txtMinClusterArea.Leave += new System.EventHandler(this.OnMinClusterAreaLeave);
            // 
            // txtCMMinMatches
            // 
            this.txtCMMinMatches.BackColor = System.Drawing.Color.GhostWhite;
            this.txtCMMinMatches.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtCMMinMatches.ForeColor = System.Drawing.Color.DodgerBlue;
            this.txtCMMinMatches.Location = new System.Drawing.Point(287, 519);
            this.txtCMMinMatches.Name = "txtCMMinMatches";
            this.txtCMMinMatches.Size = new System.Drawing.Size(53, 25);
            this.txtCMMinMatches.TabIndex = 48;
            this.txtCMMinMatches.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.ImagingConfigToolTip.SetToolTip(this.txtCMMinMatches, "Minimum number of matching clusters between two layers.\r\nAll matches are cancelle" +
        "d if the overall number falls below this number.");
            this.txtCMMinMatches.Leave += new System.EventHandler(this.OnMinClusterMatchesLeave);
            // 
            // txtMinGrainVolume
            // 
            this.txtMinGrainVolume.BackColor = System.Drawing.Color.GhostWhite;
            this.txtMinGrainVolume.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtMinGrainVolume.ForeColor = System.Drawing.Color.DodgerBlue;
            this.txtMinGrainVolume.Location = new System.Drawing.Point(287, 486);
            this.txtMinGrainVolume.Name = "txtMinGrainVolume";
            this.txtMinGrainVolume.Size = new System.Drawing.Size(53, 25);
            this.txtMinGrainVolume.TabIndex = 46;
            this.txtMinGrainVolume.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.ImagingConfigToolTip.SetToolTip(this.txtMinGrainVolume, "Minimum size of a grain.");
            this.txtMinGrainVolume.Leave += new System.EventHandler(this.OnMinGrainVolumeLeave);
            // 
            // btnViewAcquisition
            // 
            this.btnViewAcquisition.AutoSize = true;
            this.btnViewAcquisition.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnViewAcquisition.BackColor = System.Drawing.Color.Transparent;
            this.btnViewAcquisition.FocusedColor = System.Drawing.Color.Navy;
            this.btnViewAcquisition.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.btnViewAcquisition.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnViewAcquisition.Location = new System.Drawing.Point(493, 234);
            this.btnViewAcquisition.Margin = new System.Windows.Forms.Padding(6);
            this.btnViewAcquisition.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnViewAcquisition.Name = "btnViewAcquisition";
            this.btnViewAcquisition.Size = new System.Drawing.Size(127, 25);
            this.btnViewAcquisition.TabIndex = 49;
            this.btnViewAcquisition.Text = "View Acquisition";
            this.ImagingConfigToolTip.SetToolTip(this.btnViewAcquisition, "Computes cluster maps with this configuration.\r\nCluster maps are needed for next " +
        "steps, which deal with optical corrections.");
            this.btnViewAcquisition.Click += new System.EventHandler(this.btnViewAcquisition_Click);
            // 
            // txtNewName
            // 
            this.txtNewName.BackColor = System.Drawing.Color.GhostWhite;
            this.txtNewName.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtNewName.ForeColor = System.Drawing.Color.DodgerBlue;
            this.txtNewName.Location = new System.Drawing.Point(340, 68);
            this.txtNewName.Name = "txtNewName";
            this.txtNewName.Size = new System.Drawing.Size(121, 25);
            this.txtNewName.TabIndex = 4;
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.BackColor = System.Drawing.Color.Transparent;
            this.label1.ForeColor = System.Drawing.Color.DimGray;
            this.label1.Location = new System.Drawing.Point(491, 35);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(146, 21);
            this.label1.TabIndex = 10;
            this.label1.Text = "Image width-height";
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.BackColor = System.Drawing.Color.Transparent;
            this.label4.ForeColor = System.Drawing.Color.DimGray;
            this.label4.Location = new System.Drawing.Point(491, 198);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(186, 21);
            this.label4.TabIndex = 23;
            this.label4.Text = "Maximum clusters/image";
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.BackColor = System.Drawing.Color.Transparent;
            this.label5.ForeColor = System.Drawing.Color.DimGray;
            this.label5.Location = new System.Drawing.Point(491, 165);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(182, 21);
            this.label5.TabIndex = 21;
            this.label5.Text = "Maximum segments/line";
            // 
            // panel1
            // 
            this.panel1.BorderStyle = System.Windows.Forms.BorderStyle.Fixed3D;
            this.panel1.Location = new System.Drawing.Point(474, 33);
            this.panel1.Name = "panel1";
            this.panel1.Size = new System.Drawing.Size(10, 263);
            this.panel1.TabIndex = 36;
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.BackColor = System.Drawing.Color.Transparent;
            this.label3.ForeColor = System.Drawing.Color.DimGray;
            this.label3.Location = new System.Drawing.Point(491, 132);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(180, 21);
            this.label3.TabIndex = 19;
            this.label3.Text = "Grey level target median";
            // 
            // OpenEmptyImagesDlg
            // 
            this.OpenEmptyImagesDlg.Filter = "Bitmap files (*.bmp)|*.bmp|Base64 images (*.b64)|*.b64";
            this.OpenEmptyImagesDlg.Multiselect = true;
            this.OpenEmptyImagesDlg.Title = "Open empty images";
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.BackColor = System.Drawing.Color.Transparent;
            this.label6.ForeColor = System.Drawing.Color.DimGray;
            this.label6.Location = new System.Drawing.Point(19, 386);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(237, 21);
            this.label6.TabIndex = 39;
            this.label6.Text = "Cluster match: position tolerance";
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.BackColor = System.Drawing.Color.Transparent;
            this.label7.ForeColor = System.Drawing.Color.DimGray;
            this.label7.Location = new System.Drawing.Point(19, 419);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(185, 21);
            this.label7.TabIndex = 41;
            this.label7.Text = "Cluster match: max offset";
            // 
            // label8
            // 
            this.label8.AutoSize = true;
            this.label8.BackColor = System.Drawing.Color.Transparent;
            this.label8.ForeColor = System.Drawing.Color.DimGray;
            this.label8.Location = new System.Drawing.Point(19, 452);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(224, 21);
            this.label8.TabIndex = 43;
            this.label8.Text = "Cluster match: min cluster area";
            // 
            // label9
            // 
            this.label9.AutoSize = true;
            this.label9.BackColor = System.Drawing.Color.Transparent;
            this.label9.ForeColor = System.Drawing.Color.DimGray;
            this.label9.Location = new System.Drawing.Point(19, 517);
            this.label9.Name = "label9";
            this.label9.Size = new System.Drawing.Size(202, 21);
            this.label9.TabIndex = 47;
            this.label9.Text = "Cluster match: min matches";
            // 
            // label10
            // 
            this.label10.AutoSize = true;
            this.label10.BackColor = System.Drawing.Color.Transparent;
            this.label10.ForeColor = System.Drawing.Color.DimGray;
            this.label10.Location = new System.Drawing.Point(19, 484);
            this.label10.Name = "label10";
            this.label10.Size = new System.Drawing.Size(236, 21);
            this.label10.TabIndex = 45;
            this.label10.Text = "Cluster match: min grain volume";
            // 
            // chkOptimizeDMagDZ
            // 
            this.chkOptimizeDMagDZ.AutoSize = true;
            this.chkOptimizeDMagDZ.BackColor = System.Drawing.Color.Transparent;
            this.chkOptimizeDMagDZ.ForeColor = System.Drawing.Color.DodgerBlue;
            this.chkOptimizeDMagDZ.Location = new System.Drawing.Point(279, 550);
            this.chkOptimizeDMagDZ.Name = "chkOptimizeDMagDZ";
            this.chkOptimizeDMagDZ.Size = new System.Drawing.Size(164, 25);
            this.chkOptimizeDMagDZ.TabIndex = 50;
            this.chkOptimizeDMagDZ.Text = "Optimize DMag/DZ";
            this.chkOptimizeDMagDZ.UseVisualStyleBackColor = false;
            this.chkOptimizeDMagDZ.Visible = false;
            // 
            // panel2
            // 
            this.panel2.BorderStyle = System.Windows.Forms.BorderStyle.Fixed3D;
            this.panel2.Location = new System.Drawing.Point(20, 302);
            this.panel2.Name = "panel2";
            this.panel2.Size = new System.Drawing.Size(792, 10);
            this.panel2.TabIndex = 51;
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.BackColor = System.Drawing.Color.Transparent;
            this.label2.ForeColor = System.Drawing.Color.DimGray;
            this.label2.Location = new System.Drawing.Point(535, 518);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(101, 21);
            this.label2.TabIndex = 54;
            this.label2.Text = "XY Curvature";
            // 
            // label11
            // 
            this.label11.AutoSize = true;
            this.label11.BackColor = System.Drawing.Color.Transparent;
            this.label11.ForeColor = System.Drawing.Color.DimGray;
            this.label11.Location = new System.Drawing.Point(535, 483);
            this.label11.Name = "label11";
            this.label11.Size = new System.Drawing.Size(92, 21);
            this.label11.TabIndex = 53;
            this.label11.Text = "Z Curvature";
            // 
            // label12
            // 
            this.label12.AutoSize = true;
            this.label12.BackColor = System.Drawing.Color.Transparent;
            this.label12.ForeColor = System.Drawing.Color.DimGray;
            this.label12.Location = new System.Drawing.Point(535, 450);
            this.label12.Name = "label12";
            this.label12.Size = new System.Drawing.Size(142, 21);
            this.label12.TabIndex = 52;
            this.label12.Text = "DMagnification/DZ";
            // 
            // label13
            // 
            this.label13.AutoSize = true;
            this.label13.BackColor = System.Drawing.Color.Transparent;
            this.label13.ForeColor = System.Drawing.Color.DimGray;
            this.label13.Location = new System.Drawing.Point(535, 416);
            this.label13.Name = "label13";
            this.label13.Size = new System.Drawing.Size(142, 21);
            this.label13.TabIndex = 56;
            this.label13.Text = "DMagnification/DY";
            // 
            // txtDmagDY
            // 
            this.txtDmagDY.BackColor = System.Drawing.Color.GhostWhite;
            this.txtDmagDY.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtDmagDY.ForeColor = System.Drawing.Color.DodgerBlue;
            this.txtDmagDY.Location = new System.Drawing.Point(730, 416);
            this.txtDmagDY.Name = "txtDmagDY";
            this.txtDmagDY.Size = new System.Drawing.Size(82, 25);
            this.txtDmagDY.TabIndex = 55;
            this.txtDmagDY.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.ImagingConfigToolTip.SetToolTip(this.txtDmagDY, "Dependency of the magnification factor on Y");
            this.txtDmagDY.Leave += new System.EventHandler(this.OnDmagDYLeave);
            // 
            // label14
            // 
            this.label14.AutoSize = true;
            this.label14.BackColor = System.Drawing.Color.Transparent;
            this.label14.ForeColor = System.Drawing.Color.DimGray;
            this.label14.Location = new System.Drawing.Point(535, 383);
            this.label14.Name = "label14";
            this.label14.Size = new System.Drawing.Size(142, 21);
            this.label14.TabIndex = 58;
            this.label14.Text = "DMagnification/DX";
            // 
            // txtDmagDX
            // 
            this.txtDmagDX.BackColor = System.Drawing.Color.GhostWhite;
            this.txtDmagDX.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtDmagDX.ForeColor = System.Drawing.Color.DodgerBlue;
            this.txtDmagDX.Location = new System.Drawing.Point(730, 383);
            this.txtDmagDX.Name = "txtDmagDX";
            this.txtDmagDX.Size = new System.Drawing.Size(82, 25);
            this.txtDmagDX.TabIndex = 57;
            this.txtDmagDX.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.ImagingConfigToolTip.SetToolTip(this.txtDmagDX, "Dependency of the magnification factor on X");
            this.txtDmagDX.Leave += new System.EventHandler(this.OnDmagDXLeave);
            // 
            // label15
            // 
            this.label15.AutoSize = true;
            this.label15.BackColor = System.Drawing.Color.Transparent;
            this.label15.ForeColor = System.Drawing.Color.DimGray;
            this.label15.Location = new System.Drawing.Point(535, 549);
            this.label15.Name = "label15";
            this.label15.Size = new System.Drawing.Size(89, 21);
            this.label15.TabIndex = 60;
            this.label15.Text = "Camera Tilt";
            // 
            // txtCameraRotation
            // 
            this.txtCameraRotation.BackColor = System.Drawing.Color.GhostWhite;
            this.txtCameraRotation.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtCameraRotation.ForeColor = System.Drawing.Color.DodgerBlue;
            this.txtCameraRotation.Location = new System.Drawing.Point(730, 545);
            this.txtCameraRotation.Name = "txtCameraRotation";
            this.txtCameraRotation.Size = new System.Drawing.Size(82, 25);
            this.txtCameraRotation.TabIndex = 59;
            this.txtCameraRotation.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.ImagingConfigToolTip.SetToolTip(this.txtCameraRotation, "The variation of the metric factor along the radius from the center");
            this.txtCameraRotation.Leave += new System.EventHandler(this.OnCameraRotationLeave);
            // 
            // label16
            // 
            this.label16.AutoSize = true;
            this.label16.BackColor = System.Drawing.Color.Transparent;
            this.label16.ForeColor = System.Drawing.Color.DimGray;
            this.label16.Location = new System.Drawing.Point(535, 321);
            this.label16.Name = "label16";
            this.label16.Size = new System.Drawing.Size(58, 21);
            this.label16.TabIndex = 62;
            this.label16.Text = "X Slant";
            // 
            // txtXSlant
            // 
            this.txtXSlant.BackColor = System.Drawing.Color.GhostWhite;
            this.txtXSlant.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtXSlant.ForeColor = System.Drawing.Color.DodgerBlue;
            this.txtXSlant.Location = new System.Drawing.Point(730, 321);
            this.txtXSlant.Name = "txtXSlant";
            this.txtXSlant.Size = new System.Drawing.Size(82, 25);
            this.txtXSlant.TabIndex = 61;
            this.txtXSlant.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.ImagingConfigToolTip.SetToolTip(this.txtXSlant, "Deviation of the optical axis towards the X axis  (DX/DZ)");
            this.txtXSlant.Leave += new System.EventHandler(this.OnYSlantLeave);
            // 
            // label17
            // 
            this.label17.AutoSize = true;
            this.label17.BackColor = System.Drawing.Color.Transparent;
            this.label17.ForeColor = System.Drawing.Color.DimGray;
            this.label17.Location = new System.Drawing.Point(535, 352);
            this.label17.Name = "label17";
            this.label17.Size = new System.Drawing.Size(58, 21);
            this.label17.TabIndex = 64;
            this.label17.Text = "Y Slant";
            // 
            // txtYSlant
            // 
            this.txtYSlant.BackColor = System.Drawing.Color.GhostWhite;
            this.txtYSlant.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtYSlant.ForeColor = System.Drawing.Color.DodgerBlue;
            this.txtYSlant.Location = new System.Drawing.Point(730, 352);
            this.txtYSlant.Name = "txtYSlant";
            this.txtYSlant.Size = new System.Drawing.Size(82, 25);
            this.txtYSlant.TabIndex = 63;
            this.txtYSlant.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.ImagingConfigToolTip.SetToolTip(this.txtYSlant, "Deviation of the optical axis towards the Y axis  (DY/DZ)");
            this.txtYSlant.Leave += new System.EventHandler(this.OnXSlantLeave);
            // 
            // ImagingConfigurationForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(9F, 21F);
            this.ClientSize = new System.Drawing.Size(831, 585);
            this.Controls.Add(this.label17);
            this.Controls.Add(this.txtYSlant);
            this.Controls.Add(this.label16);
            this.Controls.Add(this.txtXSlant);
            this.Controls.Add(this.label15);
            this.Controls.Add(this.txtCameraRotation);
            this.Controls.Add(this.label14);
            this.Controls.Add(this.txtDmagDX);
            this.Controls.Add(this.label13);
            this.Controls.Add(this.txtDmagDY);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.label11);
            this.Controls.Add(this.label12);
            this.Controls.Add(this.panel2);
            this.Controls.Add(this.chkOptimizeDMagDZ);
            this.Controls.Add(this.btnViewAcquisition);
            this.Controls.Add(this.txtCMMinMatches);
            this.Controls.Add(this.label9);
            this.Controls.Add(this.txtMinGrainVolume);
            this.Controls.Add(this.label10);
            this.Controls.Add(this.txtMinClusterArea);
            this.Controls.Add(this.label8);
            this.Controls.Add(this.txtCMMaxOffset);
            this.Controls.Add(this.label7);
            this.Controls.Add(this.txtCMPosTol);
            this.Controls.Add(this.label6);
            this.Controls.Add(this.btnMakeChains);
            this.Controls.Add(this.txtSummaryFile);
            this.Controls.Add(this.btnComputeClusterMaps);
            this.Controls.Add(this.btnPixelToMicron);
            this.Controls.Add(this.btnExit);
            this.Controls.Add(this.panel1);
            this.Controls.Add(this.txtXYCurvature);
            this.Controls.Add(this.txtZCurvature);
            this.Controls.Add(this.txtDmagDZ);
            this.Controls.Add(this.txtPixMicronY);
            this.Controls.Add(this.txtPixMicronX);
            this.Controls.Add(this.txtMaxClusters);
            this.Controls.Add(this.label4);
            this.Controls.Add(this.txtMaxSegsLine);
            this.Controls.Add(this.label5);
            this.Controls.Add(this.txtGreyLevelTargetMedian);
            this.Controls.Add(this.label3);
            this.Controls.Add(this.btnViewThresholdImage);
            this.Controls.Add(this.txtThresholdImage);
            this.Controls.Add(this.btnThresholdImage);
            this.Controls.Add(this.btnViewEmptyImage);
            this.Controls.Add(this.txtEmptyImage);
            this.Controls.Add(this.btnEmptyImage);
            this.Controls.Add(this.txtHeight);
            this.Controls.Add(this.txtWidth);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.txtCurrentConfiguration);
            this.Controls.Add(this.btnMakeCurrent);
            this.Controls.Add(this.btnDel);
            this.Controls.Add(this.btnLoad);
            this.Controls.Add(this.txtNewName);
            this.Controls.Add(this.btnDuplicate);
            this.Controls.Add(this.lbConfigurations);
            this.Controls.Add(this.btnNew);
            this.DialogCaption = "Imaging Configuration";
            this.Name = "ImagingConfigurationForm";
            this.Load += new System.EventHandler(this.OnLoad);
            this.Controls.SetChildIndex(this.btnNew, 0);
            this.Controls.SetChildIndex(this.lbConfigurations, 0);
            this.Controls.SetChildIndex(this.btnDuplicate, 0);
            this.Controls.SetChildIndex(this.txtNewName, 0);
            this.Controls.SetChildIndex(this.btnLoad, 0);
            this.Controls.SetChildIndex(this.btnDel, 0);
            this.Controls.SetChildIndex(this.btnMakeCurrent, 0);
            this.Controls.SetChildIndex(this.txtCurrentConfiguration, 0);
            this.Controls.SetChildIndex(this.label1, 0);
            this.Controls.SetChildIndex(this.txtWidth, 0);
            this.Controls.SetChildIndex(this.txtHeight, 0);
            this.Controls.SetChildIndex(this.btnEmptyImage, 0);
            this.Controls.SetChildIndex(this.txtEmptyImage, 0);
            this.Controls.SetChildIndex(this.btnViewEmptyImage, 0);
            this.Controls.SetChildIndex(this.btnThresholdImage, 0);
            this.Controls.SetChildIndex(this.txtThresholdImage, 0);
            this.Controls.SetChildIndex(this.btnViewThresholdImage, 0);
            this.Controls.SetChildIndex(this.label3, 0);
            this.Controls.SetChildIndex(this.txtGreyLevelTargetMedian, 0);
            this.Controls.SetChildIndex(this.label5, 0);
            this.Controls.SetChildIndex(this.txtMaxSegsLine, 0);
            this.Controls.SetChildIndex(this.label4, 0);
            this.Controls.SetChildIndex(this.txtMaxClusters, 0);
            this.Controls.SetChildIndex(this.txtPixMicronX, 0);
            this.Controls.SetChildIndex(this.txtPixMicronY, 0);
            this.Controls.SetChildIndex(this.txtDmagDZ, 0);
            this.Controls.SetChildIndex(this.txtZCurvature, 0);
            this.Controls.SetChildIndex(this.txtXYCurvature, 0);
            this.Controls.SetChildIndex(this.panel1, 0);
            this.Controls.SetChildIndex(this.btnExit, 0);
            this.Controls.SetChildIndex(this.btnPixelToMicron, 0);
            this.Controls.SetChildIndex(this.btnComputeClusterMaps, 0);
            this.Controls.SetChildIndex(this.txtSummaryFile, 0);
            this.Controls.SetChildIndex(this.btnMakeChains, 0);
            this.Controls.SetChildIndex(this.label6, 0);
            this.Controls.SetChildIndex(this.txtCMPosTol, 0);
            this.Controls.SetChildIndex(this.label7, 0);
            this.Controls.SetChildIndex(this.txtCMMaxOffset, 0);
            this.Controls.SetChildIndex(this.label8, 0);
            this.Controls.SetChildIndex(this.txtMinClusterArea, 0);
            this.Controls.SetChildIndex(this.label10, 0);
            this.Controls.SetChildIndex(this.txtMinGrainVolume, 0);
            this.Controls.SetChildIndex(this.label9, 0);
            this.Controls.SetChildIndex(this.txtCMMinMatches, 0);
            this.Controls.SetChildIndex(this.btnViewAcquisition, 0);
            this.Controls.SetChildIndex(this.chkOptimizeDMagDZ, 0);
            this.Controls.SetChildIndex(this.panel2, 0);
            this.Controls.SetChildIndex(this.label12, 0);
            this.Controls.SetChildIndex(this.label11, 0);
            this.Controls.SetChildIndex(this.label2, 0);
            this.Controls.SetChildIndex(this.txtDmagDY, 0);
            this.Controls.SetChildIndex(this.label13, 0);
            this.Controls.SetChildIndex(this.txtDmagDX, 0);
            this.Controls.SetChildIndex(this.label14, 0);
            this.Controls.SetChildIndex(this.txtCameraRotation, 0);
            this.Controls.SetChildIndex(this.label15, 0);
            this.Controls.SetChildIndex(this.txtXSlant, 0);
            this.Controls.SetChildIndex(this.label16, 0);
            this.Controls.SetChildIndex(this.txtYSlant, 0);
            this.Controls.SetChildIndex(this.label17, 0);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.ToolTip ImagingConfigToolTip;
        private System.Windows.Forms.ListBox lbConfigurations;
        private SySal.SySalNExTControls.SySalButton btnNew;
        private SySal.SySalNExTControls.SySalButton btnDuplicate;
        private System.Windows.Forms.TextBox txtNewName;
        private SySal.SySalNExTControls.SySalButton btnLoad;
        private SySal.SySalNExTControls.SySalButton btnDel;
        private SySal.SySalNExTControls.SySalButton btnMakeCurrent;
        private System.Windows.Forms.TextBox txtCurrentConfiguration;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TextBox txtWidth;
        private System.Windows.Forms.TextBox txtHeight;
        private SySal.SySalNExTControls.SySalButton btnEmptyImage;
        private System.Windows.Forms.TextBox txtEmptyImage;
        private SySal.SySalNExTControls.SySalButton btnViewEmptyImage;
        private SySal.SySalNExTControls.SySalButton btnViewThresholdImage;
        private System.Windows.Forms.TextBox txtThresholdImage;
        private SySal.SySalNExTControls.SySalButton btnThresholdImage;
        private System.Windows.Forms.TextBox txtGreyLevelTargetMedian;
        private System.Windows.Forms.TextBox txtMaxClusters;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.TextBox txtMaxSegsLine;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.TextBox txtPixMicronY;
        private System.Windows.Forms.TextBox txtPixMicronX;
        private System.Windows.Forms.TextBox txtDmagDZ;
        private System.Windows.Forms.TextBox txtZCurvature;
        private System.Windows.Forms.TextBox txtXYCurvature;
        private System.Windows.Forms.Panel panel1;
        private SySal.SySalNExTControls.SySalButton btnExit;
        private SySal.SySalNExTControls.SySalButton btnPixelToMicron;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.OpenFileDialog OpenEmptyImagesDlg;
        private SySal.SySalNExTControls.SySalButton btnComputeClusterMaps;
        private System.Windows.Forms.TextBox txtSummaryFile;
        private SySal.SySalNExTControls.SySalButton btnMakeChains;
        private System.Windows.Forms.TextBox txtCMPosTol;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.TextBox txtCMMaxOffset;
        private System.Windows.Forms.Label label7;
        private System.Windows.Forms.TextBox txtMinClusterArea;
        private System.Windows.Forms.Label label8;
        private System.Windows.Forms.TextBox txtCMMinMatches;
        private System.Windows.Forms.Label label9;
        private System.Windows.Forms.TextBox txtMinGrainVolume;
        private System.Windows.Forms.Label label10;
        private SySal.SySalNExTControls.SySalButton btnViewAcquisition;
        private System.Windows.Forms.CheckBox chkOptimizeDMagDZ;
        private System.Windows.Forms.Panel panel2;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Label label11;
        private System.Windows.Forms.Label label12;
        private System.Windows.Forms.Label label13;
        private System.Windows.Forms.TextBox txtDmagDY;
        private System.Windows.Forms.Label label14;
        private System.Windows.Forms.TextBox txtDmagDX;
        private System.Windows.Forms.Label label15;
        private System.Windows.Forms.TextBox txtCameraRotation;
        private System.Windows.Forms.Label label16;
        private System.Windows.Forms.TextBox txtXSlant;
        private System.Windows.Forms.Label label17;
        private System.Windows.Forms.TextBox txtYSlant;
    }
}
