namespace SySal.Executables.NExTScanner
{
    partial class ThresholdImageForm
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
            this.btnSelImages = new SySal.SySalNExTControls.SySalButton();
            this.txtSampleImages = new System.Windows.Forms.TextBox();
            this.label1 = new System.Windows.Forms.Label();
            this.txtMinThreshold = new System.Windows.Forms.TextBox();
            this.txtMaxThreshold = new System.Windows.Forms.TextBox();
            this.label2 = new System.Windows.Forms.Label();
            this.txtThresholdSteps = new System.Windows.Forms.TextBox();
            this.label3 = new System.Windows.Forms.Label();
            this.txtXDCTWavelets = new System.Windows.Forms.TextBox();
            this.label4 = new System.Windows.Forms.Label();
            this.txtYDCTWavelets = new System.Windows.Forms.TextBox();
            this.label5 = new System.Windows.Forms.Label();
            this.txtCellHeight = new System.Windows.Forms.TextBox();
            this.label6 = new System.Windows.Forms.Label();
            this.txtCellWidth = new System.Windows.Forms.TextBox();
            this.label7 = new System.Windows.Forms.Label();
            this.btnCompute = new SySal.SySalNExTControls.SySalButton();
            this.pbProgress = new SySal.SySalNExTControls.SySalProgressBar();
            this.btnOK = new SySal.SySalNExTControls.SySalButton();
            this.btnCancel = new SySal.SySalNExTControls.SySalButton();
            this.btnViewResult = new SySal.SySalNExTControls.SySalButton();
            this.txtResult = new System.Windows.Forms.TextBox();
            this.ThresholdImageToolTip = new System.Windows.Forms.ToolTip(this.components);
            this.txtMinSize = new System.Windows.Forms.TextBox();
            this.txtMaxSize = new System.Windows.Forms.TextBox();
            this.txtYCells = new System.Windows.Forms.TextBox();
            this.txtXCells = new System.Windows.Forms.TextBox();
            this.btnDefaults = new SySal.SySalNExTControls.SySalButton();
            this.btnRemember = new SySal.SySalNExTControls.SySalButton();
            this.label8 = new System.Windows.Forms.Label();
            this.label9 = new System.Windows.Forms.Label();
            this.label10 = new System.Windows.Forms.Label();
            this.label11 = new System.Windows.Forms.Label();
            this.OpenImageFileDlg = new System.Windows.Forms.OpenFileDialog();
            this.SuspendLayout();
            // 
            // btnSelImages
            // 
            this.btnSelImages.AutoSize = true;
            this.btnSelImages.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnSelImages.BackColor = System.Drawing.Color.Transparent;
            this.btnSelImages.FocusedColor = System.Drawing.Color.Navy;
            this.btnSelImages.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.btnSelImages.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnSelImages.Location = new System.Drawing.Point(13, 33);
            this.btnSelImages.Margin = new System.Windows.Forms.Padding(6);
            this.btnSelImages.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnSelImages.Name = "btnSelImages";
            this.btnSelImages.Size = new System.Drawing.Size(106, 25);
            this.btnSelImages.TabIndex = 1;
            this.btnSelImages.Text = "Select images";
            this.ThresholdImageToolTip.SetToolTip(this.btnSelImages, "Choose the images to be used to compute the threshold image.\r\nThey should be all " +
                    "independent of each other, or at least be a set of independent tomographic seque" +
                    "nces.\r\n");
            this.btnSelImages.Click += new System.EventHandler(this.btnSelImages_Click);
            // 
            // txtSampleImages
            // 
            this.txtSampleImages.BackColor = System.Drawing.Color.LightGray;
            this.txtSampleImages.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtSampleImages.ForeColor = System.Drawing.Color.Navy;
            this.txtSampleImages.Location = new System.Drawing.Point(129, 33);
            this.txtSampleImages.Name = "txtSampleImages";
            this.txtSampleImages.ReadOnly = true;
            this.txtSampleImages.Size = new System.Drawing.Size(147, 25);
            this.txtSampleImages.TabIndex = 2;
            this.txtSampleImages.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.ThresholdImageToolTip.SetToolTip(this.txtSampleImages, "Summary on the set of images chosen for computation.");
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.BackColor = System.Drawing.Color.Transparent;
            this.label1.ForeColor = System.Drawing.Color.DimGray;
            this.label1.Location = new System.Drawing.Point(11, 64);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(148, 21);
            this.label1.TabIndex = 3;
            this.label1.Text = "Minimum threshold";
            // 
            // txtMinThreshold
            // 
            this.txtMinThreshold.BackColor = System.Drawing.Color.GhostWhite;
            this.txtMinThreshold.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtMinThreshold.ForeColor = System.Drawing.Color.DodgerBlue;
            this.txtMinThreshold.Location = new System.Drawing.Point(223, 64);
            this.txtMinThreshold.Name = "txtMinThreshold";
            this.txtMinThreshold.Size = new System.Drawing.Size(53, 25);
            this.txtMinThreshold.TabIndex = 4;
            this.txtMinThreshold.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.ThresholdImageToolTip.SetToolTip(this.txtMinThreshold, "The minimum threshold to use.\r\nSuch values usually occur near the edges, where il" +
                    "lumination is worse and spherical aberrations reach their maximum.");
            this.txtMinThreshold.Leave += new System.EventHandler(this.OnMinThresholdLeave);
            // 
            // txtMaxThreshold
            // 
            this.txtMaxThreshold.BackColor = System.Drawing.Color.GhostWhite;
            this.txtMaxThreshold.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtMaxThreshold.ForeColor = System.Drawing.Color.DodgerBlue;
            this.txtMaxThreshold.Location = new System.Drawing.Point(223, 95);
            this.txtMaxThreshold.Name = "txtMaxThreshold";
            this.txtMaxThreshold.Size = new System.Drawing.Size(53, 25);
            this.txtMaxThreshold.TabIndex = 6;
            this.txtMaxThreshold.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.ThresholdImageToolTip.SetToolTip(this.txtMaxThreshold, "The maximum threshold to use.\r\nThis is normally used around the center, where gra" +
                    "ins are best seen.");
            this.txtMaxThreshold.Leave += new System.EventHandler(this.OnMaxThresholdLeave);
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.BackColor = System.Drawing.Color.Transparent;
            this.label2.ForeColor = System.Drawing.Color.DimGray;
            this.label2.Location = new System.Drawing.Point(11, 95);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(150, 21);
            this.label2.TabIndex = 5;
            this.label2.Text = "Maximum threshold";
            // 
            // txtThresholdSteps
            // 
            this.txtThresholdSteps.BackColor = System.Drawing.Color.GhostWhite;
            this.txtThresholdSteps.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtThresholdSteps.ForeColor = System.Drawing.Color.DodgerBlue;
            this.txtThresholdSteps.Location = new System.Drawing.Point(223, 126);
            this.txtThresholdSteps.Name = "txtThresholdSteps";
            this.txtThresholdSteps.Size = new System.Drawing.Size(53, 25);
            this.txtThresholdSteps.TabIndex = 8;
            this.txtThresholdSteps.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.ThresholdImageToolTip.SetToolTip(this.txtThresholdSteps, "The gap between minimum and maximum threshold is spanned in this number of steps." +
                    "");
            this.txtThresholdSteps.Leave += new System.EventHandler(this.OnThresholdStepsLeave);
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.BackColor = System.Drawing.Color.Transparent;
            this.label3.ForeColor = System.Drawing.Color.DimGray;
            this.label3.Location = new System.Drawing.Point(11, 126);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(119, 21);
            this.label3.TabIndex = 7;
            this.label3.Text = "Threshold steps";
            // 
            // txtXDCTWavelets
            // 
            this.txtXDCTWavelets.BackColor = System.Drawing.Color.GhostWhite;
            this.txtXDCTWavelets.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtXDCTWavelets.ForeColor = System.Drawing.Color.DodgerBlue;
            this.txtXDCTWavelets.Location = new System.Drawing.Point(440, 33);
            this.txtXDCTWavelets.Name = "txtXDCTWavelets";
            this.txtXDCTWavelets.Size = new System.Drawing.Size(53, 25);
            this.txtXDCTWavelets.TabIndex = 14;
            this.txtXDCTWavelets.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.ThresholdImageToolTip.SetToolTip(this.txtXDCTWavelets, "The number of DCT wavelets along X.");
            this.txtXDCTWavelets.Leave += new System.EventHandler(this.OnXWavesLeave);
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.BackColor = System.Drawing.Color.Transparent;
            this.label4.ForeColor = System.Drawing.Color.DimGray;
            this.label4.Location = new System.Drawing.Point(288, 33);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(116, 21);
            this.label4.TabIndex = 13;
            this.label4.Text = "X DCT wavelets";
            // 
            // txtYDCTWavelets
            // 
            this.txtYDCTWavelets.BackColor = System.Drawing.Color.GhostWhite;
            this.txtYDCTWavelets.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtYDCTWavelets.ForeColor = System.Drawing.Color.DodgerBlue;
            this.txtYDCTWavelets.Location = new System.Drawing.Point(440, 64);
            this.txtYDCTWavelets.Name = "txtYDCTWavelets";
            this.txtYDCTWavelets.Size = new System.Drawing.Size(53, 25);
            this.txtYDCTWavelets.TabIndex = 16;
            this.txtYDCTWavelets.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.ThresholdImageToolTip.SetToolTip(this.txtYDCTWavelets, "The number of DCT wavelets along Y.");
            this.txtYDCTWavelets.Leave += new System.EventHandler(this.OnYWavesLeave);
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.BackColor = System.Drawing.Color.Transparent;
            this.label5.ForeColor = System.Drawing.Color.DimGray;
            this.label5.Location = new System.Drawing.Point(288, 64);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(116, 21);
            this.label5.TabIndex = 15;
            this.label5.Text = "Y DCT wavelets";
            // 
            // txtCellHeight
            // 
            this.txtCellHeight.BackColor = System.Drawing.Color.GhostWhite;
            this.txtCellHeight.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtCellHeight.ForeColor = System.Drawing.Color.DodgerBlue;
            this.txtCellHeight.Location = new System.Drawing.Point(440, 188);
            this.txtCellHeight.Name = "txtCellHeight";
            this.txtCellHeight.Size = new System.Drawing.Size(53, 25);
            this.txtCellHeight.TabIndex = 24;
            this.txtCellHeight.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.ThresholdImageToolTip.SetToolTip(this.txtCellHeight, "Height of sampling cells in pixels.\r\nA limited amount of overlapping among cells " +
                    "can help regularizing the results, \r\nbut too much inter-dependence reduces the e" +
                    "ffectiveness of the procedure.\r\n");
            this.txtCellHeight.Leave += new System.EventHandler(this.OnCellHeightLeave);
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.BackColor = System.Drawing.Color.Transparent;
            this.label6.ForeColor = System.Drawing.Color.DimGray;
            this.label6.Location = new System.Drawing.Point(288, 188);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(84, 21);
            this.label6.TabIndex = 23;
            this.label6.Text = "Cell height";
            // 
            // txtCellWidth
            // 
            this.txtCellWidth.BackColor = System.Drawing.Color.GhostWhite;
            this.txtCellWidth.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtCellWidth.ForeColor = System.Drawing.Color.DodgerBlue;
            this.txtCellWidth.Location = new System.Drawing.Point(440, 157);
            this.txtCellWidth.Name = "txtCellWidth";
            this.txtCellWidth.Size = new System.Drawing.Size(53, 25);
            this.txtCellWidth.TabIndex = 22;
            this.txtCellWidth.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.ThresholdImageToolTip.SetToolTip(this.txtCellWidth, "Width of sampling cells in pixels.\r\nA limited amount of overlapping among cells c" +
                    "an help regularizing the results, \r\nbut too much inter-dependence reduces the ef" +
                    "fectiveness of the procedure.");
            this.txtCellWidth.Leave += new System.EventHandler(this.OnCellWidthLeave);
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.BackColor = System.Drawing.Color.Transparent;
            this.label7.ForeColor = System.Drawing.Color.DimGray;
            this.label7.Location = new System.Drawing.Point(288, 157);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(79, 21);
            this.label7.TabIndex = 21;
            this.label7.Text = "Cell width";
            // 
            // btnCompute
            // 
            this.btnCompute.AutoSize = true;
            this.btnCompute.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnCompute.BackColor = System.Drawing.Color.Transparent;
            this.btnCompute.FocusedColor = System.Drawing.Color.Navy;
            this.btnCompute.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.btnCompute.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnCompute.Location = new System.Drawing.Point(15, 256);
            this.btnCompute.Margin = new System.Windows.Forms.Padding(6);
            this.btnCompute.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnCompute.Name = "btnCompute";
            this.btnCompute.Size = new System.Drawing.Size(74, 25);
            this.btnCompute.TabIndex = 25;
            this.btnCompute.Text = "Compute";
            this.ThresholdImageToolTip.SetToolTip(this.btnCompute, "Starts computing the threshold image.\r\nOnce the computation has begun, the same b" +
                    "utton can pause it.\r\nIntermediate results may be available if enough statistics " +
                    "has been collected.");
            this.btnCompute.Click += new System.EventHandler(this.btnCompute_Click);
            // 
            // pbProgress
            // 
            this.pbProgress.Direction = SySal.SySalNExTControls.SySalProgressBarDirection.LeftToRight;
            this.pbProgress.EmptyGradientColors = new System.Drawing.Color[] {
        System.Drawing.Color.Lavender,
        System.Drawing.Color.Azure,
        System.Drawing.Color.Lavender};
            this.pbProgress.EmptyGradientStops = new double[] {
        0.5};
            this.pbProgress.FillGradientColors = new System.Drawing.Color[] {
        System.Drawing.Color.PowderBlue,
        System.Drawing.Color.DodgerBlue,
        System.Drawing.Color.PowderBlue};
            this.pbProgress.FillGradientStops = new double[] {
        0.5};
            this.pbProgress.Font = new System.Drawing.Font("Segoe UI", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.pbProgress.ForeColor = System.Drawing.Color.White;
            this.pbProgress.Location = new System.Drawing.Point(98, 262);
            this.pbProgress.Name = "pbProgress";
            this.pbProgress.Size = new System.Drawing.Size(395, 19);
            this.pbProgress.TabIndex = 26;
            // 
            // btnOK
            // 
            this.btnOK.AutoSize = true;
            this.btnOK.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnOK.BackColor = System.Drawing.Color.Transparent;
            this.btnOK.FocusedColor = System.Drawing.Color.Navy;
            this.btnOK.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.btnOK.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnOK.Location = new System.Drawing.Point(15, 304);
            this.btnOK.Margin = new System.Windows.Forms.Padding(6);
            this.btnOK.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnOK.Name = "btnOK";
            this.btnOK.Size = new System.Drawing.Size(29, 25);
            this.btnOK.TabIndex = 27;
            this.btnOK.Text = "OK";
            this.btnOK.Click += new System.EventHandler(this.btnOK_Click);
            // 
            // btnCancel
            // 
            this.btnCancel.AutoSize = true;
            this.btnCancel.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnCancel.BackColor = System.Drawing.Color.Transparent;
            this.btnCancel.FocusedColor = System.Drawing.Color.Navy;
            this.btnCancel.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.btnCancel.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnCancel.Location = new System.Drawing.Point(438, 304);
            this.btnCancel.Margin = new System.Windows.Forms.Padding(6);
            this.btnCancel.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnCancel.Name = "btnCancel";
            this.btnCancel.Size = new System.Drawing.Size(55, 25);
            this.btnCancel.TabIndex = 30;
            this.btnCancel.Text = "Cancel";
            this.btnCancel.Click += new System.EventHandler(this.btnCancel_Click);
            // 
            // btnViewResult
            // 
            this.btnViewResult.AutoSize = true;
            this.btnViewResult.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnViewResult.BackColor = System.Drawing.Color.Transparent;
            this.btnViewResult.FocusedColor = System.Drawing.Color.Navy;
            this.btnViewResult.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.btnViewResult.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnViewResult.Location = new System.Drawing.Point(98, 304);
            this.btnViewResult.Margin = new System.Windows.Forms.Padding(6);
            this.btnViewResult.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnViewResult.Name = "btnViewResult";
            this.btnViewResult.Size = new System.Drawing.Size(86, 25);
            this.btnViewResult.TabIndex = 28;
            this.btnViewResult.Text = "View result";
            this.ThresholdImageToolTip.SetToolTip(this.btnViewResult, "Shows the resulting threshold image and the density of clusters in two plots.");
            this.btnViewResult.Click += new System.EventHandler(this.btnViewResult_Click);
            // 
            // txtResult
            // 
            this.txtResult.BackColor = System.Drawing.Color.LightGray;
            this.txtResult.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtResult.ForeColor = System.Drawing.Color.Navy;
            this.txtResult.Location = new System.Drawing.Point(203, 304);
            this.txtResult.Name = "txtResult";
            this.txtResult.ReadOnly = true;
            this.txtResult.Size = new System.Drawing.Size(213, 25);
            this.txtResult.TabIndex = 29;
            this.txtResult.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.ThresholdImageToolTip.SetToolTip(this.txtResult, "Information about the results of the procedure.");
            // 
            // ThresholdImageToolTip
            // 
            this.ThresholdImageToolTip.IsBalloon = true;
            // 
            // txtMinSize
            // 
            this.txtMinSize.BackColor = System.Drawing.Color.GhostWhite;
            this.txtMinSize.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtMinSize.ForeColor = System.Drawing.Color.DodgerBlue;
            this.txtMinSize.Location = new System.Drawing.Point(223, 157);
            this.txtMinSize.Name = "txtMinSize";
            this.txtMinSize.Size = new System.Drawing.Size(53, 25);
            this.txtMinSize.TabIndex = 10;
            this.txtMinSize.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.ThresholdImageToolTip.SetToolTip(this.txtMinSize, "Minimum size of the clusters.\r\nClusters that are too small are likely to be noise" +
                    ".");
            this.txtMinSize.Leave += new System.EventHandler(this.OnMinClusterSizeLeave);
            // 
            // txtMaxSize
            // 
            this.txtMaxSize.BackColor = System.Drawing.Color.GhostWhite;
            this.txtMaxSize.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtMaxSize.ForeColor = System.Drawing.Color.DodgerBlue;
            this.txtMaxSize.Location = new System.Drawing.Point(223, 188);
            this.txtMaxSize.Name = "txtMaxSize";
            this.txtMaxSize.Size = new System.Drawing.Size(53, 25);
            this.txtMaxSize.TabIndex = 12;
            this.txtMaxSize.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.ThresholdImageToolTip.SetToolTip(this.txtMaxSize, "The maximum size of clusters to be used.\r\nClusters that are too big are likely sc" +
                    "ratches.\r\nHowever, they are usually few, so their absolute number is not very cr" +
                    "itical.");
            this.txtMaxSize.Leave += new System.EventHandler(this.OnMaxClusterSizeLeave);
            // 
            // txtYCells
            // 
            this.txtYCells.BackColor = System.Drawing.Color.GhostWhite;
            this.txtYCells.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtYCells.ForeColor = System.Drawing.Color.DodgerBlue;
            this.txtYCells.Location = new System.Drawing.Point(440, 126);
            this.txtYCells.Name = "txtYCells";
            this.txtYCells.Size = new System.Drawing.Size(53, 25);
            this.txtYCells.TabIndex = 20;
            this.txtYCells.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.ThresholdImageToolTip.SetToolTip(this.txtYCells, "The number of DCT cells along Y.\r\nThis must be at least twice as much as the numb" +
                    "er of wavelets on Y.");
            this.txtYCells.Leave += new System.EventHandler(this.OnYCellsLeave);
            // 
            // txtXCells
            // 
            this.txtXCells.BackColor = System.Drawing.Color.GhostWhite;
            this.txtXCells.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtXCells.ForeColor = System.Drawing.Color.DodgerBlue;
            this.txtXCells.Location = new System.Drawing.Point(440, 95);
            this.txtXCells.Name = "txtXCells";
            this.txtXCells.Size = new System.Drawing.Size(53, 25);
            this.txtXCells.TabIndex = 18;
            this.txtXCells.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.ThresholdImageToolTip.SetToolTip(this.txtXCells, "The number of DCT cells along X.\r\nThis must be at least twice as much as the numb" +
                    "er of wavelets on X.");
            this.txtXCells.Leave += new System.EventHandler(this.OnXCellsLeave);
            // 
            // btnDefaults
            // 
            this.btnDefaults.AutoSize = true;
            this.btnDefaults.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnDefaults.BackColor = System.Drawing.Color.Transparent;
            this.btnDefaults.FocusedColor = System.Drawing.Color.Navy;
            this.btnDefaults.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.btnDefaults.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnDefaults.Location = new System.Drawing.Point(13, 219);
            this.btnDefaults.Margin = new System.Windows.Forms.Padding(6);
            this.btnDefaults.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnDefaults.Name = "btnDefaults";
            this.btnDefaults.Size = new System.Drawing.Size(121, 25);
            this.btnDefaults.TabIndex = 31;
            this.btnDefaults.Text = "Default settings";
            this.ThresholdImageToolTip.SetToolTip(this.btnDefaults, "Loads the default settings for threshold image computation.");
            this.btnDefaults.Click += new System.EventHandler(this.btnDefaults_Click);
            // 
            // btnRemember
            // 
            this.btnRemember.AutoSize = true;
            this.btnRemember.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnRemember.BackColor = System.Drawing.Color.Transparent;
            this.btnRemember.FocusedColor = System.Drawing.Color.Navy;
            this.btnRemember.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.btnRemember.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnRemember.Location = new System.Drawing.Point(288, 219);
            this.btnRemember.Margin = new System.Windows.Forms.Padding(6);
            this.btnRemember.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnRemember.Name = "btnRemember";
            this.btnRemember.Size = new System.Drawing.Size(148, 25);
            this.btnRemember.TabIndex = 32;
            this.btnRemember.Text = "Remember settings";
            this.ThresholdImageToolTip.SetToolTip(this.btnRemember, "Remember the current settings.");
            this.btnRemember.Click += new System.EventHandler(this.btnRemember_Click);
            // 
            // label8
            // 
            this.label8.AutoSize = true;
            this.label8.BackColor = System.Drawing.Color.Transparent;
            this.label8.ForeColor = System.Drawing.Color.DimGray;
            this.label8.Location = new System.Drawing.Point(11, 157);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(158, 21);
            this.label8.TabIndex = 9;
            this.label8.Text = "Minimum cluster size";
            // 
            // label9
            // 
            this.label9.AutoSize = true;
            this.label9.BackColor = System.Drawing.Color.Transparent;
            this.label9.ForeColor = System.Drawing.Color.DimGray;
            this.label9.Location = new System.Drawing.Point(11, 188);
            this.label9.Name = "label9";
            this.label9.Size = new System.Drawing.Size(160, 21);
            this.label9.TabIndex = 11;
            this.label9.Text = "Maximum cluster size";
            // 
            // label10
            // 
            this.label10.AutoSize = true;
            this.label10.BackColor = System.Drawing.Color.Transparent;
            this.label10.ForeColor = System.Drawing.Color.DimGray;
            this.label10.Location = new System.Drawing.Point(288, 126);
            this.label10.Name = "label10";
            this.label10.Size = new System.Drawing.Size(86, 21);
            this.label10.TabIndex = 19;
            this.label10.Text = "Y DCT cells";
            // 
            // label11
            // 
            this.label11.AutoSize = true;
            this.label11.BackColor = System.Drawing.Color.Transparent;
            this.label11.ForeColor = System.Drawing.Color.DimGray;
            this.label11.Location = new System.Drawing.Point(288, 95);
            this.label11.Name = "label11";
            this.label11.Size = new System.Drawing.Size(86, 21);
            this.label11.TabIndex = 17;
            this.label11.Text = "X DCT cells";
            // 
            // OpenImageFileDlg
            // 
            this.OpenImageFileDlg.Filter = "Bitmap files (*.bmp)|*.bmp|Base64 images (*.b64)|*.b64";
            this.OpenImageFileDlg.Multiselect = true;
            this.OpenImageFileDlg.Title = "Select image files for threshold image";
            // 
            // ThresholdImageForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(9F, 21F);
            this.ClientSize = new System.Drawing.Size(514, 350);
            this.Controls.Add(this.btnRemember);
            this.Controls.Add(this.btnDefaults);
            this.Controls.Add(this.txtYCells);
            this.Controls.Add(this.label10);
            this.Controls.Add(this.txtXCells);
            this.Controls.Add(this.label11);
            this.Controls.Add(this.txtMaxSize);
            this.Controls.Add(this.label9);
            this.Controls.Add(this.txtMinSize);
            this.Controls.Add(this.label8);
            this.Controls.Add(this.txtResult);
            this.Controls.Add(this.btnViewResult);
            this.Controls.Add(this.btnCancel);
            this.Controls.Add(this.btnOK);
            this.Controls.Add(this.pbProgress);
            this.Controls.Add(this.btnCompute);
            this.Controls.Add(this.txtCellHeight);
            this.Controls.Add(this.label6);
            this.Controls.Add(this.txtCellWidth);
            this.Controls.Add(this.label7);
            this.Controls.Add(this.txtYDCTWavelets);
            this.Controls.Add(this.label5);
            this.Controls.Add(this.txtXDCTWavelets);
            this.Controls.Add(this.label4);
            this.Controls.Add(this.txtThresholdSteps);
            this.Controls.Add(this.label3);
            this.Controls.Add(this.txtMaxThreshold);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.txtMinThreshold);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.txtSampleImages);
            this.Controls.Add(this.btnSelImages);
            this.DialogCaption = "Threshold Image Computation";
            this.Name = "ThresholdImageForm";
            this.NoCloseButton = true;
            this.Load += new System.EventHandler(this.OnLoad);
            this.Controls.SetChildIndex(this.btnSelImages, 0);
            this.Controls.SetChildIndex(this.txtSampleImages, 0);
            this.Controls.SetChildIndex(this.label1, 0);
            this.Controls.SetChildIndex(this.txtMinThreshold, 0);
            this.Controls.SetChildIndex(this.label2, 0);
            this.Controls.SetChildIndex(this.txtMaxThreshold, 0);
            this.Controls.SetChildIndex(this.label3, 0);
            this.Controls.SetChildIndex(this.txtThresholdSteps, 0);
            this.Controls.SetChildIndex(this.label4, 0);
            this.Controls.SetChildIndex(this.txtXDCTWavelets, 0);
            this.Controls.SetChildIndex(this.label5, 0);
            this.Controls.SetChildIndex(this.txtYDCTWavelets, 0);
            this.Controls.SetChildIndex(this.label7, 0);
            this.Controls.SetChildIndex(this.txtCellWidth, 0);
            this.Controls.SetChildIndex(this.label6, 0);
            this.Controls.SetChildIndex(this.txtCellHeight, 0);
            this.Controls.SetChildIndex(this.btnCompute, 0);
            this.Controls.SetChildIndex(this.pbProgress, 0);
            this.Controls.SetChildIndex(this.btnOK, 0);
            this.Controls.SetChildIndex(this.btnCancel, 0);
            this.Controls.SetChildIndex(this.btnViewResult, 0);
            this.Controls.SetChildIndex(this.txtResult, 0);
            this.Controls.SetChildIndex(this.label8, 0);
            this.Controls.SetChildIndex(this.txtMinSize, 0);
            this.Controls.SetChildIndex(this.label9, 0);
            this.Controls.SetChildIndex(this.txtMaxSize, 0);
            this.Controls.SetChildIndex(this.label11, 0);
            this.Controls.SetChildIndex(this.txtXCells, 0);
            this.Controls.SetChildIndex(this.label10, 0);
            this.Controls.SetChildIndex(this.txtYCells, 0);
            this.Controls.SetChildIndex(this.btnDefaults, 0);
            this.Controls.SetChildIndex(this.btnRemember, 0);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private SySal.SySalNExTControls.SySalButton btnSelImages;
        private System.Windows.Forms.TextBox txtSampleImages;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TextBox txtMinThreshold;
        private System.Windows.Forms.TextBox txtMaxThreshold;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.TextBox txtThresholdSteps;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.TextBox txtXDCTWavelets;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.TextBox txtYDCTWavelets;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.TextBox txtCellHeight;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.TextBox txtCellWidth;
        private System.Windows.Forms.Label label7;
        private SySal.SySalNExTControls.SySalButton btnCompute;
        private SySal.SySalNExTControls.SySalProgressBar pbProgress;
        private SySal.SySalNExTControls.SySalButton btnOK;
        private SySal.SySalNExTControls.SySalButton btnCancel;
        private SySal.SySalNExTControls.SySalButton btnViewResult;
        private System.Windows.Forms.TextBox txtResult;
        private System.Windows.Forms.ToolTip ThresholdImageToolTip;
        private System.Windows.Forms.TextBox txtMinSize;
        private System.Windows.Forms.Label label8;
        private System.Windows.Forms.TextBox txtMaxSize;
        private System.Windows.Forms.Label label9;
        private System.Windows.Forms.TextBox txtYCells;
        private System.Windows.Forms.Label label10;
        private System.Windows.Forms.TextBox txtXCells;
        private System.Windows.Forms.Label label11;
        private System.Windows.Forms.OpenFileDialog OpenImageFileDlg;
        private SySal.SySalNExTControls.SySalButton btnDefaults;
        private SySal.SySalNExTControls.SySalButton btnRemember;
    }
}
