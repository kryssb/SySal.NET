namespace SySal.Executables.NExTScanner
{
    partial class Pixel2MicronComputeForm
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
            this.label1 = new System.Windows.Forms.Label();
            this.txtMinArea = new System.Windows.Forms.TextBox();
            this.txtMaxArea = new System.Windows.Forms.TextBox();
            this.label2 = new System.Windows.Forms.Label();
            this.txtPosTolerance = new System.Windows.Forms.TextBox();
            this.label3 = new System.Windows.Forms.Label();
            this.txtMinConv = new System.Windows.Forms.TextBox();
            this.label4 = new System.Windows.Forms.Label();
            this.txtMaxConv = new System.Windows.Forms.TextBox();
            this.label5 = new System.Windows.Forms.Label();
            this.btnComputeX = new SySal.SySalNExTControls.SySalButton();
            this.btnComputeY = new SySal.SySalNExTControls.SySalButton();
            this.btnComputeBoth = new SySal.SySalNExTControls.SySalButton();
            this.txtPix2MiX = new System.Windows.Forms.TextBox();
            this.txtPix2MiY = new System.Windows.Forms.TextBox();
            this.pbProgress = new SySal.SySalNExTControls.SySalProgressBar();
            this.btnCancel = new SySal.SySalNExTControls.SySalButton();
            this.btnOK = new SySal.SySalNExTControls.SySalButton();
            this.btnRemember = new SySal.SySalNExTControls.SySalButton();
            this.btnDefaults = new SySal.SySalNExTControls.SySalButton();
            this.txtMinMatches = new System.Windows.Forms.TextBox();
            this.label6 = new System.Windows.Forms.Label();
            this.Pix2MicronComputeToolTip = new System.Windows.Forms.ToolTip(this.components);
            this.label7 = new System.Windows.Forms.Label();
            this.txtPixMiYErr = new System.Windows.Forms.TextBox();
            this.txtPixMiXErr = new System.Windows.Forms.TextBox();
            this.SuspendLayout();
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.BackColor = System.Drawing.Color.Transparent;
            this.label1.ForeColor = System.Drawing.Color.DimGray;
            this.label1.Location = new System.Drawing.Point(12, 37);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(121, 21);
            this.label1.TabIndex = 1;
            this.label1.Text = "Min cluster area";
            // 
            // txtMinArea
            // 
            this.txtMinArea.BackColor = System.Drawing.Color.GhostWhite;
            this.txtMinArea.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtMinArea.ForeColor = System.Drawing.Color.DodgerBlue;
            this.txtMinArea.Location = new System.Drawing.Point(167, 37);
            this.txtMinArea.Name = "txtMinArea";
            this.txtMinArea.Size = new System.Drawing.Size(53, 25);
            this.txtMinArea.TabIndex = 2;
            this.txtMinArea.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.Pix2MicronComputeToolTip.SetToolTip(this.txtMinArea, "The lower bound for the area of clusters to be used.");
            this.txtMinArea.Leave += new System.EventHandler(this.OnMinAreaLeave);
            // 
            // txtMaxArea
            // 
            this.txtMaxArea.BackColor = System.Drawing.Color.GhostWhite;
            this.txtMaxArea.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtMaxArea.ForeColor = System.Drawing.Color.DodgerBlue;
            this.txtMaxArea.Location = new System.Drawing.Point(167, 69);
            this.txtMaxArea.Name = "txtMaxArea";
            this.txtMaxArea.Size = new System.Drawing.Size(53, 25);
            this.txtMaxArea.TabIndex = 4;
            this.txtMaxArea.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.Pix2MicronComputeToolTip.SetToolTip(this.txtMaxArea, "The upper bound for the area of clusters to be used.");
            this.txtMaxArea.Leave += new System.EventHandler(this.OnMaxAreaLeave);
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.BackColor = System.Drawing.Color.Transparent;
            this.label2.ForeColor = System.Drawing.Color.DimGray;
            this.label2.Location = new System.Drawing.Point(12, 69);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(123, 21);
            this.label2.TabIndex = 3;
            this.label2.Text = "Max cluster area";
            // 
            // txtPosTolerance
            // 
            this.txtPosTolerance.BackColor = System.Drawing.Color.GhostWhite;
            this.txtPosTolerance.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtPosTolerance.ForeColor = System.Drawing.Color.DodgerBlue;
            this.txtPosTolerance.Location = new System.Drawing.Point(458, 37);
            this.txtPosTolerance.Name = "txtPosTolerance";
            this.txtPosTolerance.Size = new System.Drawing.Size(53, 25);
            this.txtPosTolerance.TabIndex = 8;
            this.txtPosTolerance.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.Pix2MicronComputeToolTip.SetToolTip(this.txtPosTolerance, "Mapping tolerance in pixels between two images.");
            this.txtPosTolerance.Leave += new System.EventHandler(this.OnPosTolLeave);
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.BackColor = System.Drawing.Color.Transparent;
            this.label3.ForeColor = System.Drawing.Color.DimGray;
            this.label3.Location = new System.Drawing.Point(240, 37);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(180, 21);
            this.label3.TabIndex = 7;
            this.label3.Text = "Position tolerance (pixel)";
            // 
            // txtMinConv
            // 
            this.txtMinConv.BackColor = System.Drawing.Color.GhostWhite;
            this.txtMinConv.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtMinConv.ForeColor = System.Drawing.Color.DodgerBlue;
            this.txtMinConv.Location = new System.Drawing.Point(458, 69);
            this.txtMinConv.Name = "txtMinConv";
            this.txtMinConv.Size = new System.Drawing.Size(53, 25);
            this.txtMinConv.TabIndex = 10;
            this.txtMinConv.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.Pix2MicronComputeToolTip.SetToolTip(this.txtMinConv, "Minimum conversion factor (always positive)");
            this.txtMinConv.Leave += new System.EventHandler(this.OnMinConvLeave);
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.BackColor = System.Drawing.Color.Transparent;
            this.label4.ForeColor = System.Drawing.Color.DimGray;
            this.label4.Location = new System.Drawing.Point(240, 69);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(161, 21);
            this.label4.TabIndex = 9;
            this.label4.Text = "Min conversion factor";
            // 
            // txtMaxConv
            // 
            this.txtMaxConv.BackColor = System.Drawing.Color.GhostWhite;
            this.txtMaxConv.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtMaxConv.ForeColor = System.Drawing.Color.DodgerBlue;
            this.txtMaxConv.Location = new System.Drawing.Point(458, 100);
            this.txtMaxConv.Name = "txtMaxConv";
            this.txtMaxConv.Size = new System.Drawing.Size(53, 25);
            this.txtMaxConv.TabIndex = 12;
            this.txtMaxConv.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.Pix2MicronComputeToolTip.SetToolTip(this.txtMaxConv, "Maximum conversion factor (always positive).");
            this.txtMaxConv.Leave += new System.EventHandler(this.OnMaxConvLeave);
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.BackColor = System.Drawing.Color.Transparent;
            this.label5.ForeColor = System.Drawing.Color.DimGray;
            this.label5.Location = new System.Drawing.Point(240, 100);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(163, 21);
            this.label5.TabIndex = 11;
            this.label5.Text = "Max conversion factor";
            // 
            // btnComputeX
            // 
            this.btnComputeX.AutoSize = true;
            this.btnComputeX.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnComputeX.BackColor = System.Drawing.Color.Transparent;
            this.btnComputeX.FocusedColor = System.Drawing.Color.Navy;
            this.btnComputeX.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.btnComputeX.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnComputeX.Location = new System.Drawing.Point(16, 171);
            this.btnComputeX.Margin = new System.Windows.Forms.Padding(6);
            this.btnComputeX.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnComputeX.Name = "btnComputeX";
            this.btnComputeX.Size = new System.Drawing.Size(88, 25);
            this.btnComputeX.TabIndex = 15;
            this.btnComputeX.Text = "Compute X";
            this.Pix2MicronComputeToolTip.SetToolTip(this.btnComputeX, "Compute only the X pixel-micron conversion factor.");
            this.btnComputeX.Click += new System.EventHandler(this.btnComputeX_Click);
            // 
            // btnComputeY
            // 
            this.btnComputeY.AutoSize = true;
            this.btnComputeY.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnComputeY.BackColor = System.Drawing.Color.Transparent;
            this.btnComputeY.FocusedColor = System.Drawing.Color.Navy;
            this.btnComputeY.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.btnComputeY.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnComputeY.Location = new System.Drawing.Point(116, 171);
            this.btnComputeY.Margin = new System.Windows.Forms.Padding(6);
            this.btnComputeY.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnComputeY.Name = "btnComputeY";
            this.btnComputeY.Size = new System.Drawing.Size(88, 25);
            this.btnComputeY.TabIndex = 16;
            this.btnComputeY.Text = "Compute Y";
            this.Pix2MicronComputeToolTip.SetToolTip(this.btnComputeY, "Compute only the Y pixel-micron conversion factor.");
            this.btnComputeY.Click += new System.EventHandler(this.btnComputeY_Click);
            // 
            // btnComputeBoth
            // 
            this.btnComputeBoth.AutoSize = true;
            this.btnComputeBoth.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnComputeBoth.BackColor = System.Drawing.Color.Transparent;
            this.btnComputeBoth.FocusedColor = System.Drawing.Color.Navy;
            this.btnComputeBoth.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.btnComputeBoth.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnComputeBoth.Location = new System.Drawing.Point(216, 171);
            this.btnComputeBoth.Margin = new System.Windows.Forms.Padding(6);
            this.btnComputeBoth.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnComputeBoth.Name = "btnComputeBoth";
            this.btnComputeBoth.Size = new System.Drawing.Size(113, 25);
            this.btnComputeBoth.TabIndex = 17;
            this.btnComputeBoth.Text = "Compute Both";
            this.Pix2MicronComputeToolTip.SetToolTip(this.btnComputeBoth, "Compute the conversion factors for both axes.");
            this.btnComputeBoth.Click += new System.EventHandler(this.btnComputeBoth_Click);
            // 
            // txtPix2MiX
            // 
            this.txtPix2MiX.BackColor = System.Drawing.Color.LightGray;
            this.txtPix2MiX.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtPix2MiX.ForeColor = System.Drawing.Color.Navy;
            this.txtPix2MiX.Location = new System.Drawing.Point(370, 171);
            this.txtPix2MiX.Name = "txtPix2MiX";
            this.txtPix2MiX.Size = new System.Drawing.Size(69, 25);
            this.txtPix2MiX.TabIndex = 18;
            this.txtPix2MiX.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // txtPix2MiY
            // 
            this.txtPix2MiY.BackColor = System.Drawing.Color.LightGray;
            this.txtPix2MiY.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtPix2MiY.ForeColor = System.Drawing.Color.Navy;
            this.txtPix2MiY.Location = new System.Drawing.Point(445, 171);
            this.txtPix2MiY.Name = "txtPix2MiY";
            this.txtPix2MiY.Size = new System.Drawing.Size(66, 25);
            this.txtPix2MiY.TabIndex = 19;
            this.txtPix2MiY.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
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
            this.pbProgress.Location = new System.Drawing.Point(116, 238);
            this.pbProgress.Name = "pbProgress";
            this.pbProgress.Size = new System.Drawing.Size(285, 19);
            this.pbProgress.TabIndex = 21;
            // 
            // btnCancel
            // 
            this.btnCancel.AutoSize = true;
            this.btnCancel.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnCancel.BackColor = System.Drawing.Color.Transparent;
            this.btnCancel.FocusedColor = System.Drawing.Color.Navy;
            this.btnCancel.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.btnCancel.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnCancel.Location = new System.Drawing.Point(456, 232);
            this.btnCancel.Margin = new System.Windows.Forms.Padding(6);
            this.btnCancel.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnCancel.Name = "btnCancel";
            this.btnCancel.Size = new System.Drawing.Size(55, 25);
            this.btnCancel.TabIndex = 22;
            this.btnCancel.Text = "Cancel";
            this.btnCancel.Click += new System.EventHandler(this.btnCancel_Click);
            // 
            // btnOK
            // 
            this.btnOK.AutoSize = true;
            this.btnOK.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnOK.BackColor = System.Drawing.Color.Transparent;
            this.btnOK.FocusedColor = System.Drawing.Color.Navy;
            this.btnOK.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.btnOK.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnOK.Location = new System.Drawing.Point(16, 232);
            this.btnOK.Margin = new System.Windows.Forms.Padding(6);
            this.btnOK.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnOK.Name = "btnOK";
            this.btnOK.Size = new System.Drawing.Size(29, 25);
            this.btnOK.TabIndex = 20;
            this.btnOK.Text = "OK";
            this.btnOK.Click += new System.EventHandler(this.btnOK_Click);
            // 
            // btnRemember
            // 
            this.btnRemember.AutoSize = true;
            this.btnRemember.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnRemember.BackColor = System.Drawing.Color.Transparent;
            this.btnRemember.FocusedColor = System.Drawing.Color.Navy;
            this.btnRemember.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.btnRemember.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnRemember.Location = new System.Drawing.Point(240, 140);
            this.btnRemember.Margin = new System.Windows.Forms.Padding(6);
            this.btnRemember.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnRemember.Name = "btnRemember";
            this.btnRemember.Size = new System.Drawing.Size(148, 25);
            this.btnRemember.TabIndex = 14;
            this.btnRemember.Text = "Remember settings";
            this.Pix2MicronComputeToolTip.SetToolTip(this.btnRemember, "Save the current settings for pixel-micron conversion computation.");
            this.btnRemember.Click += new System.EventHandler(this.btnRemember_Click);
            // 
            // btnDefaults
            // 
            this.btnDefaults.AutoSize = true;
            this.btnDefaults.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnDefaults.BackColor = System.Drawing.Color.Transparent;
            this.btnDefaults.FocusedColor = System.Drawing.Color.Navy;
            this.btnDefaults.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.btnDefaults.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnDefaults.Location = new System.Drawing.Point(13, 140);
            this.btnDefaults.Margin = new System.Windows.Forms.Padding(6);
            this.btnDefaults.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnDefaults.Name = "btnDefaults";
            this.btnDefaults.Size = new System.Drawing.Size(121, 25);
            this.btnDefaults.TabIndex = 13;
            this.btnDefaults.Text = "Default settings";
            this.Pix2MicronComputeToolTip.SetToolTip(this.btnDefaults, "Restores the default settings.");
            this.btnDefaults.Click += new System.EventHandler(this.btnDefaults_Click);
            // 
            // txtMinMatches
            // 
            this.txtMinMatches.BackColor = System.Drawing.Color.GhostWhite;
            this.txtMinMatches.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtMinMatches.ForeColor = System.Drawing.Color.DodgerBlue;
            this.txtMinMatches.Location = new System.Drawing.Point(167, 100);
            this.txtMinMatches.Name = "txtMinMatches";
            this.txtMinMatches.Size = new System.Drawing.Size(53, 25);
            this.txtMinMatches.TabIndex = 6;
            this.txtMinMatches.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.Pix2MicronComputeToolTip.SetToolTip(this.txtMinMatches, "The minimum number of clusters two image must have in common to give a valid matc" +
                    "h.");
            this.txtMinMatches.Leave += new System.EventHandler(this.OnMinMatchesLeave);
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.BackColor = System.Drawing.Color.Transparent;
            this.label6.ForeColor = System.Drawing.Color.DimGray;
            this.label6.Location = new System.Drawing.Point(12, 100);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(99, 21);
            this.label6.TabIndex = 5;
            this.label6.Text = "Min matches";
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.BackColor = System.Drawing.Color.Transparent;
            this.label7.ForeColor = System.Drawing.Color.DimGray;
            this.label7.Location = new System.Drawing.Point(320, 202);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(45, 21);
            this.label7.TabIndex = 23;
            this.label7.Text = "Error";
            // 
            // txtPixMiYErr
            // 
            this.txtPixMiYErr.BackColor = System.Drawing.Color.LightGray;
            this.txtPixMiYErr.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtPixMiYErr.ForeColor = System.Drawing.Color.Navy;
            this.txtPixMiYErr.Location = new System.Drawing.Point(445, 202);
            this.txtPixMiYErr.Name = "txtPixMiYErr";
            this.txtPixMiYErr.Size = new System.Drawing.Size(66, 25);
            this.txtPixMiYErr.TabIndex = 25;
            this.txtPixMiYErr.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // txtPixMiXErr
            // 
            this.txtPixMiXErr.BackColor = System.Drawing.Color.LightGray;
            this.txtPixMiXErr.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtPixMiXErr.ForeColor = System.Drawing.Color.Navy;
            this.txtPixMiXErr.Location = new System.Drawing.Point(370, 202);
            this.txtPixMiXErr.Name = "txtPixMiXErr";
            this.txtPixMiXErr.Size = new System.Drawing.Size(69, 25);
            this.txtPixMiXErr.TabIndex = 24;
            this.txtPixMiXErr.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // Pixel2MicronComputeForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(9F, 21F);
            this.ClientSize = new System.Drawing.Size(534, 277);
            this.Controls.Add(this.txtPixMiYErr);
            this.Controls.Add(this.txtPixMiXErr);
            this.Controls.Add(this.label7);
            this.Controls.Add(this.txtMinMatches);
            this.Controls.Add(this.label6);
            this.Controls.Add(this.btnRemember);
            this.Controls.Add(this.btnDefaults);
            this.Controls.Add(this.btnCancel);
            this.Controls.Add(this.btnOK);
            this.Controls.Add(this.pbProgress);
            this.Controls.Add(this.txtPix2MiY);
            this.Controls.Add(this.txtPix2MiX);
            this.Controls.Add(this.btnComputeBoth);
            this.Controls.Add(this.btnComputeY);
            this.Controls.Add(this.btnComputeX);
            this.Controls.Add(this.txtMaxConv);
            this.Controls.Add(this.label5);
            this.Controls.Add(this.txtMinConv);
            this.Controls.Add(this.label4);
            this.Controls.Add(this.txtPosTolerance);
            this.Controls.Add(this.label3);
            this.Controls.Add(this.txtMaxArea);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.txtMinArea);
            this.Controls.Add(this.label1);
            this.DialogCaption = "Computing Pixel/Micron conversion";
            this.Name = "Pixel2MicronComputeForm";
            this.NoCloseButton = true;
            this.Load += new System.EventHandler(this.OnLoad);
            this.Controls.SetChildIndex(this.label1, 0);
            this.Controls.SetChildIndex(this.txtMinArea, 0);
            this.Controls.SetChildIndex(this.label2, 0);
            this.Controls.SetChildIndex(this.txtMaxArea, 0);
            this.Controls.SetChildIndex(this.label3, 0);
            this.Controls.SetChildIndex(this.txtPosTolerance, 0);
            this.Controls.SetChildIndex(this.label4, 0);
            this.Controls.SetChildIndex(this.txtMinConv, 0);
            this.Controls.SetChildIndex(this.label5, 0);
            this.Controls.SetChildIndex(this.txtMaxConv, 0);
            this.Controls.SetChildIndex(this.btnComputeX, 0);
            this.Controls.SetChildIndex(this.btnComputeY, 0);
            this.Controls.SetChildIndex(this.btnComputeBoth, 0);
            this.Controls.SetChildIndex(this.txtPix2MiX, 0);
            this.Controls.SetChildIndex(this.txtPix2MiY, 0);
            this.Controls.SetChildIndex(this.pbProgress, 0);
            this.Controls.SetChildIndex(this.btnOK, 0);
            this.Controls.SetChildIndex(this.btnCancel, 0);
            this.Controls.SetChildIndex(this.btnDefaults, 0);
            this.Controls.SetChildIndex(this.btnRemember, 0);
            this.Controls.SetChildIndex(this.label6, 0);
            this.Controls.SetChildIndex(this.txtMinMatches, 0);
            this.Controls.SetChildIndex(this.label7, 0);
            this.Controls.SetChildIndex(this.txtPixMiXErr, 0);
            this.Controls.SetChildIndex(this.txtPixMiYErr, 0);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TextBox txtMinArea;
        private System.Windows.Forms.TextBox txtMaxArea;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.TextBox txtPosTolerance;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.TextBox txtMinConv;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.TextBox txtMaxConv;
        private System.Windows.Forms.Label label5;
        private SySal.SySalNExTControls.SySalButton btnComputeX;
        private SySal.SySalNExTControls.SySalButton btnComputeY;
        private SySal.SySalNExTControls.SySalButton btnComputeBoth;
        private System.Windows.Forms.TextBox txtPix2MiX;
        private System.Windows.Forms.TextBox txtPix2MiY;
        private SySal.SySalNExTControls.SySalProgressBar pbProgress;
        private SySal.SySalNExTControls.SySalButton btnCancel;
        private SySal.SySalNExTControls.SySalButton btnOK;
        private SySal.SySalNExTControls.SySalButton btnRemember;
        private SySal.SySalNExTControls.SySalButton btnDefaults;
        private System.Windows.Forms.TextBox txtMinMatches;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.ToolTip Pix2MicronComputeToolTip;
        private System.Windows.Forms.Label label7;
        private System.Windows.Forms.TextBox txtPixMiYErr;
        private System.Windows.Forms.TextBox txtPixMiXErr;
    }
}
