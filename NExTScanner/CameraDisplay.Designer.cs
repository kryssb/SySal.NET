namespace SySal.Executables.NExTScanner
{
    partial class CameraDisplay
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
                base.Dispose(disposing);
            }            
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.pnRight = new System.Windows.Forms.Panel();
            this.label5 = new System.Windows.Forms.Label();
            this.txtMouseY = new System.Windows.Forms.TextBox();
            this.label4 = new System.Windows.Forms.Label();
            this.txtMouseX = new System.Windows.Forms.TextBox();
            this.btnClearOverlay = new SySal.SySalNExTControls.SySalButton();
            this.btnSyncStage = new SySal.SySalNExTControls.SySalButton();
            this.btnLoadOverlay = new SySal.SySalNExTControls.SySalButton();
            this.btnSendToBack = new SySal.SySalNExTControls.SySalButton();
            this.btnSaveImage = new SySal.SySalNExTControls.SySalButton();
            this.txtZoom = new System.Windows.Forms.TextBox();
            this.btnZoomOut = new SySal.SySalNExTControls.SySalButton();
            this.btnZoomIn = new SySal.SySalNExTControls.SySalButton();
            this.label3 = new System.Windows.Forms.Label();
            this.chkProcess = new System.Windows.Forms.CheckBox();
            this.btnConfigure = new SySal.SySalNExTControls.SySalButton();
            this.chkShowClusters = new System.Windows.Forms.CheckBox();
            this.txtClusters = new System.Windows.Forms.TextBox();
            this.label2 = new System.Windows.Forms.Label();
            this.txtGreyLevelMedian = new System.Windows.Forms.TextBox();
            this.label1 = new System.Windows.Forms.Label();
            this.rdBinary = new System.Windows.Forms.RadioButton();
            this.rdSource = new System.Windows.Forms.RadioButton();
            this.pnBottom = new System.Windows.Forms.Panel();
            this.pbScreen = new System.Windows.Forms.PictureBox();
            this.dlgSaveImage = new System.Windows.Forms.SaveFileDialog();
            this.dlgLoadOverlay = new System.Windows.Forms.OpenFileDialog();
            this.pnRight.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pbScreen)).BeginInit();
            this.SuspendLayout();
            // 
            // pnRight
            // 
            this.pnRight.BackColor = System.Drawing.Color.Azure;
            this.pnRight.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Stretch;
            this.pnRight.Controls.Add(this.label5);
            this.pnRight.Controls.Add(this.txtMouseY);
            this.pnRight.Controls.Add(this.label4);
            this.pnRight.Controls.Add(this.txtMouseX);
            this.pnRight.Controls.Add(this.btnClearOverlay);
            this.pnRight.Controls.Add(this.btnSyncStage);
            this.pnRight.Controls.Add(this.btnLoadOverlay);
            this.pnRight.Controls.Add(this.btnSendToBack);
            this.pnRight.Controls.Add(this.btnSaveImage);
            this.pnRight.Controls.Add(this.txtZoom);
            this.pnRight.Controls.Add(this.btnZoomOut);
            this.pnRight.Controls.Add(this.btnZoomIn);
            this.pnRight.Controls.Add(this.label3);
            this.pnRight.Controls.Add(this.chkProcess);
            this.pnRight.Controls.Add(this.btnConfigure);
            this.pnRight.Controls.Add(this.chkShowClusters);
            this.pnRight.Controls.Add(this.txtClusters);
            this.pnRight.Controls.Add(this.label2);
            this.pnRight.Controls.Add(this.txtGreyLevelMedian);
            this.pnRight.Controls.Add(this.label1);
            this.pnRight.Controls.Add(this.rdBinary);
            this.pnRight.Controls.Add(this.rdSource);
            this.pnRight.Dock = System.Windows.Forms.DockStyle.Right;
            this.pnRight.Location = new System.Drawing.Point(633, 0);
            this.pnRight.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.pnRight.MinimumSize = new System.Drawing.Size(144, 492);
            this.pnRight.Name = "pnRight";
            this.pnRight.Size = new System.Drawing.Size(144, 655);
            this.pnRight.TabIndex = 1;
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.BackColor = System.Drawing.Color.Transparent;
            this.label5.ForeColor = System.Drawing.Color.DarkGray;
            this.label5.Location = new System.Drawing.Point(11, 657);
            this.label5.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(19, 21);
            this.label5.TabIndex = 24;
            this.label5.Text = "Y";
            // 
            // txtMouseY
            // 
            this.txtMouseY.Location = new System.Drawing.Point(39, 657);
            this.txtMouseY.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.txtMouseY.Name = "txtMouseY";
            this.txtMouseY.ReadOnly = true;
            this.txtMouseY.Size = new System.Drawing.Size(73, 29);
            this.txtMouseY.TabIndex = 23;
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.BackColor = System.Drawing.Color.Transparent;
            this.label4.ForeColor = System.Drawing.Color.DarkGray;
            this.label4.Location = new System.Drawing.Point(11, 618);
            this.label4.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(19, 21);
            this.label4.TabIndex = 22;
            this.label4.Text = "X";
            // 
            // txtMouseX
            // 
            this.txtMouseX.Location = new System.Drawing.Point(39, 618);
            this.txtMouseX.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.txtMouseX.Name = "txtMouseX";
            this.txtMouseX.ReadOnly = true;
            this.txtMouseX.Size = new System.Drawing.Size(73, 29);
            this.txtMouseX.TabIndex = 21;
            // 
            // btnClearOverlay
            // 
            this.btnClearOverlay.AutoSize = true;
            this.btnClearOverlay.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnClearOverlay.FocusedColor = System.Drawing.Color.Navy;
            this.btnClearOverlay.Font = new System.Drawing.Font("Segoe UI", 14.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btnClearOverlay.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnClearOverlay.Location = new System.Drawing.Point(10, 502);
            this.btnClearOverlay.Margin = new System.Windows.Forms.Padding(6);
            this.btnClearOverlay.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnClearOverlay.Name = "btnClearOverlay";
            this.btnClearOverlay.Size = new System.Drawing.Size(122, 29);
            this.btnClearOverlay.TabIndex = 20;
            this.btnClearOverlay.Text = "Clear Overlay";
            this.btnClearOverlay.Click += new System.EventHandler(this.btnClearOverlay_Click);
            // 
            // btnSyncStage
            // 
            this.btnSyncStage.AutoSize = true;
            this.btnSyncStage.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnSyncStage.FocusedColor = System.Drawing.Color.Navy;
            this.btnSyncStage.Font = new System.Drawing.Font("Segoe UI", 14.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btnSyncStage.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnSyncStage.Location = new System.Drawing.Point(10, 540);
            this.btnSyncStage.Margin = new System.Windows.Forms.Padding(6);
            this.btnSyncStage.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnSyncStage.Name = "btnSyncStage";
            this.btnSyncStage.Size = new System.Drawing.Size(116, 29);
            this.btnSyncStage.TabIndex = 19;
            this.btnSyncStage.Text = "In Stage Ref.";
            this.btnSyncStage.Click += new System.EventHandler(this.btnSyncStage_Click);
            // 
            // btnLoadOverlay
            // 
            this.btnLoadOverlay.AutoSize = true;
            this.btnLoadOverlay.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnLoadOverlay.FocusedColor = System.Drawing.Color.Navy;
            this.btnLoadOverlay.Font = new System.Drawing.Font("Segoe UI", 14.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btnLoadOverlay.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnLoadOverlay.Location = new System.Drawing.Point(10, 465);
            this.btnLoadOverlay.Margin = new System.Windows.Forms.Padding(6);
            this.btnLoadOverlay.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnLoadOverlay.Name = "btnLoadOverlay";
            this.btnLoadOverlay.Size = new System.Drawing.Size(121, 29);
            this.btnLoadOverlay.TabIndex = 18;
            this.btnLoadOverlay.Text = "Load Overlay";
            this.btnLoadOverlay.Click += new System.EventHandler(this.btnLoadOverlay_Click);
            // 
            // btnSendToBack
            // 
            this.btnSendToBack.AutoSize = true;
            this.btnSendToBack.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnSendToBack.FocusedColor = System.Drawing.Color.Navy;
            this.btnSendToBack.Font = new System.Drawing.Font("Segoe UI", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btnSendToBack.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnSendToBack.Location = new System.Drawing.Point(10, 580);
            this.btnSendToBack.Margin = new System.Windows.Forms.Padding(5, 5, 5, 5);
            this.btnSendToBack.MinimumSize = new System.Drawing.Size(3, 3);
            this.btnSendToBack.Name = "btnSendToBack";
            this.btnSendToBack.Size = new System.Drawing.Size(101, 25);
            this.btnSendToBack.TabIndex = 17;
            this.btnSendToBack.Text = "Send to back";
            this.btnSendToBack.Click += new System.EventHandler(this.btnSendToBack_Click);
            // 
            // btnSaveImage
            // 
            this.btnSaveImage.AutoSize = true;
            this.btnSaveImage.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnSaveImage.FocusedColor = System.Drawing.Color.Navy;
            this.btnSaveImage.Font = new System.Drawing.Font("Segoe UI", 14.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btnSaveImage.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnSaveImage.Location = new System.Drawing.Point(10, 424);
            this.btnSaveImage.Margin = new System.Windows.Forms.Padding(6);
            this.btnSaveImage.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnSaveImage.Name = "btnSaveImage";
            this.btnSaveImage.Size = new System.Drawing.Size(107, 29);
            this.btnSaveImage.TabIndex = 16;
            this.btnSaveImage.Text = "Save Image";
            this.btnSaveImage.Click += new System.EventHandler(this.btnSaveImage_Click);
            // 
            // txtZoom
            // 
            this.txtZoom.Location = new System.Drawing.Point(30, 343);
            this.txtZoom.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.txtZoom.Name = "txtZoom";
            this.txtZoom.ReadOnly = true;
            this.txtZoom.Size = new System.Drawing.Size(47, 29);
            this.txtZoom.TabIndex = 15;
            this.txtZoom.Text = "1.00";
            this.txtZoom.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // btnZoomOut
            // 
            this.btnZoomOut.AutoSize = true;
            this.btnZoomOut.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnZoomOut.FocusedColor = System.Drawing.Color.Navy;
            this.btnZoomOut.Font = new System.Drawing.Font("Segoe UI", 14.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btnZoomOut.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnZoomOut.Location = new System.Drawing.Point(79, 343);
            this.btnZoomOut.Margin = new System.Windows.Forms.Padding(6);
            this.btnZoomOut.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnZoomOut.Name = "btnZoomOut";
            this.btnZoomOut.Size = new System.Drawing.Size(16, 29);
            this.btnZoomOut.TabIndex = 14;
            this.btnZoomOut.Text = "-";
            this.btnZoomOut.Click += new System.EventHandler(this.btnZoomOut_Click);
            // 
            // btnZoomIn
            // 
            this.btnZoomIn.AutoSize = true;
            this.btnZoomIn.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnZoomIn.FocusedColor = System.Drawing.Color.Navy;
            this.btnZoomIn.Font = new System.Drawing.Font("Segoe UI", 14.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btnZoomIn.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnZoomIn.Location = new System.Drawing.Point(9, 343);
            this.btnZoomIn.Margin = new System.Windows.Forms.Padding(6);
            this.btnZoomIn.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnZoomIn.Name = "btnZoomIn";
            this.btnZoomIn.Size = new System.Drawing.Size(21, 29);
            this.btnZoomIn.TabIndex = 13;
            this.btnZoomIn.Text = "+";
            this.btnZoomIn.Click += new System.EventHandler(this.btnZoomIn_Click);
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.BackColor = System.Drawing.Color.Transparent;
            this.label3.ForeColor = System.Drawing.Color.DarkGray;
            this.label3.Location = new System.Drawing.Point(10, 316);
            this.label3.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(51, 21);
            this.label3.TabIndex = 12;
            this.label3.Text = "Zoom";
            // 
            // chkProcess
            // 
            this.chkProcess.AutoSize = true;
            this.chkProcess.BackColor = System.Drawing.Color.Transparent;
            this.chkProcess.ForeColor = System.Drawing.Color.DodgerBlue;
            this.chkProcess.Location = new System.Drawing.Point(14, 277);
            this.chkProcess.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.chkProcess.Name = "chkProcess";
            this.chkProcess.Size = new System.Drawing.Size(82, 25);
            this.chkProcess.TabIndex = 11;
            this.chkProcess.Text = "Process";
            this.chkProcess.UseVisualStyleBackColor = false;
            // 
            // btnConfigure
            // 
            this.btnConfigure.AutoSize = true;
            this.btnConfigure.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnConfigure.FocusedColor = System.Drawing.Color.Navy;
            this.btnConfigure.Font = new System.Drawing.Font("Segoe UI", 14.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btnConfigure.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnConfigure.Location = new System.Drawing.Point(9, 385);
            this.btnConfigure.Margin = new System.Windows.Forms.Padding(6);
            this.btnConfigure.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnConfigure.Name = "btnConfigure";
            this.btnConfigure.Size = new System.Drawing.Size(93, 29);
            this.btnConfigure.TabIndex = 10;
            this.btnConfigure.Text = "Configure";
            this.btnConfigure.Click += new System.EventHandler(this.btnConfigure_Click);
            // 
            // chkShowClusters
            // 
            this.chkShowClusters.AutoSize = true;
            this.chkShowClusters.BackColor = System.Drawing.Color.Transparent;
            this.chkShowClusters.ForeColor = System.Drawing.Color.DodgerBlue;
            this.chkShowClusters.Location = new System.Drawing.Point(15, 242);
            this.chkShowClusters.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.chkShowClusters.Name = "chkShowClusters";
            this.chkShowClusters.Size = new System.Drawing.Size(68, 25);
            this.chkShowClusters.TabIndex = 9;
            this.chkShowClusters.Text = "Show";
            this.chkShowClusters.UseVisualStyleBackColor = false;
            // 
            // txtClusters
            // 
            this.txtClusters.Location = new System.Drawing.Point(14, 202);
            this.txtClusters.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.txtClusters.Name = "txtClusters";
            this.txtClusters.ReadOnly = true;
            this.txtClusters.Size = new System.Drawing.Size(73, 29);
            this.txtClusters.TabIndex = 8;
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.BackColor = System.Drawing.Color.Transparent;
            this.label2.ForeColor = System.Drawing.Color.DarkGray;
            this.label2.Location = new System.Drawing.Point(10, 173);
            this.label2.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(66, 21);
            this.label2.TabIndex = 7;
            this.label2.Text = "Clusters";
            // 
            // txtGreyLevelMedian
            // 
            this.txtGreyLevelMedian.Location = new System.Drawing.Point(14, 129);
            this.txtGreyLevelMedian.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.txtGreyLevelMedian.Name = "txtGreyLevelMedian";
            this.txtGreyLevelMedian.ReadOnly = true;
            this.txtGreyLevelMedian.Size = new System.Drawing.Size(73, 29);
            this.txtGreyLevelMedian.TabIndex = 6;
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.BackColor = System.Drawing.Color.Transparent;
            this.label1.ForeColor = System.Drawing.Color.DarkGray;
            this.label1.Location = new System.Drawing.Point(10, 100);
            this.label1.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(83, 21);
            this.label1.TabIndex = 5;
            this.label1.Text = "Grey Level";
            // 
            // rdBinary
            // 
            this.rdBinary.AutoSize = true;
            this.rdBinary.BackColor = System.Drawing.Color.Transparent;
            this.rdBinary.ForeColor = System.Drawing.Color.DodgerBlue;
            this.rdBinary.Location = new System.Drawing.Point(9, 57);
            this.rdBinary.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.rdBinary.Name = "rdBinary";
            this.rdBinary.Size = new System.Drawing.Size(72, 25);
            this.rdBinary.TabIndex = 4;
            this.rdBinary.TabStop = true;
            this.rdBinary.Text = "Binary";
            this.rdBinary.UseVisualStyleBackColor = false;
            this.rdBinary.CheckedChanged += new System.EventHandler(this.OnBinaryCheckedChanged);
            // 
            // rdSource
            // 
            this.rdSource.AutoSize = true;
            this.rdSource.BackColor = System.Drawing.Color.Transparent;
            this.rdSource.Checked = true;
            this.rdSource.ForeColor = System.Drawing.Color.DodgerBlue;
            this.rdSource.Location = new System.Drawing.Point(9, 19);
            this.rdSource.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.rdSource.Name = "rdSource";
            this.rdSource.Size = new System.Drawing.Size(76, 25);
            this.rdSource.TabIndex = 3;
            this.rdSource.TabStop = true;
            this.rdSource.Text = "Source";
            this.rdSource.UseVisualStyleBackColor = false;
            this.rdSource.CheckedChanged += new System.EventHandler(this.OnSourceCheckedChanged);
            // 
            // pnBottom
            // 
            this.pnBottom.BackColor = System.Drawing.Color.Azure;
            this.pnBottom.Dock = System.Windows.Forms.DockStyle.Bottom;
            this.pnBottom.Location = new System.Drawing.Point(0, 584);
            this.pnBottom.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.pnBottom.Name = "pnBottom";
            this.pnBottom.Size = new System.Drawing.Size(633, 71);
            this.pnBottom.TabIndex = 5;
            // 
            // pbScreen
            // 
            this.pbScreen.BackgroundImageLayout = System.Windows.Forms.ImageLayout.None;
            this.pbScreen.Dock = System.Windows.Forms.DockStyle.Fill;
            this.pbScreen.Location = new System.Drawing.Point(0, 0);
            this.pbScreen.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.pbScreen.Name = "pbScreen";
            this.pbScreen.Size = new System.Drawing.Size(633, 584);
            this.pbScreen.TabIndex = 6;
            this.pbScreen.TabStop = false;
            this.pbScreen.Paint += new System.Windows.Forms.PaintEventHandler(this.OnImagePaint);
            this.pbScreen.MouseDown += new System.Windows.Forms.MouseEventHandler(this.OnScreenMouseDown);
            // 
            // dlgSaveImage
            // 
            this.dlgSaveImage.Filter = "Bitmap files (*.bmp)|*.bmp|Base64 files (*.b64)|*.b64|JPEG files (*.jpg)|*.jpg";
            this.dlgSaveImage.Title = "Choose a file name to save the current image";
            // 
            // dlgLoadOverlay
            // 
            this.dlgLoadOverlay.Filter = "XML 3D files (*.x3l)|*.x3l|Reader files (*.reader)|*.reader|RaW Data files (*.rwd" +
    ")|*.rwd|TLG files (*.tlg)|*.tlg|ASCII files (*.txt)|*.txt";
            this.dlgLoadOverlay.Title = "Select file with overlay graphics";
            // 
            // CameraDisplay
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(9F, 21F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(777, 655);
            this.ControlBox = false;
            this.Controls.Add(this.pbScreen);
            this.Controls.Add(this.pnBottom);
            this.Controls.Add(this.pnRight);
            this.Font = new System.Drawing.Font("Segoe UI", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.ForeColor = System.Drawing.Color.DodgerBlue;
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedSingle;
            this.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.MaximizeBox = false;
            this.MinimizeBox = false;
            this.Name = "CameraDisplay";
            this.ShowIcon = false;
            this.ShowInTaskbar = false;
            this.pnRight.ResumeLayout(false);
            this.pnRight.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pbScreen)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.Panel pnRight;
        private System.Windows.Forms.RadioButton rdBinary;
        private System.Windows.Forms.RadioButton rdSource;
        private System.Windows.Forms.TextBox txtGreyLevelMedian;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Panel pnBottom;
        private System.Windows.Forms.PictureBox pbScreen;
        private System.Windows.Forms.TextBox txtClusters;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.CheckBox chkShowClusters;
        private SySalNExTControls.SySalButton btnConfigure;
        private System.Windows.Forms.CheckBox chkProcess;
        private System.Windows.Forms.TextBox txtZoom;
        private SySalNExTControls.SySalButton btnZoomOut;
        private SySalNExTControls.SySalButton btnZoomIn;
        private System.Windows.Forms.Label label3;
        private SySalNExTControls.SySalButton btnSaveImage;
        private System.Windows.Forms.SaveFileDialog dlgSaveImage;
        private SySalNExTControls.SySalButton btnSendToBack;
        private SySalNExTControls.SySalButton btnSyncStage;
        private SySalNExTControls.SySalButton btnLoadOverlay;
        private SySalNExTControls.SySalButton btnClearOverlay;
        private System.Windows.Forms.OpenFileDialog dlgLoadOverlay;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.TextBox txtMouseY;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.TextBox txtMouseX;
    }
}