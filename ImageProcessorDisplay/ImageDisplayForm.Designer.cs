namespace SySal.ImageProcessorDisplay
{
    partial class ImageDisplayForm
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
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(ImageDisplayForm));
            this.pnRight = new System.Windows.Forms.Panel();
            this.txtClusters = new System.Windows.Forms.TextBox();
            this.label2 = new System.Windows.Forms.Label();
            this.txtGreyLevelMedian = new System.Windows.Forms.TextBox();
            this.label1 = new System.Windows.Forms.Label();
            this.rdBinary = new System.Windows.Forms.RadioButton();
            this.rdSource = new System.Windows.Forms.RadioButton();
            this.btnConfigure = new System.Windows.Forms.Button();
            this.pnBottom = new System.Windows.Forms.Panel();
            this.pbScreen = new System.Windows.Forms.PictureBox();
            this.chkShowClusters = new System.Windows.Forms.CheckBox();
            this.pnRight.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pbScreen)).BeginInit();
            this.SuspendLayout();
            // 
            // pnRight
            // 
            this.pnRight.BackColor = System.Drawing.Color.White;
            this.pnRight.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("pnRight.BackgroundImage")));
            this.pnRight.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Stretch;
            this.pnRight.Controls.Add(this.chkShowClusters);
            this.pnRight.Controls.Add(this.txtClusters);
            this.pnRight.Controls.Add(this.label2);
            this.pnRight.Controls.Add(this.txtGreyLevelMedian);
            this.pnRight.Controls.Add(this.label1);
            this.pnRight.Controls.Add(this.rdBinary);
            this.pnRight.Controls.Add(this.rdSource);
            this.pnRight.Controls.Add(this.btnConfigure);
            this.pnRight.Dock = System.Windows.Forms.DockStyle.Right;
            this.pnRight.Location = new System.Drawing.Point(529, 0);
            this.pnRight.MinimumSize = new System.Drawing.Size(72, 0);
            this.pnRight.Name = "pnRight";
            this.pnRight.Size = new System.Drawing.Size(72, 286);
            this.pnRight.TabIndex = 1;
            // 
            // txtClusters
            // 
            this.txtClusters.Location = new System.Drawing.Point(9, 125);
            this.txtClusters.Name = "txtClusters";
            this.txtClusters.ReadOnly = true;
            this.txtClusters.Size = new System.Drawing.Size(50, 20);
            this.txtClusters.TabIndex = 8;
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.BackColor = System.Drawing.Color.Transparent;
            this.label2.ForeColor = System.Drawing.Color.White;
            this.label2.Location = new System.Drawing.Point(7, 107);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(44, 13);
            this.label2.TabIndex = 7;
            this.label2.Text = "Clusters";
            // 
            // txtGreyLevelMedian
            // 
            this.txtGreyLevelMedian.Location = new System.Drawing.Point(9, 80);
            this.txtGreyLevelMedian.Name = "txtGreyLevelMedian";
            this.txtGreyLevelMedian.ReadOnly = true;
            this.txtGreyLevelMedian.Size = new System.Drawing.Size(50, 20);
            this.txtGreyLevelMedian.TabIndex = 6;
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.BackColor = System.Drawing.Color.Transparent;
            this.label1.ForeColor = System.Drawing.Color.White;
            this.label1.Location = new System.Drawing.Point(7, 62);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(58, 13);
            this.label1.TabIndex = 5;
            this.label1.Text = "Grey Level";
            // 
            // rdBinary
            // 
            this.rdBinary.AutoSize = true;
            this.rdBinary.BackColor = System.Drawing.Color.Transparent;
            this.rdBinary.ForeColor = System.Drawing.Color.White;
            this.rdBinary.Location = new System.Drawing.Point(6, 35);
            this.rdBinary.Name = "rdBinary";
            this.rdBinary.Size = new System.Drawing.Size(54, 17);
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
            this.rdSource.ForeColor = System.Drawing.Color.White;
            this.rdSource.Location = new System.Drawing.Point(6, 12);
            this.rdSource.Name = "rdSource";
            this.rdSource.Size = new System.Drawing.Size(59, 17);
            this.rdSource.TabIndex = 3;
            this.rdSource.TabStop = true;
            this.rdSource.Text = "Source";
            this.rdSource.UseVisualStyleBackColor = false;
            this.rdSource.CheckedChanged += new System.EventHandler(this.OnSourceCheckedChanged);
            // 
            // btnConfigure
            // 
            this.btnConfigure.Location = new System.Drawing.Point(6, 183);
            this.btnConfigure.Name = "btnConfigure";
            this.btnConfigure.Size = new System.Drawing.Size(60, 25);
            this.btnConfigure.TabIndex = 2;
            this.btnConfigure.Text = "Config";
            this.btnConfigure.UseVisualStyleBackColor = true;
            this.btnConfigure.Click += new System.EventHandler(this.btnConfigure_Click);
            // 
            // pnBottom
            // 
            this.pnBottom.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(46)))), ((int)(((byte)(80)))), ((int)(((byte)(123)))));
            this.pnBottom.Dock = System.Windows.Forms.DockStyle.Bottom;
            this.pnBottom.Location = new System.Drawing.Point(0, 242);
            this.pnBottom.Name = "pnBottom";
            this.pnBottom.Size = new System.Drawing.Size(529, 44);
            this.pnBottom.TabIndex = 5;
            // 
            // pbScreen
            // 
            this.pbScreen.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Zoom;
            this.pbScreen.Dock = System.Windows.Forms.DockStyle.Fill;
            this.pbScreen.Location = new System.Drawing.Point(0, 0);
            this.pbScreen.Name = "pbScreen";
            this.pbScreen.Size = new System.Drawing.Size(529, 242);
            this.pbScreen.TabIndex = 6;
            this.pbScreen.TabStop = false;
            // 
            // chkShowClusters
            // 
            this.chkShowClusters.AutoSize = true;
            this.chkShowClusters.BackColor = System.Drawing.Color.Transparent;
            this.chkShowClusters.ForeColor = System.Drawing.Color.White;
            this.chkShowClusters.Location = new System.Drawing.Point(10, 150);
            this.chkShowClusters.Name = "chkShowClusters";
            this.chkShowClusters.Size = new System.Drawing.Size(53, 17);
            this.chkShowClusters.TabIndex = 9;
            this.chkShowClusters.Text = "Show";
            this.chkShowClusters.UseVisualStyleBackColor = false;
            // 
            // ImageDisplayForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(601, 286);
            this.ControlBox = false;
            this.Controls.Add(this.pbScreen);
            this.Controls.Add(this.pnBottom);
            this.Controls.Add(this.pnRight);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedSingle;
            this.MaximizeBox = false;
            this.MinimizeBox = false;
            this.Name = "ImageDisplayForm";
            this.ShowIcon = false;
            this.ShowInTaskbar = false;
            this.pnRight.ResumeLayout(false);
            this.pnRight.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pbScreen)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.Panel pnRight;
        private System.Windows.Forms.Button btnConfigure;
        private System.Windows.Forms.RadioButton rdBinary;
        private System.Windows.Forms.RadioButton rdSource;
        private System.Windows.Forms.TextBox txtGreyLevelMedian;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Panel pnBottom;
        private System.Windows.Forms.PictureBox pbScreen;
        private System.Windows.Forms.TextBox txtClusters;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.CheckBox chkShowClusters;
    }
}