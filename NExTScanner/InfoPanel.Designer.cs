namespace SySal.Executables.NExTScanner
{
    partial class InfoPanel
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
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(InfoPanel));
            this.TitleLabel = new System.Windows.Forms.Label();
            this.ContentPanel = new System.Windows.Forms.Panel();
            this.sysBtnClose = new SySal.SySalNExTControls.SySalButton();
            this.sysBtnRefresh = new SySal.SySalNExTControls.SySalButton();
            this.sysBtnExport = new SySal.SySalNExTControls.SySalButton();
            this.SuspendLayout();
            // 
            // TitleLabel
            // 
            this.TitleLabel.Dock = System.Windows.Forms.DockStyle.Top;
            this.TitleLabel.Location = new System.Drawing.Point(0, 0);
            this.TitleLabel.Name = "TitleLabel";
            this.TitleLabel.Size = new System.Drawing.Size(426, 21);
            this.TitleLabel.TabIndex = 0;
            this.TitleLabel.Text = "title";
            this.TitleLabel.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            this.TitleLabel.MouseMove += new System.Windows.Forms.MouseEventHandler(this.OnTitleMouseMove);
            this.TitleLabel.MouseDown += new System.Windows.Forms.MouseEventHandler(this.OnTitleMouseDown);
            this.TitleLabel.MouseUp += new System.Windows.Forms.MouseEventHandler(this.OnTitleMouseUp);
            // 
            // ContentPanel
            // 
            this.ContentPanel.BackColor = System.Drawing.Color.White;
            this.ContentPanel.Dock = System.Windows.Forms.DockStyle.Fill;
            this.ContentPanel.Location = new System.Drawing.Point(0, 21);
            this.ContentPanel.Name = "ContentPanel";
            this.ContentPanel.Size = new System.Drawing.Size(426, 187);
            this.ContentPanel.TabIndex = 4;
            this.ContentPanel.Paint += new System.Windows.Forms.PaintEventHandler(this.OnPaint);
            // 
            // sysBtnClose
            // 
            this.sysBtnClose.AutoSize = true;
            this.sysBtnClose.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.sysBtnClose.BackColor = System.Drawing.Color.Transparent;
            this.sysBtnClose.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("sysBtnClose.BackgroundImage")));
            this.sysBtnClose.FocusedColor = System.Drawing.Color.Navy;
            this.sysBtnClose.Font = new System.Drawing.Font("Segoe UI", 14.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.sysBtnClose.ForeColor = System.Drawing.Color.DodgerBlue;
            this.sysBtnClose.Location = new System.Drawing.Point(410, 3);
            this.sysBtnClose.Margin = new System.Windows.Forms.Padding(2);
            this.sysBtnClose.MinimumSize = new System.Drawing.Size(12, 12);
            this.sysBtnClose.Name = "sysBtnClose";
            this.sysBtnClose.Size = new System.Drawing.Size(12, 12);
            this.sysBtnClose.TabIndex = 2;
            this.sysBtnClose.Click += new System.EventHandler(this.OnCloseClick);
            // 
            // sysBtnRefresh
            // 
            this.sysBtnRefresh.AutoSize = true;
            this.sysBtnRefresh.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.sysBtnRefresh.BackColor = System.Drawing.Color.Transparent;
            this.sysBtnRefresh.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("sysBtnRefresh.BackgroundImage")));
            this.sysBtnRefresh.FocusedColor = System.Drawing.Color.Navy;
            this.sysBtnRefresh.Font = new System.Drawing.Font("Segoe UI", 14.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.sysBtnRefresh.ForeColor = System.Drawing.Color.DodgerBlue;
            this.sysBtnRefresh.Location = new System.Drawing.Point(27, 1);
            this.sysBtnRefresh.Margin = new System.Windows.Forms.Padding(2);
            this.sysBtnRefresh.MinimumSize = new System.Drawing.Size(16, 16);
            this.sysBtnRefresh.Name = "sysBtnRefresh";
            this.sysBtnRefresh.Size = new System.Drawing.Size(16, 16);
            this.sysBtnRefresh.TabIndex = 3;
            this.sysBtnRefresh.Click += new System.EventHandler(this.OnRefreshClick);
            // 
            // sysBtnExport
            // 
            this.sysBtnExport.AutoSize = true;
            this.sysBtnExport.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.sysBtnExport.BackColor = System.Drawing.Color.Transparent;
            this.sysBtnExport.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("sysBtnExport.BackgroundImage")));
            this.sysBtnExport.FocusedColor = System.Drawing.Color.Navy;
            this.sysBtnExport.Font = new System.Drawing.Font("Segoe UI", 14.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.sysBtnExport.ForeColor = System.Drawing.Color.DodgerBlue;
            this.sysBtnExport.Location = new System.Drawing.Point(4, 1);
            this.sysBtnExport.Margin = new System.Windows.Forms.Padding(2);
            this.sysBtnExport.MinimumSize = new System.Drawing.Size(18, 16);
            this.sysBtnExport.Name = "sysBtnExport";
            this.sysBtnExport.Size = new System.Drawing.Size(18, 16);
            this.sysBtnExport.TabIndex = 4;
            this.sysBtnExport.Click += new System.EventHandler(this.OnExportClick);
            // 
            // InfoPanel
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(9F, 21F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.BackColor = System.Drawing.Color.White;
            this.ClientSize = new System.Drawing.Size(426, 208);
            this.ControlBox = false;
            this.Controls.Add(this.sysBtnExport);
            this.Controls.Add(this.sysBtnRefresh);
            this.Controls.Add(this.ContentPanel);
            this.Controls.Add(this.sysBtnClose);
            this.Controls.Add(this.TitleLabel);
            this.DoubleBuffered = true;
            this.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.ForeColor = System.Drawing.Color.Navy;
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.SizableToolWindow;
            this.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.Name = "InfoPanel";
            this.ShowIcon = false;
            this.ShowInTaskbar = false;
            this.Resize += new System.EventHandler(this.OnResize);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Label TitleLabel;
        private System.Windows.Forms.Panel ContentPanel;
        internal SySal.SySalNExTControls.SySalButton sysBtnClose;
        internal SySal.SySalNExTControls.SySalButton sysBtnRefresh;
        internal SySal.SySalNExTControls.SySalButton sysBtnExport;
    }
}