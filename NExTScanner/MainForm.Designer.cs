namespace SySal.Executables.NExTScanner
{
    partial class SySalMainForm
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
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(SySalMainForm));
            this.TopPanel = new System.Windows.Forms.Panel();
            this.BackgroundPanel = new System.Windows.Forms.Panel();
            this.MenuFlowLayout = new System.Windows.Forms.FlowLayoutPanel();
            this.SysMenuPanel = new System.Windows.Forms.Panel();
            this.sysBtnMaximize = new SySal.SySalNExTControls.SySalButton();
            this.SysMenuSeparator2 = new System.Windows.Forms.Panel();
            this.sysBtnMinimize = new SySal.SySalNExTControls.SySalButton();
            this.SysMenuSeparator1 = new System.Windows.Forms.Panel();
            this.sysBtnClose = new SySal.SySalNExTControls.SySalButton();
            this.SySalLabel = new System.Windows.Forms.Label();
            this.panel2 = new System.Windows.Forms.Panel();
            this.ClientPanel = new System.Windows.Forms.Panel();
            this.MainToolTip = new System.Windows.Forms.ToolTip(this.components);
            this.TopPanel.SuspendLayout();
            this.BackgroundPanel.SuspendLayout();
            this.SysMenuPanel.SuspendLayout();
            this.SuspendLayout();
            // 
            // TopPanel
            // 
            this.TopPanel.BackColor = System.Drawing.Color.Transparent;
            this.TopPanel.Controls.Add(this.BackgroundPanel);
            this.TopPanel.Controls.Add(this.panel2);
            this.TopPanel.Dock = System.Windows.Forms.DockStyle.Top;
            this.TopPanel.Location = new System.Drawing.Point(0, 0);
            this.TopPanel.Name = "TopPanel";
            this.TopPanel.Size = new System.Drawing.Size(584, 48);
            this.TopPanel.TabIndex = 0;
            this.TopPanel.Paint += new System.Windows.Forms.PaintEventHandler(this.OnTopPanelPaint);
            // 
            // BackgroundPanel
            // 
            this.BackgroundPanel.BackColor = System.Drawing.Color.Transparent;
            this.BackgroundPanel.Controls.Add(this.MenuFlowLayout);
            this.BackgroundPanel.Controls.Add(this.SysMenuPanel);
            this.BackgroundPanel.Controls.Add(this.SySalLabel);
            this.BackgroundPanel.Dock = System.Windows.Forms.DockStyle.Fill;
            this.BackgroundPanel.Location = new System.Drawing.Point(0, 0);
            this.BackgroundPanel.Name = "BackgroundPanel";
            this.BackgroundPanel.Size = new System.Drawing.Size(584, 47);
            this.BackgroundPanel.TabIndex = 1;
            // 
            // MenuFlowLayout
            // 
            this.MenuFlowLayout.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.MenuFlowLayout.BackColor = System.Drawing.Color.Transparent;
            this.MenuFlowLayout.Dock = System.Windows.Forms.DockStyle.Fill;
            this.MenuFlowLayout.ForeColor = System.Drawing.Color.DodgerBlue;
            this.MenuFlowLayout.Location = new System.Drawing.Point(84, 0);
            this.MenuFlowLayout.Name = "MenuFlowLayout";
            this.MenuFlowLayout.Size = new System.Drawing.Size(488, 47);
            this.MenuFlowLayout.TabIndex = 1;
            this.MenuFlowLayout.WrapContents = false;
            this.MenuFlowLayout.Paint += new System.Windows.Forms.PaintEventHandler(this.OnMainBarPaint);
            this.MenuFlowLayout.MouseDown += new System.Windows.Forms.MouseEventHandler(this.OnMainBarMouseDown);
            this.MenuFlowLayout.MouseMove += new System.Windows.Forms.MouseEventHandler(this.OnMainBarMouseMove);
            this.MenuFlowLayout.MouseUp += new System.Windows.Forms.MouseEventHandler(this.OnMainBarMouseUp);
            // 
            // SysMenuPanel
            // 
            this.SysMenuPanel.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.SysMenuPanel.Controls.Add(this.sysBtnMaximize);
            this.SysMenuPanel.Controls.Add(this.SysMenuSeparator2);
            this.SysMenuPanel.Controls.Add(this.sysBtnMinimize);
            this.SysMenuPanel.Controls.Add(this.SysMenuSeparator1);
            this.SysMenuPanel.Controls.Add(this.sysBtnClose);
            this.SysMenuPanel.Dock = System.Windows.Forms.DockStyle.Right;
            this.SysMenuPanel.Location = new System.Drawing.Point(572, 0);
            this.SysMenuPanel.Name = "SysMenuPanel";
            this.SysMenuPanel.Size = new System.Drawing.Size(12, 47);
            this.SysMenuPanel.TabIndex = 1;
            // 
            // sysBtnMaximize
            // 
            this.sysBtnMaximize.AutoSize = true;
            this.sysBtnMaximize.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.sysBtnMaximize.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("sysBtnMaximize.BackgroundImage")));
            this.sysBtnMaximize.Dock = System.Windows.Forms.DockStyle.Top;
            this.sysBtnMaximize.FocusedColor = System.Drawing.Color.Navy;
            this.sysBtnMaximize.Font = new System.Drawing.Font("Segoe UI", 14.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.sysBtnMaximize.ForeColor = System.Drawing.Color.DodgerBlue;
            this.sysBtnMaximize.Location = new System.Drawing.Point(0, 32);
            this.sysBtnMaximize.Margin = new System.Windows.Forms.Padding(2);
            this.sysBtnMaximize.MinimumSize = new System.Drawing.Size(12, 12);
            this.sysBtnMaximize.Name = "sysBtnMaximize";
            this.sysBtnMaximize.Size = new System.Drawing.Size(12, 12);
            this.sysBtnMaximize.TabIndex = 4;
            this.sysBtnMaximize.Click += new System.EventHandler(this.sysBtnMaximize_Click);
            // 
            // SysMenuSeparator2
            // 
            this.SysMenuSeparator2.Dock = System.Windows.Forms.DockStyle.Top;
            this.SysMenuSeparator2.Location = new System.Drawing.Point(0, 28);
            this.SysMenuSeparator2.Name = "SysMenuSeparator2";
            this.SysMenuSeparator2.Size = new System.Drawing.Size(12, 4);
            this.SysMenuSeparator2.TabIndex = 1;
            // 
            // sysBtnMinimize
            // 
            this.sysBtnMinimize.AutoSize = true;
            this.sysBtnMinimize.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.sysBtnMinimize.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("sysBtnMinimize.BackgroundImage")));
            this.sysBtnMinimize.Dock = System.Windows.Forms.DockStyle.Top;
            this.sysBtnMinimize.FocusedColor = System.Drawing.Color.Navy;
            this.sysBtnMinimize.Font = new System.Drawing.Font("Segoe UI", 14.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.sysBtnMinimize.ForeColor = System.Drawing.Color.DodgerBlue;
            this.sysBtnMinimize.Location = new System.Drawing.Point(0, 16);
            this.sysBtnMinimize.Margin = new System.Windows.Forms.Padding(2);
            this.sysBtnMinimize.MinimumSize = new System.Drawing.Size(12, 12);
            this.sysBtnMinimize.Name = "sysBtnMinimize";
            this.sysBtnMinimize.Size = new System.Drawing.Size(12, 12);
            this.sysBtnMinimize.TabIndex = 2;
            this.sysBtnMinimize.Click += new System.EventHandler(this.sysBtnMinimize_Click);
            // 
            // SysMenuSeparator1
            // 
            this.SysMenuSeparator1.Dock = System.Windows.Forms.DockStyle.Top;
            this.SysMenuSeparator1.Location = new System.Drawing.Point(0, 12);
            this.SysMenuSeparator1.Name = "SysMenuSeparator1";
            this.SysMenuSeparator1.Size = new System.Drawing.Size(12, 4);
            this.SysMenuSeparator1.TabIndex = 3;
            // 
            // sysBtnClose
            // 
            this.sysBtnClose.AutoSize = true;
            this.sysBtnClose.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.sysBtnClose.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("sysBtnClose.BackgroundImage")));
            this.sysBtnClose.Dock = System.Windows.Forms.DockStyle.Top;
            this.sysBtnClose.FocusedColor = System.Drawing.Color.Navy;
            this.sysBtnClose.Font = new System.Drawing.Font("Segoe UI", 14.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.sysBtnClose.ForeColor = System.Drawing.Color.DodgerBlue;
            this.sysBtnClose.Location = new System.Drawing.Point(0, 0);
            this.sysBtnClose.Margin = new System.Windows.Forms.Padding(2);
            this.sysBtnClose.MinimumSize = new System.Drawing.Size(12, 12);
            this.sysBtnClose.Name = "sysBtnClose";
            this.sysBtnClose.Size = new System.Drawing.Size(12, 12);
            this.sysBtnClose.TabIndex = 0;
            this.sysBtnClose.Click += new System.EventHandler(this.sysBtnClose_Click);
            // 
            // SySalLabel
            // 
            this.SySalLabel.BackColor = System.Drawing.Color.Transparent;
            this.SySalLabel.Dock = System.Windows.Forms.DockStyle.Left;
            this.SySalLabel.Font = new System.Drawing.Font("Segoe UI Semibold", 20.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.SySalLabel.ForeColor = System.Drawing.Color.Navy;
            this.SySalLabel.Location = new System.Drawing.Point(0, 0);
            this.SySalLabel.Name = "SySalLabel";
            this.SySalLabel.Size = new System.Drawing.Size(84, 47);
            this.SySalLabel.TabIndex = 0;
            this.SySalLabel.Text = "SySal";
            this.SySalLabel.MouseDown += new System.Windows.Forms.MouseEventHandler(this.OnMainBarMouseDown);
            this.SySalLabel.MouseMove += new System.Windows.Forms.MouseEventHandler(this.OnMainBarMouseMove);
            this.SySalLabel.MouseUp += new System.Windows.Forms.MouseEventHandler(this.OnMainBarMouseUp);
            // 
            // panel2
            // 
            this.panel2.BackColor = System.Drawing.Color.SteelBlue;
            this.panel2.Dock = System.Windows.Forms.DockStyle.Bottom;
            this.panel2.ForeColor = System.Drawing.Color.DodgerBlue;
            this.panel2.Location = new System.Drawing.Point(0, 47);
            this.panel2.Name = "panel2";
            this.panel2.Size = new System.Drawing.Size(584, 1);
            this.panel2.TabIndex = 0;
            // 
            // ClientPanel
            // 
            this.ClientPanel.BackColor = System.Drawing.Color.Transparent;
            this.ClientPanel.Dock = System.Windows.Forms.DockStyle.Fill;
            this.ClientPanel.Location = new System.Drawing.Point(0, 48);
            this.ClientPanel.Name = "ClientPanel";
            this.ClientPanel.Size = new System.Drawing.Size(584, 336);
            this.ClientPanel.TabIndex = 3;
            this.ClientPanel.Paint += new System.Windows.Forms.PaintEventHandler(this.OnBackpanelPaint);
            this.ClientPanel.Resize += new System.EventHandler(this.OnResize);
            // 
            // MainToolTip
            // 
            this.MainToolTip.IsBalloon = true;
            // 
            // SySalMainForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.BackColor = System.Drawing.Color.White;
            this.ClientSize = new System.Drawing.Size(584, 384);
            this.ControlBox = false;
            this.Controls.Add(this.ClientPanel);
            this.Controls.Add(this.TopPanel);
            this.DoubleBuffered = true;
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.SizableToolWindow;
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.MaximizeBox = false;
            this.MinimizeBox = false;
            this.Name = "SySalMainForm";
            this.FormClosing += new System.Windows.Forms.FormClosingEventHandler(this.OnMainFormClosing);
            this.Load += new System.EventHandler(this.OnMainFormLoad);
            this.TopPanel.ResumeLayout(false);
            this.BackgroundPanel.ResumeLayout(false);
            this.SysMenuPanel.ResumeLayout(false);
            this.SysMenuPanel.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.Panel TopPanel;
        private System.Windows.Forms.Panel BackgroundPanel;
        private System.Windows.Forms.Label SySalLabel;
        private System.Windows.Forms.Panel panel2;
        private System.Windows.Forms.FlowLayoutPanel MenuFlowLayout;
        private System.Windows.Forms.Panel SysMenuPanel;
        private System.Windows.Forms.Panel ClientPanel;
        private SySal.SySalNExTControls.SySalButton sysBtnMaximize;
        private SySal.SySalNExTControls.SySalButton sysBtnMinimize;
        private SySal.SySalNExTControls.SySalButton sysBtnClose;
        private System.Windows.Forms.Panel SysMenuSeparator1;
        private System.Windows.Forms.Panel SysMenuSeparator2;
        private System.Windows.Forms.ToolTip MainToolTip;
    }
}

