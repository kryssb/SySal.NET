namespace SySal.Executables.SySal2GIF
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
            this.menuStrip1 = new System.Windows.Forms.MenuStrip();
            this.fileToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.openSequenceToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.openAnimatedGIFToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.openFogImageSequenceFileToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripMenuItem5 = new System.Windows.Forms.ToolStripSeparator();
            this.recodeToAnimatedGIFToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.extractBMPImagesToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.viewToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.zoomInToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.zoomOutToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripMenuItem2 = new System.Windows.Forms.ToolStripSeparator();
            this.frameUpToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.frameRevToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.settingsToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.applySySalColorTToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.sharpenImageToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripMenuItem4 = new System.Windows.Forms.ToolStripSeparator();
            this.unlockRegionToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripMenuItem1 = new System.Windows.Forms.ToolStripSeparator();
            this.reverseToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripMenuItem3 = new System.Windows.Forms.ToolStripSeparator();
            this.commentToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.pictureBox1 = new System.Windows.Forms.PictureBox();
            this.statusStrip1 = new System.Windows.Forms.StatusStrip();
            this.statusImageN = new System.Windows.Forms.ToolStripStatusLabel();
            this.statusImageSize = new System.Windows.Forms.ToolStripStatusLabel();
            this.statusXY = new System.Windows.Forms.ToolStripStatusLabel();
            this.panel1 = new System.Windows.Forms.Panel();
            this.menuStrip1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).BeginInit();
            this.statusStrip1.SuspendLayout();
            this.panel1.SuspendLayout();
            this.SuspendLayout();
            // 
            // menuStrip1
            // 
            this.menuStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.fileToolStripMenuItem,
            this.viewToolStripMenuItem,
            this.settingsToolStripMenuItem});
            this.menuStrip1.Location = new System.Drawing.Point(0, 0);
            this.menuStrip1.Name = "menuStrip1";
            this.menuStrip1.Size = new System.Drawing.Size(761, 24);
            this.menuStrip1.TabIndex = 0;
            this.menuStrip1.Text = "menuStrip1";
            // 
            // fileToolStripMenuItem
            // 
            this.fileToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.openSequenceToolStripMenuItem,
            this.openAnimatedGIFToolStripMenuItem,
            this.openFogImageSequenceFileToolStripMenuItem,
            this.toolStripMenuItem5,
            this.recodeToAnimatedGIFToolStripMenuItem,
            this.extractBMPImagesToolStripMenuItem});
            this.fileToolStripMenuItem.Name = "fileToolStripMenuItem";
            this.fileToolStripMenuItem.Size = new System.Drawing.Size(37, 20);
            this.fileToolStripMenuItem.Text = "&File";
            // 
            // openSequenceToolStripMenuItem
            // 
            this.openSequenceToolStripMenuItem.Name = "openSequenceToolStripMenuItem";
            this.openSequenceToolStripMenuItem.ShortcutKeys = ((System.Windows.Forms.Keys)((System.Windows.Forms.Keys.Alt | System.Windows.Forms.Keys.B)));
            this.openSequenceToolStripMenuItem.Size = new System.Drawing.Size(273, 22);
            this.openSequenceToolStripMenuItem.Text = "Open &BMP Sequence";
            this.openSequenceToolStripMenuItem.Click += new System.EventHandler(this.openSequenceToolStripMenuItem_Click);
            // 
            // openAnimatedGIFToolStripMenuItem
            // 
            this.openAnimatedGIFToolStripMenuItem.Name = "openAnimatedGIFToolStripMenuItem";
            this.openAnimatedGIFToolStripMenuItem.ShortcutKeys = ((System.Windows.Forms.Keys)((System.Windows.Forms.Keys.Alt | System.Windows.Forms.Keys.A)));
            this.openAnimatedGIFToolStripMenuItem.Size = new System.Drawing.Size(273, 22);
            this.openAnimatedGIFToolStripMenuItem.Text = "Open &Animated GIF";
            this.openAnimatedGIFToolStripMenuItem.Click += new System.EventHandler(this.openAnimatedGIFToolStripMenuItem_Click);
            // 
            // openFogImageSequenceFileToolStripMenuItem
            // 
            this.openFogImageSequenceFileToolStripMenuItem.Name = "openFogImageSequenceFileToolStripMenuItem";
            this.openFogImageSequenceFileToolStripMenuItem.ShortcutKeys = ((System.Windows.Forms.Keys)((System.Windows.Forms.Keys.Alt | System.Windows.Forms.Keys.F)));
            this.openFogImageSequenceFileToolStripMenuItem.Size = new System.Drawing.Size(273, 22);
            this.openFogImageSequenceFileToolStripMenuItem.Text = "Open Fog Image Sequence File";
            this.openFogImageSequenceFileToolStripMenuItem.Click += new System.EventHandler(this.openFogImageSequenceFileToolStripMenuItem_Click);
            // 
            // toolStripMenuItem5
            // 
            this.toolStripMenuItem5.Name = "toolStripMenuItem5";
            this.toolStripMenuItem5.Size = new System.Drawing.Size(270, 6);
            // 
            // recodeToAnimatedGIFToolStripMenuItem
            // 
            this.recodeToAnimatedGIFToolStripMenuItem.Enabled = false;
            this.recodeToAnimatedGIFToolStripMenuItem.Name = "recodeToAnimatedGIFToolStripMenuItem";
            this.recodeToAnimatedGIFToolStripMenuItem.ShortcutKeys = ((System.Windows.Forms.Keys)((System.Windows.Forms.Keys.Alt | System.Windows.Forms.Keys.R)));
            this.recodeToAnimatedGIFToolStripMenuItem.Size = new System.Drawing.Size(273, 22);
            this.recodeToAnimatedGIFToolStripMenuItem.Text = "&Recode to Animated GIF";
            this.recodeToAnimatedGIFToolStripMenuItem.Click += new System.EventHandler(this.recodeToAnimatedGIFToolStripMenuItem_Click);
            // 
            // extractBMPImagesToolStripMenuItem
            // 
            this.extractBMPImagesToolStripMenuItem.Enabled = false;
            this.extractBMPImagesToolStripMenuItem.Name = "extractBMPImagesToolStripMenuItem";
            this.extractBMPImagesToolStripMenuItem.ShortcutKeys = ((System.Windows.Forms.Keys)((System.Windows.Forms.Keys.Alt | System.Windows.Forms.Keys.X)));
            this.extractBMPImagesToolStripMenuItem.Size = new System.Drawing.Size(273, 22);
            this.extractBMPImagesToolStripMenuItem.Text = "E&xtract BMP Images";
            this.extractBMPImagesToolStripMenuItem.Click += new System.EventHandler(this.extractBMPImagesToolStripMenuItem_Click);
            // 
            // viewToolStripMenuItem
            // 
            this.viewToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.zoomInToolStripMenuItem,
            this.zoomOutToolStripMenuItem,
            this.toolStripMenuItem2,
            this.frameUpToolStripMenuItem,
            this.frameRevToolStripMenuItem});
            this.viewToolStripMenuItem.Name = "viewToolStripMenuItem";
            this.viewToolStripMenuItem.Size = new System.Drawing.Size(44, 20);
            this.viewToolStripMenuItem.Text = "&View";
            // 
            // zoomInToolStripMenuItem
            // 
            this.zoomInToolStripMenuItem.Name = "zoomInToolStripMenuItem";
            this.zoomInToolStripMenuItem.ShortcutKeys = ((System.Windows.Forms.Keys)((System.Windows.Forms.Keys.Control | System.Windows.Forms.Keys.I)));
            this.zoomInToolStripMenuItem.Size = new System.Drawing.Size(215, 22);
            this.zoomInToolStripMenuItem.Text = "Zoom &In";
            this.zoomInToolStripMenuItem.Click += new System.EventHandler(this.zoomInToolStripMenuItem_Click);
            // 
            // zoomOutToolStripMenuItem
            // 
            this.zoomOutToolStripMenuItem.Name = "zoomOutToolStripMenuItem";
            this.zoomOutToolStripMenuItem.ShortcutKeys = ((System.Windows.Forms.Keys)((System.Windows.Forms.Keys.Control | System.Windows.Forms.Keys.O)));
            this.zoomOutToolStripMenuItem.Size = new System.Drawing.Size(215, 22);
            this.zoomOutToolStripMenuItem.Text = "Zoom &Out";
            this.zoomOutToolStripMenuItem.Click += new System.EventHandler(this.zoomOutToolStripMenuItem_Click);
            // 
            // toolStripMenuItem2
            // 
            this.toolStripMenuItem2.Name = "toolStripMenuItem2";
            this.toolStripMenuItem2.Size = new System.Drawing.Size(212, 6);
            // 
            // frameUpToolStripMenuItem
            // 
            this.frameUpToolStripMenuItem.Name = "frameUpToolStripMenuItem";
            this.frameUpToolStripMenuItem.ShortcutKeys = ((System.Windows.Forms.Keys)((System.Windows.Forms.Keys.Control | System.Windows.Forms.Keys.Right)));
            this.frameUpToolStripMenuItem.Size = new System.Drawing.Size(215, 22);
            this.frameUpToolStripMenuItem.Text = "&Frame Forward";
            this.frameUpToolStripMenuItem.Click += new System.EventHandler(this.frameUpToolStripMenuItem_Click);
            // 
            // frameRevToolStripMenuItem
            // 
            this.frameRevToolStripMenuItem.Name = "frameRevToolStripMenuItem";
            this.frameRevToolStripMenuItem.ShortcutKeys = ((System.Windows.Forms.Keys)((System.Windows.Forms.Keys.Control | System.Windows.Forms.Keys.Left)));
            this.frameRevToolStripMenuItem.Size = new System.Drawing.Size(215, 22);
            this.frameRevToolStripMenuItem.Text = "&Frame Reverse";
            this.frameRevToolStripMenuItem.Click += new System.EventHandler(this.frameRevToolStripMenuItem_Click);
            // 
            // settingsToolStripMenuItem
            // 
            this.settingsToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.applySySalColorTToolStripMenuItem,
            this.sharpenImageToolStripMenuItem,
            this.toolStripMenuItem4,
            this.unlockRegionToolStripMenuItem,
            this.toolStripMenuItem1,
            this.reverseToolStripMenuItem,
            this.toolStripMenuItem3,
            this.commentToolStripMenuItem});
            this.settingsToolStripMenuItem.Name = "settingsToolStripMenuItem";
            this.settingsToolStripMenuItem.Size = new System.Drawing.Size(61, 20);
            this.settingsToolStripMenuItem.Text = "&Settings";
            // 
            // applySySalColorTToolStripMenuItem
            // 
            this.applySySalColorTToolStripMenuItem.Name = "applySySalColorTToolStripMenuItem";
            this.applySySalColorTToolStripMenuItem.Size = new System.Drawing.Size(199, 22);
            this.applySySalColorTToolStripMenuItem.Text = "Apply SySal Color Table";
            this.applySySalColorTToolStripMenuItem.Click += new System.EventHandler(this.applySySalColorTToolStripMenuItem_Click);
            // 
            // sharpenImageToolStripMenuItem
            // 
            this.sharpenImageToolStripMenuItem.Name = "sharpenImageToolStripMenuItem";
            this.sharpenImageToolStripMenuItem.ShortcutKeys = ((System.Windows.Forms.Keys)((System.Windows.Forms.Keys.Control | System.Windows.Forms.Keys.H)));
            this.sharpenImageToolStripMenuItem.Size = new System.Drawing.Size(199, 22);
            this.sharpenImageToolStripMenuItem.Text = "S&harpen Image";
            this.sharpenImageToolStripMenuItem.Click += new System.EventHandler(this.sharpenImageToolStripMenuItem_Click);
            // 
            // toolStripMenuItem4
            // 
            this.toolStripMenuItem4.Name = "toolStripMenuItem4";
            this.toolStripMenuItem4.Size = new System.Drawing.Size(196, 6);
            // 
            // unlockRegionToolStripMenuItem
            // 
            this.unlockRegionToolStripMenuItem.Name = "unlockRegionToolStripMenuItem";
            this.unlockRegionToolStripMenuItem.ShortcutKeys = ((System.Windows.Forms.Keys)((System.Windows.Forms.Keys.Control | System.Windows.Forms.Keys.U)));
            this.unlockRegionToolStripMenuItem.Size = new System.Drawing.Size(199, 22);
            this.unlockRegionToolStripMenuItem.Text = "&Unlock Region";
            this.unlockRegionToolStripMenuItem.Click += new System.EventHandler(this.unlockRegionToolStripMenuItem_Click);
            // 
            // toolStripMenuItem1
            // 
            this.toolStripMenuItem1.Name = "toolStripMenuItem1";
            this.toolStripMenuItem1.Size = new System.Drawing.Size(196, 6);
            // 
            // reverseToolStripMenuItem
            // 
            this.reverseToolStripMenuItem.Enabled = false;
            this.reverseToolStripMenuItem.Name = "reverseToolStripMenuItem";
            this.reverseToolStripMenuItem.ShortcutKeys = ((System.Windows.Forms.Keys)((System.Windows.Forms.Keys.Control | System.Windows.Forms.Keys.R)));
            this.reverseToolStripMenuItem.Size = new System.Drawing.Size(199, 22);
            this.reverseToolStripMenuItem.Text = "&Reverse Order";
            this.reverseToolStripMenuItem.Click += new System.EventHandler(this.commentToolStripMenuItem_Click);
            // 
            // toolStripMenuItem3
            // 
            this.toolStripMenuItem3.Name = "toolStripMenuItem3";
            this.toolStripMenuItem3.Size = new System.Drawing.Size(196, 6);
            // 
            // commentToolStripMenuItem
            // 
            this.commentToolStripMenuItem.Name = "commentToolStripMenuItem";
            this.commentToolStripMenuItem.ShortcutKeys = ((System.Windows.Forms.Keys)((System.Windows.Forms.Keys.Control | System.Windows.Forms.Keys.C)));
            this.commentToolStripMenuItem.Size = new System.Drawing.Size(199, 22);
            this.commentToolStripMenuItem.Text = "&Comment";
            this.commentToolStripMenuItem.Click += new System.EventHandler(this.commentToolStripMenuItem_Click_1);
            // 
            // pictureBox1
            // 
            this.pictureBox1.BackColor = System.Drawing.Color.DarkSeaGreen;
            this.pictureBox1.BorderStyle = System.Windows.Forms.BorderStyle.Fixed3D;
            this.pictureBox1.Cursor = System.Windows.Forms.Cursors.Cross;
            this.pictureBox1.Location = new System.Drawing.Point(3, 3);
            this.pictureBox1.Name = "pictureBox1";
            this.pictureBox1.Size = new System.Drawing.Size(544, 263);
            this.pictureBox1.TabIndex = 1;
            this.pictureBox1.TabStop = false;
            this.pictureBox1.MouseLeave += new System.EventHandler(this.OnMouseLeave);
            this.pictureBox1.MouseMove += new System.Windows.Forms.MouseEventHandler(this.OnMouseMove);
            this.pictureBox1.MouseDown += new System.Windows.Forms.MouseEventHandler(this.OnMouseDown);
            this.pictureBox1.Paint += new System.Windows.Forms.PaintEventHandler(this.OnPaint);
            this.pictureBox1.MouseUp += new System.Windows.Forms.MouseEventHandler(this.OnMouseUp);
            this.pictureBox1.MouseEnter += new System.EventHandler(this.OnMouseEnter);
            // 
            // statusStrip1
            // 
            this.statusStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.statusImageN,
            this.statusImageSize,
            this.statusXY});
            this.statusStrip1.Location = new System.Drawing.Point(0, 308);
            this.statusStrip1.Name = "statusStrip1";
            this.statusStrip1.Size = new System.Drawing.Size(761, 22);
            this.statusStrip1.TabIndex = 2;
            this.statusStrip1.Text = "statusStrip1";
            // 
            // statusImageN
            // 
            this.statusImageN.AutoSize = false;
            this.statusImageN.Name = "statusImageN";
            this.statusImageN.Size = new System.Drawing.Size(120, 17);
            this.statusImageN.Text = "Image #";
            this.statusImageN.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // statusImageSize
            // 
            this.statusImageSize.AutoSize = false;
            this.statusImageSize.BorderSides = System.Windows.Forms.ToolStripStatusLabelBorderSides.Left;
            this.statusImageSize.Name = "statusImageSize";
            this.statusImageSize.Size = new System.Drawing.Size(120, 17);
            this.statusImageSize.Text = "WxH =";
            this.statusImageSize.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // statusXY
            // 
            this.statusXY.AutoSize = false;
            this.statusXY.BorderSides = ((System.Windows.Forms.ToolStripStatusLabelBorderSides)((System.Windows.Forms.ToolStripStatusLabelBorderSides.Left | System.Windows.Forms.ToolStripStatusLabelBorderSides.Right)));
            this.statusXY.Name = "statusXY";
            this.statusXY.Size = new System.Drawing.Size(200, 17);
            this.statusXY.Text = "X;Y;G =";
            this.statusXY.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // panel1
            // 
            this.panel1.AutoScroll = true;
            this.panel1.Controls.Add(this.pictureBox1);
            this.panel1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.panel1.Location = new System.Drawing.Point(0, 24);
            this.panel1.Name = "panel1";
            this.panel1.Size = new System.Drawing.Size(761, 284);
            this.panel1.TabIndex = 3;
            // 
            // MainForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(761, 330);
            this.Controls.Add(this.panel1);
            this.Controls.Add(this.statusStrip1);
            this.Controls.Add(this.menuStrip1);
            this.MainMenuStrip = this.menuStrip1;
            this.Name = "MainForm";
            this.Text = "SySal BMP to Animated GIF Recoder";
            this.Load += new System.EventHandler(this.OnLoad);
            this.Resize += new System.EventHandler(this.OnResize);
            this.menuStrip1.ResumeLayout(false);
            this.menuStrip1.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).EndInit();
            this.statusStrip1.ResumeLayout(false);
            this.statusStrip1.PerformLayout();
            this.panel1.ResumeLayout(false);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.MenuStrip menuStrip1;
        private System.Windows.Forms.ToolStripMenuItem fileToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem openSequenceToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem recodeToAnimatedGIFToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem viewToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem zoomInToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem zoomOutToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem settingsToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem unlockRegionToolStripMenuItem;
        private System.Windows.Forms.ToolStripSeparator toolStripMenuItem1;
        private System.Windows.Forms.ToolStripMenuItem reverseToolStripMenuItem;
        private System.Windows.Forms.PictureBox pictureBox1;
        private System.Windows.Forms.ToolStripSeparator toolStripMenuItem2;
        private System.Windows.Forms.ToolStripMenuItem frameUpToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem frameRevToolStripMenuItem;
        private System.Windows.Forms.ToolStripSeparator toolStripMenuItem3;
        private System.Windows.Forms.ToolStripMenuItem commentToolStripMenuItem;
        private System.Windows.Forms.StatusStrip statusStrip1;
        private System.Windows.Forms.ToolStripStatusLabel statusImageN;
        private System.Windows.Forms.ToolStripStatusLabel statusImageSize;
        private System.Windows.Forms.ToolStripStatusLabel statusXY;
        private System.Windows.Forms.ToolStripMenuItem applySySalColorTToolStripMenuItem;
        private System.Windows.Forms.ToolStripSeparator toolStripMenuItem4;
        private System.Windows.Forms.Panel panel1;
        private System.Windows.Forms.ToolStripMenuItem openAnimatedGIFToolStripMenuItem;
        private System.Windows.Forms.ToolStripSeparator toolStripMenuItem5;
        private System.Windows.Forms.ToolStripMenuItem extractBMPImagesToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem openFogImageSequenceFileToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem sharpenImageToolStripMenuItem;
    }
}

