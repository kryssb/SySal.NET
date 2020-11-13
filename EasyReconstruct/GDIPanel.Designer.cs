namespace SySal.Executables.EasyReconstruct
{
    partial class GDIPanel
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
            this.SuspendLayout();
            // 
            // gdiDisplay1
            // 
            this.gdiDisplay1.Alpha = 0.49803921568627452;
            this.gdiDisplay1.AutoRender = false;
            this.gdiDisplay1.BackColor = System.Drawing.Color.Black;
            this.gdiDisplay1.BorderWidth = 1;
            this.gdiDisplay1.ClickSelect = null;
            this.gdiDisplay1.Distance = 100;
            this.gdiDisplay1.DoubleClickSelect = null;
            this.gdiDisplay1.Infinity = false;
            this.gdiDisplay1.LabelFontName = "Arial";
            this.gdiDisplay1.LabelFontSize = 12;
            this.gdiDisplay1.LineWidth = 1;
            this.gdiDisplay1.Location = new System.Drawing.Point(12, 12);
            this.gdiDisplay1.MouseMode = GDI3D.Control.MouseMotion.Rotate;
            this.gdiDisplay1.MouseMultiplier = 0.01;
            this.gdiDisplay1.Name = "gdiDisplay1";
            this.gdiDisplay1.NextClickSetsCenter = false;
            this.gdiDisplay1.PointSize = 5;
            this.gdiDisplay1.Size = new System.Drawing.Size(400, 400);
            this.gdiDisplay1.TabIndex = 9;
            this.gdiDisplay1.Zoom = 2000;
            // 
            // GDIPanel
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(426, 424);
            this.Controls.Add(this.gdiDisplay1);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.Fixed3D;
            this.Name = "GDIPanel";
            this.Text = "DisplayPanel";
            this.FormClosing += new System.Windows.Forms.FormClosingEventHandler(this.OnClose);
            this.ResumeLayout(false);

        }

        #endregion

        public GDI3D.Control.GDIDisplay gdiDisplay1;
    }
}