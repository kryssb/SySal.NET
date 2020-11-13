namespace SySal.SySalNExTControls
{
    partial class SySalDialog
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
            this.TitleLabel = new System.Windows.Forms.Label();
            this.sysBtnClose = new SySal.SySalNExTControls.SySalButton();
            this.SuspendLayout();
            // 
            // TitleLabel
            // 
            this.TitleLabel.Dock = System.Windows.Forms.DockStyle.Top;
            this.TitleLabel.Font = new System.Drawing.Font("Segoe UI", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.TitleLabel.ForeColor = System.Drawing.Color.Navy;
            this.TitleLabel.Location = new System.Drawing.Point(0, 0);
            this.TitleLabel.Name = "TitleLabel";
            this.TitleLabel.Size = new System.Drawing.Size(426, 21);
            this.TitleLabel.TabIndex = 0;
            this.TitleLabel.Text = "Dialog";
            this.TitleLabel.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            this.TitleLabel.MouseMove += new System.Windows.Forms.MouseEventHandler(this.OnDialogMouseMove);
            this.TitleLabel.MouseDown += new System.Windows.Forms.MouseEventHandler(this.OnDialogMouseDown);
            this.TitleLabel.MouseUp += new System.Windows.Forms.MouseEventHandler(this.OnDialogMouseUp);
            // 
            // sysBtnClose
            // 
            this.sysBtnClose.AutoSize = true;
            this.sysBtnClose.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.sysBtnClose.BackColor = System.Drawing.Color.Transparent;
            this.sysBtnClose.BackgroundImage = global::SySal.SySalNExTControls.SySalResources.close1;
            this.sysBtnClose.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
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
            // SySalDialog
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(9F, 21F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.BackColor = System.Drawing.Color.White;
            this.ClientSize = new System.Drawing.Size(426, 422);
            this.ControlBox = false;
            this.Controls.Add(this.sysBtnClose);
            this.Controls.Add(this.TitleLabel);
            this.DoubleBuffered = true;
            this.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.ForeColor = System.Drawing.Color.Navy;
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
            this.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.Name = "SySalDialog";
            this.ShowIcon = false;
            this.ShowInTaskbar = false;
            this.MouseUp += new System.Windows.Forms.MouseEventHandler(this.OnDialogMouseUp);
            this.Paint += new System.Windows.Forms.PaintEventHandler(this.OnPaint);
            this.MouseDown += new System.Windows.Forms.MouseEventHandler(this.OnDialogMouseDown);
            this.Resize += new System.EventHandler(this.OnResize);
            this.MouseMove += new System.Windows.Forms.MouseEventHandler(this.OnDialogMouseMove);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Label TitleLabel;
        private SySalButton sysBtnClose;
    }
}