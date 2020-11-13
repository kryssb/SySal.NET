namespace SySal.Executables.NExTScanner
{
    partial class LongOperationForm
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
            this.pbProgress = new SySal.SySalNExTControls.SySalProgressBar();
            this.btnStop = new SySal.SySalNExTControls.SySalButton();
            this.SuspendLayout();
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
            this.pbProgress.Location = new System.Drawing.Point(12, 40);
            this.pbProgress.Name = "pbProgress";
            this.pbProgress.Size = new System.Drawing.Size(349, 19);
            this.pbProgress.TabIndex = 3;
            // 
            // btnStop
            // 
            this.btnStop.AutoSize = true;
            this.btnStop.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnStop.BackColor = System.Drawing.Color.Transparent;
            this.btnStop.FocusedColor = System.Drawing.Color.Navy;
            this.btnStop.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.btnStop.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnStop.Location = new System.Drawing.Point(166, 68);
            this.btnStop.Margin = new System.Windows.Forms.Padding(6);
            this.btnStop.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnStop.Name = "btnStop";
            this.btnStop.Size = new System.Drawing.Size(41, 25);
            this.btnStop.TabIndex = 4;
            this.btnStop.Text = "Stop";
            this.btnStop.Click += new System.EventHandler(this.btnStop_Click);
            // 
            // LongOperationForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(9F, 21F);
            this.ClientSize = new System.Drawing.Size(373, 104);
            this.Controls.Add(this.pbProgress);
            this.Controls.Add(this.btnStop);
            this.DialogCaption = "Long Operation";
            this.Name = "LongOperationForm";
            this.NoCloseButton = true;
            this.TopMost = true;
            this.Load += new System.EventHandler(this.OnLoad);
            this.Controls.SetChildIndex(this.btnStop, 0);
            this.Controls.SetChildIndex(this.pbProgress, 0);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private SySal.SySalNExTControls.SySalProgressBar pbProgress;
        private SySal.SySalNExTControls.SySalButton btnStop;
    }
}
