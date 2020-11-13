namespace SySal.Executables.SySal2GIF
{
    partial class TimerPieForm
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
            this.StopButton = new System.Windows.Forms.Button();
            this.pieBox = new System.Windows.Forms.PictureBox();
            ((System.ComponentModel.ISupportInitialize)(this.pieBox)).BeginInit();
            this.SuspendLayout();
            // 
            // StopButton
            // 
            this.StopButton.Location = new System.Drawing.Point(12, 72);
            this.StopButton.Name = "StopButton";
            this.StopButton.Size = new System.Drawing.Size(58, 24);
            this.StopButton.TabIndex = 0;
            this.StopButton.Text = "Cancel";
            this.StopButton.UseVisualStyleBackColor = true;
            this.StopButton.Click += new System.EventHandler(this.StopButton_Click);
            // 
            // pieBox
            // 
            this.pieBox.Location = new System.Drawing.Point(12, 7);
            this.pieBox.Name = "pieBox";
            this.pieBox.Size = new System.Drawing.Size(58, 59);
            this.pieBox.TabIndex = 1;
            this.pieBox.TabStop = false;
            // 
            // TimerPieForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(82, 103);
            this.ControlBox = false;
            this.Controls.Add(this.pieBox);
            this.Controls.Add(this.StopButton);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedSingle;
            this.MaximizeBox = false;
            this.MinimizeBox = false;
            this.Name = "TimerPieForm";
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterScreen;
            this.TopMost = true;
            ((System.ComponentModel.ISupportInitialize)(this.pieBox)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.Button StopButton;
        private System.Windows.Forms.PictureBox pieBox;
    }
}