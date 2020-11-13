namespace SySal.Executables.X3LView
{
    partial class QuickDisplayChoose
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
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.comboCS = new System.Windows.Forms.ComboBox();
            this.label1 = new System.Windows.Forms.Label();
            this.OKScanDisplayButton = new System.Windows.Forms.Button();
            this.CancelBtn = new System.Windows.Forms.Button();
            this.groupBox1.SuspendLayout();
            this.SuspendLayout();
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.OKScanDisplayButton);
            this.groupBox1.Controls.Add(this.label1);
            this.groupBox1.Controls.Add(this.comboCS);
            this.groupBox1.Location = new System.Drawing.Point(12, 12);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(417, 51);
            this.groupBox1.TabIndex = 0;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Scan Display";
            // 
            // comboCS
            // 
            this.comboCS.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.comboCS.FormattingEnabled = true;
            this.comboCS.Location = new System.Drawing.Point(148, 19);
            this.comboCS.Name = "comboCS";
            this.comboCS.Size = new System.Drawing.Size(146, 21);
            this.comboCS.TabIndex = 0;
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(14, 23);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(73, 13);
            this.label1.TabIndex = 1;
            this.label1.Text = "CS Doublet Id";
            // 
            // OKScanDisplayButton
            // 
            this.OKScanDisplayButton.Location = new System.Drawing.Point(319, 19);
            this.OKScanDisplayButton.Name = "OKScanDisplayButton";
            this.OKScanDisplayButton.Size = new System.Drawing.Size(78, 20);
            this.OKScanDisplayButton.TabIndex = 2;
            this.OKScanDisplayButton.Text = "OK";
            this.OKScanDisplayButton.UseVisualStyleBackColor = true;
            this.OKScanDisplayButton.Click += new System.EventHandler(this.OKScanDisplayButton_Click);
            // 
            // CancelBtn
            // 
            this.CancelBtn.DialogResult = System.Windows.Forms.DialogResult.Cancel;
            this.CancelBtn.Location = new System.Drawing.Point(331, 87);
            this.CancelBtn.Name = "CancelBtn";
            this.CancelBtn.Size = new System.Drawing.Size(78, 20);
            this.CancelBtn.TabIndex = 3;
            this.CancelBtn.Text = "Cancel";
            this.CancelBtn.UseVisualStyleBackColor = true;
            // 
            // QuickDisplayChoose
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.CancelButton = this.CancelBtn;
            this.ClientSize = new System.Drawing.Size(441, 119);
            this.Controls.Add(this.CancelBtn);
            this.Controls.Add(this.groupBox1);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedToolWindow;
            this.Name = "QuickDisplayChoose";
            this.Text = "Choose Quick Display";
            this.Load += new System.EventHandler(this.OnLoad);
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.Button OKScanDisplayButton;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.ComboBox comboCS;
        private System.Windows.Forms.Button CancelBtn;
    }
}