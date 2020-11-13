namespace SySal.Processing.MapMerge
{
    partial class EditConfigForm
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
            this.label1 = new System.Windows.Forms.Label();
            this.txtPosTolerance = new System.Windows.Forms.TextBox();
            this.btnOK = new System.Windows.Forms.Button();
            this.btnCancel = new System.Windows.Forms.Button();
            this.txtSlopeTolerance = new System.Windows.Forms.TextBox();
            this.label2 = new System.Windows.Forms.Label();
            this.txtMaxOffset = new System.Windows.Forms.TextBox();
            this.label3 = new System.Windows.Forms.Label();
            this.txtMapSize = new System.Windows.Forms.TextBox();
            this.label4 = new System.Windows.Forms.Label();
            this.txtMinMatches = new System.Windows.Forms.TextBox();
            this.label5 = new System.Windows.Forms.Label();
            this.chkFavorSpeed = new System.Windows.Forms.CheckBox();
            this.SuspendLayout();
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(12, 17);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(91, 13);
            this.label1.TabIndex = 0;
            this.label1.Text = "Position tolerance";
            // 
            // txtPosTolerance
            // 
            this.txtPosTolerance.Location = new System.Drawing.Point(110, 14);
            this.txtPosTolerance.Name = "txtPosTolerance";
            this.txtPosTolerance.Size = new System.Drawing.Size(45, 20);
            this.txtPosTolerance.TabIndex = 1;
            // 
            // btnOK
            // 
            this.btnOK.Location = new System.Drawing.Point(270, 105);
            this.btnOK.Name = "btnOK";
            this.btnOK.Size = new System.Drawing.Size(70, 24);
            this.btnOK.TabIndex = 7;
            this.btnOK.Text = "OK";
            this.btnOK.UseVisualStyleBackColor = true;
            this.btnOK.Click += new System.EventHandler(this.btnOK_Click);
            // 
            // btnCancel
            // 
            this.btnCancel.Location = new System.Drawing.Point(15, 105);
            this.btnCancel.Name = "btnCancel";
            this.btnCancel.Size = new System.Drawing.Size(70, 24);
            this.btnCancel.TabIndex = 8;
            this.btnCancel.Text = "Cancel";
            this.btnCancel.UseVisualStyleBackColor = true;
            this.btnCancel.Click += new System.EventHandler(this.btnCancel_Click);
            // 
            // txtSlopeTolerance
            // 
            this.txtSlopeTolerance.Location = new System.Drawing.Point(110, 40);
            this.txtSlopeTolerance.Name = "txtSlopeTolerance";
            this.txtSlopeTolerance.Size = new System.Drawing.Size(45, 20);
            this.txtSlopeTolerance.TabIndex = 10;
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(12, 43);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(81, 13);
            this.label2.TabIndex = 9;
            this.label2.Text = "Slope tolerance";
            // 
            // txtMaxOffset
            // 
            this.txtMaxOffset.Location = new System.Drawing.Point(110, 66);
            this.txtMaxOffset.Name = "txtMaxOffset";
            this.txtMaxOffset.Size = new System.Drawing.Size(45, 20);
            this.txtMaxOffset.TabIndex = 12;
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(12, 69);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(80, 13);
            this.label3.TabIndex = 11;
            this.label3.Text = "Maximum offset";
            // 
            // txtMapSize
            // 
            this.txtMapSize.Location = new System.Drawing.Point(295, 14);
            this.txtMapSize.Name = "txtMapSize";
            this.txtMapSize.Size = new System.Drawing.Size(45, 20);
            this.txtMapSize.TabIndex = 14;
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(170, 17);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(49, 13);
            this.label4.TabIndex = 13;
            this.label4.Text = "Map size";
            // 
            // txtMinMatches
            // 
            this.txtMinMatches.Location = new System.Drawing.Point(295, 40);
            this.txtMinMatches.Name = "txtMinMatches";
            this.txtMinMatches.Size = new System.Drawing.Size(45, 20);
            this.txtMinMatches.TabIndex = 16;
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(170, 43);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(67, 13);
            this.label5.TabIndex = 15;
            this.label5.Text = "Min matches";
            // 
            // chkFavorSpeed
            // 
            this.chkFavorSpeed.AutoSize = true;
            this.chkFavorSpeed.Location = new System.Drawing.Point(173, 68);
            this.chkFavorSpeed.Name = "chkFavorSpeed";
            this.chkFavorSpeed.Size = new System.Drawing.Size(156, 17);
            this.chkFavorSpeed.TabIndex = 17;
            this.chkFavorSpeed.Text = "Favor speed over accuracy";
            this.chkFavorSpeed.UseVisualStyleBackColor = true;
            // 
            // EditConfigForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(352, 142);
            this.Controls.Add(this.chkFavorSpeed);
            this.Controls.Add(this.txtMinMatches);
            this.Controls.Add(this.label5);
            this.Controls.Add(this.txtMapSize);
            this.Controls.Add(this.label4);
            this.Controls.Add(this.txtMaxOffset);
            this.Controls.Add(this.label3);
            this.Controls.Add(this.txtSlopeTolerance);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.btnCancel);
            this.Controls.Add(this.btnOK);
            this.Controls.Add(this.txtPosTolerance);
            this.Controls.Add(this.label1);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
            this.Name = "EditConfigForm";
            this.Text = "Edit Configuration for Map Merging";
            this.Load += new System.EventHandler(this.OnLoad);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TextBox txtPosTolerance;
        private System.Windows.Forms.Button btnOK;
        private System.Windows.Forms.Button btnCancel;
        private System.Windows.Forms.TextBox txtSlopeTolerance;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.TextBox txtMaxOffset;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.TextBox txtMapSize;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.TextBox txtMinMatches;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.CheckBox chkFavorSpeed;
    }
}