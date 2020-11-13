namespace SySal.Executables.EasyReconstruct
{
    partial class MomentumFitForm
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
            this.btnCompute = new System.Windows.Forms.Button();
            this.label2 = new System.Windows.Forms.Label();
            this.txtResults = new System.Windows.Forms.TextBox();
            this.btnExport = new System.Windows.Forms.Button();
            this.btnDumpLikelihood = new System.Windows.Forms.Button();
            this.btnRemove = new System.Windows.Forms.Button();
            this.clSlopeSets = new System.Windows.Forms.CheckedListBox();
            this.pbLkDisplay = new System.Windows.Forms.PictureBox();
            ((System.ComponentModel.ISupportInitialize)(this.pbLkDisplay)).BeginInit();
            this.SuspendLayout();
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(12, 9);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(58, 13);
            this.label1.TabIndex = 1;
            this.label1.Text = "Slope Sets";
            // 
            // btnCompute
            // 
            this.btnCompute.Location = new System.Drawing.Point(12, 199);
            this.btnCompute.Name = "btnCompute";
            this.btnCompute.Size = new System.Drawing.Size(123, 23);
            this.btnCompute.TabIndex = 2;
            this.btnCompute.Text = "Compute";
            this.btnCompute.UseVisualStyleBackColor = true;
            this.btnCompute.Click += new System.EventHandler(this.btnCompute_Click);
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(9, 240);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(79, 13);
            this.label2.TabIndex = 3;
            this.label2.Text = "P; Intervals; CL";
            // 
            // txtResults
            // 
            this.txtResults.Location = new System.Drawing.Point(141, 237);
            this.txtResults.Name = "txtResults";
            this.txtResults.ReadOnly = true;
            this.txtResults.Size = new System.Drawing.Size(328, 20);
            this.txtResults.TabIndex = 4;
            // 
            // btnExport
            // 
            this.btnExport.Location = new System.Drawing.Point(484, 235);
            this.btnExport.Name = "btnExport";
            this.btnExport.Size = new System.Drawing.Size(145, 23);
            this.btnExport.TabIndex = 5;
            this.btnExport.Text = "Update track momenta";
            this.btnExport.UseVisualStyleBackColor = true;
            this.btnExport.Click += new System.EventHandler(this.btnExport_Click);
            // 
            // btnDumpLikelihood
            // 
            this.btnDumpLikelihood.Location = new System.Drawing.Point(12, 170);
            this.btnDumpLikelihood.Name = "btnDumpLikelihood";
            this.btnDumpLikelihood.Size = new System.Drawing.Size(123, 23);
            this.btnDumpLikelihood.TabIndex = 6;
            this.btnDumpLikelihood.Text = "Dump Likelihood";
            this.btnDumpLikelihood.UseVisualStyleBackColor = true;
            this.btnDumpLikelihood.Click += new System.EventHandler(this.btnDumpLikelihood_Click);
            // 
            // btnRemove
            // 
            this.btnRemove.Location = new System.Drawing.Point(12, 141);
            this.btnRemove.Name = "btnRemove";
            this.btnRemove.Size = new System.Drawing.Size(123, 23);
            this.btnRemove.TabIndex = 7;
            this.btnRemove.Text = "Remove";
            this.btnRemove.UseVisualStyleBackColor = true;
            this.btnRemove.Click += new System.EventHandler(this.btnRemove_Click);
            // 
            // clSlopeSets
            // 
            this.clSlopeSets.FormattingEnabled = true;
            this.clSlopeSets.Location = new System.Drawing.Point(12, 25);
            this.clSlopeSets.Name = "clSlopeSets";
            this.clSlopeSets.Size = new System.Drawing.Size(123, 109);
            this.clSlopeSets.TabIndex = 8;
            this.clSlopeSets.SelectedIndexChanged += new System.EventHandler(this.OnSelectedSlopeSetChanged);
            // 
            // pbLkDisplay
            // 
            this.pbLkDisplay.Location = new System.Drawing.Point(141, 10);
            this.pbLkDisplay.Name = "pbLkDisplay";
            this.pbLkDisplay.Size = new System.Drawing.Size(488, 211);
            this.pbLkDisplay.TabIndex = 9;
            this.pbLkDisplay.TabStop = false;
            // 
            // MomentumFitForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(641, 265);
            this.Controls.Add(this.pbLkDisplay);
            this.Controls.Add(this.clSlopeSets);
            this.Controls.Add(this.btnRemove);
            this.Controls.Add(this.btnDumpLikelihood);
            this.Controls.Add(this.btnExport);
            this.Controls.Add(this.txtResults);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.btnCompute);
            this.Controls.Add(this.label1);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.Fixed3D;
            this.Name = "MomentumFitForm";
            this.Text = "Momentum Fit";
            this.Load += new System.EventHandler(this.OnLoad);
            this.FormClosed += new System.Windows.Forms.FormClosedEventHandler(this.OnClose);
            ((System.ComponentModel.ISupportInitialize)(this.pbLkDisplay)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Button btnCompute;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.TextBox txtResults;
        private System.Windows.Forms.Button btnExport;
        private System.Windows.Forms.Button btnDumpLikelihood;
        private System.Windows.Forms.Button btnRemove;
        private System.Windows.Forms.CheckedListBox clSlopeSets;
        private System.Windows.Forms.PictureBox pbLkDisplay;
    }
}