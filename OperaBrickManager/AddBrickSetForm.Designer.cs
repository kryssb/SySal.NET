namespace OperaBrickManager
{
    partial class AddBrickSetForm
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
            this.txtSetName = new System.Windows.Forms.TextBox();
            this.txtTablespaceExt = new System.Windows.Forms.TextBox();
            this.label2 = new System.Windows.Forms.Label();
            this.txtRangeMin = new System.Windows.Forms.TextBox();
            this.label3 = new System.Windows.Forms.Label();
            this.txtRangeMax = new System.Windows.Forms.TextBox();
            this.label4 = new System.Windows.Forms.Label();
            this.txtDefaultId = new System.Windows.Forms.TextBox();
            this.label5 = new System.Windows.Forms.Label();
            this.btnCancel = new System.Windows.Forms.Button();
            this.btnOK = new System.Windows.Forms.Button();
            this.SuspendLayout();
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(12, 15);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(81, 13);
            this.label1.TabIndex = 0;
            this.label1.Text = "Brick Set Name";
            // 
            // txtSetName
            // 
            this.txtSetName.Location = new System.Drawing.Point(140, 12);
            this.txtSetName.Name = "txtSetName";
            this.txtSetName.Size = new System.Drawing.Size(104, 20);
            this.txtSetName.TabIndex = 1;
            this.txtSetName.Leave += new System.EventHandler(this.OnDataChanged);
            // 
            // txtTablespaceExt
            // 
            this.txtTablespaceExt.Location = new System.Drawing.Point(140, 38);
            this.txtTablespaceExt.MaxLength = 4;
            this.txtTablespaceExt.Name = "txtTablespaceExt";
            this.txtTablespaceExt.Size = new System.Drawing.Size(104, 20);
            this.txtTablespaceExt.TabIndex = 3;
            this.txtTablespaceExt.Leave += new System.EventHandler(this.OnDataChanged);
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(12, 41);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(112, 13);
            this.label2.TabIndex = 2;
            this.label2.Text = "Tablespace Extension";
            // 
            // txtRangeMin
            // 
            this.txtRangeMin.Location = new System.Drawing.Point(140, 70);
            this.txtRangeMin.Name = "txtRangeMin";
            this.txtRangeMin.Size = new System.Drawing.Size(104, 20);
            this.txtRangeMin.TabIndex = 5;
            this.txtRangeMin.Leave += new System.EventHandler(this.OnDataChanged);
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(12, 73);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(59, 13);
            this.label3.TabIndex = 4;
            this.label3.Text = "Range Min";
            // 
            // txtRangeMax
            // 
            this.txtRangeMax.Location = new System.Drawing.Point(140, 96);
            this.txtRangeMax.Name = "txtRangeMax";
            this.txtRangeMax.Size = new System.Drawing.Size(104, 20);
            this.txtRangeMax.TabIndex = 7;
            this.txtRangeMax.Leave += new System.EventHandler(this.OnDataChanged);
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(12, 99);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(62, 13);
            this.label4.TabIndex = 6;
            this.label4.Text = "Range Max";
            // 
            // txtDefaultId
            // 
            this.txtDefaultId.Location = new System.Drawing.Point(140, 133);
            this.txtDefaultId.Name = "txtDefaultId";
            this.txtDefaultId.Size = new System.Drawing.Size(104, 20);
            this.txtDefaultId.TabIndex = 9;
            this.txtDefaultId.Leave += new System.EventHandler(this.OnDataChanged);
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(12, 136);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(53, 13);
            this.label5.TabIndex = 8;
            this.label5.Text = "Default Id";
            // 
            // btnCancel
            // 
            this.btnCancel.Location = new System.Drawing.Point(195, 169);
            this.btnCancel.Name = "btnCancel";
            this.btnCancel.Size = new System.Drawing.Size(49, 24);
            this.btnCancel.TabIndex = 10;
            this.btnCancel.Text = "Cancel";
            this.btnCancel.UseVisualStyleBackColor = true;
            this.btnCancel.Click += new System.EventHandler(this.btnCancel_Click);
            // 
            // btnOK
            // 
            this.btnOK.Enabled = false;
            this.btnOK.Location = new System.Drawing.Point(12, 169);
            this.btnOK.Name = "btnOK";
            this.btnOK.Size = new System.Drawing.Size(49, 24);
            this.btnOK.TabIndex = 11;
            this.btnOK.Text = "OK";
            this.btnOK.UseVisualStyleBackColor = true;
            this.btnOK.Click += new System.EventHandler(this.btnOK_Click);
            // 
            // AddBrickSetForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(264, 209);
            this.Controls.Add(this.btnOK);
            this.Controls.Add(this.btnCancel);
            this.Controls.Add(this.txtDefaultId);
            this.Controls.Add(this.label5);
            this.Controls.Add(this.txtRangeMax);
            this.Controls.Add(this.label4);
            this.Controls.Add(this.txtRangeMin);
            this.Controls.Add(this.label3);
            this.Controls.Add(this.txtTablespaceExt);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.txtSetName);
            this.Controls.Add(this.label1);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
            this.Name = "AddBrickSetForm";
            this.Text = "AddBrickSetForm";
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TextBox txtSetName;
        private System.Windows.Forms.TextBox txtTablespaceExt;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.TextBox txtRangeMin;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.TextBox txtRangeMax;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.TextBox txtDefaultId;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.Button btnCancel;
        private System.Windows.Forms.Button btnOK;
    }
}