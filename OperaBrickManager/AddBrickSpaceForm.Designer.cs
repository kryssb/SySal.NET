namespace OperaBrickManager
{
    partial class AddBrickSpaceForm
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
            this.btnOK = new System.Windows.Forms.Button();
            this.btnCancel = new System.Windows.Forms.Button();
            this.lblNewId = new System.Windows.Forms.Label();
            this.txtNewID = new System.Windows.Forms.TextBox();
            this.lblBrickSet = new System.Windows.Forms.Label();
            this.txtBrickSet = new System.Windows.Forms.TextBox();
            this.SuspendLayout();
            // 
            // btnOK
            // 
            this.btnOK.Location = new System.Drawing.Point(12, 182);
            this.btnOK.Name = "btnOK";
            this.btnOK.Size = new System.Drawing.Size(88, 26);
            this.btnOK.TabIndex = 2;
            this.btnOK.Text = "OK";
            this.btnOK.UseVisualStyleBackColor = true;
            this.btnOK.Click += new System.EventHandler(this.btnOK_Click);
            // 
            // btnCancel
            // 
            this.btnCancel.Location = new System.Drawing.Point(136, 182);
            this.btnCancel.Name = "btnCancel";
            this.btnCancel.Size = new System.Drawing.Size(88, 26);
            this.btnCancel.TabIndex = 3;
            this.btnCancel.Text = "Cancel";
            this.btnCancel.UseVisualStyleBackColor = true;
            this.btnCancel.Click += new System.EventHandler(this.btnCancel_Click);
            // 
            // lblNewId
            // 
            this.lblNewId.AutoSize = true;
            this.lblNewId.Location = new System.Drawing.Point(29, 56);
            this.lblNewId.Name = "lblNewId";
            this.lblNewId.Size = new System.Drawing.Size(43, 13);
            this.lblNewId.TabIndex = 2;
            this.lblNewId.Text = "New ID";
            // 
            // txtNewID
            // 
            this.txtNewID.Location = new System.Drawing.Point(86, 56);
            this.txtNewID.Name = "txtNewID";
            this.txtNewID.Size = new System.Drawing.Size(106, 20);
            this.txtNewID.TabIndex = 1;
            // 
            // lblBrickSet
            // 
            this.lblBrickSet.AutoSize = true;
            this.lblBrickSet.Location = new System.Drawing.Point(27, 116);
            this.lblBrickSet.Name = "lblBrickSet";
            this.lblBrickSet.Size = new System.Drawing.Size(50, 13);
            this.lblBrickSet.TabIndex = 4;
            this.lblBrickSet.Text = "Brick Set";
            // 
            // txtBrickSet
            // 
            this.txtBrickSet.Location = new System.Drawing.Point(86, 109);
            this.txtBrickSet.Name = "txtBrickSet";
            this.txtBrickSet.ReadOnly = true;
            this.txtBrickSet.Size = new System.Drawing.Size(106, 20);
            this.txtBrickSet.TabIndex = 5;
            // 
            // AddBrickSpaceForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(236, 231);
            this.Controls.Add(this.txtBrickSet);
            this.Controls.Add(this.lblBrickSet);
            this.Controls.Add(this.txtNewID);
            this.Controls.Add(this.lblNewId);
            this.Controls.Add(this.btnCancel);
            this.Controls.Add(this.btnOK);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
            this.Name = "AddBrickSpaceForm";
            this.Text = "Add Brick Space";
            this.Load += new System.EventHandler(this.AddBrickSpaceForm_Load);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Button btnOK;
        private System.Windows.Forms.Button btnCancel;
        private System.Windows.Forms.Label lblNewId;
        private System.Windows.Forms.TextBox txtNewID;
        private System.Windows.Forms.Label lblBrickSet;
        private System.Windows.Forms.TextBox txtBrickSet;
    }
}