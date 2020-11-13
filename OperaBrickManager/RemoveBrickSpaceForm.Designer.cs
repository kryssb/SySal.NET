namespace OperaBrickManager
{
    partial class RemoveBrickSpaceForm
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
            this.btnOk = new System.Windows.Forms.Button();
            this.btnCancel = new System.Windows.Forms.Button();
            this.lblIDSpace = new System.Windows.Forms.Label();
            this.lblSetBrick = new System.Windows.Forms.Label();
            this.txtIdSpace = new System.Windows.Forms.TextBox();
            this.txtBrickSet = new System.Windows.Forms.TextBox();
            this.SuspendLayout();
            // 
            // btnOk
            // 
            this.btnOk.Location = new System.Drawing.Point(153, 146);
            this.btnOk.Name = "btnOk";
            this.btnOk.Size = new System.Drawing.Size(73, 37);
            this.btnOk.TabIndex = 3;
            this.btnOk.Text = "Ok";
            this.btnOk.UseVisualStyleBackColor = true;
            this.btnOk.Click += new System.EventHandler(this.btnOk_Click);
            // 
            // btnCancel
            // 
            this.btnCancel.Location = new System.Drawing.Point(24, 146);
            this.btnCancel.Name = "btnCancel";
            this.btnCancel.Size = new System.Drawing.Size(79, 37);
            this.btnCancel.TabIndex = 4;
            this.btnCancel.Text = "Cancel";
            this.btnCancel.UseVisualStyleBackColor = true;
            this.btnCancel.Click += new System.EventHandler(this.btnCancel_Click);
            // 
            // lblIDSpace
            // 
            this.lblIDSpace.AutoSize = true;
            this.lblIDSpace.Location = new System.Drawing.Point(38, 37);
            this.lblIDSpace.Name = "lblIDSpace";
            this.lblIDSpace.Size = new System.Drawing.Size(50, 13);
            this.lblIDSpace.TabIndex = 2;
            this.lblIDSpace.Text = "Id Space";
            // 
            // lblSetBrick
            // 
            this.lblSetBrick.AutoSize = true;
            this.lblSetBrick.Location = new System.Drawing.Point(38, 91);
            this.lblSetBrick.Name = "lblSetBrick";
            this.lblSetBrick.Size = new System.Drawing.Size(50, 13);
            this.lblSetBrick.TabIndex = 3;
            this.lblSetBrick.Text = "Set Brick";
            // 
            // txtIdSpace
            // 
            this.txtIdSpace.Location = new System.Drawing.Point(108, 37);
            this.txtIdSpace.Name = "txtIdSpace";
            this.txtIdSpace.Size = new System.Drawing.Size(77, 20);
            this.txtIdSpace.TabIndex = 1;
            // 
            // txtBrickSet
            // 
            this.txtBrickSet.Location = new System.Drawing.Point(108, 89);
            this.txtBrickSet.Name = "txtBrickSet";
            this.txtBrickSet.ReadOnly = true;
            this.txtBrickSet.Size = new System.Drawing.Size(77, 20);
            this.txtBrickSet.TabIndex = 2;
            // 
            // RemoveBrickSpaceForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(247, 209);
            this.Controls.Add(this.txtBrickSet);
            this.Controls.Add(this.txtIdSpace);
            this.Controls.Add(this.lblSetBrick);
            this.Controls.Add(this.lblIDSpace);
            this.Controls.Add(this.btnCancel);
            this.Controls.Add(this.btnOk);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
            this.Name = "RemoveBrickSpaceForm";
            this.Text = "Remove Brick Space";
            this.Load += new System.EventHandler(this.RemoveBrickSpaceForm_Load);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Button btnOk;
        private System.Windows.Forms.Button btnCancel;
        private System.Windows.Forms.Label lblIDSpace;
        private System.Windows.Forms.Label lblSetBrick;
        private System.Windows.Forms.TextBox txtIdSpace;
        private System.Windows.Forms.TextBox txtBrickSet;
    }
}