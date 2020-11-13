namespace OperaBrickManager
{
    partial class frmLoadMarkFromFile
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
            this.btnLoadFromFile = new System.Windows.Forms.Button();
            this.FileOpen = new System.Windows.Forms.OpenFileDialog();
            this.rdbLateralMark = new System.Windows.Forms.RadioButton();
            this.rdbCSMark = new System.Windows.Forms.RadioButton();
            this.rdbSpotMark = new System.Windows.Forms.RadioButton();
            this.txtNcount = new System.Windows.Forms.TextBox();
            this.lblNcount = new System.Windows.Forms.Label();
            this.txtIdBrick = new System.Windows.Forms.TextBox();
            this.lblIdBrick = new System.Windows.Forms.Label();
            this.btnOk = new System.Windows.Forms.Button();
            this.btnCancel = new System.Windows.Forms.Button();
            this.btnClear = new System.Windows.Forms.Button();
            this.SuspendLayout();
            // 
            // btnLoadFromFile
            // 
            this.btnLoadFromFile.Location = new System.Drawing.Point(23, 186);
            this.btnLoadFromFile.Name = "btnLoadFromFile";
            this.btnLoadFromFile.Size = new System.Drawing.Size(122, 31);
            this.btnLoadFromFile.TabIndex = 3;
            this.btnLoadFromFile.Text = "Load From File";
            this.btnLoadFromFile.UseVisualStyleBackColor = true;
            this.btnLoadFromFile.Click += new System.EventHandler(this.btnLoadFromFile_Click);
            // 
            // FileOpen
            // 
            this.FileOpen.Filter = "Text file(s)|*.txt";
            // 
            // rdbLateralMark
            // 
            this.rdbLateralMark.AutoSize = true;
            this.rdbLateralMark.Location = new System.Drawing.Point(43, 72);
            this.rdbLateralMark.Name = "rdbLateralMark";
            this.rdbLateralMark.Size = new System.Drawing.Size(116, 17);
            this.rdbLateralMark.TabIndex = 4;
            this.rdbLateralMark.TabStop = true;
            this.rdbLateralMark.Text = "Lateral X-ray Marks";
            this.rdbLateralMark.UseVisualStyleBackColor = true;
            // 
            // rdbCSMark
            // 
            this.rdbCSMark.AutoSize = true;
            this.rdbCSMark.Location = new System.Drawing.Point(43, 107);
            this.rdbCSMark.Name = "rdbCSMark";
            this.rdbCSMark.Size = new System.Drawing.Size(115, 17);
            this.rdbCSMark.TabIndex = 5;
            this.rdbCSMark.TabStop = true;
            this.rdbCSMark.Text = "Frontal X-ray marks";
            this.rdbCSMark.UseVisualStyleBackColor = true;
            // 
            // rdbSpotMark
            // 
            this.rdbSpotMark.AutoSize = true;
            this.rdbSpotMark.Location = new System.Drawing.Point(43, 141);
            this.rdbSpotMark.Name = "rdbSpotMark";
            this.rdbSpotMark.Size = new System.Drawing.Size(115, 17);
            this.rdbSpotMark.TabIndex = 6;
            this.rdbSpotMark.TabStop = true;
            this.rdbSpotMark.Text = "Spot Optical Marks";
            this.rdbSpotMark.UseVisualStyleBackColor = true;
            // 
            // txtNcount
            // 
            this.txtNcount.Location = new System.Drawing.Point(187, 234);
            this.txtNcount.Name = "txtNcount";
            this.txtNcount.ReadOnly = true;
            this.txtNcount.Size = new System.Drawing.Size(55, 20);
            this.txtNcount.TabIndex = 10;
            this.txtNcount.TextChanged += new System.EventHandler(this.noChange2);
            // 
            // lblNcount
            // 
            this.lblNcount.AutoSize = true;
            this.lblNcount.Location = new System.Drawing.Point(40, 237);
            this.lblNcount.Name = "lblNcount";
            this.lblNcount.Size = new System.Drawing.Size(122, 13);
            this.lblNcount.TabIndex = 9;
            this.lblNcount.Text = "Number of marks loaded";
            // 
            // txtIdBrick
            // 
            this.txtIdBrick.Location = new System.Drawing.Point(93, 23);
            this.txtIdBrick.Name = "txtIdBrick";
            this.txtIdBrick.ReadOnly = true;
            this.txtIdBrick.Size = new System.Drawing.Size(58, 20);
            this.txtIdBrick.TabIndex = 12;
            this.txtIdBrick.TextChanged += new System.EventHandler(this.noChange);
            // 
            // lblIdBrick
            // 
            this.lblIdBrick.AutoSize = true;
            this.lblIdBrick.Location = new System.Drawing.Point(43, 26);
            this.lblIdBrick.Name = "lblIdBrick";
            this.lblIdBrick.Size = new System.Drawing.Size(42, 13);
            this.lblIdBrick.TabIndex = 11;
            this.lblIdBrick.Text = "IDBrick";
            // 
            // btnOk
            // 
            this.btnOk.Enabled = false;
            this.btnOk.Location = new System.Drawing.Point(187, 23);
            this.btnOk.Name = "btnOk";
            this.btnOk.Size = new System.Drawing.Size(86, 31);
            this.btnOk.TabIndex = 13;
            this.btnOk.Text = "OK";
            this.btnOk.UseVisualStyleBackColor = true;
            this.btnOk.Click += new System.EventHandler(this.btnOk_Click);
            // 
            // btnCancel
            // 
            this.btnCancel.Location = new System.Drawing.Point(187, 74);
            this.btnCancel.Name = "btnCancel";
            this.btnCancel.Size = new System.Drawing.Size(86, 31);
            this.btnCancel.TabIndex = 14;
            this.btnCancel.Text = "Cancel";
            this.btnCancel.UseVisualStyleBackColor = true;
            this.btnCancel.Click += new System.EventHandler(this.btnCancel_Click);
            // 
            // btnClear
            // 
            this.btnClear.Location = new System.Drawing.Point(187, 122);
            this.btnClear.Name = "btnClear";
            this.btnClear.Size = new System.Drawing.Size(86, 31);
            this.btnClear.TabIndex = 15;
            this.btnClear.Text = "Clear";
            this.btnClear.UseVisualStyleBackColor = true;
            this.btnClear.Click += new System.EventHandler(this.btnClear_Click);
            // 
            // frmLoadMarkFromFile
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(285, 266);
            this.Controls.Add(this.btnClear);
            this.Controls.Add(this.btnCancel);
            this.Controls.Add(this.btnOk);
            this.Controls.Add(this.txtIdBrick);
            this.Controls.Add(this.lblIdBrick);
            this.Controls.Add(this.txtNcount);
            this.Controls.Add(this.lblNcount);
            this.Controls.Add(this.rdbSpotMark);
            this.Controls.Add(this.rdbCSMark);
            this.Controls.Add(this.rdbLateralMark);
            this.Controls.Add(this.btnLoadFromFile);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
            this.MaximizeBox = false;
            this.Name = "frmLoadMarkFromFile";
            this.Text = "Load Marks From File";
            this.Load += new System.EventHandler(this.frmLoadMarkFromFile_Load);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Button btnLoadFromFile;
        private System.Windows.Forms.OpenFileDialog FileOpen;
        private System.Windows.Forms.RadioButton rdbLateralMark;
        private System.Windows.Forms.RadioButton rdbCSMark;
        private System.Windows.Forms.RadioButton rdbSpotMark;
        private System.Windows.Forms.TextBox txtNcount;
        private System.Windows.Forms.Label lblNcount;
        private System.Windows.Forms.TextBox txtIdBrick;
        private System.Windows.Forms.Label lblIdBrick;
        private System.Windows.Forms.Button btnOk;
        private System.Windows.Forms.Button btnCancel;
        private System.Windows.Forms.Button btnClear;
    }
}