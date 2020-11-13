namespace SySal.DAQSystem.Drivers.CSScanDriver
{
    partial class frmEasyInterrupt
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
            this.buttonSend = new System.Windows.Forms.Button();
            this.buttonCancel = new System.Windows.Forms.Button();
            this.textOPERAPwd = new System.Windows.Forms.TextBox();
            this.textOPERAUsername = new System.Windows.Forms.TextBox();
            this.label4 = new System.Windows.Forms.Label();
            this.label3 = new System.Windows.Forms.Label();
            this.comboProcOpId = new System.Windows.Forms.ComboBox();
            this.label2 = new System.Windows.Forms.Label();
            this.comboBatchManager = new System.Windows.Forms.ComboBox();
            this.label1 = new System.Windows.Forms.Label();
            this.groupScanningArea = new System.Windows.Forms.GroupBox();
            this.textMaxY = new System.Windows.Forms.TextBox();
            this.labelMaxY = new System.Windows.Forms.Label();
            this.textMinY = new System.Windows.Forms.TextBox();
            this.labelMinY = new System.Windows.Forms.Label();
            this.textMaxX = new System.Windows.Forms.TextBox();
            this.labelMaxX = new System.Windows.Forms.Label();
            this.textMinX = new System.Windows.Forms.TextBox();
            this.labelMinX = new System.Windows.Forms.Label();
            this.groupScanningArea.SuspendLayout();
            this.SuspendLayout();
            // 
            // buttonSend
            // 
            this.buttonSend.Location = new System.Drawing.Point(325, 86);
            this.buttonSend.Name = "buttonSend";
            this.buttonSend.Size = new System.Drawing.Size(64, 24);
            this.buttonSend.TabIndex = 20;
            this.buttonSend.Text = "Send";
            this.buttonSend.Click += new System.EventHandler(this.buttonSend_Click);
            // 
            // buttonCancel
            // 
            this.buttonCancel.Location = new System.Drawing.Point(12, 298);
            this.buttonCancel.Name = "buttonCancel";
            this.buttonCancel.Size = new System.Drawing.Size(64, 24);
            this.buttonCancel.TabIndex = 19;
            this.buttonCancel.Text = "Cancel";
            this.buttonCancel.Click += new System.EventHandler(this.buttonCancel_Click);
            // 
            // textOPERAPwd
            // 
            this.textOPERAPwd.Location = new System.Drawing.Point(213, 265);
            this.textOPERAPwd.Name = "textOPERAPwd";
            this.textOPERAPwd.PasswordChar = '*';
            this.textOPERAPwd.Size = new System.Drawing.Size(200, 20);
            this.textOPERAPwd.TabIndex = 18;
            // 
            // textOPERAUsername
            // 
            this.textOPERAUsername.Location = new System.Drawing.Point(213, 232);
            this.textOPERAUsername.Name = "textOPERAUsername";
            this.textOPERAUsername.Size = new System.Drawing.Size(200, 20);
            this.textOPERAUsername.TabIndex = 16;
            // 
            // label4
            // 
            this.label4.Location = new System.Drawing.Point(12, 265);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(104, 24);
            this.label4.TabIndex = 17;
            this.label4.Text = "OPERA Password";
            this.label4.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // label3
            // 
            this.label3.Location = new System.Drawing.Point(12, 232);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(104, 24);
            this.label3.TabIndex = 15;
            this.label3.Text = "OPERA Username";
            this.label3.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // comboProcOpId
            // 
            this.comboProcOpId.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.comboProcOpId.Location = new System.Drawing.Point(149, 57);
            this.comboProcOpId.Name = "comboProcOpId";
            this.comboProcOpId.Size = new System.Drawing.Size(264, 21);
            this.comboProcOpId.TabIndex = 14;
            // 
            // label2
            // 
            this.label2.Location = new System.Drawing.Point(12, 57);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(120, 24);
            this.label2.TabIndex = 13;
            this.label2.Text = "Process Operation";
            this.label2.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // comboBatchManager
            // 
            this.comboBatchManager.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.comboBatchManager.Location = new System.Drawing.Point(149, 21);
            this.comboBatchManager.Name = "comboBatchManager";
            this.comboBatchManager.Size = new System.Drawing.Size(264, 21);
            this.comboBatchManager.TabIndex = 12;
            this.comboBatchManager.SelectionChangeCommitted += new System.EventHandler(this.OnBatchManagerSelected);
            this.comboBatchManager.SelectedIndexChanged += new System.EventHandler(this.OnBatchManagerSelected);
            // 
            // label1
            // 
            this.label1.Location = new System.Drawing.Point(12, 21);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(88, 24);
            this.label1.TabIndex = 11;
            this.label1.Text = "Batch Manager";
            this.label1.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // groupScanningArea
            // 
            this.groupScanningArea.Controls.Add(this.textMaxY);
            this.groupScanningArea.Controls.Add(this.buttonSend);
            this.groupScanningArea.Controls.Add(this.labelMaxY);
            this.groupScanningArea.Controls.Add(this.textMinY);
            this.groupScanningArea.Controls.Add(this.labelMinY);
            this.groupScanningArea.Controls.Add(this.textMaxX);
            this.groupScanningArea.Controls.Add(this.labelMaxX);
            this.groupScanningArea.Controls.Add(this.textMinX);
            this.groupScanningArea.Controls.Add(this.labelMinX);
            this.groupScanningArea.Location = new System.Drawing.Point(12, 94);
            this.groupScanningArea.Name = "groupScanningArea";
            this.groupScanningArea.Size = new System.Drawing.Size(401, 122);
            this.groupScanningArea.TabIndex = 21;
            this.groupScanningArea.TabStop = false;
            this.groupScanningArea.Text = "Scanning Area";
            // 
            // textMaxY
            // 
            this.textMaxY.Location = new System.Drawing.Point(260, 50);
            this.textMaxY.Name = "textMaxY";
            this.textMaxY.Size = new System.Drawing.Size(129, 20);
            this.textMaxY.TabIndex = 15;
            // 
            // labelMaxY
            // 
            this.labelMaxY.AutoSize = true;
            this.labelMaxY.Location = new System.Drawing.Point(207, 56);
            this.labelMaxY.Name = "labelMaxY";
            this.labelMaxY.Size = new System.Drawing.Size(34, 13);
            this.labelMaxY.TabIndex = 14;
            this.labelMaxY.Text = "MaxY";
            // 
            // textMinY
            // 
            this.textMinY.Location = new System.Drawing.Point(260, 17);
            this.textMinY.Name = "textMinY";
            this.textMinY.Size = new System.Drawing.Size(129, 20);
            this.textMinY.TabIndex = 13;
            // 
            // labelMinY
            // 
            this.labelMinY.AutoSize = true;
            this.labelMinY.Location = new System.Drawing.Point(207, 20);
            this.labelMinY.Name = "labelMinY";
            this.labelMinY.Size = new System.Drawing.Size(31, 13);
            this.labelMinY.TabIndex = 12;
            this.labelMinY.Text = "MinY";
            // 
            // textMaxX
            // 
            this.textMaxX.Location = new System.Drawing.Point(65, 53);
            this.textMaxX.Name = "textMaxX";
            this.textMaxX.Size = new System.Drawing.Size(129, 20);
            this.textMaxX.TabIndex = 11;
            // 
            // labelMaxX
            // 
            this.labelMaxX.AutoSize = true;
            this.labelMaxX.Location = new System.Drawing.Point(15, 54);
            this.labelMaxX.Name = "labelMaxX";
            this.labelMaxX.Size = new System.Drawing.Size(34, 13);
            this.labelMaxX.TabIndex = 10;
            this.labelMaxX.Text = "MaxX";
            // 
            // textMinX
            // 
            this.textMinX.Location = new System.Drawing.Point(65, 19);
            this.textMinX.Name = "textMinX";
            this.textMinX.Size = new System.Drawing.Size(129, 20);
            this.textMinX.TabIndex = 9;
            // 
            // labelMinX
            // 
            this.labelMinX.AutoSize = true;
            this.labelMinX.Location = new System.Drawing.Point(15, 25);
            this.labelMinX.Name = "labelMinX";
            this.labelMinX.Size = new System.Drawing.Size(31, 13);
            this.labelMinX.TabIndex = 8;
            this.labelMinX.Text = "MinX";
            // 
            // frmEasyInterrupt
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(425, 330);
            this.Controls.Add(this.groupScanningArea);
            this.Controls.Add(this.buttonCancel);
            this.Controls.Add(this.textOPERAPwd);
            this.Controls.Add(this.textOPERAUsername);
            this.Controls.Add(this.label4);
            this.Controls.Add(this.label3);
            this.Controls.Add(this.comboProcOpId);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.comboBatchManager);
            this.Controls.Add(this.label1);
            this.Name = "frmEasyInterrupt";
            this.Text = "frmEasyInterrupt";
            this.Load += new System.EventHandler(this.OnLoad);
            this.groupScanningArea.ResumeLayout(false);
            this.groupScanningArea.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Button buttonSend;
        private System.Windows.Forms.Button buttonCancel;
        private System.Windows.Forms.TextBox textOPERAPwd;
        private System.Windows.Forms.TextBox textOPERAUsername;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.ComboBox comboProcOpId;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.ComboBox comboBatchManager;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.GroupBox groupScanningArea;
        private System.Windows.Forms.TextBox textMaxY;
        private System.Windows.Forms.Label labelMaxY;
        private System.Windows.Forms.TextBox textMinY;
        private System.Windows.Forms.Label labelMinY;
        private System.Windows.Forms.TextBox textMaxX;
        private System.Windows.Forms.Label labelMaxX;
        private System.Windows.Forms.TextBox textMinX;
        private System.Windows.Forms.Label labelMinX;
    }
}