namespace SySal.DAQSystem.Drivers.PredictionScan3Driver
{
    partial class CSBrickSelProcOp
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
            this.lvOps = new System.Windows.Forms.ListView();
            this.columnHeader1 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader2 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader3 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader4 = new System.Windows.Forms.ColumnHeader();
            this.btnOK = new System.Windows.Forms.Button();
            this.btnCancel = new System.Windows.Forms.Button();
            this.label1 = new System.Windows.Forms.Label();
            this.txtMinGrains = new System.Windows.Forms.TextBox();
            this.chkIncludeMicrotracks = new System.Windows.Forms.CheckBox();
            this.SuspendLayout();
            // 
            // lvOps
            // 
            this.lvOps.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.columnHeader1,
            this.columnHeader2,
            this.columnHeader3,
            this.columnHeader4});
            this.lvOps.FullRowSelect = true;
            this.lvOps.GridLines = true;
            this.lvOps.HideSelection = false;
            this.lvOps.Location = new System.Drawing.Point(18, 21);
            this.lvOps.Name = "lvOps";
            this.lvOps.Size = new System.Drawing.Size(367, 166);
            this.lvOps.TabIndex = 0;
            this.lvOps.UseCompatibleStateImageBehavior = false;
            this.lvOps.View = System.Windows.Forms.View.Details;
            // 
            // columnHeader1
            // 
            this.columnHeader1.Text = "Id";
            this.columnHeader1.Width = 134;
            // 
            // columnHeader2
            // 
            this.columnHeader2.Text = "Machine";
            this.columnHeader2.Width = 73;
            // 
            // columnHeader3
            // 
            this.columnHeader3.Text = "Date";
            this.columnHeader3.Width = 68;
            // 
            // columnHeader4
            // 
            this.columnHeader4.Text = "Success";
            // 
            // btnOK
            // 
            this.btnOK.Location = new System.Drawing.Point(315, 193);
            this.btnOK.Name = "btnOK";
            this.btnOK.Size = new System.Drawing.Size(70, 33);
            this.btnOK.TabIndex = 1;
            this.btnOK.Text = "OK";
            this.btnOK.UseVisualStyleBackColor = true;
            this.btnOK.Click += new System.EventHandler(this.btnOK_Click);
            // 
            // btnCancel
            // 
            this.btnCancel.Location = new System.Drawing.Point(18, 193);
            this.btnCancel.Name = "btnCancel";
            this.btnCancel.Size = new System.Drawing.Size(70, 33);
            this.btnCancel.TabIndex = 2;
            this.btnCancel.Text = "Cancel";
            this.btnCancel.UseVisualStyleBackColor = true;
            this.btnCancel.Click += new System.EventHandler(this.btnCancel_Click);
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(91, 201);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(60, 13);
            this.label1.TabIndex = 3;
            this.label1.Text = "Min. Grains";
            // 
            // txtMinGrains
            // 
            this.txtMinGrains.Location = new System.Drawing.Point(158, 199);
            this.txtMinGrains.Name = "txtMinGrains";
            this.txtMinGrains.Size = new System.Drawing.Size(45, 20);
            this.txtMinGrains.TabIndex = 4;
            this.txtMinGrains.Leave += new System.EventHandler(this.OnMinGrainsLeave);
            // 
            // chkIncludeMicrotracks
            // 
            this.chkIncludeMicrotracks.AutoSize = true;
            this.chkIncludeMicrotracks.Location = new System.Drawing.Point(209, 200);
            this.chkIncludeMicrotracks.Name = "chkIncludeMicrotracks";
            this.chkIncludeMicrotracks.Size = new System.Drawing.Size(102, 17);
            this.chkIncludeMicrotracks.TabIndex = 5;
            this.chkIncludeMicrotracks.Text = "Use microtracks";
            this.chkIncludeMicrotracks.UseVisualStyleBackColor = true;
            // 
            // CSBrickSelProcOp
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(405, 234);
            this.Controls.Add(this.chkIncludeMicrotracks);
            this.Controls.Add(this.txtMinGrains);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.btnCancel);
            this.Controls.Add(this.btnOK);
            this.Controls.Add(this.lvOps);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedToolWindow;
            this.Name = "CSBrickSelProcOp";
            this.Text = "Select CS-Brick connection operation";
            this.Load += new System.EventHandler(this.OnLoad);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.ListView lvOps;
        private System.Windows.Forms.ColumnHeader columnHeader1;
        private System.Windows.Forms.ColumnHeader columnHeader2;
        private System.Windows.Forms.Button btnOK;
        private System.Windows.Forms.Button btnCancel;
        private System.Windows.Forms.ColumnHeader columnHeader3;
        private System.Windows.Forms.ColumnHeader columnHeader4;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TextBox txtMinGrains;
        private System.Windows.Forms.CheckBox chkIncludeMicrotracks;
    }
}