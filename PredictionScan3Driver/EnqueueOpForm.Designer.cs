namespace SySal.DAQSystem.Drivers.PredictionScan3Driver
{
    partial class EnqueueOpForm
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
            this.lvProgramSettings = new System.Windows.Forms.ListView();
            this.columnHeader1 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader2 = new System.Windows.Forms.ColumnHeader();
            this.chkFavorites = new System.Windows.Forms.CheckBox();
            this.groupBox2 = new System.Windows.Forms.GroupBox();
            this.lvMachines = new System.Windows.Forms.ListView();
            this.columnHeader3 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader4 = new System.Windows.Forms.ColumnHeader();
            this.label1 = new System.Windows.Forms.Label();
            this.txtWidthHeight = new System.Windows.Forms.TextBox();
            this.btnOK = new System.Windows.Forms.Button();
            this.btnCancel = new System.Windows.Forms.Button();
            this.txtNotes = new System.Windows.Forms.TextBox();
            this.label2 = new System.Windows.Forms.Label();
            this.groupBox1.SuspendLayout();
            this.groupBox2.SuspendLayout();
            this.SuspendLayout();
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.lvProgramSettings);
            this.groupBox1.Controls.Add(this.chkFavorites);
            this.groupBox1.Location = new System.Drawing.Point(9, 11);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(282, 306);
            this.groupBox1.TabIndex = 0;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Program Settings";
            // 
            // lvProgramSettings
            // 
            this.lvProgramSettings.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.columnHeader1,
            this.columnHeader2});
            this.lvProgramSettings.FullRowSelect = true;
            this.lvProgramSettings.GridLines = true;
            this.lvProgramSettings.HideSelection = false;
            this.lvProgramSettings.Location = new System.Drawing.Point(12, 54);
            this.lvProgramSettings.Name = "lvProgramSettings";
            this.lvProgramSettings.Size = new System.Drawing.Size(257, 241);
            this.lvProgramSettings.TabIndex = 1;
            this.lvProgramSettings.UseCompatibleStateImageBehavior = false;
            this.lvProgramSettings.View = System.Windows.Forms.View.Details;
            // 
            // columnHeader1
            // 
            this.columnHeader1.Text = "Id";
            // 
            // columnHeader2
            // 
            this.columnHeader2.Text = "Name";
            this.columnHeader2.Width = 180;
            // 
            // chkFavorites
            // 
            this.chkFavorites.AutoSize = true;
            this.chkFavorites.Checked = true;
            this.chkFavorites.CheckState = System.Windows.Forms.CheckState.Checked;
            this.chkFavorites.Location = new System.Drawing.Point(11, 21);
            this.chkFavorites.Name = "chkFavorites";
            this.chkFavorites.Size = new System.Drawing.Size(69, 17);
            this.chkFavorites.TabIndex = 0;
            this.chkFavorites.Text = "&Favorites";
            this.chkFavorites.UseVisualStyleBackColor = true;
            this.chkFavorites.CheckedChanged += new System.EventHandler(this.OnFavoriteChanged);
            // 
            // groupBox2
            // 
            this.groupBox2.Controls.Add(this.lvMachines);
            this.groupBox2.Location = new System.Drawing.Point(297, 11);
            this.groupBox2.Name = "groupBox2";
            this.groupBox2.Size = new System.Drawing.Size(282, 306);
            this.groupBox2.TabIndex = 1;
            this.groupBox2.TabStop = false;
            this.groupBox2.Text = "Microscope";
            // 
            // lvMachines
            // 
            this.lvMachines.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.columnHeader3,
            this.columnHeader4});
            this.lvMachines.FullRowSelect = true;
            this.lvMachines.GridLines = true;
            this.lvMachines.HideSelection = false;
            this.lvMachines.Location = new System.Drawing.Point(12, 21);
            this.lvMachines.Name = "lvMachines";
            this.lvMachines.Size = new System.Drawing.Size(257, 274);
            this.lvMachines.TabIndex = 1;
            this.lvMachines.UseCompatibleStateImageBehavior = false;
            this.lvMachines.View = System.Windows.Forms.View.Details;
            // 
            // columnHeader3
            // 
            this.columnHeader3.Text = "Id";
            // 
            // columnHeader4
            // 
            this.columnHeader4.Text = "Name";
            this.columnHeader4.Width = 180;
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(17, 331);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(71, 13);
            this.label1.TabIndex = 0;
            this.label1.Text = "Width/Height";
            // 
            // txtWidthHeight
            // 
            this.txtWidthHeight.Location = new System.Drawing.Point(103, 326);
            this.txtWidthHeight.Name = "txtWidthHeight";
            this.txtWidthHeight.Size = new System.Drawing.Size(55, 20);
            this.txtWidthHeight.TabIndex = 1;
            this.txtWidthHeight.Leave += new System.EventHandler(this.OnWidthHeightLeave);
            // 
            // btnOK
            // 
            this.btnOK.Location = new System.Drawing.Point(503, 325);
            this.btnOK.Name = "btnOK";
            this.btnOK.Size = new System.Drawing.Size(76, 21);
            this.btnOK.TabIndex = 2;
            this.btnOK.Text = "OK";
            this.btnOK.UseVisualStyleBackColor = true;
            this.btnOK.Click += new System.EventHandler(this.btnOK_Click);
            // 
            // btnCancel
            // 
            this.btnCancel.Location = new System.Drawing.Point(421, 325);
            this.btnCancel.Name = "btnCancel";
            this.btnCancel.Size = new System.Drawing.Size(76, 21);
            this.btnCancel.TabIndex = 3;
            this.btnCancel.Text = "Cancel";
            this.btnCancel.UseVisualStyleBackColor = true;
            this.btnCancel.Click += new System.EventHandler(this.btnCancel_Click);
            // 
            // txtNotes
            // 
            this.txtNotes.Location = new System.Drawing.Point(212, 326);
            this.txtNotes.Name = "txtNotes";
            this.txtNotes.Size = new System.Drawing.Size(203, 20);
            this.txtNotes.TabIndex = 5;
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(171, 331);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(35, 13);
            this.label2.TabIndex = 4;
            this.label2.Text = "Notes";
            // 
            // EnqueueOpForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(589, 360);
            this.Controls.Add(this.txtNotes);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.btnCancel);
            this.Controls.Add(this.btnOK);
            this.Controls.Add(this.groupBox2);
            this.Controls.Add(this.txtWidthHeight);
            this.Controls.Add(this.groupBox1);
            this.Controls.Add(this.label1);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedToolWindow;
            this.Name = "EnqueueOpForm";
            this.Text = "Enqueue Operation";
            this.Load += new System.EventHandler(this.OnLoad);
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            this.groupBox2.ResumeLayout(false);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.ListView lvProgramSettings;
        private System.Windows.Forms.ColumnHeader columnHeader1;
        private System.Windows.Forms.ColumnHeader columnHeader2;
        private System.Windows.Forms.CheckBox chkFavorites;
        private System.Windows.Forms.GroupBox groupBox2;
        private System.Windows.Forms.ListView lvMachines;
        private System.Windows.Forms.ColumnHeader columnHeader3;
        private System.Windows.Forms.ColumnHeader columnHeader4;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TextBox txtWidthHeight;
        private System.Windows.Forms.Button btnOK;
        private System.Windows.Forms.Button btnCancel;
        private System.Windows.Forms.TextBox txtNotes;
        private System.Windows.Forms.Label label2;
    }
}