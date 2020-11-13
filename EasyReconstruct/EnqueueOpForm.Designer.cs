namespace SySal.Executables.EasyReconstruct
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
            this.btnOK = new System.Windows.Forms.Button();
            this.btnCancel = new System.Windows.Forms.Button();
            this.label2 = new System.Windows.Forms.Label();
            this.label3 = new System.Windows.Forms.Label();
            this.label4 = new System.Windows.Forms.Label();
            this.cmbWidthHeight = new System.Windows.Forms.ComboBox();
            this.cmbPlates = new System.Windows.Forms.ComboBox();
            this.cmbSkew = new System.Windows.Forms.ComboBox();
            this.cmbNotes = new System.Windows.Forms.ComboBox();
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
            this.lvProgramSettings.TabIndex = 2;
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
            this.chkFavorites.TabIndex = 1;
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
            this.lvMachines.TabIndex = 3;
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
            this.label1.Location = new System.Drawing.Point(6, 331);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(71, 13);
            this.label1.TabIndex = 0;
            this.label1.Text = "Width/Height";
            // 
            // btnOK
            // 
            this.btnOK.Location = new System.Drawing.Point(503, 351);
            this.btnOK.Name = "btnOK";
            this.btnOK.Size = new System.Drawing.Size(76, 21);
            this.btnOK.TabIndex = 9;
            this.btnOK.Text = "OK";
            this.btnOK.UseVisualStyleBackColor = true;
            this.btnOK.Click += new System.EventHandler(this.btnOK_Click);
            // 
            // btnCancel
            // 
            this.btnCancel.Location = new System.Drawing.Point(421, 351);
            this.btnCancel.Name = "btnCancel";
            this.btnCancel.Size = new System.Drawing.Size(76, 21);
            this.btnCancel.TabIndex = 8;
            this.btnCancel.Text = "Cancel";
            this.btnCancel.UseVisualStyleBackColor = true;
            this.btnCancel.Click += new System.EventHandler(this.btnCancel_Click);
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(10, 355);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(35, 13);
            this.label2.TabIndex = 4;
            this.label2.Text = "Notes";
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(194, 331);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(115, 13);
            this.label3.TabIndex = 6;
            this.label3.Text = "Plates Down/upstream";
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(390, 329);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(84, 13);
            this.label4.TabIndex = 8;
            this.label4.Text = "Volume skewing";
            // 
            // cmbWidthHeight
            // 
            this.cmbWidthHeight.FormattingEnabled = true;
            this.cmbWidthHeight.Items.AddRange(new object[] {
            "10000/10000",
            "3000/3000"});
            this.cmbWidthHeight.Location = new System.Drawing.Point(85, 326);
            this.cmbWidthHeight.Name = "cmbWidthHeight";
            this.cmbWidthHeight.Size = new System.Drawing.Size(95, 21);
            this.cmbWidthHeight.TabIndex = 4;
            this.cmbWidthHeight.Leave += new System.EventHandler(this.OnWidthHeightLeave);
            // 
            // cmbPlates
            // 
            this.cmbPlates.FormattingEnabled = true;
            this.cmbPlates.Items.AddRange(new object[] {
            "10/5",
            "5/5"});
            this.cmbPlates.Location = new System.Drawing.Point(318, 326);
            this.cmbPlates.Name = "cmbPlates";
            this.cmbPlates.Size = new System.Drawing.Size(60, 21);
            this.cmbPlates.TabIndex = 5;
            this.cmbPlates.Leave += new System.EventHandler(this.OnPlatesLeave);
            // 
            // cmbSkew
            // 
            this.cmbSkew.FormattingEnabled = true;
            this.cmbSkew.Items.AddRange(new object[] {
            "0.000/0.059",
            "0.000/0.000"});
            this.cmbSkew.Location = new System.Drawing.Point(491, 326);
            this.cmbSkew.Name = "cmbSkew";
            this.cmbSkew.Size = new System.Drawing.Size(86, 21);
            this.cmbSkew.TabIndex = 6;
            this.cmbSkew.Leave += new System.EventHandler(this.OnSkewLeave);
            // 
            // cmbNotes
            // 
            this.cmbNotes.FormattingEnabled = true;
            this.cmbNotes.Location = new System.Drawing.Point(56, 353);
            this.cmbNotes.Name = "cmbNotes";
            this.cmbNotes.Size = new System.Drawing.Size(354, 21);
            this.cmbNotes.TabIndex = 7;
            // 
            // EnqueueOpForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(589, 379);
            this.Controls.Add(this.cmbNotes);
            this.Controls.Add(this.cmbSkew);
            this.Controls.Add(this.cmbPlates);
            this.Controls.Add(this.cmbWidthHeight);
            this.Controls.Add(this.label4);
            this.Controls.Add(this.label3);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.btnCancel);
            this.Controls.Add(this.btnOK);
            this.Controls.Add(this.groupBox2);
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
        private System.Windows.Forms.Button btnOK;
        private System.Windows.Forms.Button btnCancel;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.ComboBox cmbWidthHeight;
        private System.Windows.Forms.ComboBox cmbPlates;
        private System.Windows.Forms.ComboBox cmbSkew;
        private System.Windows.Forms.ComboBox cmbNotes;
    }
}