namespace SySal.Executables.EasyReconstruct
{
    partial class SegReplaceForm
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
            this.lvReplacements = new System.Windows.Forms.ListView();
            this.columnHeader1 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader2 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader3 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader4 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader9 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader7 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader10 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader11 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader5 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader6 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader8 = new System.Windows.Forms.ColumnHeader();
            this.btnOK = new System.Windows.Forms.Button();
            this.btnCancel = new System.Windows.Forms.Button();
            this.SuspendLayout();
            // 
            // lvReplacements
            // 
            this.lvReplacements.CheckBoxes = true;
            this.lvReplacements.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.columnHeader1,
            this.columnHeader2,
            this.columnHeader3,
            this.columnHeader4,
            this.columnHeader9,
            this.columnHeader7,
            this.columnHeader10,
            this.columnHeader11,
            this.columnHeader5,
            this.columnHeader6,
            this.columnHeader8});
            this.lvReplacements.FullRowSelect = true;
            this.lvReplacements.GridLines = true;
            this.lvReplacements.HideSelection = false;
            this.lvReplacements.Location = new System.Drawing.Point(12, 12);
            this.lvReplacements.MultiSelect = false;
            this.lvReplacements.Name = "lvReplacements";
            this.lvReplacements.Size = new System.Drawing.Size(678, 229);
            this.lvReplacements.TabIndex = 0;
            this.lvReplacements.UseCompatibleStateImageBehavior = false;
            this.lvReplacements.View = System.Windows.Forms.View.Details;
            // 
            // columnHeader1
            // 
            this.columnHeader1.Text = "Man Tk";
            // 
            // columnHeader2
            // 
            this.columnHeader2.Text = "SX";
            // 
            // columnHeader3
            // 
            this.columnHeader3.Text = "SY";
            // 
            // columnHeader4
            // 
            this.columnHeader4.Text = "TSR Tk";
            // 
            // columnHeader9
            // 
            this.columnHeader9.Text = "Layer";
            // 
            // columnHeader7
            // 
            this.columnHeader7.Text = "DSlope";
            // 
            // columnHeader10
            // 
            this.columnHeader10.Text = "DPX";
            // 
            // columnHeader11
            // 
            this.columnHeader11.Text = "DPY";
            // 
            // columnHeader5
            // 
            this.columnHeader5.Text = "SX";
            // 
            // columnHeader6
            // 
            this.columnHeader6.Text = "SY";
            // 
            // columnHeader8
            // 
            this.columnHeader8.Text = "Grains";
            // 
            // btnOK
            // 
            this.btnOK.Location = new System.Drawing.Point(600, 260);
            this.btnOK.Name = "btnOK";
            this.btnOK.Size = new System.Drawing.Size(90, 29);
            this.btnOK.TabIndex = 1;
            this.btnOK.Text = "OK";
            this.btnOK.UseVisualStyleBackColor = true;
            this.btnOK.Click += new System.EventHandler(this.btnOK_Click);
            // 
            // btnCancel
            // 
            this.btnCancel.Location = new System.Drawing.Point(12, 260);
            this.btnCancel.Name = "btnCancel";
            this.btnCancel.Size = new System.Drawing.Size(90, 29);
            this.btnCancel.TabIndex = 2;
            this.btnCancel.Text = "Cancel";
            this.btnCancel.UseVisualStyleBackColor = true;
            this.btnCancel.Click += new System.EventHandler(this.btnCancel_Click);
            // 
            // SegReplaceForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(702, 301);
            this.ControlBox = false;
            this.Controls.Add(this.btnCancel);
            this.Controls.Add(this.btnOK);
            this.Controls.Add(this.lvReplacements);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedToolWindow;
            this.Name = "SegReplaceForm";
            this.Text = "Segment Replacement Confirmation";
            this.Load += new System.EventHandler(this.OnLoad);
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.ListView lvReplacements;
        private System.Windows.Forms.ColumnHeader columnHeader1;
        private System.Windows.Forms.Button btnOK;
        private System.Windows.Forms.Button btnCancel;
        private System.Windows.Forms.ColumnHeader columnHeader2;
        private System.Windows.Forms.ColumnHeader columnHeader3;
        private System.Windows.Forms.ColumnHeader columnHeader4;
        private System.Windows.Forms.ColumnHeader columnHeader5;
        private System.Windows.Forms.ColumnHeader columnHeader6;
        private System.Windows.Forms.ColumnHeader columnHeader7;
        private System.Windows.Forms.ColumnHeader columnHeader8;
        private System.Windows.Forms.ColumnHeader columnHeader9;
        private System.Windows.Forms.ColumnHeader columnHeader10;
        private System.Windows.Forms.ColumnHeader columnHeader11;
    }
}