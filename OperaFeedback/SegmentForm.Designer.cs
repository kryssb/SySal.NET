namespace SySal.Executables.OperaFeedback
{
    partial class SegmentForm
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
            this.lvSegments = new System.Windows.Forms.ListView();
            this.columnHeader1 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader2 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader3 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader4 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader5 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader6 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader7 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader8 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader9 = new System.Windows.Forms.ColumnHeader();
            this.btnSegRemove = new System.Windows.Forms.Button();
            this.btnSegAdd = new System.Windows.Forms.Button();
            this.txtSegAdd = new System.Windows.Forms.TextBox();
            this.SuspendLayout();
            // 
            // lvSegments
            // 
            this.lvSegments.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.columnHeader1,
            this.columnHeader2,
            this.columnHeader3,
            this.columnHeader4,
            this.columnHeader5,
            this.columnHeader6,
            this.columnHeader7,
            this.columnHeader8,
            this.columnHeader9});
            this.lvSegments.FullRowSelect = true;
            this.lvSegments.GridLines = true;
            this.lvSegments.Location = new System.Drawing.Point(12, 15);
            this.lvSegments.Name = "lvSegments";
            this.lvSegments.Size = new System.Drawing.Size(525, 217);
            this.lvSegments.TabIndex = 0;
            this.lvSegments.UseCompatibleStateImageBehavior = false;
            this.lvSegments.View = System.Windows.Forms.View.Details;
            this.lvSegments.DoubleClick += new System.EventHandler(this.lvSegments_DoubleClick);
            // 
            // columnHeader1
            // 
            this.columnHeader1.Text = "Plate";
            // 
            // columnHeader2
            // 
            this.columnHeader2.Text = "Type";
            this.columnHeader2.Width = 40;
            // 
            // columnHeader3
            // 
            this.columnHeader3.Text = "Mode";
            this.columnHeader3.Width = 44;
            // 
            // columnHeader4
            // 
            this.columnHeader4.Text = "Grains";
            // 
            // columnHeader5
            // 
            this.columnHeader5.Text = "X";
            // 
            // columnHeader6
            // 
            this.columnHeader6.Text = "Y";
            // 
            // columnHeader7
            // 
            this.columnHeader7.Text = "Z";
            // 
            // columnHeader8
            // 
            this.columnHeader8.Text = "SX";
            // 
            // columnHeader9
            // 
            this.columnHeader9.Text = "SY";
            // 
            // btnSegRemove
            // 
            this.btnSegRemove.Location = new System.Drawing.Point(475, 239);
            this.btnSegRemove.Name = "btnSegRemove";
            this.btnSegRemove.Size = new System.Drawing.Size(62, 24);
            this.btnSegRemove.TabIndex = 1;
            this.btnSegRemove.Text = "Remove";
            this.btnSegRemove.UseVisualStyleBackColor = true;
            this.btnSegRemove.Click += new System.EventHandler(this.btnSegRemove_Click);
            // 
            // btnSegAdd
            // 
            this.btnSegAdd.Location = new System.Drawing.Point(12, 238);
            this.btnSegAdd.Name = "btnSegAdd";
            this.btnSegAdd.Size = new System.Drawing.Size(174, 24);
            this.btnSegAdd.TabIndex = 2;
            this.btnSegAdd.Text = "Add (use \";\" to separate fields)";
            this.btnSegAdd.UseVisualStyleBackColor = true;
            this.btnSegAdd.Click += new System.EventHandler(this.btnSegAdd_Click);
            // 
            // txtSegAdd
            // 
            this.txtSegAdd.Location = new System.Drawing.Point(192, 241);
            this.txtSegAdd.Name = "txtSegAdd";
            this.txtSegAdd.Size = new System.Drawing.Size(277, 20);
            this.txtSegAdd.TabIndex = 3;
            // 
            // SegmentForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(550, 275);
            this.Controls.Add(this.txtSegAdd);
            this.Controls.Add(this.btnSegAdd);
            this.Controls.Add(this.btnSegRemove);
            this.Controls.Add(this.lvSegments);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedToolWindow;
            this.Name = "SegmentForm";
            this.Text = "Segments of track#";
            this.Load += new System.EventHandler(this.OnLoad);
            this.FormClosing += new System.Windows.Forms.FormClosingEventHandler(this.OnClose);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.ListView lvSegments;
        private System.Windows.Forms.ColumnHeader columnHeader1;
        private System.Windows.Forms.ColumnHeader columnHeader2;
        private System.Windows.Forms.ColumnHeader columnHeader3;
        private System.Windows.Forms.ColumnHeader columnHeader4;
        private System.Windows.Forms.ColumnHeader columnHeader5;
        private System.Windows.Forms.ColumnHeader columnHeader6;
        private System.Windows.Forms.ColumnHeader columnHeader7;
        private System.Windows.Forms.ColumnHeader columnHeader8;
        private System.Windows.Forms.ColumnHeader columnHeader9;
        private System.Windows.Forms.Button btnSegRemove;
        private System.Windows.Forms.Button btnSegAdd;
        private System.Windows.Forms.TextBox txtSegAdd;
    }
}