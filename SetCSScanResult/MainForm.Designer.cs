namespace SetCSScanResult
{
    partial class MainForm
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
            this.labelUser = new System.Windows.Forms.Label();
            this.btnReset = new System.Windows.Forms.Button();
            this.btnNoCand = new System.Windows.Forms.Button();
            this.btnBlack = new System.Windows.Forms.Button();
            this.btnCandOK = new System.Windows.Forms.Button();
            this.btnRefresh = new System.Windows.Forms.Button();
            this.lvCSScanOps = new System.Windows.Forms.ListView();
            this.columnHeader1 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader2 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader6 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader5 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader4 = new System.Windows.Forms.ColumnHeader();
            this.btnSubmitChanges = new System.Windows.Forms.Button();
            this.btnCancel = new System.Windows.Forms.Button();
            this.groupBox1.SuspendLayout();
            this.SuspendLayout();
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.labelUser);
            this.groupBox1.Controls.Add(this.btnReset);
            this.groupBox1.Controls.Add(this.btnNoCand);
            this.groupBox1.Controls.Add(this.btnBlack);
            this.groupBox1.Controls.Add(this.btnCandOK);
            this.groupBox1.Controls.Add(this.btnRefresh);
            this.groupBox1.Controls.Add(this.lvCSScanOps);
            this.groupBox1.Location = new System.Drawing.Point(13, 12);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(697, 378);
            this.groupBox1.TabIndex = 0;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "CS scan operations waiting for result";
            // 
            // labelUser
            // 
            this.labelUser.AutoSize = true;
            this.labelUser.Location = new System.Drawing.Point(14, 25);
            this.labelUser.Name = "labelUser";
            this.labelUser.Size = new System.Drawing.Size(19, 13);
            this.labelUser.TabIndex = 6;
            this.labelUser.Text = "----";
            // 
            // btnReset
            // 
            this.btnReset.Location = new System.Drawing.Point(162, 336);
            this.btnReset.Name = "btnReset";
            this.btnReset.Size = new System.Drawing.Size(127, 27);
            this.btnReset.TabIndex = 5;
            this.btnReset.Text = "Reset";
            this.btnReset.UseVisualStyleBackColor = true;
            this.btnReset.Click += new System.EventHandler(this.btnReset_Click);
            // 
            // btnNoCand
            // 
            this.btnNoCand.Location = new System.Drawing.Point(589, 336);
            this.btnNoCand.Name = "btnNoCand";
            this.btnNoCand.Size = new System.Drawing.Size(85, 27);
            this.btnNoCand.TabIndex = 4;
            this.btnNoCand.Text = "Not found";
            this.btnNoCand.UseVisualStyleBackColor = true;
            this.btnNoCand.Click += new System.EventHandler(this.btnNoCand_Click);
            // 
            // btnBlack
            // 
            this.btnBlack.Location = new System.Drawing.Point(472, 336);
            this.btnBlack.Name = "btnBlack";
            this.btnBlack.Size = new System.Drawing.Size(85, 27);
            this.btnBlack.TabIndex = 3;
            this.btnBlack.Text = "Black CS";
            this.btnBlack.UseVisualStyleBackColor = true;
            this.btnBlack.Click += new System.EventHandler(this.btnBlack_Click);
            // 
            // btnCandOK
            // 
            this.btnCandOK.Location = new System.Drawing.Point(351, 336);
            this.btnCandOK.Name = "btnCandOK";
            this.btnCandOK.Size = new System.Drawing.Size(85, 27);
            this.btnCandOK.TabIndex = 2;
            this.btnCandOK.Text = "Candidate OK";
            this.btnCandOK.UseVisualStyleBackColor = true;
            this.btnCandOK.Click += new System.EventHandler(this.btnCandOK_Click);
            // 
            // btnRefresh
            // 
            this.btnRefresh.Location = new System.Drawing.Point(17, 336);
            this.btnRefresh.Name = "btnRefresh";
            this.btnRefresh.Size = new System.Drawing.Size(117, 27);
            this.btnRefresh.TabIndex = 1;
            this.btnRefresh.Text = "Refresh";
            this.btnRefresh.UseVisualStyleBackColor = true;
            this.btnRefresh.Click += new System.EventHandler(this.btnRefresh_Click);
            // 
            // lvCSScanOps
            // 
            this.lvCSScanOps.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.columnHeader1,
            this.columnHeader2,
            this.columnHeader6,
            this.columnHeader5,
            this.columnHeader4});
            this.lvCSScanOps.FullRowSelect = true;
            this.lvCSScanOps.GridLines = true;
            this.lvCSScanOps.Location = new System.Drawing.Point(17, 59);
            this.lvCSScanOps.MultiSelect = false;
            this.lvCSScanOps.Name = "lvCSScanOps";
            this.lvCSScanOps.Size = new System.Drawing.Size(657, 261);
            this.lvCSScanOps.TabIndex = 0;
            this.lvCSScanOps.UseCompatibleStateImageBehavior = false;
            this.lvCSScanOps.View = System.Windows.Forms.View.Details;
            // 
            // columnHeader1
            // 
            this.columnHeader1.Text = "CS";
            this.columnHeader1.Width = 80;
            // 
            // columnHeader2
            // 
            this.columnHeader2.Text = "Event";
            this.columnHeader2.Width = 120;
            // 
            // columnHeader6
            // 
            this.columnHeader6.Text = "Extraction";
            // 
            // columnHeader5
            // 
            this.columnHeader5.Text = "Pred#(ID_EVENT)";
            this.columnHeader5.Width = 120;
            // 
            // columnHeader4
            // 
            this.columnHeader4.Text = "Result";
            this.columnHeader4.Width = 100;
            // 
            // btnSubmitChanges
            // 
            this.btnSubmitChanges.Location = new System.Drawing.Point(593, 396);
            this.btnSubmitChanges.Name = "btnSubmitChanges";
            this.btnSubmitChanges.Size = new System.Drawing.Size(117, 27);
            this.btnSubmitChanges.TabIndex = 2;
            this.btnSubmitChanges.Text = "Submit changes";
            this.btnSubmitChanges.UseVisualStyleBackColor = true;
            this.btnSubmitChanges.Click += new System.EventHandler(this.btnSubmitChanges_Click);
            // 
            // btnCancel
            // 
            this.btnCancel.Location = new System.Drawing.Point(13, 396);
            this.btnCancel.Name = "btnCancel";
            this.btnCancel.Size = new System.Drawing.Size(117, 27);
            this.btnCancel.TabIndex = 3;
            this.btnCancel.Text = "Cancel changes";
            this.btnCancel.UseVisualStyleBackColor = true;
            this.btnCancel.Click += new System.EventHandler(this.btnCancel_Click);
            // 
            // MainForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(722, 435);
            this.Controls.Add(this.btnCancel);
            this.Controls.Add(this.btnSubmitChanges);
            this.Controls.Add(this.groupBox1);
            this.Name = "MainForm";
            this.Text = "Set CS Scan Result";
            this.Load += new System.EventHandler(this.OnLoad);
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.Button btnNoCand;
        private System.Windows.Forms.Button btnBlack;
        private System.Windows.Forms.Button btnCandOK;
        private System.Windows.Forms.Button btnRefresh;
        private System.Windows.Forms.ListView lvCSScanOps;
        private System.Windows.Forms.Button btnSubmitChanges;
        private System.Windows.Forms.Button btnCancel;
        private System.Windows.Forms.ColumnHeader columnHeader1;
        private System.Windows.Forms.ColumnHeader columnHeader2;
        private System.Windows.Forms.ColumnHeader columnHeader4;
        private System.Windows.Forms.Button btnReset;
        private System.Windows.Forms.ColumnHeader columnHeader6;
        private System.Windows.Forms.ColumnHeader columnHeader5;
        private System.Windows.Forms.Label labelUser;
    }
}

