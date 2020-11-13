namespace OperaBrickManager
{
    partial class frmAddFromDb
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
            this.lvZeroCoordinate = new System.Windows.Forms.ListView();
            this.columnHeader1 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader2 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader3 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader4 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader5 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader6 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader7 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader8 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader9 = new System.Windows.Forms.ColumnHeader();
            this.lblBrickId = new System.Windows.Forms.Label();
            this.txtIdBrick = new System.Windows.Forms.TextBox();
            this.btnConnect = new System.Windows.Forms.Button();
            this.btnClear = new System.Windows.Forms.Button();
            this.SuspendLayout();
            // 
            // btnOk
            // 
            this.btnOk.DialogResult = System.Windows.Forms.DialogResult.OK;
            this.btnOk.Enabled = false;
            this.btnOk.Location = new System.Drawing.Point(642, 84);
            this.btnOk.Name = "btnOk";
            this.btnOk.Size = new System.Drawing.Size(78, 34);
            this.btnOk.TabIndex = 0;
            this.btnOk.Text = "&Ok";
            this.btnOk.UseVisualStyleBackColor = true;
            this.btnOk.Click += new System.EventHandler(this.btnOk_Click);
            // 
            // btnCancel
            // 
            this.btnCancel.DialogResult = System.Windows.Forms.DialogResult.Cancel;
            this.btnCancel.Location = new System.Drawing.Point(642, 143);
            this.btnCancel.Name = "btnCancel";
            this.btnCancel.Size = new System.Drawing.Size(78, 34);
            this.btnCancel.TabIndex = 1;
            this.btnCancel.Text = "&Cancel";
            this.btnCancel.UseVisualStyleBackColor = true;
            // 
            // lvZeroCoordinate
            // 
            this.lvZeroCoordinate.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.columnHeader1,
            this.columnHeader2,
            this.columnHeader3,
            this.columnHeader4,
            this.columnHeader5,
            this.columnHeader6,
            this.columnHeader7,
            this.columnHeader8,
            this.columnHeader9});
            this.lvZeroCoordinate.FullRowSelect = true;
            this.lvZeroCoordinate.GridLines = true;
            this.lvZeroCoordinate.HideSelection = false;
            this.lvZeroCoordinate.Location = new System.Drawing.Point(10, 70);
            this.lvZeroCoordinate.MultiSelect = false;
            this.lvZeroCoordinate.Name = "lvZeroCoordinate";
            this.lvZeroCoordinate.Size = new System.Drawing.Size(607, 201);
            this.lvZeroCoordinate.TabIndex = 2;
            this.lvZeroCoordinate.Tag = "0";
            this.lvZeroCoordinate.UseCompatibleStateImageBehavior = false;
            this.lvZeroCoordinate.View = System.Windows.Forms.View.Details;
            this.lvZeroCoordinate.Click += new System.EventHandler(this.lvZeroCoordinate_DoubleClick);
            // 
            // columnHeader1
            // 
            this.columnHeader1.Text = "Id";
            this.columnHeader1.Width = 57;
            // 
            // columnHeader2
            // 
            this.columnHeader2.Text = "MinX";
            // 
            // columnHeader3
            // 
            this.columnHeader3.Text = "MaxX";
            // 
            // columnHeader4
            // 
            this.columnHeader4.Text = "MinY";
            // 
            // columnHeader5
            // 
            this.columnHeader5.Text = "MaxY";
            // 
            // columnHeader6
            // 
            this.columnHeader6.Text = "Id Set";
            this.columnHeader6.Width = 77;
            // 
            // columnHeader7
            // 
            this.columnHeader7.Text = "Id Brick";
            // 
            // columnHeader8
            // 
            this.columnHeader8.Text = "ZeroX";
            // 
            // columnHeader9
            // 
            this.columnHeader9.Text = "ZeroY";
            // 
            // lblBrickId
            // 
            this.lblBrickId.AutoSize = true;
            this.lblBrickId.Location = new System.Drawing.Point(7, 23);
            this.lblBrickId.Name = "lblBrickId";
            this.lblBrickId.Size = new System.Drawing.Size(45, 13);
            this.lblBrickId.TabIndex = 3;
            this.lblBrickId.Text = "Brick ID";
            // 
            // txtIdBrick
            // 
            this.txtIdBrick.Location = new System.Drawing.Point(58, 16);
            this.txtIdBrick.Name = "txtIdBrick";
            this.txtIdBrick.Size = new System.Drawing.Size(58, 20);
            this.txtIdBrick.TabIndex = 4;
            this.txtIdBrick.TextChanged += new System.EventHandler(this.Filled3);
            // 
            // btnConnect
            // 
            this.btnConnect.Enabled = false;
            this.btnConnect.Location = new System.Drawing.Point(175, 12);
            this.btnConnect.Name = "btnConnect";
            this.btnConnect.Size = new System.Drawing.Size(105, 35);
            this.btnConnect.TabIndex = 5;
            this.btnConnect.Text = "Download from DB";
            this.btnConnect.UseVisualStyleBackColor = true;
            this.btnConnect.Click += new System.EventHandler(this.btnConnect_Click);
            // 
            // btnClear
            // 
            this.btnClear.Location = new System.Drawing.Point(642, 199);
            this.btnClear.Name = "btnClear";
            this.btnClear.Size = new System.Drawing.Size(76, 33);
            this.btnClear.TabIndex = 6;
            this.btnClear.Text = "Clear All";
            this.btnClear.UseVisualStyleBackColor = true;
            this.btnClear.Click += new System.EventHandler(this.btnClear_Click);
            // 
            // frmAddFromDb
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.CancelButton = this.btnCancel;
            this.ClientSize = new System.Drawing.Size(738, 292);
            this.Controls.Add(this.btnClear);
            this.Controls.Add(this.btnConnect);
            this.Controls.Add(this.txtIdBrick);
            this.Controls.Add(this.lblBrickId);
            this.Controls.Add(this.lvZeroCoordinate);
            this.Controls.Add(this.btnCancel);
            this.Controls.Add(this.btnOk);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
            this.Name = "frmAddFromDb";
            this.Text = "Add From Database";
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Button btnOk;
        private System.Windows.Forms.Button btnCancel;
        private System.Windows.Forms.ListView lvZeroCoordinate;
        private System.Windows.Forms.ColumnHeader columnHeader1;
        private System.Windows.Forms.ColumnHeader columnHeader2;
        private System.Windows.Forms.ColumnHeader columnHeader3;
        private System.Windows.Forms.ColumnHeader columnHeader4;
        private System.Windows.Forms.ColumnHeader columnHeader5;
        private System.Windows.Forms.ColumnHeader columnHeader6;
        private System.Windows.Forms.ColumnHeader columnHeader7;
        private System.Windows.Forms.Label lblBrickId;
        private System.Windows.Forms.TextBox txtIdBrick;
        private System.Windows.Forms.Button btnConnect;
        private System.Windows.Forms.Button btnClear;
        private System.Windows.Forms.ColumnHeader columnHeader8;
        private System.Windows.Forms.ColumnHeader columnHeader9;
    }
}