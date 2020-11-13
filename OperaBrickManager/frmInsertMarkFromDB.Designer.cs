namespace OperaBrickManager
{
    partial class frmInsertMarkFromDB
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
            this.btnConnectDB = new System.Windows.Forms.Button();
            this.txtIDBrick = new System.Windows.Forms.TextBox();
            this.lblIDBrick = new System.Windows.Forms.Label();
            this.lvCSMarks = new System.Windows.Forms.ListView();
            this.ColumnHeader0 = new System.Windows.Forms.ColumnHeader();
            this.btnClearAll = new System.Windows.Forms.Button();
            this.btnCancel = new System.Windows.Forms.Button();
            this.btnOK = new System.Windows.Forms.Button();
            this.SuspendLayout();
            // 
            // btnConnectDB
            // 
            this.btnConnectDB.Location = new System.Drawing.Point(178, 16);
            this.btnConnectDB.Name = "btnConnectDB";
            this.btnConnectDB.Size = new System.Drawing.Size(95, 29);
            this.btnConnectDB.TabIndex = 5;
            this.btnConnectDB.Text = "Connect DB";
            this.btnConnectDB.UseVisualStyleBackColor = true;
            this.btnConnectDB.Click += new System.EventHandler(this.btnConnectDB_Click);
            // 
            // txtIDBrick
            // 
            this.txtIDBrick.Location = new System.Drawing.Point(61, 21);
            this.txtIDBrick.Name = "txtIDBrick";
            this.txtIDBrick.ReadOnly = true;
            this.txtIDBrick.Size = new System.Drawing.Size(87, 20);
            this.txtIDBrick.TabIndex = 4;
            // 
            // lblIDBrick
            // 
            this.lblIDBrick.AutoSize = true;
            this.lblIDBrick.Location = new System.Drawing.Point(12, 24);
            this.lblIDBrick.Name = "lblIDBrick";
            this.lblIDBrick.Size = new System.Drawing.Size(43, 13);
            this.lblIDBrick.TabIndex = 3;
            this.lblIDBrick.Text = "Id Brick";
            // 
            // lvCSMarks
            // 
            this.lvCSMarks.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.ColumnHeader0});
            this.lvCSMarks.FullRowSelect = true;
            this.lvCSMarks.GridLines = true;
            this.lvCSMarks.HideSelection = false;
            this.lvCSMarks.Location = new System.Drawing.Point(15, 71);
            this.lvCSMarks.MultiSelect = false;
            this.lvCSMarks.Name = "lvCSMarks";
            this.lvCSMarks.Size = new System.Drawing.Size(99, 180);
            this.lvCSMarks.TabIndex = 6;
            this.lvCSMarks.Tag = "0";
            this.lvCSMarks.UseCompatibleStateImageBehavior = false;
            this.lvCSMarks.View = System.Windows.Forms.View.Details;
            this.lvCSMarks.SelectedIndexChanged += new System.EventHandler(this.lvCSMarks_SelectedIndexChanged);
            // 
            // ColumnHeader0
            // 
            this.ColumnHeader0.Text = "CS_ID";
            this.ColumnHeader0.Width = 127;
            // 
            // btnClearAll
            // 
            this.btnClearAll.Location = new System.Drawing.Point(144, 186);
            this.btnClearAll.Name = "btnClearAll";
            this.btnClearAll.Size = new System.Drawing.Size(92, 33);
            this.btnClearAll.TabIndex = 9;
            this.btnClearAll.Text = "Clear All";
            this.btnClearAll.UseVisualStyleBackColor = true;
            this.btnClearAll.Click += new System.EventHandler(this.btnClearAll_Click);
            // 
            // btnCancel
            // 
            this.btnCancel.Location = new System.Drawing.Point(144, 135);
            this.btnCancel.Name = "btnCancel";
            this.btnCancel.Size = new System.Drawing.Size(92, 33);
            this.btnCancel.TabIndex = 8;
            this.btnCancel.Text = "Cancel";
            this.btnCancel.UseVisualStyleBackColor = true;
            this.btnCancel.Click += new System.EventHandler(this.btnCancel_Click);
            // 
            // btnOK
            // 
            this.btnOK.Enabled = false;
            this.btnOK.Location = new System.Drawing.Point(144, 85);
            this.btnOK.Name = "btnOK";
            this.btnOK.Size = new System.Drawing.Size(92, 33);
            this.btnOK.TabIndex = 7;
            this.btnOK.Text = "OK";
            this.btnOK.UseVisualStyleBackColor = true;
            this.btnOK.Click += new System.EventHandler(this.btnOK_Click);
            // 
            // frmInsertMarkFromDB
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(321, 266);
            this.Controls.Add(this.btnClearAll);
            this.Controls.Add(this.btnCancel);
            this.Controls.Add(this.btnOK);
            this.Controls.Add(this.lvCSMarks);
            this.Controls.Add(this.btnConnectDB);
            this.Controls.Add(this.txtIDBrick);
            this.Controls.Add(this.lblIDBrick);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
            this.MaximizeBox = false;
            this.Name = "frmInsertMarkFromDB";
            this.Text = "Insert mark from DB";
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Button btnConnectDB;
        private System.Windows.Forms.TextBox txtIDBrick;
        private System.Windows.Forms.Label lblIDBrick;
        private System.Windows.Forms.ListView lvCSMarks;
        private System.Windows.Forms.Button btnClearAll;
        private System.Windows.Forms.Button btnCancel;
        private System.Windows.Forms.Button btnOK;
        private System.Windows.Forms.ColumnHeader ColumnHeader0;
    }
}