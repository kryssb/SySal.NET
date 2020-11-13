namespace SySal.Services.NExTSrvHost_Win
{
    partial class NExTSrvHost_Form
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
            this.lvServices = new System.Windows.Forms.ListView();
            this.columnHeader1 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader2 = new System.Windows.Forms.ColumnHeader();
            this.btnExit = new System.Windows.Forms.Button();
            this.SuspendLayout();
            // 
            // lvServices
            // 
            this.lvServices.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.columnHeader1,
            this.columnHeader2});
            this.lvServices.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lvServices.FullRowSelect = true;
            this.lvServices.GridLines = true;
            this.lvServices.Location = new System.Drawing.Point(12, 12);
            this.lvServices.Name = "lvServices";
            this.lvServices.Size = new System.Drawing.Size(486, 201);
            this.lvServices.TabIndex = 0;
            this.lvServices.UseCompatibleStateImageBehavior = false;
            this.lvServices.View = System.Windows.Forms.View.Details;
            // 
            // columnHeader1
            // 
            this.columnHeader1.Text = "Name";
            this.columnHeader1.Width = 100;
            // 
            // columnHeader2
            // 
            this.columnHeader2.Text = "Status";
            this.columnHeader2.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.columnHeader2.Width = 360;
            // 
            // btnExit
            // 
            this.btnExit.Location = new System.Drawing.Point(432, 229);
            this.btnExit.Name = "btnExit";
            this.btnExit.Size = new System.Drawing.Size(66, 23);
            this.btnExit.TabIndex = 1;
            this.btnExit.Text = "Exit";
            this.btnExit.UseVisualStyleBackColor = true;
            this.btnExit.Click += new System.EventHandler(this.btnExit_Click);
            // 
            // NExTSrvHost_Form
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(510, 262);
            this.Controls.Add(this.btnExit);
            this.Controls.Add(this.lvServices);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.Fixed3D;
            this.Name = "NExTSrvHost_Form";
            this.Text = "SySal.NExT Host";
            this.Load += new System.EventHandler(this.OnLoad);
            this.FormClosing += new System.Windows.Forms.FormClosingEventHandler(this.OnClosing);
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.ListView lvServices;
        private System.Windows.Forms.ColumnHeader columnHeader1;
        private System.Windows.Forms.ColumnHeader columnHeader2;
        private System.Windows.Forms.Button btnExit;
    }
}

