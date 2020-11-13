namespace SySal.Executables.OperaFeedback
{
    partial class SQLReportForm
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
            this.rtSQL = new System.Windows.Forms.RichTextBox();
            this.SuspendLayout();
            // 
            // rtSQL
            // 
            this.rtSQL.Font = new System.Drawing.Font("Courier New", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.rtSQL.Location = new System.Drawing.Point(12, 12);
            this.rtSQL.Name = "rtSQL";
            this.rtSQL.ReadOnly = true;
            this.rtSQL.Size = new System.Drawing.Size(825, 354);
            this.rtSQL.TabIndex = 0;
            this.rtSQL.Text = "";
            this.rtSQL.WordWrap = false;
            // 
            // SQLReportForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(852, 378);
            this.Controls.Add(this.rtSQL);
            this.Name = "SQLReportForm";
            this.Text = "SQLReportForm";
            this.ResumeLayout(false);

        }

        #endregion

        internal System.Windows.Forms.RichTextBox rtSQL;

    }
}