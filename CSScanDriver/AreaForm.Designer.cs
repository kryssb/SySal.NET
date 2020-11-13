namespace SySal.DAQSystem.Drivers.CSScanDriver
{
    partial class AreaForm
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
            this.labelMinX = new System.Windows.Forms.Label();
            this.textMinX = new System.Windows.Forms.TextBox();
            this.textMaxX = new System.Windows.Forms.TextBox();
            this.labelMaxX = new System.Windows.Forms.Label();
            this.textMinY = new System.Windows.Forms.TextBox();
            this.labelMinY = new System.Windows.Forms.Label();
            this.textMaxY = new System.Windows.Forms.TextBox();
            this.labelMaxY = new System.Windows.Forms.Label();
            this.buttonCancel = new System.Windows.Forms.Button();
            this.buttonOk = new System.Windows.Forms.Button();
            this.SuspendLayout();
            // 
            // labelMinX
            // 
            this.labelMinX.AutoSize = true;
            this.labelMinX.Location = new System.Drawing.Point(22, 24);
            this.labelMinX.Name = "labelMinX";
            this.labelMinX.Size = new System.Drawing.Size(31, 13);
            this.labelMinX.TabIndex = 0;
            this.labelMinX.Text = "MinX";
            // 
            // textMinX
            // 
            this.textMinX.Location = new System.Drawing.Point(82, 21);
            this.textMinX.Name = "textMinX";
            this.textMinX.Size = new System.Drawing.Size(124, 20);
            this.textMinX.TabIndex = 1;
            this.textMinX.Leave += new System.EventHandler(this.textMinX_TextChanged);
            // 
            // textMaxX
            // 
            this.textMaxX.Location = new System.Drawing.Point(82, 65);
            this.textMaxX.Name = "textMaxX";
            this.textMaxX.Size = new System.Drawing.Size(124, 20);
            this.textMaxX.TabIndex = 3;
            this.textMaxX.Leave += new System.EventHandler(this.textMaxX_TextChanged);
            // 
            // labelMaxX
            // 
            this.labelMaxX.AutoSize = true;
            this.labelMaxX.Location = new System.Drawing.Point(22, 68);
            this.labelMaxX.Name = "labelMaxX";
            this.labelMaxX.Size = new System.Drawing.Size(34, 13);
            this.labelMaxX.TabIndex = 2;
            this.labelMaxX.Text = "MaxX";
            // 
            // textMinY
            // 
            this.textMinY.Location = new System.Drawing.Point(324, 21);
            this.textMinY.Name = "textMinY";
            this.textMinY.Size = new System.Drawing.Size(124, 20);
            this.textMinY.TabIndex = 5;
            this.textMinY.Leave += new System.EventHandler(this.textMinY_TextChanged);
            // 
            // labelMinY
            // 
            this.labelMinY.AutoSize = true;
            this.labelMinY.Location = new System.Drawing.Point(264, 24);
            this.labelMinY.Name = "labelMinY";
            this.labelMinY.Size = new System.Drawing.Size(31, 13);
            this.labelMinY.TabIndex = 4;
            this.labelMinY.Text = "MinY";
            // 
            // textMaxY
            // 
            this.textMaxY.Location = new System.Drawing.Point(324, 68);
            this.textMaxY.Name = "textMaxY";
            this.textMaxY.Size = new System.Drawing.Size(124, 20);
            this.textMaxY.TabIndex = 7;
            this.textMaxY.Leave += new System.EventHandler(this.textMaxY_TextChanged);
            // 
            // labelMaxY
            // 
            this.labelMaxY.AutoSize = true;
            this.labelMaxY.Location = new System.Drawing.Point(264, 71);
            this.labelMaxY.Name = "labelMaxY";
            this.labelMaxY.Size = new System.Drawing.Size(34, 13);
            this.labelMaxY.TabIndex = 6;
            this.labelMaxY.Text = "MaxY";
            // 
            // buttonCancel
            // 
            this.buttonCancel.DialogResult = System.Windows.Forms.DialogResult.Cancel;
            this.buttonCancel.Location = new System.Drawing.Point(373, 120);
            this.buttonCancel.Name = "buttonCancel";
            this.buttonCancel.Size = new System.Drawing.Size(75, 23);
            this.buttonCancel.TabIndex = 9;
            this.buttonCancel.Text = "Cancel";
            this.buttonCancel.UseVisualStyleBackColor = true;
            this.buttonCancel.Click += new System.EventHandler(this.buttonCancel_Click);
            // 
            // buttonOk
            // 
            this.buttonOk.DialogResult = System.Windows.Forms.DialogResult.OK;
            this.buttonOk.Enabled = false;
            this.buttonOk.Location = new System.Drawing.Point(25, 120);
            this.buttonOk.Name = "buttonOk";
            this.buttonOk.Size = new System.Drawing.Size(75, 23);
            this.buttonOk.TabIndex = 8;
            this.buttonOk.Text = "OK";
            this.buttonOk.UseVisualStyleBackColor = true;
            this.buttonOk.Click += new System.EventHandler(this.buttonOk_Click);
            // 
            // AreaForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(467, 159);
            this.Controls.Add(this.buttonOk);
            this.Controls.Add(this.buttonCancel);
            this.Controls.Add(this.textMaxY);
            this.Controls.Add(this.labelMaxY);
            this.Controls.Add(this.textMinY);
            this.Controls.Add(this.labelMinY);
            this.Controls.Add(this.textMaxX);
            this.Controls.Add(this.labelMaxX);
            this.Controls.Add(this.textMinX);
            this.Controls.Add(this.labelMinX);
            this.Name = "AreaForm";
            this.Text = "Scanning area";
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Label labelMinX;
        private System.Windows.Forms.TextBox textMinX;
        private System.Windows.Forms.TextBox textMaxX;
        private System.Windows.Forms.Label labelMaxX;
        private System.Windows.Forms.TextBox textMinY;
        private System.Windows.Forms.Label labelMinY;
        private System.Windows.Forms.TextBox textMaxY;
        private System.Windows.Forms.Label labelMaxY;
        private System.Windows.Forms.Button buttonCancel;
        private System.Windows.Forms.Button buttonOk;
    }
}