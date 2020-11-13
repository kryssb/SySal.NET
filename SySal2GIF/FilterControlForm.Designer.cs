namespace SySal.Executables.SySal2GIF
{
    partial class FilterControlForm
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
            this.components = new System.ComponentModel.Container();
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(FilterControlForm));
            this.label1 = new System.Windows.Forms.Label();
            this.txtFilterMult = new System.Windows.Forms.TextBox();
            this.txtFilterZeroThreshold = new System.Windows.Forms.TextBox();
            this.label2 = new System.Windows.Forms.Label();
            this.txtImageMult = new System.Windows.Forms.TextBox();
            this.label3 = new System.Windows.Forms.Label();
            this.txtImageOffset = new System.Windows.Forms.TextBox();
            this.label4 = new System.Windows.Forms.Label();
            this.btnOK = new System.Windows.Forms.Button();
            this.btnCancel = new System.Windows.Forms.Button();
            this.toolTip1 = new System.Windows.Forms.ToolTip(this.components);
            this.label5 = new System.Windows.Forms.Label();
            this.SuspendLayout();
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(12, 16);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(72, 13);
            this.label1.TabIndex = 0;
            this.label1.Text = "Filter multiplier";
            // 
            // txtFilterMult
            // 
            this.txtFilterMult.Location = new System.Drawing.Point(119, 13);
            this.txtFilterMult.Name = "txtFilterMult";
            this.txtFilterMult.Size = new System.Drawing.Size(46, 20);
            this.txtFilterMult.TabIndex = 1;
            this.toolTip1.SetToolTip(this.txtFilterMult, "The output of the filter is multiplied by \r\nthis factor (called \"k\").");
            this.txtFilterMult.Leave += new System.EventHandler(this.OnParamTextLeave);
            // 
            // txtFilterZeroThreshold
            // 
            this.txtFilterZeroThreshold.Location = new System.Drawing.Point(119, 39);
            this.txtFilterZeroThreshold.Name = "txtFilterZeroThreshold";
            this.txtFilterZeroThreshold.Size = new System.Drawing.Size(46, 20);
            this.txtFilterZeroThreshold.TabIndex = 3;
            this.toolTip1.SetToolTip(this.txtFilterZeroThreshold, "If the filter output falls below this threshold, \r\nit is replaced with zero. The " +
                    "purpose is to \r\navoid that small disturbances are amplified.\r\nThis parameter is " +
                    "called \"z\".\r\n");
            this.txtFilterZeroThreshold.Leave += new System.EventHandler(this.OnParamTextLeave);
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(12, 42);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(98, 13);
            this.label2.TabIndex = 2;
            this.label2.Text = "Filter zero threshold";
            // 
            // txtImageMult
            // 
            this.txtImageMult.Location = new System.Drawing.Point(119, 65);
            this.txtImageMult.Name = "txtImageMult";
            this.txtImageMult.Size = new System.Drawing.Size(46, 20);
            this.txtImageMult.TabIndex = 5;
            this.toolTip1.SetToolTip(this.txtImageMult, resources.GetString("txtImageMult.ToolTip"));
            this.txtImageMult.Leave += new System.EventHandler(this.OnParamTextLeave);
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(12, 68);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(79, 13);
            this.label3.TabIndex = 4;
            this.label3.Text = "Image multiplier";
            // 
            // txtImageOffset
            // 
            this.txtImageOffset.Location = new System.Drawing.Point(119, 91);
            this.txtImageOffset.Name = "txtImageOffset";
            this.txtImageOffset.Size = new System.Drawing.Size(46, 20);
            this.txtImageOffset.TabIndex = 7;
            this.toolTip1.SetToolTip(this.txtImageOffset, "The image is offset by this pedestal, to compensate\r\nthe overall scale reduction " +
                    "by the multiplier, and to \r\nincrease the brightness for the human eye.\r\nThis par" +
                    "ameter is called \"p\".");
            this.txtImageOffset.Leave += new System.EventHandler(this.OnParamTextLeave);
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(12, 94);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(65, 13);
            this.label4.TabIndex = 6;
            this.label4.Text = "Image offset";
            // 
            // btnOK
            // 
            this.btnOK.Location = new System.Drawing.Point(190, 12);
            this.btnOK.Name = "btnOK";
            this.btnOK.Size = new System.Drawing.Size(59, 21);
            this.btnOK.TabIndex = 8;
            this.btnOK.Text = "OK";
            this.btnOK.UseVisualStyleBackColor = true;
            this.btnOK.Click += new System.EventHandler(this.btnOK_Click);
            // 
            // btnCancel
            // 
            this.btnCancel.Location = new System.Drawing.Point(190, 39);
            this.btnCancel.Name = "btnCancel";
            this.btnCancel.Size = new System.Drawing.Size(59, 21);
            this.btnCancel.TabIndex = 9;
            this.btnCancel.Text = "Cancel";
            this.btnCancel.UseVisualStyleBackColor = true;
            this.btnCancel.Click += new System.EventHandler(this.btnCancel_Click);
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.label5.Location = new System.Drawing.Point(12, 124);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(166, 67);
            this.label5.TabIndex = 10;
            this.label5.Text = "Final formula: \r\n   m g(x,y) + p + thr(k f(x,y))\r\n   g(x,y) = grey level at pixel" +
                " x,y\r\n   f(x,y) = filter output at pixel x,y\r\n   thr(w) = 0 if w >= z, w otherwi" +
                "se";
            this.label5.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // FilterControlForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(267, 201);
            this.Controls.Add(this.label5);
            this.Controls.Add(this.btnCancel);
            this.Controls.Add(this.btnOK);
            this.Controls.Add(this.txtImageOffset);
            this.Controls.Add(this.label4);
            this.Controls.Add(this.txtImageMult);
            this.Controls.Add(this.label3);
            this.Controls.Add(this.txtFilterZeroThreshold);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.txtFilterMult);
            this.Controls.Add(this.label1);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedToolWindow;
            this.Name = "FilterControlForm";
            this.Text = "Filter Control";
            this.Load += new System.EventHandler(this.OnLoad);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TextBox txtFilterMult;
        private System.Windows.Forms.TextBox txtFilterZeroThreshold;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.TextBox txtImageMult;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.TextBox txtImageOffset;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.Button btnOK;
        private System.Windows.Forms.Button btnCancel;
        private System.Windows.Forms.ToolTip toolTip1;
        private System.Windows.Forms.Label label5;
    }
}