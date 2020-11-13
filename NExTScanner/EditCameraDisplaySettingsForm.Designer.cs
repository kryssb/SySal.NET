namespace SySal.Executables.NExTScanner
{
    partial class EditCameraDisplaySettingsForm
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
            this.txtHeight = new System.Windows.Forms.TextBox();
            this.txtWidth = new System.Windows.Forms.TextBox();
            this.label1 = new System.Windows.Forms.Label();
            this.txtTop = new System.Windows.Forms.TextBox();
            this.txtLeft = new System.Windows.Forms.TextBox();
            this.label2 = new System.Windows.Forms.Label();
            this.btnCancel = new SySal.SySalNExTControls.SySalButton();
            this.btnOK = new SySal.SySalNExTControls.SySalButton();
            this.mainToolTip = new System.Windows.Forms.ToolTip(this.components);
            this.SuspendLayout();
            // 
            // txtHeight
            // 
            this.txtHeight.BackColor = System.Drawing.Color.GhostWhite;
            this.txtHeight.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtHeight.ForeColor = System.Drawing.Color.DodgerBlue;
            this.txtHeight.Location = new System.Drawing.Point(280, 40);
            this.txtHeight.Name = "txtHeight";
            this.txtHeight.Size = new System.Drawing.Size(53, 25);
            this.txtHeight.TabIndex = 15;
            this.txtHeight.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.mainToolTip.SetToolTip(this.txtHeight, "Height of the display panel");
            this.txtHeight.Leave += new System.EventHandler(this.OnHeightLeave);
            // 
            // txtWidth
            // 
            this.txtWidth.BackColor = System.Drawing.Color.GhostWhite;
            this.txtWidth.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtWidth.ForeColor = System.Drawing.Color.DodgerBlue;
            this.txtWidth.Location = new System.Drawing.Point(221, 40);
            this.txtWidth.Name = "txtWidth";
            this.txtWidth.Size = new System.Drawing.Size(53, 25);
            this.txtWidth.TabIndex = 14;
            this.txtWidth.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.mainToolTip.SetToolTip(this.txtWidth, "Width of the display panel");
            this.txtWidth.Leave += new System.EventHandler(this.OnWidthLeave);
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.BackColor = System.Drawing.Color.Transparent;
            this.label1.ForeColor = System.Drawing.Color.DimGray;
            this.label1.Location = new System.Drawing.Point(12, 41);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(146, 21);
            this.label1.TabIndex = 13;
            this.label1.Text = "Image width-height";
            // 
            // txtTop
            // 
            this.txtTop.BackColor = System.Drawing.Color.GhostWhite;
            this.txtTop.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtTop.ForeColor = System.Drawing.Color.DodgerBlue;
            this.txtTop.Location = new System.Drawing.Point(280, 72);
            this.txtTop.Name = "txtTop";
            this.txtTop.Size = new System.Drawing.Size(53, 25);
            this.txtTop.TabIndex = 18;
            this.txtTop.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.mainToolTip.SetToolTip(this.txtTop, "Top corner of the display panel");
            this.txtTop.Leave += new System.EventHandler(this.OnTopLeave);
            // 
            // txtLeft
            // 
            this.txtLeft.BackColor = System.Drawing.Color.GhostWhite;
            this.txtLeft.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtLeft.ForeColor = System.Drawing.Color.DodgerBlue;
            this.txtLeft.Location = new System.Drawing.Point(221, 72);
            this.txtLeft.Name = "txtLeft";
            this.txtLeft.Size = new System.Drawing.Size(53, 25);
            this.txtLeft.TabIndex = 17;
            this.txtLeft.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.mainToolTip.SetToolTip(this.txtLeft, "Left corner of the display panel");
            this.txtLeft.Leave += new System.EventHandler(this.OnLeftLeave);
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.BackColor = System.Drawing.Color.Transparent;
            this.label2.ForeColor = System.Drawing.Color.DimGray;
            this.label2.Location = new System.Drawing.Point(12, 73);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(120, 21);
            this.label2.TabIndex = 16;
            this.label2.Text = "Left/Top Corner";
            // 
            // btnCancel
            // 
            this.btnCancel.AutoSize = true;
            this.btnCancel.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnCancel.BackColor = System.Drawing.Color.Transparent;
            this.btnCancel.FocusedColor = System.Drawing.Color.Navy;
            this.btnCancel.Font = new System.Drawing.Font("Segoe UI", 14.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btnCancel.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnCancel.Location = new System.Drawing.Point(268, 119);
            this.btnCancel.Margin = new System.Windows.Forms.Padding(7);
            this.btnCancel.MinimumSize = new System.Drawing.Size(5, 5);
            this.btnCancel.Name = "btnCancel";
            this.btnCancel.Size = new System.Drawing.Size(65, 29);
            this.btnCancel.TabIndex = 20;
            this.btnCancel.Text = "Cancel";
            this.btnCancel.Click += new System.EventHandler(this.btnCancel_Click);
            // 
            // btnOK
            // 
            this.btnOK.AutoSize = true;
            this.btnOK.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnOK.BackColor = System.Drawing.Color.Transparent;
            this.btnOK.FocusedColor = System.Drawing.Color.Navy;
            this.btnOK.Font = new System.Drawing.Font("Segoe UI", 14.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btnOK.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnOK.Location = new System.Drawing.Point(16, 119);
            this.btnOK.Margin = new System.Windows.Forms.Padding(7);
            this.btnOK.MinimumSize = new System.Drawing.Size(5, 5);
            this.btnOK.Name = "btnOK";
            this.btnOK.Size = new System.Drawing.Size(34, 29);
            this.btnOK.TabIndex = 19;
            this.btnOK.Text = "OK";
            this.btnOK.Click += new System.EventHandler(this.btnOK_Click);
            // 
            // EditCameraDisplaySettingsForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(9F, 21F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(347, 170);
            this.Controls.Add(this.btnCancel);
            this.Controls.Add(this.btnOK);
            this.Controls.Add(this.txtTop);
            this.Controls.Add(this.txtLeft);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.txtHeight);
            this.Controls.Add(this.txtWidth);
            this.Controls.Add(this.label1);
            this.DialogCaption = "Camera Display Settings";
            this.Name = "EditCameraDisplaySettingsForm";
            this.Load += new System.EventHandler(this.OnLoad);
            this.Controls.SetChildIndex(this.label1, 0);
            this.Controls.SetChildIndex(this.txtWidth, 0);
            this.Controls.SetChildIndex(this.txtHeight, 0);
            this.Controls.SetChildIndex(this.label2, 0);
            this.Controls.SetChildIndex(this.txtLeft, 0);
            this.Controls.SetChildIndex(this.txtTop, 0);
            this.Controls.SetChildIndex(this.btnOK, 0);
            this.Controls.SetChildIndex(this.btnCancel, 0);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.TextBox txtHeight;
        private System.Windows.Forms.TextBox txtWidth;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TextBox txtTop;
        private System.Windows.Forms.TextBox txtLeft;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.ToolTip mainToolTip;
        private SySalNExTControls.SySalButton btnCancel;
        private SySalNExTControls.SySalButton btnOK;
    }
}