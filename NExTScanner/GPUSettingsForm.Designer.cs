namespace SySal.Executables.NExTScanner
{
    partial class GPUSettingsForm
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
            this.label1 = new System.Windows.Forms.Label();
            this.clvEnabledGPUs = new System.Windows.Forms.CheckedListBox();
            this.btnCancel = new SySal.SySalNExTControls.SySalButton();
            this.btnOK = new SySal.SySalNExTControls.SySalButton();
            this.btnGPUSetup = new SySal.SySalNExTControls.SySalButton();
            this.GPUSettingsToolTip = new System.Windows.Forms.ToolTip(this.components);
            this.SuspendLayout();
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.ForeColor = System.Drawing.Color.DimGray;
            this.label1.Location = new System.Drawing.Point(21, 31);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(163, 21);
            this.label1.TabIndex = 3;
            this.label1.Text = "Available GPU devices";
            // 
            // clvEnabledGPUs
            // 
            this.clvEnabledGPUs.ForeColor = System.Drawing.Color.DodgerBlue;
            this.clvEnabledGPUs.FormattingEnabled = true;
            this.clvEnabledGPUs.Location = new System.Drawing.Point(192, 29);
            this.clvEnabledGPUs.Name = "clvEnabledGPUs";
            this.clvEnabledGPUs.Size = new System.Drawing.Size(674, 124);
            this.clvEnabledGPUs.TabIndex = 4;
            this.GPUSettingsToolTip.SetToolTip(this.clvEnabledGPUs, "Select the GPU devices that you want to be used by this Scanner.");
            // 
            // btnCancel
            // 
            this.btnCancel.AutoSize = true;
            this.btnCancel.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnCancel.BackColor = System.Drawing.Color.Transparent;
            this.btnCancel.FocusedColor = System.Drawing.Color.Navy;
            this.btnCancel.Font = new System.Drawing.Font("Segoe UI", 14.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btnCancel.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnCancel.Location = new System.Drawing.Point(801, 164);
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
            this.btnOK.Location = new System.Drawing.Point(25, 164);
            this.btnOK.Margin = new System.Windows.Forms.Padding(7);
            this.btnOK.MinimumSize = new System.Drawing.Size(5, 5);
            this.btnOK.Name = "btnOK";
            this.btnOK.Size = new System.Drawing.Size(34, 29);
            this.btnOK.TabIndex = 19;
            this.btnOK.Text = "OK";
            this.btnOK.Click += new System.EventHandler(this.btnOK_Click);
            // 
            // btnGPUSetup
            // 
            this.btnGPUSetup.AutoSize = true;
            this.btnGPUSetup.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnGPUSetup.BackColor = System.Drawing.Color.Transparent;
            this.btnGPUSetup.FocusedColor = System.Drawing.Color.Navy;
            this.btnGPUSetup.Font = new System.Drawing.Font("Segoe UI", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btnGPUSetup.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnGPUSetup.Location = new System.Drawing.Point(25, 68);
            this.btnGPUSetup.Margin = new System.Windows.Forms.Padding(6);
            this.btnGPUSetup.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnGPUSetup.Name = "btnGPUSetup";
            this.btnGPUSetup.Size = new System.Drawing.Size(85, 25);
            this.btnGPUSetup.TabIndex = 18;
            this.btnGPUSetup.Text = "GPU Setup";
            this.GPUSettingsToolTip.SetToolTip(this.btnGPUSetup, "Configure GPU-specific options that will apply to all selected devices.");
            // 
            // GPUSettingsToolTip
            // 
            this.GPUSettingsToolTip.IsBalloon = true;
            // 
            // GPUSettingsForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(9F, 21F);
            this.ClientSize = new System.Drawing.Size(870, 209);
            this.Controls.Add(this.btnCancel);
            this.Controls.Add(this.btnOK);
            this.Controls.Add(this.btnGPUSetup);
            this.Controls.Add(this.clvEnabledGPUs);
            this.Controls.Add(this.label1);
            this.DialogCaption = "GPU Settings";
            this.Name = "GPUSettingsForm";
            this.Load += new System.EventHandler(this.OnLoad);
            this.Controls.SetChildIndex(this.label1, 0);
            this.Controls.SetChildIndex(this.clvEnabledGPUs, 0);
            this.Controls.SetChildIndex(this.btnGPUSetup, 0);
            this.Controls.SetChildIndex(this.btnOK, 0);
            this.Controls.SetChildIndex(this.btnCancel, 0);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.CheckedListBox clvEnabledGPUs;
        private SySal.SySalNExTControls.SySalButton btnCancel;
        private SySal.SySalNExTControls.SySalButton btnOK;
        private SySal.SySalNExTControls.SySalButton btnGPUSetup;
        private System.Windows.Forms.ToolTip GPUSettingsToolTip;
    }
}
