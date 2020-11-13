namespace SySal.Executables.EasyReconstruct
{
    partial class MovieForm
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
            this.chkEvent = new System.Windows.Forms.CheckBox();
            this.chkBrick = new System.Windows.Forms.CheckBox();
            this.trkSpeed = new System.Windows.Forms.TrackBar();
            this.label1 = new System.Windows.Forms.Label();
            this.btnMake = new System.Windows.Forms.Button();
            this.pbGeneration = new System.Windows.Forms.ProgressBar();
            this.groupViewControl = new System.Windows.Forms.GroupBox();
            this.chkAddViewExplanation = new System.Windows.Forms.CheckBox();
            this.chkViewXYPan = new System.Windows.Forms.CheckBox();
            this.txtXYPanFrames = new System.Windows.Forms.TextBox();
            this.txtZRotFrames = new System.Windows.Forms.TextBox();
            this.txtYZFrames = new System.Windows.Forms.TextBox();
            this.txtXZFrames = new System.Windows.Forms.TextBox();
            this.txtXYFrames = new System.Windows.Forms.TextBox();
            this.chkViewYRot = new System.Windows.Forms.CheckBox();
            this.chkViewYZ = new System.Windows.Forms.CheckBox();
            this.chkViewXZ = new System.Windows.Forms.CheckBox();
            this.chkViewXY = new System.Windows.Forms.CheckBox();
            this.txtComments = new System.Windows.Forms.TextBox();
            this.label2 = new System.Windows.Forms.Label();
            ((System.ComponentModel.ISupportInitialize)(this.trkSpeed)).BeginInit();
            this.groupViewControl.SuspendLayout();
            this.SuspendLayout();
            // 
            // chkEvent
            // 
            this.chkEvent.AutoSize = true;
            this.chkEvent.Checked = true;
            this.chkEvent.CheckState = System.Windows.Forms.CheckState.Checked;
            this.chkEvent.Location = new System.Drawing.Point(13, 11);
            this.chkEvent.Name = "chkEvent";
            this.chkEvent.Size = new System.Drawing.Size(86, 17);
            this.chkEvent.TabIndex = 0;
            this.chkEvent.Text = "Add Event #";
            this.chkEvent.UseVisualStyleBackColor = true;
            // 
            // chkBrick
            // 
            this.chkBrick.AutoSize = true;
            this.chkBrick.Checked = true;
            this.chkBrick.CheckState = System.Windows.Forms.CheckState.Checked;
            this.chkBrick.Location = new System.Drawing.Point(13, 34);
            this.chkBrick.Name = "chkBrick";
            this.chkBrick.Size = new System.Drawing.Size(82, 17);
            this.chkBrick.TabIndex = 1;
            this.chkBrick.Text = "Add Brick #";
            this.chkBrick.UseVisualStyleBackColor = true;
            // 
            // trkSpeed
            // 
            this.trkSpeed.Location = new System.Drawing.Point(167, 6);
            this.trkSpeed.Minimum = 1;
            this.trkSpeed.Name = "trkSpeed";
            this.trkSpeed.Size = new System.Drawing.Size(114, 45);
            this.trkSpeed.TabIndex = 2;
            this.trkSpeed.Value = 1;
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(123, 12);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(38, 13);
            this.label1.TabIndex = 3;
            this.label1.Text = "Speed";
            // 
            // btnMake
            // 
            this.btnMake.Location = new System.Drawing.Point(300, 11);
            this.btnMake.Name = "btnMake";
            this.btnMake.Size = new System.Drawing.Size(47, 40);
            this.btnMake.TabIndex = 4;
            this.btnMake.Text = "Make!";
            this.btnMake.UseVisualStyleBackColor = true;
            this.btnMake.Click += new System.EventHandler(this.btnMake_Click);
            // 
            // pbGeneration
            // 
            this.pbGeneration.Location = new System.Drawing.Point(126, 43);
            this.pbGeneration.Name = "pbGeneration";
            this.pbGeneration.Size = new System.Drawing.Size(154, 8);
            this.pbGeneration.Step = 1;
            this.pbGeneration.Style = System.Windows.Forms.ProgressBarStyle.Continuous;
            this.pbGeneration.TabIndex = 5;
            // 
            // groupViewControl
            // 
            this.groupViewControl.Controls.Add(this.chkAddViewExplanation);
            this.groupViewControl.Controls.Add(this.chkViewXYPan);
            this.groupViewControl.Controls.Add(this.txtXYPanFrames);
            this.groupViewControl.Controls.Add(this.txtZRotFrames);
            this.groupViewControl.Controls.Add(this.txtYZFrames);
            this.groupViewControl.Controls.Add(this.txtXZFrames);
            this.groupViewControl.Controls.Add(this.txtXYFrames);
            this.groupViewControl.Controls.Add(this.chkViewYRot);
            this.groupViewControl.Controls.Add(this.chkViewYZ);
            this.groupViewControl.Controls.Add(this.chkViewXZ);
            this.groupViewControl.Controls.Add(this.chkViewXY);
            this.groupViewControl.Location = new System.Drawing.Point(12, 106);
            this.groupViewControl.Name = "groupViewControl";
            this.groupViewControl.Size = new System.Drawing.Size(335, 95);
            this.groupViewControl.TabIndex = 6;
            this.groupViewControl.TabStop = false;
            this.groupViewControl.Text = "Movie details";
            // 
            // chkAddViewExplanation
            // 
            this.chkAddViewExplanation.AutoSize = true;
            this.chkAddViewExplanation.Checked = true;
            this.chkAddViewExplanation.CheckState = System.Windows.Forms.CheckState.Checked;
            this.chkAddViewExplanation.Location = new System.Drawing.Point(186, 71);
            this.chkAddViewExplanation.Name = "chkAddViewExplanation";
            this.chkAddViewExplanation.Size = new System.Drawing.Size(129, 17);
            this.chkAddViewExplanation.TabIndex = 12;
            this.chkAddViewExplanation.Text = "Add View Explanation";
            this.chkAddViewExplanation.UseVisualStyleBackColor = true;
            // 
            // chkViewXYPan
            // 
            this.chkViewXYPan.AutoSize = true;
            this.chkViewXYPan.Checked = true;
            this.chkViewXYPan.CheckState = System.Windows.Forms.CheckState.Checked;
            this.chkViewXYPan.Location = new System.Drawing.Point(186, 43);
            this.chkViewXYPan.Name = "chkViewXYPan";
            this.chkViewXYPan.Size = new System.Drawing.Size(95, 17);
            this.chkViewXYPan.TabIndex = 11;
            this.chkViewXYPan.Text = "XY pan frames";
            this.chkViewXYPan.UseVisualStyleBackColor = true;
            // 
            // txtXYPanFrames
            // 
            this.txtXYPanFrames.Location = new System.Drawing.Point(287, 41);
            this.txtXYPanFrames.Name = "txtXYPanFrames";
            this.txtXYPanFrames.Size = new System.Drawing.Size(37, 20);
            this.txtXYPanFrames.TabIndex = 10;
            this.txtXYPanFrames.Leave += new System.EventHandler(this.OnTextBoxLeave);
            // 
            // txtZRotFrames
            // 
            this.txtZRotFrames.Location = new System.Drawing.Point(287, 15);
            this.txtZRotFrames.Name = "txtZRotFrames";
            this.txtZRotFrames.Size = new System.Drawing.Size(37, 20);
            this.txtZRotFrames.TabIndex = 9;
            this.txtZRotFrames.Leave += new System.EventHandler(this.OnTextBoxLeave);
            // 
            // txtYZFrames
            // 
            this.txtYZFrames.Location = new System.Drawing.Point(107, 69);
            this.txtYZFrames.Name = "txtYZFrames";
            this.txtYZFrames.Size = new System.Drawing.Size(37, 20);
            this.txtYZFrames.TabIndex = 8;
            this.txtYZFrames.Leave += new System.EventHandler(this.OnTextBoxLeave);
            // 
            // txtXZFrames
            // 
            this.txtXZFrames.Location = new System.Drawing.Point(107, 43);
            this.txtXZFrames.Name = "txtXZFrames";
            this.txtXZFrames.Size = new System.Drawing.Size(37, 20);
            this.txtXZFrames.TabIndex = 7;
            this.txtXZFrames.Leave += new System.EventHandler(this.OnTextBoxLeave);
            // 
            // txtXYFrames
            // 
            this.txtXYFrames.Location = new System.Drawing.Point(107, 17);
            this.txtXYFrames.Name = "txtXYFrames";
            this.txtXYFrames.Size = new System.Drawing.Size(37, 20);
            this.txtXYFrames.TabIndex = 6;
            this.txtXYFrames.Leave += new System.EventHandler(this.OnTextBoxLeave);
            // 
            // chkViewYRot
            // 
            this.chkViewYRot.AutoSize = true;
            this.chkViewYRot.Location = new System.Drawing.Point(186, 17);
            this.chkViewYRot.Name = "chkViewYRot";
            this.chkViewYRot.Size = new System.Drawing.Size(82, 17);
            this.chkViewYRot.TabIndex = 5;
            this.chkViewYRot.Text = "Y rot frames";
            this.chkViewYRot.UseVisualStyleBackColor = true;
            // 
            // chkViewYZ
            // 
            this.chkViewYZ.AutoSize = true;
            this.chkViewYZ.Checked = true;
            this.chkViewYZ.CheckState = System.Windows.Forms.CheckState.Checked;
            this.chkViewYZ.Location = new System.Drawing.Point(6, 71);
            this.chkViewYZ.Name = "chkViewYZ";
            this.chkViewYZ.Size = new System.Drawing.Size(74, 17);
            this.chkViewYZ.TabIndex = 4;
            this.chkViewYZ.Text = "YZ frames";
            this.chkViewYZ.UseVisualStyleBackColor = true;
            // 
            // chkViewXZ
            // 
            this.chkViewXZ.AutoSize = true;
            this.chkViewXZ.Checked = true;
            this.chkViewXZ.CheckState = System.Windows.Forms.CheckState.Checked;
            this.chkViewXZ.Location = new System.Drawing.Point(6, 45);
            this.chkViewXZ.Name = "chkViewXZ";
            this.chkViewXZ.Size = new System.Drawing.Size(74, 17);
            this.chkViewXZ.TabIndex = 3;
            this.chkViewXZ.Text = "XZ frames";
            this.chkViewXZ.UseVisualStyleBackColor = true;
            // 
            // chkViewXY
            // 
            this.chkViewXY.AutoSize = true;
            this.chkViewXY.Checked = true;
            this.chkViewXY.CheckState = System.Windows.Forms.CheckState.Checked;
            this.chkViewXY.Location = new System.Drawing.Point(6, 19);
            this.chkViewXY.Name = "chkViewXY";
            this.chkViewXY.Size = new System.Drawing.Size(74, 17);
            this.chkViewXY.TabIndex = 2;
            this.chkViewXY.Text = "XY frames";
            this.chkViewXY.UseVisualStyleBackColor = true;
            // 
            // txtComments
            // 
            this.txtComments.Location = new System.Drawing.Point(86, 60);
            this.txtComments.Multiline = true;
            this.txtComments.Name = "txtComments";
            this.txtComments.Size = new System.Drawing.Size(260, 46);
            this.txtComments.TabIndex = 7;
            this.txtComments.Leave += new System.EventHandler(this.OnTxtCommentChanged);
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(15, 63);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(56, 13);
            this.label2.TabIndex = 8;
            this.label2.Text = "Comments";
            // 
            // MovieForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(356, 214);
            this.ControlBox = false;
            this.Controls.Add(this.label2);
            this.Controls.Add(this.txtComments);
            this.Controls.Add(this.groupViewControl);
            this.Controls.Add(this.pbGeneration);
            this.Controls.Add(this.btnMake);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.trkSpeed);
            this.Controls.Add(this.chkBrick);
            this.Controls.Add(this.chkEvent);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedToolWindow;
            this.Name = "MovieForm";
            this.Text = "Movie Control";
            ((System.ComponentModel.ISupportInitialize)(this.trkSpeed)).EndInit();
            this.groupViewControl.ResumeLayout(false);
            this.groupViewControl.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.CheckBox chkEvent;
        private System.Windows.Forms.CheckBox chkBrick;
        private System.Windows.Forms.TrackBar trkSpeed;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Button btnMake;
        private System.Windows.Forms.ProgressBar pbGeneration;
        private System.Windows.Forms.GroupBox groupViewControl;
        private System.Windows.Forms.TextBox txtXYPanFrames;
        private System.Windows.Forms.TextBox txtZRotFrames;
        private System.Windows.Forms.TextBox txtYZFrames;
        private System.Windows.Forms.TextBox txtXZFrames;
        private System.Windows.Forms.TextBox txtXYFrames;
        private System.Windows.Forms.CheckBox chkViewYRot;
        private System.Windows.Forms.CheckBox chkViewYZ;
        private System.Windows.Forms.CheckBox chkViewXZ;
        private System.Windows.Forms.CheckBox chkViewXY;
        private System.Windows.Forms.CheckBox chkViewXYPan;
        private System.Windows.Forms.CheckBox chkAddViewExplanation;
        private System.Windows.Forms.TextBox txtComments;
        private System.Windows.Forms.Label label2;
    }
}