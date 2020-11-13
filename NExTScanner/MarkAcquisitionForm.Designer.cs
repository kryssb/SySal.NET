namespace SySal.Executables.NExTScanner
{
    partial class MarkAcquisitionForm
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
            this.label2 = new System.Windows.Forms.Label();
            this.txtMarkString = new System.Windows.Forms.TextBox();
            this.btnStart = new SySal.SySalNExTControls.SySalButton();
            this.btnStop = new SySal.SySalNExTControls.SySalButton();
            this.btnDone = new SySal.SySalNExTControls.SySalButton();
            this.btnCancel = new SySal.SySalNExTControls.SySalButton();
            this.txtID = new System.Windows.Forms.TextBox();
            this.label1 = new System.Windows.Forms.Label();
            this.label3 = new System.Windows.Forms.Label();
            this.txtMapX = new System.Windows.Forms.TextBox();
            this.txtMapY = new System.Windows.Forms.TextBox();
            this.label4 = new System.Windows.Forms.Label();
            this.txtSide = new System.Windows.Forms.TextBox();
            this.txtExpY = new System.Windows.Forms.TextBox();
            this.label5 = new System.Windows.Forms.Label();
            this.txtExpX = new System.Windows.Forms.TextBox();
            this.txtFoundY = new System.Windows.Forms.TextBox();
            this.label6 = new System.Windows.Forms.Label();
            this.txtFoundX = new System.Windows.Forms.TextBox();
            this.txtFlag = new System.Windows.Forms.TextBox();
            this.label7 = new System.Windows.Forms.Label();
            this.btnSetX = new SySal.SySalNExTControls.SySalButton();
            this.btnSetY = new SySal.SySalNExTControls.SySalButton();
            this.btnSetNotFound = new SySal.SySalNExTControls.SySalButton();
            this.btnPrevious = new SySal.SySalNExTControls.SySalButton();
            this.btnNext = new SySal.SySalNExTControls.SySalButton();
            this.btnPause = new SySal.SySalNExTControls.SySalButton();
            this.btnSetNotSearched = new SySal.SySalNExTControls.SySalButton();
            this.textBox1 = new System.Windows.Forms.TextBox();
            this.btnContinue = new SySal.SySalNExTControls.SySalButton();
            this.SuspendLayout();
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.BackColor = System.Drawing.Color.Transparent;
            this.label2.Location = new System.Drawing.Point(12, 34);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(90, 21);
            this.label2.TabIndex = 7;
            this.label2.Text = "Mark string";
            // 
            // txtMarkString
            // 
            this.txtMarkString.BackColor = System.Drawing.Color.GhostWhite;
            this.txtMarkString.Location = new System.Drawing.Point(108, 31);
            this.txtMarkString.Multiline = true;
            this.txtMarkString.Name = "txtMarkString";
            this.txtMarkString.Size = new System.Drawing.Size(526, 66);
            this.txtMarkString.TabIndex = 8;
            this.txtMarkString.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // btnStart
            // 
            this.btnStart.AutoSize = true;
            this.btnStart.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnStart.BackColor = System.Drawing.Color.Transparent;
            this.btnStart.FocusedColor = System.Drawing.Color.Navy;
            this.btnStart.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.btnStart.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnStart.Location = new System.Drawing.Point(15, 113);
            this.btnStart.Margin = new System.Windows.Forms.Padding(6);
            this.btnStart.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnStart.Name = "btnStart";
            this.btnStart.Size = new System.Drawing.Size(41, 25);
            this.btnStart.TabIndex = 110;
            this.btnStart.Text = "Start";
            this.btnStart.Click += new System.EventHandler(this.btnStart_Click);
            // 
            // btnStop
            // 
            this.btnStop.AutoSize = true;
            this.btnStop.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnStop.BackColor = System.Drawing.Color.Transparent;
            this.btnStop.Enabled = false;
            this.btnStop.FocusedColor = System.Drawing.Color.Navy;
            this.btnStop.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.btnStop.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnStop.Location = new System.Drawing.Point(15, 141);
            this.btnStop.Margin = new System.Windows.Forms.Padding(6);
            this.btnStop.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnStop.Name = "btnStop";
            this.btnStop.Size = new System.Drawing.Size(41, 25);
            this.btnStop.TabIndex = 111;
            this.btnStop.Text = "Stop";
            // 
            // btnDone
            // 
            this.btnDone.AutoSize = true;
            this.btnDone.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnDone.BackColor = System.Drawing.Color.Transparent;
            this.btnDone.Enabled = false;
            this.btnDone.FocusedColor = System.Drawing.Color.Navy;
            this.btnDone.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.btnDone.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnDone.Location = new System.Drawing.Point(15, 320);
            this.btnDone.Margin = new System.Windows.Forms.Padding(6);
            this.btnDone.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnDone.Name = "btnDone";
            this.btnDone.Size = new System.Drawing.Size(46, 25);
            this.btnDone.TabIndex = 112;
            this.btnDone.Text = "Done";
            this.btnDone.Click += new System.EventHandler(this.btnDone_Click);
            // 
            // btnCancel
            // 
            this.btnCancel.AutoSize = true;
            this.btnCancel.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnCancel.BackColor = System.Drawing.Color.Transparent;
            this.btnCancel.FocusedColor = System.Drawing.Color.Navy;
            this.btnCancel.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.btnCancel.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnCancel.Location = new System.Drawing.Point(579, 322);
            this.btnCancel.Margin = new System.Windows.Forms.Padding(6);
            this.btnCancel.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnCancel.Name = "btnCancel";
            this.btnCancel.Size = new System.Drawing.Size(55, 25);
            this.btnCancel.TabIndex = 113;
            this.btnCancel.Text = "Cancel";
            this.btnCancel.Click += new System.EventHandler(this.btnCancel_Click);
            // 
            // txtID
            // 
            this.txtID.BackColor = System.Drawing.Color.GhostWhite;
            this.txtID.Location = new System.Drawing.Point(175, 119);
            this.txtID.Name = "txtID";
            this.txtID.ReadOnly = true;
            this.txtID.Size = new System.Drawing.Size(42, 29);
            this.txtID.TabIndex = 114;
            this.txtID.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.BackColor = System.Drawing.Color.Transparent;
            this.label1.Location = new System.Drawing.Point(104, 122);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(65, 21);
            this.label1.TabIndex = 115;
            this.label1.Text = "Mark ID";
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.BackColor = System.Drawing.Color.Transparent;
            this.label3.Location = new System.Drawing.Point(234, 122);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(108, 21);
            this.label3.TabIndex = 117;
            this.label3.Text = "Map Pos (X/Y)";
            // 
            // txtMapX
            // 
            this.txtMapX.BackColor = System.Drawing.Color.GhostWhite;
            this.txtMapX.Location = new System.Drawing.Point(351, 119);
            this.txtMapX.Name = "txtMapX";
            this.txtMapX.ReadOnly = true;
            this.txtMapX.Size = new System.Drawing.Size(68, 29);
            this.txtMapX.TabIndex = 116;
            this.txtMapX.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // txtMapY
            // 
            this.txtMapY.BackColor = System.Drawing.Color.GhostWhite;
            this.txtMapY.Location = new System.Drawing.Point(425, 119);
            this.txtMapY.Name = "txtMapY";
            this.txtMapY.ReadOnly = true;
            this.txtMapY.Size = new System.Drawing.Size(68, 29);
            this.txtMapY.TabIndex = 118;
            this.txtMapY.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.BackColor = System.Drawing.Color.Transparent;
            this.label4.Location = new System.Drawing.Point(516, 122);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(40, 21);
            this.label4.TabIndex = 119;
            this.label4.Text = "Side";
            // 
            // txtSide
            // 
            this.txtSide.BackColor = System.Drawing.Color.GhostWhite;
            this.txtSide.Location = new System.Drawing.Point(562, 119);
            this.txtSide.Name = "txtSide";
            this.txtSide.ReadOnly = true;
            this.txtSide.Size = new System.Drawing.Size(72, 29);
            this.txtSide.TabIndex = 120;
            this.txtSide.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // txtExpY
            // 
            this.txtExpY.BackColor = System.Drawing.Color.GhostWhite;
            this.txtExpY.Location = new System.Drawing.Point(425, 154);
            this.txtExpY.Name = "txtExpY";
            this.txtExpY.ReadOnly = true;
            this.txtExpY.Size = new System.Drawing.Size(68, 29);
            this.txtExpY.TabIndex = 123;
            this.txtExpY.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.BackColor = System.Drawing.Color.Transparent;
            this.label5.Location = new System.Drawing.Point(164, 157);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(178, 21);
            this.label5.TabIndex = 122;
            this.label5.Text = "Expected Stage Pos (X,Y)";
            // 
            // txtExpX
            // 
            this.txtExpX.BackColor = System.Drawing.Color.GhostWhite;
            this.txtExpX.Location = new System.Drawing.Point(351, 154);
            this.txtExpX.Name = "txtExpX";
            this.txtExpX.ReadOnly = true;
            this.txtExpX.Size = new System.Drawing.Size(68, 29);
            this.txtExpX.TabIndex = 121;
            this.txtExpX.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // txtFoundY
            // 
            this.txtFoundY.BackColor = System.Drawing.Color.GhostWhite;
            this.txtFoundY.Location = new System.Drawing.Point(425, 189);
            this.txtFoundY.Name = "txtFoundY";
            this.txtFoundY.ReadOnly = true;
            this.txtFoundY.Size = new System.Drawing.Size(68, 29);
            this.txtFoundY.TabIndex = 126;
            this.txtFoundY.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.BackColor = System.Drawing.Color.Transparent;
            this.label6.Location = new System.Drawing.Point(164, 192);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(161, 21);
            this.label6.TabIndex = 125;
            this.label6.Text = "Found Stage Pos (X,Y)";
            // 
            // txtFoundX
            // 
            this.txtFoundX.BackColor = System.Drawing.Color.GhostWhite;
            this.txtFoundX.Location = new System.Drawing.Point(351, 189);
            this.txtFoundX.Name = "txtFoundX";
            this.txtFoundX.ReadOnly = true;
            this.txtFoundX.Size = new System.Drawing.Size(68, 29);
            this.txtFoundX.TabIndex = 124;
            this.txtFoundX.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // txtFlag
            // 
            this.txtFlag.BackColor = System.Drawing.Color.GhostWhite;
            this.txtFlag.Location = new System.Drawing.Point(562, 189);
            this.txtFlag.Name = "txtFlag";
            this.txtFlag.ReadOnly = true;
            this.txtFlag.Size = new System.Drawing.Size(72, 29);
            this.txtFlag.TabIndex = 128;
            this.txtFlag.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.BackColor = System.Drawing.Color.Transparent;
            this.label7.Location = new System.Drawing.Point(516, 192);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(39, 21);
            this.label7.TabIndex = 127;
            this.label7.Text = "Flag";
            // 
            // btnSetX
            // 
            this.btnSetX.AutoSize = true;
            this.btnSetX.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnSetX.BackColor = System.Drawing.Color.Transparent;
            this.btnSetX.Enabled = false;
            this.btnSetX.FocusedColor = System.Drawing.Color.Navy;
            this.btnSetX.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.btnSetX.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnSetX.Location = new System.Drawing.Point(203, 233);
            this.btnSetX.Margin = new System.Windows.Forms.Padding(6);
            this.btnSetX.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnSetX.Name = "btnSetX";
            this.btnSetX.Size = new System.Drawing.Size(40, 25);
            this.btnSetX.TabIndex = 129;
            this.btnSetX.Text = "SetX";
            this.btnSetX.Click += new System.EventHandler(this.btnSetX_Click);
            // 
            // btnSetY
            // 
            this.btnSetY.AutoSize = true;
            this.btnSetY.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnSetY.BackColor = System.Drawing.Color.Transparent;
            this.btnSetY.Enabled = false;
            this.btnSetY.FocusedColor = System.Drawing.Color.Navy;
            this.btnSetY.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.btnSetY.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnSetY.Location = new System.Drawing.Point(276, 233);
            this.btnSetY.Margin = new System.Windows.Forms.Padding(6);
            this.btnSetY.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnSetY.Name = "btnSetY";
            this.btnSetY.Size = new System.Drawing.Size(39, 25);
            this.btnSetY.TabIndex = 130;
            this.btnSetY.Text = "SetY";
            this.btnSetY.Click += new System.EventHandler(this.btnSetY_Click);
            // 
            // btnSetNotFound
            // 
            this.btnSetNotFound.AutoSize = true;
            this.btnSetNotFound.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnSetNotFound.BackColor = System.Drawing.Color.Transparent;
            this.btnSetNotFound.Enabled = false;
            this.btnSetNotFound.FocusedColor = System.Drawing.Color.Navy;
            this.btnSetNotFound.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.btnSetNotFound.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnSetNotFound.Location = new System.Drawing.Point(351, 233);
            this.btnSetNotFound.Margin = new System.Windows.Forms.Padding(6);
            this.btnSetNotFound.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnSetNotFound.Name = "btnSetNotFound";
            this.btnSetNotFound.Size = new System.Drawing.Size(112, 25);
            this.btnSetNotFound.TabIndex = 132;
            this.btnSetNotFound.Text = "Set Not Found";
            this.btnSetNotFound.Click += new System.EventHandler(this.btnSetNotFound_Click);
            // 
            // btnPrevious
            // 
            this.btnPrevious.AutoSize = true;
            this.btnPrevious.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnPrevious.BackColor = System.Drawing.Color.Transparent;
            this.btnPrevious.Enabled = false;
            this.btnPrevious.FocusedColor = System.Drawing.Color.Navy;
            this.btnPrevious.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.btnPrevious.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnPrevious.Location = new System.Drawing.Point(122, 233);
            this.btnPrevious.Margin = new System.Windows.Forms.Padding(6);
            this.btnPrevious.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnPrevious.Name = "btnPrevious";
            this.btnPrevious.Size = new System.Drawing.Size(18, 25);
            this.btnPrevious.TabIndex = 133;
            this.btnPrevious.Text = "<";
            this.btnPrevious.Click += new System.EventHandler(this.btnPrevious_Click);
            // 
            // btnNext
            // 
            this.btnNext.AutoSize = true;
            this.btnNext.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnNext.BackColor = System.Drawing.Color.Transparent;
            this.btnNext.Enabled = false;
            this.btnNext.FocusedColor = System.Drawing.Color.Navy;
            this.btnNext.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.btnNext.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnNext.Location = new System.Drawing.Point(151, 233);
            this.btnNext.Margin = new System.Windows.Forms.Padding(6);
            this.btnNext.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnNext.Name = "btnNext";
            this.btnNext.Size = new System.Drawing.Size(18, 25);
            this.btnNext.TabIndex = 134;
            this.btnNext.Text = ">";
            this.btnNext.Click += new System.EventHandler(this.btnNext_Click);
            // 
            // btnPause
            // 
            this.btnPause.AutoSize = true;
            this.btnPause.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnPause.BackColor = System.Drawing.Color.Transparent;
            this.btnPause.Enabled = false;
            this.btnPause.FocusedColor = System.Drawing.Color.Navy;
            this.btnPause.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.btnPause.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnPause.Location = new System.Drawing.Point(15, 169);
            this.btnPause.Margin = new System.Windows.Forms.Padding(6);
            this.btnPause.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnPause.Name = "btnPause";
            this.btnPause.Size = new System.Drawing.Size(49, 25);
            this.btnPause.TabIndex = 135;
            this.btnPause.Text = "Pause";
            this.btnPause.Click += new System.EventHandler(this.btnPause_Click);
            // 
            // btnSetNotSearched
            // 
            this.btnSetNotSearched.AutoSize = true;
            this.btnSetNotSearched.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnSetNotSearched.BackColor = System.Drawing.Color.Transparent;
            this.btnSetNotSearched.Enabled = false;
            this.btnSetNotSearched.FocusedColor = System.Drawing.Color.Navy;
            this.btnSetNotSearched.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.btnSetNotSearched.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnSetNotSearched.Location = new System.Drawing.Point(501, 233);
            this.btnSetNotSearched.Margin = new System.Windows.Forms.Padding(6);
            this.btnSetNotSearched.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnSetNotSearched.Name = "btnSetNotSearched";
            this.btnSetNotSearched.Size = new System.Drawing.Size(133, 25);
            this.btnSetNotSearched.TabIndex = 136;
            this.btnSetNotSearched.Text = "Set Not Searched";
            this.btnSetNotSearched.Click += new System.EventHandler(this.btnSetNotSearched_Click);
            // 
            // textBox1
            // 
            this.textBox1.BackColor = System.Drawing.Color.Wheat;
            this.textBox1.Location = new System.Drawing.Point(16, 282);
            this.textBox1.Name = "textBox1";
            this.textBox1.ReadOnly = true;
            this.textBox1.Size = new System.Drawing.Size(618, 29);
            this.textBox1.TabIndex = 137;
            this.textBox1.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // btnContinue
            // 
            this.btnContinue.AutoSize = true;
            this.btnContinue.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnContinue.BackColor = System.Drawing.Color.Transparent;
            this.btnContinue.Enabled = false;
            this.btnContinue.FocusedColor = System.Drawing.Color.Navy;
            this.btnContinue.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.btnContinue.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnContinue.Location = new System.Drawing.Point(15, 197);
            this.btnContinue.Margin = new System.Windows.Forms.Padding(6);
            this.btnContinue.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnContinue.Name = "btnContinue";
            this.btnContinue.Size = new System.Drawing.Size(73, 25);
            this.btnContinue.TabIndex = 138;
            this.btnContinue.Text = "Continue";
            this.btnContinue.Click += new System.EventHandler(this.btnContinue_Click);
            // 
            // MarkAcquisitionForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(9F, 21F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(646, 362);
            this.Controls.Add(this.btnContinue);
            this.Controls.Add(this.textBox1);
            this.Controls.Add(this.btnSetNotSearched);
            this.Controls.Add(this.btnPause);
            this.Controls.Add(this.btnNext);
            this.Controls.Add(this.btnPrevious);
            this.Controls.Add(this.btnSetNotFound);
            this.Controls.Add(this.btnSetY);
            this.Controls.Add(this.btnSetX);
            this.Controls.Add(this.txtFlag);
            this.Controls.Add(this.label7);
            this.Controls.Add(this.txtFoundY);
            this.Controls.Add(this.label6);
            this.Controls.Add(this.txtFoundX);
            this.Controls.Add(this.txtExpY);
            this.Controls.Add(this.label5);
            this.Controls.Add(this.txtExpX);
            this.Controls.Add(this.txtSide);
            this.Controls.Add(this.label4);
            this.Controls.Add(this.txtMapY);
            this.Controls.Add(this.label3);
            this.Controls.Add(this.txtMapX);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.txtID);
            this.Controls.Add(this.btnCancel);
            this.Controls.Add(this.btnDone);
            this.Controls.Add(this.btnStop);
            this.Controls.Add(this.btnStart);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.txtMarkString);
            this.DialogCaption = "Mark Acquisition";
            this.Name = "MarkAcquisitionForm";
            this.NoCloseButton = true;
            this.Load += new System.EventHandler(this.OnLoad);
            this.Controls.SetChildIndex(this.txtMarkString, 0);
            this.Controls.SetChildIndex(this.label2, 0);
            this.Controls.SetChildIndex(this.btnStart, 0);
            this.Controls.SetChildIndex(this.btnStop, 0);
            this.Controls.SetChildIndex(this.btnDone, 0);
            this.Controls.SetChildIndex(this.btnCancel, 0);
            this.Controls.SetChildIndex(this.txtID, 0);
            this.Controls.SetChildIndex(this.label1, 0);
            this.Controls.SetChildIndex(this.txtMapX, 0);
            this.Controls.SetChildIndex(this.label3, 0);
            this.Controls.SetChildIndex(this.txtMapY, 0);
            this.Controls.SetChildIndex(this.label4, 0);
            this.Controls.SetChildIndex(this.txtSide, 0);
            this.Controls.SetChildIndex(this.txtExpX, 0);
            this.Controls.SetChildIndex(this.label5, 0);
            this.Controls.SetChildIndex(this.txtExpY, 0);
            this.Controls.SetChildIndex(this.txtFoundX, 0);
            this.Controls.SetChildIndex(this.label6, 0);
            this.Controls.SetChildIndex(this.txtFoundY, 0);
            this.Controls.SetChildIndex(this.label7, 0);
            this.Controls.SetChildIndex(this.txtFlag, 0);
            this.Controls.SetChildIndex(this.btnSetX, 0);
            this.Controls.SetChildIndex(this.btnSetY, 0);
            this.Controls.SetChildIndex(this.btnSetNotFound, 0);
            this.Controls.SetChildIndex(this.btnPrevious, 0);
            this.Controls.SetChildIndex(this.btnNext, 0);
            this.Controls.SetChildIndex(this.btnPause, 0);
            this.Controls.SetChildIndex(this.btnSetNotSearched, 0);
            this.Controls.SetChildIndex(this.textBox1, 0);
            this.Controls.SetChildIndex(this.btnContinue, 0);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.TextBox txtMarkString;
        private SySalNExTControls.SySalButton btnStart;
        private SySalNExTControls.SySalButton btnStop;
        private SySalNExTControls.SySalButton btnDone;
        private SySalNExTControls.SySalButton btnCancel;
        private System.Windows.Forms.TextBox txtID;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.TextBox txtMapX;
        private System.Windows.Forms.TextBox txtMapY;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.TextBox txtSide;
        private System.Windows.Forms.TextBox txtExpY;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.TextBox txtExpX;
        private System.Windows.Forms.TextBox txtFoundY;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.TextBox txtFoundX;
        private System.Windows.Forms.TextBox txtFlag;
        private System.Windows.Forms.Label label7;
        private SySalNExTControls.SySalButton btnSetX;
        private SySalNExTControls.SySalButton btnSetY;
        private SySalNExTControls.SySalButton btnSetNotFound;
        private SySalNExTControls.SySalButton btnPrevious;
        private SySalNExTControls.SySalButton btnNext;
        private SySalNExTControls.SySalButton btnPause;
        private SySalNExTControls.SySalButton btnSetNotSearched;
        private System.Windows.Forms.TextBox textBox1;
        private SySalNExTControls.SySalButton btnContinue;
    }
}