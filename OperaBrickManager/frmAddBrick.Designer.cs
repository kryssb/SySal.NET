namespace OperaBrickManager
{
    partial class frmAddBrick
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
            this.lblNPlate = new System.Windows.Forms.Label();
            this.lblMaxZ = new System.Windows.Forms.Label();
            this.lblDownstreamPlate = new System.Windows.Forms.Label();
            this.lblBrickSet = new System.Windows.Forms.Label();
            this.cmbNumberOfPlates = new System.Windows.Forms.ComboBox();
            this.txtMaxZ = new System.Windows.Forms.TextBox();
            this.cmbDownstreamPlate = new System.Windows.Forms.ComboBox();
            this.dlgOpenFile = new System.Windows.Forms.OpenFileDialog();
            this.lblBrick = new System.Windows.Forms.Label();
            this.txtBrick = new System.Windows.Forms.TextBox();
            this.grbCompute = new System.Windows.Forms.GroupBox();
            this.btnClearAll = new System.Windows.Forms.Button();
            this.btnCompute = new System.Windows.Forms.Button();
            this.grpZeroXY = new System.Windows.Forms.GroupBox();
            this.btnLoadZeroCoordinate = new System.Windows.Forms.Button();
            this.btnDBLink = new System.Windows.Forms.Button();
            this.txtZeroY = new System.Windows.Forms.TextBox();
            this.lblZeroY = new System.Windows.Forms.Label();
            this.txtZeroX = new System.Windows.Forms.TextBox();
            this.lblZeroX = new System.Windows.Forms.Label();
            this.grpXY = new System.Windows.Forms.GroupBox();
            this.btnInputDataFiles = new System.Windows.Forms.Button();
            this.txtEmuMaxY = new System.Windows.Forms.TextBox();
            this.lblEmuMaxY = new System.Windows.Forms.Label();
            this.txtEmuMaxX = new System.Windows.Forms.TextBox();
            this.lblEmuMaxX = new System.Windows.Forms.Label();
            this.txtEmuMinY = new System.Windows.Forms.TextBox();
            this.lblEmuMinY = new System.Windows.Forms.Label();
            this.txtEmuMinX = new System.Windows.Forms.TextBox();
            this.lblEmuMinX = new System.Windows.Forms.Label();
            this.lblMinX = new System.Windows.Forms.Label();
            this.txtMinX = new System.Windows.Forms.TextBox();
            this.lblMinY = new System.Windows.Forms.Label();
            this.txtMinY = new System.Windows.Forms.TextBox();
            this.lblMaxX = new System.Windows.Forms.Label();
            this.lbMaxY = new System.Windows.Forms.Label();
            this.txtMaxX = new System.Windows.Forms.TextBox();
            this.txtMaxY = new System.Windows.Forms.TextBox();
            this.grpComputedValues = new System.Windows.Forms.GroupBox();
            this.grpInitValues = new System.Windows.Forms.GroupBox();
            this.txtBrickSet = new System.Windows.Forms.TextBox();
            this.btnOk = new System.Windows.Forms.Button();
            this.btnCancel = new System.Windows.Forms.Button();
            this.grbCompute.SuspendLayout();
            this.grpZeroXY.SuspendLayout();
            this.grpXY.SuspendLayout();
            this.grpComputedValues.SuspendLayout();
            this.grpInitValues.SuspendLayout();
            this.SuspendLayout();
            // 
            // lblNPlate
            // 
            this.lblNPlate.AutoSize = true;
            this.lblNPlate.Location = new System.Drawing.Point(165, 26);
            this.lblNPlate.Name = "lblNPlate";
            this.lblNPlate.Size = new System.Drawing.Size(93, 13);
            this.lblNPlate.TabIndex = 3;
            this.lblNPlate.Text = "Number Of Plates:";
            // 
            // lblMaxZ
            // 
            this.lblMaxZ.AutoSize = true;
            this.lblMaxZ.Location = new System.Drawing.Point(340, 27);
            this.lblMaxZ.Name = "lblMaxZ";
            this.lblMaxZ.Size = new System.Drawing.Size(40, 13);
            this.lblMaxZ.TabIndex = 5;
            this.lblMaxZ.Text = "Max Z:";
            // 
            // lblDownstreamPlate
            // 
            this.lblDownstreamPlate.AutoSize = true;
            this.lblDownstreamPlate.Location = new System.Drawing.Point(162, 73);
            this.lblDownstreamPlate.Name = "lblDownstreamPlate";
            this.lblDownstreamPlate.Size = new System.Drawing.Size(96, 13);
            this.lblDownstreamPlate.TabIndex = 9;
            this.lblDownstreamPlate.Text = "Downstream Plate:";
            // 
            // lblBrickSet
            // 
            this.lblBrickSet.AutoSize = true;
            this.lblBrickSet.Location = new System.Drawing.Point(10, 73);
            this.lblBrickSet.Name = "lblBrickSet";
            this.lblBrickSet.Size = new System.Drawing.Size(53, 13);
            this.lblBrickSet.TabIndex = 7;
            this.lblBrickSet.Text = "Brick Set:";
            // 
            // cmbNumberOfPlates
            // 
            this.cmbNumberOfPlates.FormattingEnabled = true;
            this.cmbNumberOfPlates.Items.AddRange(new object[] {
            "2",
            "57",
            "93"});
            this.cmbNumberOfPlates.Location = new System.Drawing.Point(262, 26);
            this.cmbNumberOfPlates.Name = "cmbNumberOfPlates";
            this.cmbNumberOfPlates.Size = new System.Drawing.Size(59, 21);
            this.cmbNumberOfPlates.TabIndex = 4;
            // 
            // txtMaxZ
            // 
            this.txtMaxZ.CausesValidation = false;
            this.txtMaxZ.Location = new System.Drawing.Point(385, 27);
            this.txtMaxZ.Name = "txtMaxZ";
            this.txtMaxZ.Size = new System.Drawing.Size(57, 20);
            this.txtMaxZ.TabIndex = 6;
            // 
            // cmbDownstreamPlate
            // 
            this.cmbDownstreamPlate.FormattingEnabled = true;
            this.cmbDownstreamPlate.Items.AddRange(new object[] {
            "1",
            "57"});
            this.cmbDownstreamPlate.Location = new System.Drawing.Point(262, 73);
            this.cmbDownstreamPlate.Name = "cmbDownstreamPlate";
            this.cmbDownstreamPlate.Size = new System.Drawing.Size(59, 21);
            this.cmbDownstreamPlate.TabIndex = 10;
            // 
            // dlgOpenFile
            // 
            this.dlgOpenFile.Filter = "Text file(s)|*.txt";
            this.dlgOpenFile.Title = "Open file...";
            // 
            // lblBrick
            // 
            this.lblBrick.AutoSize = true;
            this.lblBrick.Location = new System.Drawing.Point(13, 31);
            this.lblBrick.Name = "lblBrick";
            this.lblBrick.Size = new System.Drawing.Size(45, 13);
            this.lblBrick.TabIndex = 1;
            this.lblBrick.Text = "ID Brick";
            // 
            // txtBrick
            // 
            this.txtBrick.CausesValidation = false;
            this.txtBrick.Location = new System.Drawing.Point(67, 28);
            this.txtBrick.Name = "txtBrick";
            this.txtBrick.Size = new System.Drawing.Size(84, 20);
            this.txtBrick.TabIndex = 2;
            // 
            // grbCompute
            // 
            this.grbCompute.Controls.Add(this.btnClearAll);
            this.grbCompute.Controls.Add(this.btnCompute);
            this.grbCompute.Controls.Add(this.grpZeroXY);
            this.grbCompute.Controls.Add(this.grpXY);
            this.grbCompute.Location = new System.Drawing.Point(20, 140);
            this.grbCompute.Name = "grbCompute";
            this.grbCompute.Size = new System.Drawing.Size(455, 246);
            this.grbCompute.TabIndex = 11;
            this.grbCompute.TabStop = false;
            this.grbCompute.Text = "Emulsion coordinates";
            // 
            // btnClearAll
            // 
            this.btnClearAll.Location = new System.Drawing.Point(378, 77);
            this.btnClearAll.Name = "btnClearAll";
            this.btnClearAll.Size = new System.Drawing.Size(73, 32);
            this.btnClearAll.TabIndex = 30;
            this.btnClearAll.Text = "Clear All";
            this.btnClearAll.UseVisualStyleBackColor = true;
            this.btnClearAll.Click += new System.EventHandler(this.btnClearAll_Click);
            // 
            // btnCompute
            // 
            this.btnCompute.Location = new System.Drawing.Point(376, 27);
            this.btnCompute.Name = "btnCompute";
            this.btnCompute.Size = new System.Drawing.Size(73, 32);
            this.btnCompute.TabIndex = 29;
            this.btnCompute.Text = "Compute";
            this.btnCompute.UseVisualStyleBackColor = true;
            this.btnCompute.Click += new System.EventHandler(this.btnCompute_Click);
            // 
            // grpZeroXY
            // 
            this.grpZeroXY.Controls.Add(this.btnLoadZeroCoordinate);
            this.grpZeroXY.Controls.Add(this.btnDBLink);
            this.grpZeroXY.Controls.Add(this.txtZeroY);
            this.grpZeroXY.Controls.Add(this.lblZeroY);
            this.grpZeroXY.Controls.Add(this.txtZeroX);
            this.grpZeroXY.Controls.Add(this.lblZeroX);
            this.grpZeroXY.Location = new System.Drawing.Point(202, 24);
            this.grpZeroXY.Name = "grpZeroXY";
            this.grpZeroXY.Size = new System.Drawing.Size(168, 212);
            this.grpZeroXY.TabIndex = 22;
            this.grpZeroXY.TabStop = false;
            this.grpZeroXY.Tag = "0";
            this.grpZeroXY.Text = "Zero XY Value";
            // 
            // btnLoadZeroCoordinate
            // 
            this.btnLoadZeroCoordinate.Location = new System.Drawing.Point(19, 169);
            this.btnLoadZeroCoordinate.Name = "btnLoadZeroCoordinate";
            this.btnLoadZeroCoordinate.Size = new System.Drawing.Size(135, 30);
            this.btnLoadZeroCoordinate.TabIndex = 28;
            this.btnLoadZeroCoordinate.Text = "Load from file";
            this.btnLoadZeroCoordinate.UseVisualStyleBackColor = true;
            this.btnLoadZeroCoordinate.Click += new System.EventHandler(this.btnLoadZeroCoordinate_Click);
            // 
            // btnDBLink
            // 
            this.btnDBLink.Location = new System.Drawing.Point(19, 122);
            this.btnDBLink.Name = "btnDBLink";
            this.btnDBLink.Size = new System.Drawing.Size(135, 30);
            this.btnDBLink.TabIndex = 27;
            this.btnDBLink.Text = "Load from database";
            this.btnDBLink.UseVisualStyleBackColor = true;
            this.btnDBLink.Click += new System.EventHandler(this.btnDBLink_Click);
            // 
            // txtZeroY
            // 
            this.txtZeroY.CausesValidation = false;
            this.txtZeroY.Location = new System.Drawing.Point(74, 63);
            this.txtZeroY.Name = "txtZeroY";
            this.txtZeroY.Size = new System.Drawing.Size(80, 20);
            this.txtZeroY.TabIndex = 26;
            this.txtZeroY.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.txtZeroY.TextChanged += new System.EventHandler(this.Filled2);
            // 
            // lblZeroY
            // 
            this.lblZeroY.AutoSize = true;
            this.lblZeroY.Location = new System.Drawing.Point(16, 66);
            this.lblZeroY.Name = "lblZeroY";
            this.lblZeroY.Size = new System.Drawing.Size(36, 13);
            this.lblZeroY.TabIndex = 25;
            this.lblZeroY.Text = "ZeroY";
            // 
            // txtZeroX
            // 
            this.txtZeroX.CausesValidation = false;
            this.txtZeroX.Location = new System.Drawing.Point(74, 29);
            this.txtZeroX.Name = "txtZeroX";
            this.txtZeroX.Size = new System.Drawing.Size(80, 20);
            this.txtZeroX.TabIndex = 24;
            this.txtZeroX.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.txtZeroX.TextChanged += new System.EventHandler(this.Filled2);
            // 
            // lblZeroX
            // 
            this.lblZeroX.AutoSize = true;
            this.lblZeroX.Location = new System.Drawing.Point(16, 32);
            this.lblZeroX.Name = "lblZeroX";
            this.lblZeroX.Size = new System.Drawing.Size(36, 13);
            this.lblZeroX.TabIndex = 23;
            this.lblZeroX.Text = "ZeroX";
            // 
            // grpXY
            // 
            this.grpXY.Controls.Add(this.btnInputDataFiles);
            this.grpXY.Controls.Add(this.txtEmuMaxY);
            this.grpXY.Controls.Add(this.lblEmuMaxY);
            this.grpXY.Controls.Add(this.txtEmuMaxX);
            this.grpXY.Controls.Add(this.lblEmuMaxX);
            this.grpXY.Controls.Add(this.txtEmuMinY);
            this.grpXY.Controls.Add(this.lblEmuMinY);
            this.grpXY.Controls.Add(this.txtEmuMinX);
            this.grpXY.Controls.Add(this.lblEmuMinX);
            this.grpXY.Location = new System.Drawing.Point(14, 27);
            this.grpXY.Name = "grpXY";
            this.grpXY.Size = new System.Drawing.Size(168, 209);
            this.grpXY.TabIndex = 12;
            this.grpXY.TabStop = false;
            this.grpXY.Tag = "0";
            this.grpXY.Text = "Local X-Y Extreme";
            // 
            // btnInputDataFiles
            // 
            this.btnInputDataFiles.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btnInputDataFiles.Location = new System.Drawing.Point(13, 166);
            this.btnInputDataFiles.Name = "btnInputDataFiles";
            this.btnInputDataFiles.Size = new System.Drawing.Size(139, 30);
            this.btnInputDataFiles.TabIndex = 21;
            this.btnInputDataFiles.Text = "Load from file";
            this.btnInputDataFiles.UseVisualStyleBackColor = true;
            this.btnInputDataFiles.Click += new System.EventHandler(this.btnInputDataFiles_Click);
            // 
            // txtEmuMaxY
            // 
            this.txtEmuMaxY.CausesValidation = false;
            this.txtEmuMaxY.Location = new System.Drawing.Point(72, 127);
            this.txtEmuMaxY.Name = "txtEmuMaxY";
            this.txtEmuMaxY.Size = new System.Drawing.Size(80, 20);
            this.txtEmuMaxY.TabIndex = 20;
            this.txtEmuMaxY.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.txtEmuMaxY.TextChanged += new System.EventHandler(this.Filled1);
            // 
            // lblEmuMaxY
            // 
            this.lblEmuMaxY.AutoSize = true;
            this.lblEmuMaxY.Location = new System.Drawing.Point(14, 130);
            this.lblEmuMaxY.Name = "lblEmuMaxY";
            this.lblEmuMaxY.Size = new System.Drawing.Size(34, 13);
            this.lblEmuMaxY.TabIndex = 19;
            this.lblEmuMaxY.Text = "MaxY";
            // 
            // txtEmuMaxX
            // 
            this.txtEmuMaxX.CausesValidation = false;
            this.txtEmuMaxX.Location = new System.Drawing.Point(72, 93);
            this.txtEmuMaxX.Name = "txtEmuMaxX";
            this.txtEmuMaxX.Size = new System.Drawing.Size(80, 20);
            this.txtEmuMaxX.TabIndex = 18;
            this.txtEmuMaxX.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.txtEmuMaxX.TextChanged += new System.EventHandler(this.Filled1);
            // 
            // lblEmuMaxX
            // 
            this.lblEmuMaxX.AutoSize = true;
            this.lblEmuMaxX.Location = new System.Drawing.Point(14, 96);
            this.lblEmuMaxX.Name = "lblEmuMaxX";
            this.lblEmuMaxX.Size = new System.Drawing.Size(34, 13);
            this.lblEmuMaxX.TabIndex = 17;
            this.lblEmuMaxX.Text = "MaxX";
            // 
            // txtEmuMinY
            // 
            this.txtEmuMinY.CausesValidation = false;
            this.txtEmuMinY.Location = new System.Drawing.Point(72, 60);
            this.txtEmuMinY.Name = "txtEmuMinY";
            this.txtEmuMinY.Size = new System.Drawing.Size(80, 20);
            this.txtEmuMinY.TabIndex = 16;
            this.txtEmuMinY.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.txtEmuMinY.TextChanged += new System.EventHandler(this.Filled1);
            // 
            // lblEmuMinY
            // 
            this.lblEmuMinY.AutoSize = true;
            this.lblEmuMinY.Location = new System.Drawing.Point(14, 63);
            this.lblEmuMinY.Name = "lblEmuMinY";
            this.lblEmuMinY.Size = new System.Drawing.Size(31, 13);
            this.lblEmuMinY.TabIndex = 15;
            this.lblEmuMinY.Text = "MinY";
            // 
            // txtEmuMinX
            // 
            this.txtEmuMinX.CausesValidation = false;
            this.txtEmuMinX.Location = new System.Drawing.Point(72, 26);
            this.txtEmuMinX.Name = "txtEmuMinX";
            this.txtEmuMinX.Size = new System.Drawing.Size(80, 20);
            this.txtEmuMinX.TabIndex = 14;
            this.txtEmuMinX.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.txtEmuMinX.TextChanged += new System.EventHandler(this.Filled1);
            // 
            // lblEmuMinX
            // 
            this.lblEmuMinX.AutoSize = true;
            this.lblEmuMinX.Location = new System.Drawing.Point(14, 29);
            this.lblEmuMinX.Name = "lblEmuMinX";
            this.lblEmuMinX.Size = new System.Drawing.Size(31, 13);
            this.lblEmuMinX.TabIndex = 13;
            this.lblEmuMinX.Text = "MinX";
            // 
            // lblMinX
            // 
            this.lblMinX.AutoSize = true;
            this.lblMinX.Location = new System.Drawing.Point(66, 36);
            this.lblMinX.Name = "lblMinX";
            this.lblMinX.Size = new System.Drawing.Size(31, 13);
            this.lblMinX.TabIndex = 32;
            this.lblMinX.Text = "MinX";
            // 
            // txtMinX
            // 
            this.txtMinX.Location = new System.Drawing.Point(105, 29);
            this.txtMinX.Name = "txtMinX";
            this.txtMinX.ReadOnly = true;
            this.txtMinX.Size = new System.Drawing.Size(110, 20);
            this.txtMinX.TabIndex = 33;
            this.txtMinX.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // lblMinY
            // 
            this.lblMinY.AutoSize = true;
            this.lblMinY.Location = new System.Drawing.Point(66, 61);
            this.lblMinY.Name = "lblMinY";
            this.lblMinY.Size = new System.Drawing.Size(31, 13);
            this.lblMinY.TabIndex = 34;
            this.lblMinY.Text = "MinY";
            // 
            // txtMinY
            // 
            this.txtMinY.Location = new System.Drawing.Point(105, 57);
            this.txtMinY.Name = "txtMinY";
            this.txtMinY.ReadOnly = true;
            this.txtMinY.Size = new System.Drawing.Size(110, 20);
            this.txtMinY.TabIndex = 35;
            this.txtMinY.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // lblMaxX
            // 
            this.lblMaxX.AutoSize = true;
            this.lblMaxX.Location = new System.Drawing.Point(252, 32);
            this.lblMaxX.Name = "lblMaxX";
            this.lblMaxX.Size = new System.Drawing.Size(34, 13);
            this.lblMaxX.TabIndex = 36;
            this.lblMaxX.Text = "MaxX";
            // 
            // lbMaxY
            // 
            this.lbMaxY.AutoSize = true;
            this.lbMaxY.Location = new System.Drawing.Point(252, 61);
            this.lbMaxY.Name = "lbMaxY";
            this.lbMaxY.Size = new System.Drawing.Size(34, 13);
            this.lbMaxY.TabIndex = 38;
            this.lbMaxY.Text = "MaxY";
            // 
            // txtMaxX
            // 
            this.txtMaxX.Location = new System.Drawing.Point(292, 29);
            this.txtMaxX.Name = "txtMaxX";
            this.txtMaxX.ReadOnly = true;
            this.txtMaxX.Size = new System.Drawing.Size(110, 20);
            this.txtMaxX.TabIndex = 37;
            this.txtMaxX.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // txtMaxY
            // 
            this.txtMaxY.Location = new System.Drawing.Point(292, 57);
            this.txtMaxY.Name = "txtMaxY";
            this.txtMaxY.ReadOnly = true;
            this.txtMaxY.Size = new System.Drawing.Size(110, 20);
            this.txtMaxY.TabIndex = 39;
            this.txtMaxY.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // grpComputedValues
            // 
            this.grpComputedValues.Controls.Add(this.txtMinX);
            this.grpComputedValues.Controls.Add(this.txtMaxY);
            this.grpComputedValues.Controls.Add(this.lblMinX);
            this.grpComputedValues.Controls.Add(this.txtMaxX);
            this.grpComputedValues.Controls.Add(this.lblMinY);
            this.grpComputedValues.Controls.Add(this.lbMaxY);
            this.grpComputedValues.Controls.Add(this.txtMinY);
            this.grpComputedValues.Controls.Add(this.lblMaxX);
            this.grpComputedValues.Location = new System.Drawing.Point(22, 401);
            this.grpComputedValues.Name = "grpComputedValues";
            this.grpComputedValues.Size = new System.Drawing.Size(453, 98);
            this.grpComputedValues.TabIndex = 31;
            this.grpComputedValues.TabStop = false;
            this.grpComputedValues.Text = "Computed Values";
            // 
            // grpInitValues
            // 
            this.grpInitValues.Controls.Add(this.txtBrickSet);
            this.grpInitValues.Controls.Add(this.txtBrick);
            this.grpInitValues.Controls.Add(this.lblBrick);
            this.grpInitValues.Controls.Add(this.cmbDownstreamPlate);
            this.grpInitValues.Controls.Add(this.txtMaxZ);
            this.grpInitValues.Controls.Add(this.cmbNumberOfPlates);
            this.grpInitValues.Controls.Add(this.lblBrickSet);
            this.grpInitValues.Controls.Add(this.lblDownstreamPlate);
            this.grpInitValues.Controls.Add(this.lblMaxZ);
            this.grpInitValues.Controls.Add(this.lblNPlate);
            this.grpInitValues.Location = new System.Drawing.Point(17, 8);
            this.grpInitValues.Name = "grpInitValues";
            this.grpInitValues.Size = new System.Drawing.Size(458, 115);
            this.grpInitValues.TabIndex = 0;
            this.grpInitValues.TabStop = false;
            this.grpInitValues.Text = "Initial Values";
            // 
            // txtBrickSet
            // 
            this.txtBrickSet.CausesValidation = false;
            this.txtBrickSet.Location = new System.Drawing.Point(67, 70);
            this.txtBrickSet.Name = "txtBrickSet";
            this.txtBrickSet.ReadOnly = true;
            this.txtBrickSet.Size = new System.Drawing.Size(84, 20);
            this.txtBrickSet.TabIndex = 11;
            this.txtBrickSet.TextChanged += new System.EventHandler(this.NoChange);
            // 
            // btnOk
            // 
            this.btnOk.Location = new System.Drawing.Point(517, 37);
            this.btnOk.Name = "btnOk";
            this.btnOk.Size = new System.Drawing.Size(84, 30);
            this.btnOk.TabIndex = 40;
            this.btnOk.Text = "&OK";
            this.btnOk.UseVisualStyleBackColor = true;
            this.btnOk.Click += new System.EventHandler(this.btnOk_Click);
            // 
            // btnCancel
            // 
            this.btnCancel.DialogResult = System.Windows.Forms.DialogResult.Cancel;
            this.btnCancel.Location = new System.Drawing.Point(519, 85);
            this.btnCancel.Name = "btnCancel";
            this.btnCancel.Size = new System.Drawing.Size(81, 28);
            this.btnCancel.TabIndex = 41;
            this.btnCancel.Text = "&Cancel";
            this.btnCancel.UseVisualStyleBackColor = true;
            this.btnCancel.Click += new System.EventHandler(this.btnCancel_Click);
            // 
            // frmAddBrick
            // 
            this.AcceptButton = this.btnOk;
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.CancelButton = this.btnCancel;
            this.ClientSize = new System.Drawing.Size(625, 512);
            this.Controls.Add(this.btnCancel);
            this.Controls.Add(this.btnOk);
            this.Controls.Add(this.grpInitValues);
            this.Controls.Add(this.grpComputedValues);
            this.Controls.Add(this.grbCompute);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
            this.MaximizeBox = false;
            this.Name = "frmAddBrick";
            this.Text = "Add New Brick";
            this.grbCompute.ResumeLayout(false);
            this.grpZeroXY.ResumeLayout(false);
            this.grpZeroXY.PerformLayout();
            this.grpXY.ResumeLayout(false);
            this.grpXY.PerformLayout();
            this.grpComputedValues.ResumeLayout(false);
            this.grpComputedValues.PerformLayout();
            this.grpInitValues.ResumeLayout(false);
            this.grpInitValues.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.Label lblNPlate;
        private System.Windows.Forms.Label lblMaxZ;
        private System.Windows.Forms.Label lblDownstreamPlate;
        private System.Windows.Forms.Label lblBrickSet;
        private System.Windows.Forms.ComboBox cmbNumberOfPlates;
        private System.Windows.Forms.TextBox txtMaxZ;
        private System.Windows.Forms.ComboBox cmbDownstreamPlate;
        private System.Windows.Forms.OpenFileDialog dlgOpenFile;
        private System.Windows.Forms.Label lblBrick;
        private System.Windows.Forms.TextBox txtBrick;
        private System.Windows.Forms.GroupBox grbCompute;
        private System.Windows.Forms.GroupBox grpXY;
        private System.Windows.Forms.Button btnCompute;
        private System.Windows.Forms.GroupBox grpZeroXY;
        private System.Windows.Forms.Button btnLoadZeroCoordinate;
        private System.Windows.Forms.Button btnDBLink;
        private System.Windows.Forms.TextBox txtZeroY;
        private System.Windows.Forms.Label lblZeroY;
        private System.Windows.Forms.TextBox txtZeroX;
        private System.Windows.Forms.Label lblZeroX;
        private System.Windows.Forms.Button btnInputDataFiles;
        private System.Windows.Forms.TextBox txtEmuMaxY;
        private System.Windows.Forms.Label lblEmuMaxY;
        private System.Windows.Forms.TextBox txtEmuMaxX;
        private System.Windows.Forms.Label lblEmuMaxX;
        private System.Windows.Forms.TextBox txtEmuMinY;
        private System.Windows.Forms.Label lblEmuMinY;
        private System.Windows.Forms.TextBox txtEmuMinX;
        private System.Windows.Forms.Label lblEmuMinX;
        private System.Windows.Forms.Label lblMinX;
        private System.Windows.Forms.TextBox txtMinX;
        private System.Windows.Forms.Label lblMinY;
        private System.Windows.Forms.TextBox txtMinY;
        private System.Windows.Forms.Label lblMaxX;
        private System.Windows.Forms.Label lbMaxY;
        private System.Windows.Forms.TextBox txtMaxX;
        private System.Windows.Forms.TextBox txtMaxY;
        private System.Windows.Forms.GroupBox grpComputedValues;
        private System.Windows.Forms.Button btnClearAll;
        private System.Windows.Forms.GroupBox grpInitValues;
        private System.Windows.Forms.Button btnOk;
        private System.Windows.Forms.Button btnCancel;
        private System.Windows.Forms.TextBox txtBrickSet;

    }
}