namespace SySal.Processing.MCSLikelihood
{
    partial class EditConfigForm
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
            this.btnGeometryAdd = new System.Windows.Forms.Button();
            this.txtRadLen = new System.Windows.Forms.TextBox();
            this.label8 = new System.Windows.Forms.Label();
            this.txtZMin = new System.Windows.Forms.TextBox();
            this.btnDefaultStack = new System.Windows.Forms.Button();
            this.label7 = new System.Windows.Forms.Label();
            this.btnDefaultOPERA = new System.Windows.Forms.Button();
            this.btnGeometryDel = new System.Windows.Forms.Button();
            this.lvGeometry = new System.Windows.Forms.ListView();
            this.columnHeader1 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader2 = new System.Windows.Forms.ColumnHeader();
            this.btnOK = new System.Windows.Forms.Button();
            this.btnCancel = new System.Windows.Forms.Button();
            this.label1 = new System.Windows.Forms.Label();
            this.txtSlopeErrors = new System.Windows.Forms.TextBox();
            this.label2 = new System.Windows.Forms.Label();
            this.txtCL = new System.Windows.Forms.TextBox();
            this.label3 = new System.Windows.Forms.Label();
            this.txtPMin = new System.Windows.Forms.TextBox();
            this.label4 = new System.Windows.Forms.Label();
            this.txtPMax = new System.Windows.Forms.TextBox();
            this.label5 = new System.Windows.Forms.Label();
            this.txtPStep = new System.Windows.Forms.TextBox();
            this.label6 = new System.Windows.Forms.Label();
            this.txtMinRadLen = new System.Windows.Forms.TextBox();
            this.tabControl1 = new System.Windows.Forms.TabControl();
            this.tabPage1 = new System.Windows.Forms.TabPage();
            this.tabPage2 = new System.Windows.Forms.TabPage();
            this.columnHeader3 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader4 = new System.Windows.Forms.ColumnHeader();
            this.txtBrick = new System.Windows.Forms.TextBox();
            this.label9 = new System.Windows.Forms.Label();
            this.txtPlate = new System.Windows.Forms.TextBox();
            this.label10 = new System.Windows.Forms.Label();
            this.btnFromDB = new System.Windows.Forms.Button();
            this.tabControl1.SuspendLayout();
            this.tabPage1.SuspendLayout();
            this.tabPage2.SuspendLayout();
            this.SuspendLayout();
            // 
            // btnGeometryAdd
            // 
            this.btnGeometryAdd.Location = new System.Drawing.Point(8, 288);
            this.btnGeometryAdd.Name = "btnGeometryAdd";
            this.btnGeometryAdd.Size = new System.Drawing.Size(75, 23);
            this.btnGeometryAdd.TabIndex = 17;
            this.btnGeometryAdd.Text = "Add";
            this.btnGeometryAdd.UseVisualStyleBackColor = true;
            this.btnGeometryAdd.Click += new System.EventHandler(this.btnGeometryAdd_Click);
            // 
            // txtRadLen
            // 
            this.txtRadLen.Location = new System.Drawing.Point(113, 255);
            this.txtRadLen.Name = "txtRadLen";
            this.txtRadLen.Size = new System.Drawing.Size(67, 20);
            this.txtRadLen.TabIndex = 16;
            this.txtRadLen.Text = "5600";
            // 
            // label8
            // 
            this.label8.AutoSize = true;
            this.label8.Location = new System.Drawing.Point(10, 260);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(88, 13);
            this.label8.TabIndex = 15;
            this.label8.Text = "Radiation Length";
            // 
            // txtZMin
            // 
            this.txtZMin.Location = new System.Drawing.Point(113, 228);
            this.txtZMin.Name = "txtZMin";
            this.txtZMin.Size = new System.Drawing.Size(67, 20);
            this.txtZMin.TabIndex = 14;
            this.txtZMin.Text = "0";
            // 
            // btnDefaultStack
            // 
            this.btnDefaultStack.Location = new System.Drawing.Point(157, 319);
            this.btnDefaultStack.Name = "btnDefaultStack";
            this.btnDefaultStack.Size = new System.Drawing.Size(133, 23);
            this.btnDefaultStack.TabIndex = 20;
            this.btnDefaultStack.Text = "10-plate stack";
            this.btnDefaultStack.UseVisualStyleBackColor = true;
            this.btnDefaultStack.Click += new System.EventHandler(this.btnDefaultStack_Click);
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.Location = new System.Drawing.Point(10, 231);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(31, 13);
            this.label7.TabIndex = 13;
            this.label7.Text = "ZMin";
            // 
            // btnDefaultOPERA
            // 
            this.btnDefaultOPERA.Location = new System.Drawing.Point(8, 319);
            this.btnDefaultOPERA.Name = "btnDefaultOPERA";
            this.btnDefaultOPERA.Size = new System.Drawing.Size(133, 23);
            this.btnDefaultOPERA.TabIndex = 19;
            this.btnDefaultOPERA.Text = "Default OPERA";
            this.btnDefaultOPERA.UseVisualStyleBackColor = true;
            this.btnDefaultOPERA.Click += new System.EventHandler(this.btnDefaultOPERA_Click);
            // 
            // btnGeometryDel
            // 
            this.btnGeometryDel.Location = new System.Drawing.Point(369, 288);
            this.btnGeometryDel.Name = "btnGeometryDel";
            this.btnGeometryDel.Size = new System.Drawing.Size(75, 23);
            this.btnGeometryDel.TabIndex = 18;
            this.btnGeometryDel.Text = "Delete";
            this.btnGeometryDel.UseVisualStyleBackColor = true;
            this.btnGeometryDel.Click += new System.EventHandler(this.btnGeometryDel_Click);
            // 
            // lvGeometry
            // 
            this.lvGeometry.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.columnHeader1,
            this.columnHeader2,
            this.columnHeader3,
            this.columnHeader4});
            this.lvGeometry.FullRowSelect = true;
            this.lvGeometry.GridLines = true;
            this.lvGeometry.Location = new System.Drawing.Point(8, 6);
            this.lvGeometry.Name = "lvGeometry";
            this.lvGeometry.Size = new System.Drawing.Size(436, 214);
            this.lvGeometry.TabIndex = 12;
            this.lvGeometry.UseCompatibleStateImageBehavior = false;
            this.lvGeometry.View = System.Windows.Forms.View.Details;
            // 
            // columnHeader1
            // 
            this.columnHeader1.Text = "Volume start (Min Z-µm)";
            this.columnHeader1.Width = 131;
            // 
            // columnHeader2
            // 
            this.columnHeader2.Text = "Radiation Length (µm)";
            this.columnHeader2.Width = 157;
            // 
            // btnOK
            // 
            this.btnOK.Location = new System.Drawing.Point(6, 390);
            this.btnOK.Name = "btnOK";
            this.btnOK.Size = new System.Drawing.Size(75, 23);
            this.btnOK.TabIndex = 21;
            this.btnOK.Text = "OK";
            this.btnOK.UseVisualStyleBackColor = true;
            this.btnOK.Click += new System.EventHandler(this.btnOK_Click);
            // 
            // btnCancel
            // 
            this.btnCancel.Location = new System.Drawing.Point(381, 390);
            this.btnCancel.Name = "btnCancel";
            this.btnCancel.Size = new System.Drawing.Size(75, 23);
            this.btnCancel.TabIndex = 22;
            this.btnCancel.Text = "Cancel";
            this.btnCancel.UseVisualStyleBackColor = true;
            this.btnCancel.Click += new System.EventHandler(this.btnCancel_Click);
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(6, 8);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(188, 13);
            this.label1.TabIndex = 0;
            this.label1.Text = "Slope Measurement Error (Transverse)";
            // 
            // txtSlopeErrors
            // 
            this.txtSlopeErrors.Location = new System.Drawing.Point(205, 6);
            this.txtSlopeErrors.Name = "txtSlopeErrors";
            this.txtSlopeErrors.Size = new System.Drawing.Size(67, 20);
            this.txtSlopeErrors.TabIndex = 1;
            this.txtSlopeErrors.Leave += new System.EventHandler(this.OnSlopeErrLeave);
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(6, 34);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(114, 13);
            this.label2.TabIndex = 2;
            this.label2.Text = "Confidence Level [0-1]";
            // 
            // txtCL
            // 
            this.txtCL.Location = new System.Drawing.Point(205, 32);
            this.txtCL.Name = "txtCL";
            this.txtCL.Size = new System.Drawing.Size(67, 20);
            this.txtCL.TabIndex = 3;
            this.txtCL.Leave += new System.EventHandler(this.OnCLLeave);
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(6, 60);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(144, 13);
            this.label3.TabIndex = 4;
            this.label3.Text = "Minimum Momentum (GeV/c)";
            // 
            // txtPMin
            // 
            this.txtPMin.Location = new System.Drawing.Point(205, 58);
            this.txtPMin.Name = "txtPMin";
            this.txtPMin.Size = new System.Drawing.Size(67, 20);
            this.txtPMin.TabIndex = 5;
            this.txtPMin.Leave += new System.EventHandler(this.OnMinPLeave);
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(6, 86);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(147, 13);
            this.label4.TabIndex = 6;
            this.label4.Text = "Maximum Momentum (GeV/c)";
            // 
            // txtPMax
            // 
            this.txtPMax.Location = new System.Drawing.Point(205, 84);
            this.txtPMax.Name = "txtPMax";
            this.txtPMax.Size = new System.Drawing.Size(67, 20);
            this.txtPMax.TabIndex = 7;
            this.txtPMax.Leave += new System.EventHandler(this.OnMaxPLeave);
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(6, 112);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(125, 13);
            this.label5.TabIndex = 8;
            this.label5.Text = "Momentum Step (GeV/c)";
            // 
            // txtPStep
            // 
            this.txtPStep.Location = new System.Drawing.Point(205, 110);
            this.txtPStep.Name = "txtPStep";
            this.txtPStep.Size = new System.Drawing.Size(67, 20);
            this.txtPStep.TabIndex = 9;
            this.txtPStep.Leave += new System.EventHandler(this.OnPStepLeave);
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Location = new System.Drawing.Point(6, 138);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(170, 26);
            this.label6.TabIndex = 10;
            this.label6.Text = "Scattering path in radiation lengths\r\n(0 turns autotuning on)";
            // 
            // txtMinRadLen
            // 
            this.txtMinRadLen.Location = new System.Drawing.Point(205, 136);
            this.txtMinRadLen.Name = "txtMinRadLen";
            this.txtMinRadLen.Size = new System.Drawing.Size(67, 20);
            this.txtMinRadLen.TabIndex = 11;
            this.txtMinRadLen.Leave += new System.EventHandler(this.OnMinRadLenLeave);
            // 
            // tabControl1
            // 
            this.tabControl1.Controls.Add(this.tabPage1);
            this.tabControl1.Controls.Add(this.tabPage2);
            this.tabControl1.Location = new System.Drawing.Point(2, 7);
            this.tabControl1.Name = "tabControl1";
            this.tabControl1.SelectedIndex = 0;
            this.tabControl1.Size = new System.Drawing.Size(458, 377);
            this.tabControl1.TabIndex = 24;
            // 
            // tabPage1
            // 
            this.tabPage1.BackColor = System.Drawing.SystemColors.Control;
            this.tabPage1.Controls.Add(this.txtMinRadLen);
            this.tabPage1.Controls.Add(this.label1);
            this.tabPage1.Controls.Add(this.label6);
            this.tabPage1.Controls.Add(this.txtSlopeErrors);
            this.tabPage1.Controls.Add(this.txtPStep);
            this.tabPage1.Controls.Add(this.label2);
            this.tabPage1.Controls.Add(this.label5);
            this.tabPage1.Controls.Add(this.txtCL);
            this.tabPage1.Controls.Add(this.txtPMax);
            this.tabPage1.Controls.Add(this.label3);
            this.tabPage1.Controls.Add(this.label4);
            this.tabPage1.Controls.Add(this.txtPMin);
            this.tabPage1.Location = new System.Drawing.Point(4, 22);
            this.tabPage1.Name = "tabPage1";
            this.tabPage1.Padding = new System.Windows.Forms.Padding(3);
            this.tabPage1.Size = new System.Drawing.Size(450, 351);
            this.tabPage1.TabIndex = 0;
            this.tabPage1.Text = "Fit parameters";
            // 
            // tabPage2
            // 
            this.tabPage2.BackColor = System.Drawing.SystemColors.Control;
            this.tabPage2.Controls.Add(this.btnFromDB);
            this.tabPage2.Controls.Add(this.txtBrick);
            this.tabPage2.Controls.Add(this.label9);
            this.tabPage2.Controls.Add(this.txtPlate);
            this.tabPage2.Controls.Add(this.label10);
            this.tabPage2.Controls.Add(this.btnGeometryAdd);
            this.tabPage2.Controls.Add(this.lvGeometry);
            this.tabPage2.Controls.Add(this.txtRadLen);
            this.tabPage2.Controls.Add(this.btnGeometryDel);
            this.tabPage2.Controls.Add(this.label8);
            this.tabPage2.Controls.Add(this.btnDefaultOPERA);
            this.tabPage2.Controls.Add(this.txtZMin);
            this.tabPage2.Controls.Add(this.label7);
            this.tabPage2.Controls.Add(this.btnDefaultStack);
            this.tabPage2.Location = new System.Drawing.Point(4, 22);
            this.tabPage2.Name = "tabPage2";
            this.tabPage2.Padding = new System.Windows.Forms.Padding(3);
            this.tabPage2.Size = new System.Drawing.Size(450, 351);
            this.tabPage2.TabIndex = 1;
            this.tabPage2.Text = "Geometry";
            // 
            // columnHeader3
            // 
            this.columnHeader3.Text = "Plate";
            // 
            // columnHeader4
            // 
            this.columnHeader4.Text = "Brick";
            // 
            // txtBrick
            // 
            this.txtBrick.Location = new System.Drawing.Point(376, 255);
            this.txtBrick.Name = "txtBrick";
            this.txtBrick.Size = new System.Drawing.Size(67, 20);
            this.txtBrick.TabIndex = 24;
            this.txtBrick.Text = "0";
            // 
            // label9
            // 
            this.label9.AutoSize = true;
            this.label9.Location = new System.Drawing.Point(247, 260);
            this.label9.Name = "label9";
            this.label9.Size = new System.Drawing.Size(31, 13);
            this.label9.TabIndex = 23;
            this.label9.Text = "Brick";
            // 
            // txtPlate
            // 
            this.txtPlate.Location = new System.Drawing.Point(376, 228);
            this.txtPlate.Name = "txtPlate";
            this.txtPlate.Size = new System.Drawing.Size(67, 20);
            this.txtPlate.TabIndex = 22;
            this.txtPlate.Text = "0";
            // 
            // label10
            // 
            this.label10.AutoSize = true;
            this.label10.Location = new System.Drawing.Point(247, 231);
            this.label10.Name = "label10";
            this.label10.Size = new System.Drawing.Size(116, 13);
            this.label10.TabIndex = 21;
            this.label10.Text = "Plate (upstream bound)";
            // 
            // btnFromDB
            // 
            this.btnFromDB.Location = new System.Drawing.Point(310, 319);
            this.btnFromDB.Name = "btnFromDB";
            this.btnFromDB.Size = new System.Drawing.Size(133, 23);
            this.btnFromDB.TabIndex = 25;
            this.btnFromDB.Text = "Brick from OPERA DB";
            this.btnFromDB.UseVisualStyleBackColor = true;
            this.btnFromDB.Click += new System.EventHandler(this.btnFromDB_Click);
            // 
            // EditConfigForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(472, 425);
            this.Controls.Add(this.tabControl1);
            this.Controls.Add(this.btnCancel);
            this.Controls.Add(this.btnOK);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
            this.Name = "EditConfigForm";
            this.Text = "Edit Momentum Estimator Configuration";
            this.Load += new System.EventHandler(this.EditConfigForm_Load);
            this.tabControl1.ResumeLayout(false);
            this.tabPage1.ResumeLayout(false);
            this.tabPage1.PerformLayout();
            this.tabPage2.ResumeLayout(false);
            this.tabPage2.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.Button btnOK;
        private System.Windows.Forms.Button btnCancel;
        private System.Windows.Forms.ListView lvGeometry;
        private System.Windows.Forms.ColumnHeader columnHeader1;
        private System.Windows.Forms.ColumnHeader columnHeader2;
        private System.Windows.Forms.Button btnDefaultOPERA;
        private System.Windows.Forms.Button btnGeometryDel;
        private System.Windows.Forms.Button btnDefaultStack;
        private System.Windows.Forms.Button btnGeometryAdd;
        private System.Windows.Forms.TextBox txtRadLen;
        private System.Windows.Forms.Label label8;
        private System.Windows.Forms.TextBox txtZMin;
        private System.Windows.Forms.Label label7;
        private System.Windows.Forms.ColumnHeader columnHeader3;
        private System.Windows.Forms.ColumnHeader columnHeader4;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TextBox txtSlopeErrors;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.TextBox txtCL;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.TextBox txtPMin;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.TextBox txtPMax;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.TextBox txtPStep;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.TextBox txtMinRadLen;
        private System.Windows.Forms.TabControl tabControl1;
        private System.Windows.Forms.TabPage tabPage1;
        private System.Windows.Forms.TabPage tabPage2;
        private System.Windows.Forms.TextBox txtBrick;
        private System.Windows.Forms.Label label9;
        private System.Windows.Forms.TextBox txtPlate;
        private System.Windows.Forms.Label label10;
        private System.Windows.Forms.Button btnFromDB;
    }
}