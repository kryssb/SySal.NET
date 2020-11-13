namespace SySal.Executables.EasyReconstruct
{
    partial class VertexFitForm
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
            this.TrackList = new System.Windows.Forms.ListView();
            this.columnHeader1 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader2 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader3 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader4 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader5 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader6 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader7 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader8 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader9 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader10 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader11 = new System.Windows.Forms.ColumnHeader();
            this.label1 = new System.Windows.Forms.Label();
            this.textX = new System.Windows.Forms.TextBox();
            this.textY = new System.Windows.Forms.TextBox();
            this.label2 = new System.Windows.Forms.Label();
            this.textZ = new System.Windows.Forms.TextBox();
            this.label3 = new System.Windows.Forms.Label();
            this.cmdRemove = new System.Windows.Forms.Button();
            this.cmdAddToPlot = new System.Windows.Forms.Button();
            this.DumpFileButton = new System.Windows.Forms.Button();
            this.DumpFileText = new System.Windows.Forms.TextBox();
            this.DumpSelButton = new System.Windows.Forms.Button();
            this.CheckHighight = new System.Windows.Forms.CheckBox();
            this.CheckShowLabels = new System.Windows.Forms.CheckBox();
            this.cmdTagColor = new System.Windows.Forms.Button();
            this.txtPz = new System.Windows.Forms.TextBox();
            this.label4 = new System.Windows.Forms.Label();
            this.txtPy = new System.Windows.Forms.TextBox();
            this.label5 = new System.Windows.Forms.Label();
            this.txtPx = new System.Windows.Forms.TextBox();
            this.label6 = new System.Windows.Forms.Label();
            this.label7 = new System.Windows.Forms.Label();
            this.txtParentSY = new System.Windows.Forms.TextBox();
            this.txtParentSX = new System.Windows.Forms.TextBox();
            this.label8 = new System.Windows.Forms.Label();
            this.txtPT = new System.Windows.Forms.TextBox();
            this.txtP = new System.Windows.Forms.TextBox();
            this.label30 = new System.Windows.Forms.Label();
            this.cmdToVertex = new System.Windows.Forms.Button();
            this.SuspendLayout();
            // 
            // TrackList
            // 
            this.TrackList.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.columnHeader1,
            this.columnHeader2,
            this.columnHeader3,
            this.columnHeader4,
            this.columnHeader5,
            this.columnHeader6,
            this.columnHeader7,
            this.columnHeader8,
            this.columnHeader9,
            this.columnHeader10,
            this.columnHeader11});
            this.TrackList.FullRowSelect = true;
            this.TrackList.GridLines = true;
            this.TrackList.Location = new System.Drawing.Point(12, 12);
            this.TrackList.MultiSelect = false;
            this.TrackList.Name = "TrackList";
            this.TrackList.Size = new System.Drawing.Size(762, 159);
            this.TrackList.TabIndex = 1;
            this.TrackList.UseCompatibleStateImageBehavior = false;
            this.TrackList.View = System.Windows.Forms.View.Details;
            // 
            // columnHeader1
            // 
            this.columnHeader1.Text = "Track";
            this.columnHeader1.Width = 80;
            // 
            // columnHeader2
            // 
            this.columnHeader2.Text = "X";
            // 
            // columnHeader3
            // 
            this.columnHeader3.Text = "Y";
            // 
            // columnHeader4
            // 
            this.columnHeader4.Text = "Z";
            // 
            // columnHeader5
            // 
            this.columnHeader5.Text = "SX";
            // 
            // columnHeader6
            // 
            this.columnHeader6.Text = "SY";
            // 
            // columnHeader7
            // 
            this.columnHeader7.Text = "Weight";
            // 
            // columnHeader8
            // 
            this.columnHeader8.Text = "P";
            // 
            // columnHeader9
            // 
            this.columnHeader9.Text = "IP";
            this.columnHeader9.Width = 40;
            // 
            // columnHeader10
            // 
            this.columnHeader10.Text = "D_IP";
            this.columnHeader10.Width = 40;
            // 
            // columnHeader11
            // 
            this.columnHeader11.Text = "DZ";
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(12, 188);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(14, 13);
            this.label1.TabIndex = 2;
            this.label1.Text = "X";
            // 
            // textX
            // 
            this.textX.Location = new System.Drawing.Point(35, 185);
            this.textX.Name = "textX";
            this.textX.ReadOnly = true;
            this.textX.Size = new System.Drawing.Size(70, 20);
            this.textX.TabIndex = 3;
            // 
            // textY
            // 
            this.textY.Location = new System.Drawing.Point(35, 211);
            this.textY.Name = "textY";
            this.textY.ReadOnly = true;
            this.textY.Size = new System.Drawing.Size(70, 20);
            this.textY.TabIndex = 5;
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(12, 214);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(14, 13);
            this.label2.TabIndex = 4;
            this.label2.Text = "Y";
            // 
            // textZ
            // 
            this.textZ.Location = new System.Drawing.Point(35, 237);
            this.textZ.Name = "textZ";
            this.textZ.ReadOnly = true;
            this.textZ.Size = new System.Drawing.Size(70, 20);
            this.textZ.TabIndex = 7;
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(12, 240);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(14, 13);
            this.label3.TabIndex = 6;
            this.label3.Text = "Z";
            // 
            // cmdRemove
            // 
            this.cmdRemove.Location = new System.Drawing.Point(654, 184);
            this.cmdRemove.Name = "cmdRemove";
            this.cmdRemove.Size = new System.Drawing.Size(120, 23);
            this.cmdRemove.TabIndex = 8;
            this.cmdRemove.Text = "Remove track";
            this.cmdRemove.UseVisualStyleBackColor = true;
            this.cmdRemove.Click += new System.EventHandler(this.cmdRemove_Click);
            // 
            // cmdAddToPlot
            // 
            this.cmdAddToPlot.Location = new System.Drawing.Point(400, 185);
            this.cmdAddToPlot.Name = "cmdAddToPlot";
            this.cmdAddToPlot.Size = new System.Drawing.Size(120, 23);
            this.cmdAddToPlot.TabIndex = 9;
            this.cmdAddToPlot.Text = "Add to Plot";
            this.cmdAddToPlot.UseVisualStyleBackColor = true;
            this.cmdAddToPlot.Click += new System.EventHandler(this.cmdAddToPlot_Click);
            // 
            // DumpFileButton
            // 
            this.DumpFileButton.Location = new System.Drawing.Point(725, 234);
            this.DumpFileButton.Name = "DumpFileButton";
            this.DumpFileButton.Size = new System.Drawing.Size(48, 24);
            this.DumpFileButton.TabIndex = 12;
            this.DumpFileButton.Text = "Dump";
            this.DumpFileButton.Click += new System.EventHandler(this.DumpFileButton_Click);
            // 
            // DumpFileText
            // 
            this.DumpFileText.Location = new System.Drawing.Point(438, 235);
            this.DumpFileText.Name = "DumpFileText";
            this.DumpFileText.Size = new System.Drawing.Size(281, 20);
            this.DumpFileText.TabIndex = 11;
            // 
            // DumpSelButton
            // 
            this.DumpSelButton.Location = new System.Drawing.Point(400, 233);
            this.DumpSelButton.Name = "DumpSelButton";
            this.DumpSelButton.Size = new System.Drawing.Size(32, 24);
            this.DumpSelButton.TabIndex = 10;
            this.DumpSelButton.Text = "...";
            this.DumpSelButton.Click += new System.EventHandler(this.DumpSelButton_Click);
            // 
            // CheckHighight
            // 
            this.CheckHighight.AutoSize = true;
            this.CheckHighight.Location = new System.Drawing.Point(400, 212);
            this.CheckHighight.Name = "CheckHighight";
            this.CheckHighight.Size = new System.Drawing.Size(67, 17);
            this.CheckHighight.TabIndex = 13;
            this.CheckHighight.Text = "Highlight";
            this.CheckHighight.UseVisualStyleBackColor = true;
            this.CheckHighight.CheckedChanged += new System.EventHandler(this.OnHighlightChanged);
            // 
            // CheckShowLabels
            // 
            this.CheckShowLabels.AutoSize = true;
            this.CheckShowLabels.Location = new System.Drawing.Point(495, 212);
            this.CheckShowLabels.Name = "CheckShowLabels";
            this.CheckShowLabels.Size = new System.Drawing.Size(83, 17);
            this.CheckShowLabels.TabIndex = 14;
            this.CheckShowLabels.Text = "Show labels";
            this.CheckShowLabels.UseVisualStyleBackColor = true;
            this.CheckShowLabels.CheckedChanged += new System.EventHandler(this.OnShowLabelChanged);
            // 
            // cmdTagColor
            // 
            this.cmdTagColor.BackColor = System.Drawing.Color.White;
            this.cmdTagColor.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.cmdTagColor.Location = new System.Drawing.Point(526, 186);
            this.cmdTagColor.Name = "cmdTagColor";
            this.cmdTagColor.Size = new System.Drawing.Size(24, 22);
            this.cmdTagColor.TabIndex = 15;
            this.cmdTagColor.UseVisualStyleBackColor = false;
            this.cmdTagColor.Click += new System.EventHandler(this.cmdTagColor_Click);
            // 
            // txtPz
            // 
            this.txtPz.Location = new System.Drawing.Point(142, 237);
            this.txtPz.Name = "txtPz";
            this.txtPz.ReadOnly = true;
            this.txtPz.Size = new System.Drawing.Size(49, 20);
            this.txtPz.TabIndex = 21;
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(119, 240);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(19, 13);
            this.label4.TabIndex = 20;
            this.label4.Text = "Pz";
            // 
            // txtPy
            // 
            this.txtPy.Location = new System.Drawing.Point(142, 211);
            this.txtPy.Name = "txtPy";
            this.txtPy.ReadOnly = true;
            this.txtPy.Size = new System.Drawing.Size(49, 20);
            this.txtPy.TabIndex = 19;
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(119, 214);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(19, 13);
            this.label5.TabIndex = 18;
            this.label5.Text = "Py";
            // 
            // txtPx
            // 
            this.txtPx.Location = new System.Drawing.Point(142, 185);
            this.txtPx.Name = "txtPx";
            this.txtPx.ReadOnly = true;
            this.txtPx.Size = new System.Drawing.Size(49, 20);
            this.txtPx.TabIndex = 17;
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Location = new System.Drawing.Point(119, 188);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(19, 13);
            this.label6.TabIndex = 16;
            this.label6.Text = "Px";
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.Location = new System.Drawing.Point(211, 188);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(102, 13);
            this.label7.TabIndex = 22;
            this.label7.Text = "Parent slope (Sx,Sy)";
            // 
            // txtParentSY
            // 
            this.txtParentSY.Location = new System.Drawing.Point(282, 211);
            this.txtParentSY.Name = "txtParentSY";
            this.txtParentSY.Size = new System.Drawing.Size(62, 20);
            this.txtParentSY.TabIndex = 24;
            this.txtParentSY.Leave += new System.EventHandler(this.OnParentSYLeave);
            // 
            // txtParentSX
            // 
            this.txtParentSX.Location = new System.Drawing.Point(214, 211);
            this.txtParentSX.Name = "txtParentSX";
            this.txtParentSX.Size = new System.Drawing.Size(62, 20);
            this.txtParentSX.TabIndex = 23;
            this.txtParentSX.Leave += new System.EventHandler(this.OnParentSXLeave);
            // 
            // label8
            // 
            this.label8.AutoSize = true;
            this.label8.Location = new System.Drawing.Point(211, 240);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(55, 13);
            this.label8.TabIndex = 25;
            this.label8.Text = "Missing Pt";
            // 
            // txtPT
            // 
            this.txtPT.Location = new System.Drawing.Point(282, 237);
            this.txtPT.Name = "txtPT";
            this.txtPT.ReadOnly = true;
            this.txtPT.Size = new System.Drawing.Size(49, 20);
            this.txtPT.TabIndex = 26;
            // 
            // txtP
            // 
            this.txtP.Location = new System.Drawing.Point(142, 263);
            this.txtP.Name = "txtP";
            this.txtP.ReadOnly = true;
            this.txtP.Size = new System.Drawing.Size(49, 20);
            this.txtP.TabIndex = 28;
            // 
            // label30
            // 
            this.label30.AutoSize = true;
            this.label30.Location = new System.Drawing.Point(119, 266);
            this.label30.Name = "label30";
            this.label30.Size = new System.Drawing.Size(14, 13);
            this.label30.TabIndex = 27;
            this.label30.Text = "P";
            // 
            // cmdToVertex
            // 
            this.cmdToVertex.Location = new System.Drawing.Point(214, 261);
            this.cmdToVertex.Name = "cmdToVertex";
            this.cmdToVertex.Size = new System.Drawing.Size(130, 23);
            this.cmdToVertex.TabIndex = 29;
            this.cmdToVertex.Text = "Convert to Vertex";
            this.cmdToVertex.UseVisualStyleBackColor = true;
            this.cmdToVertex.Click += new System.EventHandler(this.cmdToVertex_Click);
            // 
            // VertexFitForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(786, 294);
            this.Controls.Add(this.cmdToVertex);
            this.Controls.Add(this.txtP);
            this.Controls.Add(this.label30);
            this.Controls.Add(this.txtPT);
            this.Controls.Add(this.label8);
            this.Controls.Add(this.txtParentSY);
            this.Controls.Add(this.txtParentSX);
            this.Controls.Add(this.label7);
            this.Controls.Add(this.txtPz);
            this.Controls.Add(this.label4);
            this.Controls.Add(this.txtPy);
            this.Controls.Add(this.label5);
            this.Controls.Add(this.txtPx);
            this.Controls.Add(this.label6);
            this.Controls.Add(this.cmdTagColor);
            this.Controls.Add(this.CheckShowLabels);
            this.Controls.Add(this.CheckHighight);
            this.Controls.Add(this.DumpFileButton);
            this.Controls.Add(this.DumpFileText);
            this.Controls.Add(this.DumpSelButton);
            this.Controls.Add(this.cmdAddToPlot);
            this.Controls.Add(this.cmdRemove);
            this.Controls.Add(this.textZ);
            this.Controls.Add(this.label3);
            this.Controls.Add(this.textY);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.textX);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.TrackList);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.Fixed3D;
            this.Name = "VertexFitForm";
            this.Text = "VertexFit";
            this.Closed += new System.EventHandler(this.OnClose);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.ListView TrackList;
        private System.Windows.Forms.ColumnHeader columnHeader1;
        private System.Windows.Forms.ColumnHeader columnHeader2;
        private System.Windows.Forms.ColumnHeader columnHeader3;
        private System.Windows.Forms.ColumnHeader columnHeader4;
        private System.Windows.Forms.ColumnHeader columnHeader5;
        private System.Windows.Forms.ColumnHeader columnHeader6;
        private System.Windows.Forms.ColumnHeader columnHeader9;
        private System.Windows.Forms.ColumnHeader columnHeader10;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TextBox textX;
        private System.Windows.Forms.TextBox textY;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.TextBox textZ;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.Button cmdRemove;
        private System.Windows.Forms.Button cmdAddToPlot;
        private System.Windows.Forms.Button DumpFileButton;
        private System.Windows.Forms.TextBox DumpFileText;
        private System.Windows.Forms.Button DumpSelButton;
        private System.Windows.Forms.ColumnHeader columnHeader11;
        private System.Windows.Forms.CheckBox CheckHighight;
        private System.Windows.Forms.CheckBox CheckShowLabels;
        private System.Windows.Forms.Button cmdTagColor;
        private System.Windows.Forms.ColumnHeader columnHeader7;
        private System.Windows.Forms.ColumnHeader columnHeader8;
        private System.Windows.Forms.TextBox txtPz;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.TextBox txtPy;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.TextBox txtPx;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.Label label7;
        private System.Windows.Forms.TextBox txtParentSY;
        private System.Windows.Forms.TextBox txtParentSX;
        private System.Windows.Forms.Label label8;
        private System.Windows.Forms.TextBox txtPT;
        private System.Windows.Forms.TextBox txtP;
        private System.Windows.Forms.Label label30;
        private System.Windows.Forms.Button cmdToVertex;

    }
}