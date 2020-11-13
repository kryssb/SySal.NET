namespace SySal.Executables.ScanbackViewer
{
    partial class MomentumForm
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
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.buttonEditConfig = new System.Windows.Forms.Button();
            this.buttonSaveConfig = new System.Windows.Forms.Button();
            this.buttonLoadConfig = new System.Windows.Forms.Button();
            this.textConfigFile = new System.Windows.Forms.TextBox();
            this.buttonConfigFileSel = new System.Windows.Forms.Button();
            this.groupBox2 = new System.Windows.Forms.GroupBox();
            this.textIgnoreDeltaSlope = new System.Windows.Forms.TextBox();
            this.buttonIgnoreDeltaSlope = new System.Windows.Forms.Button();
            this.textMeasIgnoreGrains = new System.Windows.Forms.TextBox();
            this.lvSlopes = new System.Windows.Forms.ListView();
            this.columnHeader1 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader2 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader3 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader4 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader5 = new System.Windows.Forms.ColumnHeader();
            this.buttonMeasIgnoreGrains = new System.Windows.Forms.Button();
            this.buttonMeasSelAll = new System.Windows.Forms.Button();
            this.groupBox3 = new System.Windows.Forms.GroupBox();
            this.label1 = new System.Windows.Forms.Label();
            this.textResult = new System.Windows.Forms.TextBox();
            this.buttonCompute = new System.Windows.Forms.Button();
            this.sfn = new System.Windows.Forms.SaveFileDialog();
            this.label2 = new System.Windows.Forms.Label();
            this.rdAlgLikelihood = new System.Windows.Forms.RadioButton();
            this.rdAlgAnnecy = new System.Windows.Forms.RadioButton();
            this.groupBox1.SuspendLayout();
            this.groupBox2.SuspendLayout();
            this.groupBox3.SuspendLayout();
            this.SuspendLayout();
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.rdAlgAnnecy);
            this.groupBox1.Controls.Add(this.rdAlgLikelihood);
            this.groupBox1.Controls.Add(this.label2);
            this.groupBox1.Controls.Add(this.buttonEditConfig);
            this.groupBox1.Controls.Add(this.buttonSaveConfig);
            this.groupBox1.Controls.Add(this.buttonLoadConfig);
            this.groupBox1.Controls.Add(this.textConfigFile);
            this.groupBox1.Controls.Add(this.buttonConfigFileSel);
            this.groupBox1.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.groupBox1.Location = new System.Drawing.Point(1, 6);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(506, 100);
            this.groupBox1.TabIndex = 0;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Configuration";
            // 
            // buttonEditConfig
            // 
            this.buttonEditConfig.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.buttonEditConfig.Location = new System.Drawing.Point(402, 61);
            this.buttonEditConfig.Name = "buttonEditConfig";
            this.buttonEditConfig.Size = new System.Drawing.Size(90, 27);
            this.buttonEditConfig.TabIndex = 3;
            this.buttonEditConfig.Text = "Edit";
            this.buttonEditConfig.UseVisualStyleBackColor = true;
            this.buttonEditConfig.Click += new System.EventHandler(this.buttonEditConfig_Click);
            // 
            // buttonSaveConfig
            // 
            this.buttonSaveConfig.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.buttonSaveConfig.Location = new System.Drawing.Point(352, 61);
            this.buttonSaveConfig.Name = "buttonSaveConfig";
            this.buttonSaveConfig.Size = new System.Drawing.Size(44, 27);
            this.buttonSaveConfig.TabIndex = 2;
            this.buttonSaveConfig.Text = "Save";
            this.buttonSaveConfig.UseVisualStyleBackColor = true;
            this.buttonSaveConfig.Click += new System.EventHandler(this.buttonSaveConfig_Click);
            // 
            // buttonLoadConfig
            // 
            this.buttonLoadConfig.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.buttonLoadConfig.Location = new System.Drawing.Point(302, 61);
            this.buttonLoadConfig.Name = "buttonLoadConfig";
            this.buttonLoadConfig.Size = new System.Drawing.Size(44, 27);
            this.buttonLoadConfig.TabIndex = 1;
            this.buttonLoadConfig.Text = "Load";
            this.buttonLoadConfig.UseVisualStyleBackColor = true;
            this.buttonLoadConfig.Click += new System.EventHandler(this.buttonLoadConfig_Click);
            // 
            // textConfigFile
            // 
            this.textConfigFile.Location = new System.Drawing.Point(49, 65);
            this.textConfigFile.Name = "textConfigFile";
            this.textConfigFile.Size = new System.Drawing.Size(247, 20);
            this.textConfigFile.TabIndex = 1;
            // 
            // buttonConfigFileSel
            // 
            this.buttonConfigFileSel.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.buttonConfigFileSel.Location = new System.Drawing.Point(6, 61);
            this.buttonConfigFileSel.Name = "buttonConfigFileSel";
            this.buttonConfigFileSel.Size = new System.Drawing.Size(30, 27);
            this.buttonConfigFileSel.TabIndex = 0;
            this.buttonConfigFileSel.Text = "...";
            this.buttonConfigFileSel.UseVisualStyleBackColor = true;
            this.buttonConfigFileSel.Click += new System.EventHandler(this.buttonConfigFileSel_Click);
            // 
            // groupBox2
            // 
            this.groupBox2.Controls.Add(this.textIgnoreDeltaSlope);
            this.groupBox2.Controls.Add(this.buttonIgnoreDeltaSlope);
            this.groupBox2.Controls.Add(this.textMeasIgnoreGrains);
            this.groupBox2.Controls.Add(this.lvSlopes);
            this.groupBox2.Controls.Add(this.buttonMeasIgnoreGrains);
            this.groupBox2.Controls.Add(this.buttonMeasSelAll);
            this.groupBox2.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.groupBox2.Location = new System.Drawing.Point(1, 112);
            this.groupBox2.Name = "groupBox2";
            this.groupBox2.Size = new System.Drawing.Size(506, 214);
            this.groupBox2.TabIndex = 1;
            this.groupBox2.TabStop = false;
            this.groupBox2.Text = "Slope measurements";
            // 
            // textIgnoreDeltaSlope
            // 
            this.textIgnoreDeltaSlope.Location = new System.Drawing.Point(453, 92);
            this.textIgnoreDeltaSlope.Name = "textIgnoreDeltaSlope";
            this.textIgnoreDeltaSlope.Size = new System.Drawing.Size(39, 20);
            this.textIgnoreDeltaSlope.TabIndex = 7;
            this.textIgnoreDeltaSlope.Leave += new System.EventHandler(this.OnDeltaSlopeLeave);
            // 
            // buttonIgnoreDeltaSlope
            // 
            this.buttonIgnoreDeltaSlope.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.buttonIgnoreDeltaSlope.Location = new System.Drawing.Point(326, 88);
            this.buttonIgnoreDeltaSlope.Name = "buttonIgnoreDeltaSlope";
            this.buttonIgnoreDeltaSlope.Size = new System.Drawing.Size(120, 27);
            this.buttonIgnoreDeltaSlope.TabIndex = 6;
            this.buttonIgnoreDeltaSlope.Text = "Ignore delta slope >";
            this.buttonIgnoreDeltaSlope.UseVisualStyleBackColor = true;
            this.buttonIgnoreDeltaSlope.Click += new System.EventHandler(this.buttonIgnoreDeltaSlope_Click);
            // 
            // textMeasIgnoreGrains
            // 
            this.textMeasIgnoreGrains.Location = new System.Drawing.Point(453, 59);
            this.textMeasIgnoreGrains.Name = "textMeasIgnoreGrains";
            this.textMeasIgnoreGrains.Size = new System.Drawing.Size(39, 20);
            this.textMeasIgnoreGrains.TabIndex = 5;
            this.textMeasIgnoreGrains.Leave += new System.EventHandler(this.OnGrainsLeave);
            // 
            // lvSlopes
            // 
            this.lvSlopes.CheckBoxes = true;
            this.lvSlopes.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.columnHeader1,
            this.columnHeader2,
            this.columnHeader3,
            this.columnHeader4,
            this.columnHeader5});
            this.lvSlopes.FullRowSelect = true;
            this.lvSlopes.GridLines = true;
            this.lvSlopes.Location = new System.Drawing.Point(14, 22);
            this.lvSlopes.Name = "lvSlopes";
            this.lvSlopes.Size = new System.Drawing.Size(306, 171);
            this.lvSlopes.TabIndex = 4;
            this.lvSlopes.UseCompatibleStateImageBehavior = false;
            this.lvSlopes.View = System.Windows.Forms.View.Details;
            // 
            // columnHeader1
            // 
            this.columnHeader1.Text = "Plate";
            // 
            // columnHeader2
            // 
            this.columnHeader2.Text = "Z";
            // 
            // columnHeader3
            // 
            this.columnHeader3.Text = "Grains";
            // 
            // columnHeader4
            // 
            this.columnHeader4.Text = "SlopeX";
            // 
            // columnHeader5
            // 
            this.columnHeader5.Text = "SlopeY";
            // 
            // buttonMeasIgnoreGrains
            // 
            this.buttonMeasIgnoreGrains.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.buttonMeasIgnoreGrains.Location = new System.Drawing.Point(326, 55);
            this.buttonMeasIgnoreGrains.Name = "buttonMeasIgnoreGrains";
            this.buttonMeasIgnoreGrains.Size = new System.Drawing.Size(120, 27);
            this.buttonMeasIgnoreGrains.TabIndex = 2;
            this.buttonMeasIgnoreGrains.Text = "Ignore grains <";
            this.buttonMeasIgnoreGrains.UseVisualStyleBackColor = true;
            this.buttonMeasIgnoreGrains.Click += new System.EventHandler(this.buttonMeasIgnoreGrains_Click);
            // 
            // buttonMeasSelAll
            // 
            this.buttonMeasSelAll.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.buttonMeasSelAll.Location = new System.Drawing.Point(326, 22);
            this.buttonMeasSelAll.Name = "buttonMeasSelAll";
            this.buttonMeasSelAll.Size = new System.Drawing.Size(167, 27);
            this.buttonMeasSelAll.TabIndex = 1;
            this.buttonMeasSelAll.Text = "Select all";
            this.buttonMeasSelAll.UseVisualStyleBackColor = true;
            this.buttonMeasSelAll.Click += new System.EventHandler(this.buttonMeasSelAll_Click);
            // 
            // groupBox3
            // 
            this.groupBox3.Controls.Add(this.label1);
            this.groupBox3.Controls.Add(this.textResult);
            this.groupBox3.Controls.Add(this.buttonCompute);
            this.groupBox3.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.groupBox3.Location = new System.Drawing.Point(1, 332);
            this.groupBox3.Name = "groupBox3";
            this.groupBox3.Size = new System.Drawing.Size(506, 58);
            this.groupBox3.TabIndex = 2;
            this.groupBox3.TabStop = false;
            this.groupBox3.Text = "Compute";
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(86, 26);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(112, 13);
            this.label1.TabIndex = 2;
            this.label1.Text = "P ;  P min ; P max ; CL";
            // 
            // textResult
            // 
            this.textResult.Location = new System.Drawing.Point(245, 23);
            this.textResult.Name = "textResult";
            this.textResult.ReadOnly = true;
            this.textResult.Size = new System.Drawing.Size(247, 20);
            this.textResult.TabIndex = 1;
            // 
            // buttonCompute
            // 
            this.buttonCompute.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.buttonCompute.Location = new System.Drawing.Point(6, 19);
            this.buttonCompute.Name = "buttonCompute";
            this.buttonCompute.Size = new System.Drawing.Size(74, 27);
            this.buttonCompute.TabIndex = 0;
            this.buttonCompute.Text = "Compute";
            this.buttonCompute.UseVisualStyleBackColor = true;
            this.buttonCompute.Click += new System.EventHandler(this.buttonCompute_Click);
            // 
            // sfn
            // 
            this.sfn.Filter = "XML Files (*.xml)|*.xml|All files (*.*)|*.*";
            this.sfn.Title = "Select configuration file";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(6, 31);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(50, 13);
            this.label2.TabIndex = 4;
            this.label2.Text = "Algorithm";
            // 
            // rdAlgLikelihood
            // 
            this.rdAlgLikelihood.Appearance = System.Windows.Forms.Appearance.Button;
            this.rdAlgLikelihood.AutoSize = true;
            this.rdAlgLikelihood.Location = new System.Drawing.Point(89, 26);
            this.rdAlgLikelihood.Name = "rdAlgLikelihood";
            this.rdAlgLikelihood.Size = new System.Drawing.Size(65, 23);
            this.rdAlgLikelihood.TabIndex = 5;
            this.rdAlgLikelihood.TabStop = true;
            this.rdAlgLikelihood.Text = "Likelihood";
            this.rdAlgLikelihood.UseVisualStyleBackColor = true;
            this.rdAlgLikelihood.CheckedChanged += new System.EventHandler(this.rdAlgLikelihood_CheckedChanged);
            // 
            // rdAlgAnnecy
            // 
            this.rdAlgAnnecy.Appearance = System.Windows.Forms.Appearance.Button;
            this.rdAlgAnnecy.AutoSize = true;
            this.rdAlgAnnecy.Location = new System.Drawing.Point(160, 26);
            this.rdAlgAnnecy.Name = "rdAlgAnnecy";
            this.rdAlgAnnecy.Size = new System.Drawing.Size(53, 23);
            this.rdAlgAnnecy.TabIndex = 6;
            this.rdAlgAnnecy.TabStop = true;
            this.rdAlgAnnecy.Text = "Annecy";
            this.rdAlgAnnecy.UseVisualStyleBackColor = true;
            this.rdAlgAnnecy.CheckedChanged += new System.EventHandler(this.rdAlgAnnecy_CheckedChanged);
            // 
            // MomentumForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(515, 399);
            this.Controls.Add(this.groupBox3);
            this.Controls.Add(this.groupBox2);
            this.Controls.Add(this.groupBox1);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedToolWindow;
            this.Name = "MomentumForm";
            this.Text = "Momentum Estimation";
            this.Load += new System.EventHandler(this.OnLoad);
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            this.groupBox2.ResumeLayout(false);
            this.groupBox2.PerformLayout();
            this.groupBox3.ResumeLayout(false);
            this.groupBox3.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.Button buttonConfigFileSel;
        private System.Windows.Forms.Button buttonEditConfig;
        private System.Windows.Forms.Button buttonSaveConfig;
        private System.Windows.Forms.Button buttonLoadConfig;
        private System.Windows.Forms.TextBox textConfigFile;
        private System.Windows.Forms.GroupBox groupBox2;
        private System.Windows.Forms.ListView lvSlopes;
        private System.Windows.Forms.ColumnHeader columnHeader1;
        private System.Windows.Forms.ColumnHeader columnHeader2;
        private System.Windows.Forms.ColumnHeader columnHeader3;
        private System.Windows.Forms.ColumnHeader columnHeader4;
        private System.Windows.Forms.ColumnHeader columnHeader5;
        private System.Windows.Forms.Button buttonMeasIgnoreGrains;
        private System.Windows.Forms.Button buttonMeasSelAll;
        private System.Windows.Forms.TextBox textMeasIgnoreGrains;
        private System.Windows.Forms.TextBox textIgnoreDeltaSlope;
        private System.Windows.Forms.Button buttonIgnoreDeltaSlope;
        private System.Windows.Forms.GroupBox groupBox3;
        private System.Windows.Forms.TextBox textResult;
        private System.Windows.Forms.Button buttonCompute;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.SaveFileDialog sfn;
        private System.Windows.Forms.RadioButton rdAlgAnnecy;
        private System.Windows.Forms.RadioButton rdAlgLikelihood;
        private System.Windows.Forms.Label label2;
    }
}