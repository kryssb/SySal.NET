namespace OperaBrickManager
{
    partial class MainForm
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
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(MainForm));
            this.lvBrickSets = new System.Windows.Forms.ListView();
            this.columnHeader1 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader2 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader3 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader4 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader5 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader6 = new System.Windows.Forms.ColumnHeader();
            this.btnRefreshBrickSets = new System.Windows.Forms.Button();
            this.btnAddBrickSet = new System.Windows.Forms.Button();
            this.btnRemoveBrickSet = new System.Windows.Forms.Button();
            this.lvBricks = new System.Windows.Forms.ListView();
            this.columnHeader25 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader14 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader15 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader16 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader17 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader18 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader19 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader20 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader21 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader22 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader23 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader24 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader7 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader8 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader9 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader10 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader11 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader12 = new System.Windows.Forms.ColumnHeader();
            this.lbAvalaibleBricksSets = new System.Windows.Forms.Label();
            this.lbBricksInBrickset = new System.Windows.Forms.Label();
            this.btnAddBrick = new System.Windows.Forms.Button();
            this.btnRemoveBrick = new System.Windows.Forms.Button();
            this.btnExit = new System.Windows.Forms.Button();
            this.btnAddBrickSpace = new System.Windows.Forms.Button();
            this.btnRemoveBrickSpace = new System.Windows.Forms.Button();
            this.btnShowAllBrick = new System.Windows.Forms.Button();
            this.lvMarkSet = new System.Windows.Forms.ListView();
            this.columnHeader13 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader26 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader27 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader28 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader29 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader30 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader31 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader32 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader33 = new System.Windows.Forms.ColumnHeader();
            this.lblMark = new System.Windows.Forms.Label();
            this.btnInsertCSMark = new System.Windows.Forms.Button();
            this.btnLoadMarkFromFile = new System.Windows.Forms.Button();
            this.SuspendLayout();
            // 
            // lvBrickSets
            // 
            this.lvBrickSets.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.columnHeader1,
            this.columnHeader2,
            this.columnHeader3,
            this.columnHeader4,
            this.columnHeader5,
            this.columnHeader6});
            this.lvBrickSets.FullRowSelect = true;
            this.lvBrickSets.GridLines = true;
            this.lvBrickSets.HideSelection = false;
            this.lvBrickSets.Location = new System.Drawing.Point(12, 30);
            this.lvBrickSets.MultiSelect = false;
            this.lvBrickSets.Name = "lvBrickSets";
            this.lvBrickSets.Size = new System.Drawing.Size(605, 120);
            this.lvBrickSets.TabIndex = 0;
            this.lvBrickSets.UseCompatibleStateImageBehavior = false;
            this.lvBrickSets.View = System.Windows.Forms.View.Details;
            this.lvBrickSets.SelectedIndexChanged += new System.EventHandler(this.lvBrickSets_SelectedIndexChanged);
            // 
            // columnHeader1
            // 
            this.columnHeader1.Text = "Brick Set";
            this.columnHeader1.Width = 120;
            // 
            // columnHeader2
            // 
            this.columnHeader2.Text = "Tablespace Extension";
            this.columnHeader2.Width = 120;
            // 
            // columnHeader3
            // 
            this.columnHeader3.Text = "Bricks";
            // 
            // columnHeader4
            // 
            this.columnHeader4.Text = "Scan Size (GB)";
            this.columnHeader4.Width = 100;
            // 
            // columnHeader5
            // 
            this.columnHeader5.Text = "Proc Size (GB)";
            this.columnHeader5.Width = 100;
            // 
            // columnHeader6
            // 
            this.columnHeader6.Text = "Rec Size (GB)";
            this.columnHeader6.Width = 100;
            // 
            // btnRefreshBrickSets
            // 
            this.btnRefreshBrickSets.Location = new System.Drawing.Point(633, 224);
            this.btnRefreshBrickSets.Name = "btnRefreshBrickSets";
            this.btnRefreshBrickSets.Size = new System.Drawing.Size(132, 27);
            this.btnRefreshBrickSets.TabIndex = 1;
            this.btnRefreshBrickSets.Text = "Re&fresh Brick Sets";
            this.btnRefreshBrickSets.UseVisualStyleBackColor = true;
            this.btnRefreshBrickSets.Click += new System.EventHandler(this.btnRefreshBrickSets_Click);
            // 
            // btnAddBrickSet
            // 
            this.btnAddBrickSet.Location = new System.Drawing.Point(633, 30);
            this.btnAddBrickSet.Name = "btnAddBrickSet";
            this.btnAddBrickSet.Size = new System.Drawing.Size(132, 27);
            this.btnAddBrickSet.TabIndex = 2;
            this.btnAddBrickSet.Text = "Add Brick Set";
            this.btnAddBrickSet.UseVisualStyleBackColor = true;
            this.btnAddBrickSet.Click += new System.EventHandler(this.btnAddBrickSet_Click);
            // 
            // btnRemoveBrickSet
            // 
            this.btnRemoveBrickSet.Location = new System.Drawing.Point(633, 61);
            this.btnRemoveBrickSet.Name = "btnRemoveBrickSet";
            this.btnRemoveBrickSet.Size = new System.Drawing.Size(132, 27);
            this.btnRemoveBrickSet.TabIndex = 3;
            this.btnRemoveBrickSet.Text = "Remove Brick Set";
            this.btnRemoveBrickSet.UseVisualStyleBackColor = true;
            this.btnRemoveBrickSet.Click += new System.EventHandler(this.txtRemoveBrickSet_Click);
            // 
            // lvBricks
            // 
            this.lvBricks.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.columnHeader25,
            this.columnHeader14,
            this.columnHeader15,
            this.columnHeader16,
            this.columnHeader17,
            this.columnHeader18,
            this.columnHeader19,
            this.columnHeader20,
            this.columnHeader21,
            this.columnHeader22,
            this.columnHeader23,
            this.columnHeader24});
            this.lvBricks.FullRowSelect = true;
            this.lvBricks.GridLines = true;
            this.lvBricks.HideSelection = false;
            this.lvBricks.Location = new System.Drawing.Point(12, 192);
            this.lvBricks.MultiSelect = false;
            this.lvBricks.Name = "lvBricks";
            this.lvBricks.Size = new System.Drawing.Size(605, 163);
            this.lvBricks.TabIndex = 5;
            this.lvBricks.UseCompatibleStateImageBehavior = false;
            this.lvBricks.View = System.Windows.Forms.View.Details;
            this.lvBricks.SelectedIndexChanged += new System.EventHandler(this.lvBricks_SelectedIndexChanged);
            // 
            // columnHeader25
            // 
            this.columnHeader25.Text = "ID";
            // 
            // columnHeader14
            // 
            this.columnHeader14.Text = "Brick Set";
            this.columnHeader14.Width = 120;
            // 
            // columnHeader15
            // 
            this.columnHeader15.Text = "Brick Id (within set)";
            this.columnHeader15.Width = 119;
            // 
            // columnHeader16
            // 
            this.columnHeader16.Text = "MinX";
            this.columnHeader16.Width = 100;
            // 
            // columnHeader17
            // 
            this.columnHeader17.Text = "MaxX";
            this.columnHeader17.Width = 100;
            // 
            // columnHeader18
            // 
            this.columnHeader18.Text = "MinY";
            this.columnHeader18.Width = 100;
            // 
            // columnHeader19
            // 
            this.columnHeader19.Text = "MaxY";
            // 
            // columnHeader20
            // 
            this.columnHeader20.Text = "MinZ";
            // 
            // columnHeader21
            // 
            this.columnHeader21.Text = "MaxZ";
            // 
            // columnHeader22
            // 
            this.columnHeader22.Text = "ZeroX";
            // 
            // columnHeader23
            // 
            this.columnHeader23.Text = "ZeroY";
            // 
            // columnHeader24
            // 
            this.columnHeader24.Text = "ZeroZ";
            // 
            // columnHeader7
            // 
            this.columnHeader7.Text = "Brick Set";
            this.columnHeader7.Width = 120;
            // 
            // columnHeader8
            // 
            this.columnHeader8.Text = "Tablespace Extension";
            this.columnHeader8.Width = 120;
            // 
            // columnHeader9
            // 
            this.columnHeader9.Text = "Bricks";
            // 
            // columnHeader10
            // 
            this.columnHeader10.Text = "Scan Size (GB)";
            this.columnHeader10.Width = 100;
            // 
            // columnHeader11
            // 
            this.columnHeader11.Text = "Proc Size (GB)";
            this.columnHeader11.Width = 100;
            // 
            // columnHeader12
            // 
            this.columnHeader12.Text = "Rec Size (GB)";
            this.columnHeader12.Width = 100;
            // 
            // lbAvalaibleBricksSets
            // 
            this.lbAvalaibleBricksSets.AutoSize = true;
            this.lbAvalaibleBricksSets.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lbAvalaibleBricksSets.Location = new System.Drawing.Point(12, 14);
            this.lbAvalaibleBricksSets.Name = "lbAvalaibleBricksSets";
            this.lbAvalaibleBricksSets.Size = new System.Drawing.Size(127, 13);
            this.lbAvalaibleBricksSets.TabIndex = 6;
            this.lbAvalaibleBricksSets.Text = "&Available Bricks Sets";
            // 
            // lbBricksInBrickset
            // 
            this.lbBricksInBrickset.AutoSize = true;
            this.lbBricksInBrickset.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lbBricksInBrickset.Location = new System.Drawing.Point(9, 176);
            this.lbBricksInBrickset.Name = "lbBricksInBrickset";
            this.lbBricksInBrickset.Size = new System.Drawing.Size(160, 13);
            this.lbBricksInBrickset.TabIndex = 7;
            this.lbBricksInBrickset.Text = "Bricks in selected BrickSet";
            // 
            // btnAddBrick
            // 
            this.btnAddBrick.Location = new System.Drawing.Point(633, 318);
            this.btnAddBrick.Name = "btnAddBrick";
            this.btnAddBrick.Size = new System.Drawing.Size(132, 27);
            this.btnAddBrick.TabIndex = 8;
            this.btnAddBrick.Text = "Add  Brick";
            this.btnAddBrick.UseVisualStyleBackColor = true;
            this.btnAddBrick.Click += new System.EventHandler(this.btnAddBrick_Click);
            // 
            // btnRemoveBrick
            // 
            this.btnRemoveBrick.Location = new System.Drawing.Point(633, 351);
            this.btnRemoveBrick.Name = "btnRemoveBrick";
            this.btnRemoveBrick.Size = new System.Drawing.Size(132, 27);
            this.btnRemoveBrick.TabIndex = 9;
            this.btnRemoveBrick.Text = "Remove Brick";
            this.btnRemoveBrick.UseVisualStyleBackColor = true;
            this.btnRemoveBrick.Click += new System.EventHandler(this.btnRemoveBrick_Click);
            // 
            // btnExit
            // 
            this.btnExit.Location = new System.Drawing.Point(633, 539);
            this.btnExit.Name = "btnExit";
            this.btnExit.Size = new System.Drawing.Size(132, 27);
            this.btnExit.TabIndex = 10;
            this.btnExit.Text = "Exit";
            this.btnExit.UseVisualStyleBackColor = true;
            this.btnExit.Click += new System.EventHandler(this.btnExit_Click);
            // 
            // btnAddBrickSpace
            // 
            this.btnAddBrickSpace.Location = new System.Drawing.Point(633, 107);
            this.btnAddBrickSpace.Name = "btnAddBrickSpace";
            this.btnAddBrickSpace.Size = new System.Drawing.Size(132, 27);
            this.btnAddBrickSpace.TabIndex = 11;
            this.btnAddBrickSpace.Text = "Add Brick Space";
            this.btnAddBrickSpace.UseVisualStyleBackColor = true;
            this.btnAddBrickSpace.Click += new System.EventHandler(this.btnAddBrickSpace_Click);
            // 
            // btnRemoveBrickSpace
            // 
            this.btnRemoveBrickSpace.Location = new System.Drawing.Point(633, 140);
            this.btnRemoveBrickSpace.Name = "btnRemoveBrickSpace";
            this.btnRemoveBrickSpace.Size = new System.Drawing.Size(132, 27);
            this.btnRemoveBrickSpace.TabIndex = 12;
            this.btnRemoveBrickSpace.Text = "Remove Brick Space";
            this.btnRemoveBrickSpace.UseVisualStyleBackColor = true;
            this.btnRemoveBrickSpace.Click += new System.EventHandler(this.btnRemoveBrickSpace_Click);
            // 
            // btnShowAllBrick
            // 
            this.btnShowAllBrick.Location = new System.Drawing.Point(633, 281);
            this.btnShowAllBrick.Name = "btnShowAllBrick";
            this.btnShowAllBrick.Size = new System.Drawing.Size(132, 27);
            this.btnShowAllBrick.TabIndex = 13;
            this.btnShowAllBrick.Text = "Show All Bricks";
            this.btnShowAllBrick.UseVisualStyleBackColor = true;
            this.btnShowAllBrick.Click += new System.EventHandler(this.btnShowAllBrick_Click);
            // 
            // lvMarkSet
            // 
            this.lvMarkSet.AllowColumnReorder = true;
            this.lvMarkSet.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.columnHeader13,
            this.columnHeader26,
            this.columnHeader27,
            this.columnHeader28,
            this.columnHeader29,
            this.columnHeader30,
            this.columnHeader31,
            this.columnHeader32,
            this.columnHeader33});
            this.lvMarkSet.FullRowSelect = true;
            this.lvMarkSet.GridLines = true;
            this.lvMarkSet.HideSelection = false;
            this.lvMarkSet.Location = new System.Drawing.Point(12, 405);
            this.lvMarkSet.MultiSelect = false;
            this.lvMarkSet.Name = "lvMarkSet";
            this.lvMarkSet.Size = new System.Drawing.Size(605, 161);
            this.lvMarkSet.TabIndex = 16;
            this.lvMarkSet.UseCompatibleStateImageBehavior = false;
            this.lvMarkSet.View = System.Windows.Forms.View.Details;
            // 
            // columnHeader13
            // 
            this.columnHeader13.Text = "ID";
            // 
            // columnHeader26
            // 
            this.columnHeader26.Text = "Id_Brick";
            this.columnHeader26.Width = 120;
            // 
            // columnHeader27
            // 
            this.columnHeader27.Text = "ID Mark";
            this.columnHeader27.Width = 119;
            // 
            // columnHeader28
            // 
            this.columnHeader28.Text = "PosX";
            this.columnHeader28.Width = 100;
            // 
            // columnHeader29
            // 
            this.columnHeader29.Text = "PosY";
            this.columnHeader29.Width = 100;
            // 
            // columnHeader30
            // 
            this.columnHeader30.Text = "Mark Row";
            this.columnHeader30.Width = 100;
            // 
            // columnHeader31
            // 
            this.columnHeader31.Text = "Mark Col";
            // 
            // columnHeader32
            // 
            this.columnHeader32.Text = "Shape";
            // 
            // columnHeader33
            // 
            this.columnHeader33.Text = "Side";
            // 
            // lblMark
            // 
            this.lblMark.AutoSize = true;
            this.lblMark.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblMark.Location = new System.Drawing.Point(12, 380);
            this.lblMark.Name = "lblMark";
            this.lblMark.Size = new System.Drawing.Size(205, 13);
            this.lblMark.TabIndex = 17;
            this.lblMark.Text = "Marks associated to selected brick";
            // 
            // btnInsertCSMark
            // 
            this.btnInsertCSMark.Location = new System.Drawing.Point(633, 432);
            this.btnInsertCSMark.Name = "btnInsertCSMark";
            this.btnInsertCSMark.Size = new System.Drawing.Size(132, 27);
            this.btnInsertCSMark.TabIndex = 18;
            this.btnInsertCSMark.Text = "Import CS Marks";
            this.btnInsertCSMark.UseVisualStyleBackColor = true;
            this.btnInsertCSMark.Click += new System.EventHandler(this.btnInsertCSMark_Click);
            // 
            // btnLoadMarkFromFile
            // 
            this.btnLoadMarkFromFile.Location = new System.Drawing.Point(633, 477);
            this.btnLoadMarkFromFile.Name = "btnLoadMarkFromFile";
            this.btnLoadMarkFromFile.Size = new System.Drawing.Size(132, 27);
            this.btnLoadMarkFromFile.TabIndex = 19;
            this.btnLoadMarkFromFile.Text = "Load Marks from File";
            this.btnLoadMarkFromFile.UseVisualStyleBackColor = true;
            this.btnLoadMarkFromFile.Click += new System.EventHandler(this.btnLoadMarkFromFile_Click);
            // 
            // MainForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(777, 582);
            this.Controls.Add(this.btnLoadMarkFromFile);
            this.Controls.Add(this.btnInsertCSMark);
            this.Controls.Add(this.lblMark);
            this.Controls.Add(this.lvMarkSet);
            this.Controls.Add(this.btnShowAllBrick);
            this.Controls.Add(this.btnRemoveBrickSpace);
            this.Controls.Add(this.btnAddBrickSpace);
            this.Controls.Add(this.btnExit);
            this.Controls.Add(this.btnRemoveBrick);
            this.Controls.Add(this.btnAddBrick);
            this.Controls.Add(this.lbBricksInBrickset);
            this.Controls.Add(this.lbAvalaibleBricksSets);
            this.Controls.Add(this.btnRefreshBrickSets);
            this.Controls.Add(this.btnRemoveBrickSet);
            this.Controls.Add(this.lvBricks);
            this.Controls.Add(this.lvBrickSets);
            this.Controls.Add(this.btnAddBrickSet);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.Fixed3D;
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.Name = "MainForm";
            this.Text = "Opera Brick Manager";
            this.Load += new System.EventHandler(this.OnLoad);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.ListView lvBrickSets;
        private System.Windows.Forms.ColumnHeader columnHeader1;
        private System.Windows.Forms.ColumnHeader columnHeader2;
        private System.Windows.Forms.ColumnHeader columnHeader3;
        private System.Windows.Forms.ColumnHeader columnHeader4;
        private System.Windows.Forms.ColumnHeader columnHeader5;
        private System.Windows.Forms.ColumnHeader columnHeader6;
        private System.Windows.Forms.Button btnRefreshBrickSets;
        private System.Windows.Forms.Button btnAddBrickSet;
        private System.Windows.Forms.Button btnRemoveBrickSet;
        private System.Windows.Forms.ListView lvBricks;
        private System.Windows.Forms.ColumnHeader columnHeader14;
        private System.Windows.Forms.ColumnHeader columnHeader15;
        private System.Windows.Forms.ColumnHeader columnHeader16;
        private System.Windows.Forms.ColumnHeader columnHeader17;
        private System.Windows.Forms.ColumnHeader columnHeader18;
        private System.Windows.Forms.ColumnHeader columnHeader19;
        private System.Windows.Forms.ColumnHeader columnHeader20;
        private System.Windows.Forms.ColumnHeader columnHeader21;
        private System.Windows.Forms.ColumnHeader columnHeader22;
        private System.Windows.Forms.ColumnHeader columnHeader23;
        private System.Windows.Forms.ColumnHeader columnHeader7;
        private System.Windows.Forms.ColumnHeader columnHeader8;
        private System.Windows.Forms.ColumnHeader columnHeader9;
        private System.Windows.Forms.ColumnHeader columnHeader10;
        private System.Windows.Forms.ColumnHeader columnHeader11;
        private System.Windows.Forms.ColumnHeader columnHeader12;
        private System.Windows.Forms.ColumnHeader columnHeader24;
        private System.Windows.Forms.Label lbAvalaibleBricksSets;
        private System.Windows.Forms.Label lbBricksInBrickset;
        private System.Windows.Forms.Button btnAddBrick;
        private System.Windows.Forms.Button btnRemoveBrick;
        private System.Windows.Forms.Button btnExit;
        private System.Windows.Forms.ColumnHeader columnHeader25;
        private System.Windows.Forms.Button btnAddBrickSpace;
        private System.Windows.Forms.Button btnRemoveBrickSpace;
        private System.Windows.Forms.Button btnShowAllBrick;
        private System.Windows.Forms.ListView lvMarkSet;
        private System.Windows.Forms.ColumnHeader columnHeader13;
        private System.Windows.Forms.ColumnHeader columnHeader26;
        private System.Windows.Forms.ColumnHeader columnHeader27;
        private System.Windows.Forms.ColumnHeader columnHeader28;
        private System.Windows.Forms.ColumnHeader columnHeader29;
        private System.Windows.Forms.ColumnHeader columnHeader30;
        private System.Windows.Forms.ColumnHeader columnHeader31;
        private System.Windows.Forms.ColumnHeader columnHeader32;
        private System.Windows.Forms.ColumnHeader columnHeader33;
        private System.Windows.Forms.Label lblMark;
        private System.Windows.Forms.Button btnInsertCSMark;
        private System.Windows.Forms.Button btnLoadMarkFromFile;
    }
}

