namespace SySal.Executables.QuickDataCheck
{
    partial class ViewAsTracksDataSelector
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
            System.Windows.Forms.ListViewItem listViewItem11 = new System.Windows.Forms.ListViewItem(new string[] {
            "Xstart",
            ""}, -1);
            System.Windows.Forms.ListViewItem listViewItem12 = new System.Windows.Forms.ListViewItem(new string[] {
            "Ystart",
            ""}, -1);
            System.Windows.Forms.ListViewItem listViewItem13 = new System.Windows.Forms.ListViewItem(new string[] {
            "Zstart",
            ""}, -1);
            System.Windows.Forms.ListViewItem listViewItem14 = new System.Windows.Forms.ListViewItem(new string[] {
            "Xslope",
            ""}, -1);
            System.Windows.Forms.ListViewItem listViewItem15 = new System.Windows.Forms.ListViewItem(new string[] {
            "Yslope",
            ""}, -1);
            System.Windows.Forms.ListViewItem listViewItem16 = new System.Windows.Forms.ListViewItem(new string[] {
            "Length",
            ""}, -1);
            System.Windows.Forms.ListViewItem listViewItem17 = new System.Windows.Forms.ListViewItem(new string[] {
            "Xend",
            ""}, -1);
            System.Windows.Forms.ListViewItem listViewItem18 = new System.Windows.Forms.ListViewItem(new string[] {
            "Yend",
            ""}, -1);
            System.Windows.Forms.ListViewItem listViewItem19 = new System.Windows.Forms.ListViewItem(new string[] {
            "Zend",
            ""}, -1);
            System.Windows.Forms.ListViewItem listViewItem20 = new System.Windows.Forms.ListViewItem(new string[] {
            "Hue",
            ""}, -1);
            this.listVars = new System.Windows.Forms.ListView();
            this.columnHeader1 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader2 = new System.Windows.Forms.ColumnHeader();
            this.comboVars = new System.Windows.Forms.ComboBox();
            this.buttonSetVar = new System.Windows.Forms.Button();
            this.buttonRemove = new System.Windows.Forms.Button();
            this.label1 = new System.Windows.Forms.Label();
            this.textConstant = new System.Windows.Forms.TextBox();
            this.buttonSetConst = new System.Windows.Forms.Button();
            this.buttonOk = new System.Windows.Forms.Button();
            this.buttonCancel = new System.Windows.Forms.Button();
            this.SuspendLayout();
            // 
            // listVars
            // 
            this.listVars.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.columnHeader1,
            this.columnHeader2});
            this.listVars.FullRowSelect = true;
            this.listVars.GridLines = true;
            this.listVars.HideSelection = false;
            this.listVars.Items.AddRange(new System.Windows.Forms.ListViewItem[] {
            listViewItem11,
            listViewItem12,
            listViewItem13,
            listViewItem14,
            listViewItem15,
            listViewItem16,
            listViewItem17,
            listViewItem18,
            listViewItem19,
            listViewItem20});
            this.listVars.Location = new System.Drawing.Point(12, 12);
            this.listVars.Name = "listVars";
            this.listVars.Size = new System.Drawing.Size(272, 200);
            this.listVars.TabIndex = 0;
            this.listVars.UseCompatibleStateImageBehavior = false;
            this.listVars.View = System.Windows.Forms.View.Details;
            // 
            // columnHeader1
            // 
            this.columnHeader1.Text = "Track property";
            this.columnHeader1.Width = 120;
            // 
            // columnHeader2
            // 
            this.columnHeader2.Text = "Data Variable";
            this.columnHeader2.Width = 147;
            // 
            // comboVars
            // 
            this.comboVars.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.comboVars.FormattingEnabled = true;
            this.comboVars.Location = new System.Drawing.Point(304, 14);
            this.comboVars.Name = "comboVars";
            this.comboVars.Size = new System.Drawing.Size(162, 21);
            this.comboVars.TabIndex = 1;
            // 
            // buttonSetVar
            // 
            this.buttonSetVar.Location = new System.Drawing.Point(304, 54);
            this.buttonSetVar.Name = "buttonSetVar";
            this.buttonSetVar.Size = new System.Drawing.Size(162, 26);
            this.buttonSetVar.TabIndex = 2;
            this.buttonSetVar.Text = "Set selected variable";
            this.buttonSetVar.UseVisualStyleBackColor = true;
            this.buttonSetVar.Click += new System.EventHandler(this.buttonSetVar_Click);
            // 
            // buttonRemove
            // 
            this.buttonRemove.Location = new System.Drawing.Point(304, 186);
            this.buttonRemove.Name = "buttonRemove";
            this.buttonRemove.Size = new System.Drawing.Size(162, 26);
            this.buttonRemove.TabIndex = 3;
            this.buttonRemove.Text = "Remove selected";
            this.buttonRemove.UseVisualStyleBackColor = true;
            this.buttonRemove.Click += new System.EventHandler(this.buttonRemove_Click);
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(307, 133);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(49, 13);
            this.label1.TabIndex = 4;
            this.label1.Text = "Constant";
            // 
            // textConstant
            // 
            this.textConstant.Location = new System.Drawing.Point(372, 131);
            this.textConstant.Name = "textConstant";
            this.textConstant.Size = new System.Drawing.Size(94, 20);
            this.textConstant.TabIndex = 5;
            // 
            // buttonSetConst
            // 
            this.buttonSetConst.Location = new System.Drawing.Point(304, 99);
            this.buttonSetConst.Name = "buttonSetConst";
            this.buttonSetConst.Size = new System.Drawing.Size(162, 26);
            this.buttonSetConst.TabIndex = 6;
            this.buttonSetConst.Text = "Set constant";
            this.buttonSetConst.UseVisualStyleBackColor = true;
            this.buttonSetConst.Click += new System.EventHandler(this.buttonSetConst_Click);
            // 
            // buttonOk
            // 
            this.buttonOk.Location = new System.Drawing.Point(12, 226);
            this.buttonOk.Name = "buttonOk";
            this.buttonOk.Size = new System.Drawing.Size(110, 26);
            this.buttonOk.TabIndex = 7;
            this.buttonOk.Text = "Ok";
            this.buttonOk.UseVisualStyleBackColor = true;
            this.buttonOk.Click += new System.EventHandler(this.buttonOk_Click);
            // 
            // buttonCancel
            // 
            this.buttonCancel.Location = new System.Drawing.Point(356, 226);
            this.buttonCancel.Name = "buttonCancel";
            this.buttonCancel.Size = new System.Drawing.Size(110, 26);
            this.buttonCancel.TabIndex = 8;
            this.buttonCancel.Text = "Cancel";
            this.buttonCancel.UseVisualStyleBackColor = true;
            this.buttonCancel.Click += new System.EventHandler(this.buttonCancel_Click);
            // 
            // ViewAsTracksDataSelector
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(475, 264);
            this.Controls.Add(this.buttonCancel);
            this.Controls.Add(this.buttonOk);
            this.Controls.Add(this.buttonSetConst);
            this.Controls.Add(this.textConstant);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.buttonRemove);
            this.Controls.Add(this.buttonSetVar);
            this.Controls.Add(this.comboVars);
            this.Controls.Add(this.listVars);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
            this.Name = "ViewAsTracksDataSelector";
            this.Text = "Select data to view as tracks";
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.ListView listVars;
        private System.Windows.Forms.ColumnHeader columnHeader1;
        private System.Windows.Forms.ColumnHeader columnHeader2;
        private System.Windows.Forms.ComboBox comboVars;
        private System.Windows.Forms.Button buttonSetVar;
        private System.Windows.Forms.Button buttonRemove;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TextBox textConstant;
        private System.Windows.Forms.Button buttonSetConst;
        private System.Windows.Forms.Button buttonOk;
        private System.Windows.Forms.Button buttonCancel;
    }
}