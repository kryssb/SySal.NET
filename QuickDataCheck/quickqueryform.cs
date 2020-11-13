using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;

namespace SySal.Executables.QuickDataCheck
{
	/// <summary>
	/// Query form to retrieve datasets from DBs.
	/// </summary>
	/// <remarks>
	/// <para>The form allows insertion of an SQL query text.</para>
	/// <para>The dataset generated can be given a custom name for identification.</para>
	/// </remarks>
	public class QuickQueryForm : System.Windows.Forms.Form
	{
		private System.Windows.Forms.Label label1;
		private System.Windows.Forms.TextBox txtQQ;
		private System.Windows.Forms.Button cmdOK;
		private System.Windows.Forms.Button cmdCancel;
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;
		private System.Windows.Forms.Label label2;
		private System.Windows.Forms.TextBox txtDN;

        public SySal.OperaDb.OperaDbConnection dbConn;
		public string QueryText;
        internal RichTextBox rtSQL;
        private Button cmdPreview;
		public string DataSetName;
		public QuickQueryForm()
		{
			//
			// Required for Windows Form Designer support
			//
			InitializeComponent();

			//
			// TODO: Add any constructor code after InitializeComponent call
			//
		}

		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		protected override void Dispose( bool disposing )
		{
			if( disposing )
			{
				if(components != null)
				{
					components.Dispose();
				}
			}
			base.Dispose( disposing );
		}

		#region Windows Form Designer generated code
		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		private void InitializeComponent()
		{
            this.label1 = new System.Windows.Forms.Label();
            this.txtQQ = new System.Windows.Forms.TextBox();
            this.cmdOK = new System.Windows.Forms.Button();
            this.cmdCancel = new System.Windows.Forms.Button();
            this.label2 = new System.Windows.Forms.Label();
            this.txtDN = new System.Windows.Forms.TextBox();
            this.rtSQL = new System.Windows.Forms.RichTextBox();
            this.cmdPreview = new System.Windows.Forms.Button();
            this.SuspendLayout();
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(8, 40);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(93, 13);
            this.label1.TabIndex = 0;
            this.label1.Text = "Quick Query Text:";
            // 
            // txtQQ
            // 
            this.txtQQ.Location = new System.Drawing.Point(8, 56);
            this.txtQQ.Multiline = true;
            this.txtQQ.Name = "txtQQ";
            this.txtQQ.ScrollBars = System.Windows.Forms.ScrollBars.Both;
            this.txtQQ.Size = new System.Drawing.Size(576, 120);
            this.txtQQ.TabIndex = 1;
            // 
            // cmdOK
            // 
            this.cmdOK.DialogResult = System.Windows.Forms.DialogResult.OK;
            this.cmdOK.Location = new System.Drawing.Point(8, 184);
            this.cmdOK.Name = "cmdOK";
            this.cmdOK.Size = new System.Drawing.Size(80, 24);
            this.cmdOK.TabIndex = 2;
            this.cmdOK.Text = "OK";
            this.cmdOK.Click += new System.EventHandler(this.cmdOK_Click);
            // 
            // cmdCancel
            // 
            this.cmdCancel.DialogResult = System.Windows.Forms.DialogResult.Cancel;
            this.cmdCancel.Location = new System.Drawing.Point(496, 184);
            this.cmdCancel.Name = "cmdCancel";
            this.cmdCancel.Size = new System.Drawing.Size(88, 24);
            this.cmdCancel.TabIndex = 3;
            this.cmdCancel.Text = "Cancel";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(8, 8);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(78, 13);
            this.label2.TabIndex = 4;
            this.label2.Text = "Dataset Name:";
            // 
            // txtDN
            // 
            this.txtDN.Location = new System.Drawing.Point(112, 8);
            this.txtDN.Name = "txtDN";
            this.txtDN.Size = new System.Drawing.Size(200, 20);
            this.txtDN.TabIndex = 5;
            // 
            // rtSQL
            // 
            this.rtSQL.Font = new System.Drawing.Font("Courier New", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.rtSQL.Location = new System.Drawing.Point(8, 214);
            this.rtSQL.Name = "rtSQL";
            this.rtSQL.ReadOnly = true;
            this.rtSQL.Size = new System.Drawing.Size(572, 218);
            this.rtSQL.TabIndex = 6;
            this.rtSQL.Text = "";
            this.rtSQL.WordWrap = false;
            // 
            // cmdPreview
            // 
            this.cmdPreview.Location = new System.Drawing.Point(410, 184);
            this.cmdPreview.Name = "cmdPreview";
            this.cmdPreview.Size = new System.Drawing.Size(80, 24);
            this.cmdPreview.TabIndex = 7;
            this.cmdPreview.Text = "Preview";
            this.cmdPreview.Click += new System.EventHandler(this.cmdPreview_Click);
            // 
            // QuickQueryForm
            // 
            this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
            this.ClientSize = new System.Drawing.Size(592, 444);
            this.Controls.Add(this.cmdPreview);
            this.Controls.Add(this.rtSQL);
            this.Controls.Add(this.txtDN);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.cmdCancel);
            this.Controls.Add(this.cmdOK);
            this.Controls.Add(this.txtQQ);
            this.Controls.Add(this.label1);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedToolWindow;
            this.Name = "QuickQueryForm";
            this.Text = "Quick Query";
            this.ResumeLayout(false);
            this.PerformLayout();

		}
		#endregion

		private void cmdOK_Click(object sender, System.EventArgs e)
		{
			QueryText = txtQQ.Text;
			DataSetName = txtDN.Text;
			DialogResult = DialogResult.OK;
			Close();
		}

        private void cmdPreview_Click(object sender, EventArgs e)
        {
            try
            {
                dbConn.Open();
                System.Data.DataSet ds = new System.Data.DataSet();
                new SySal.OperaDb.OperaDbDataAdapter(txtQQ.Text, dbConn).Fill(ds);
                int ncol = ds.Tables[0].Columns.Count;
                int i;
                string text = "";
                for (i = 0; i < ncol; i++)
                    if (i == 0) text += ds.Tables[0].Columns[0].ColumnName;
                    else text += "\t" + ds.Tables[0].Columns[i].ColumnName;
                foreach (System.Data.DataRow dr in ds.Tables[0].Rows)
                {
                    text += "\r\n";
                    for (i = 0; i < ncol; i++)
                    {
                        if (i > 0) text += "\t";
                        text += dr[i].ToString();
                    }
                }
                text += "\r\n\r\n" + ds.Tables[0].Rows.Count + " row(s) returned.";
                rtSQL.Text = text;
            }
            catch (Exception x)
            {
                rtSQL.Text = x.ToString();
            }
        }
	}
}
